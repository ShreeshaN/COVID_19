# -*- coding: utf-8 -*-
"""
@created on: 02/16/20,
@author: Shreesha N,
@version: v0.0.1
@system name: badgod
Description:

..todo::

"""
import json
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from covid_19.networks.convnet import ConvNet
from covid_19.utils import file_utils
from covid_19.utils.data_utils import read_pkl
from covid_19.utils.logger import Logger
from covid_19.utils.network_utils import accuracy_fn, log_summary, custom_confusion_matrix, \
    log_conf_matrix, write_to_npy, to_tensor, to_numpy, log_learnable_parameter

import wandb as wnb

# setting seed
torch.manual_seed(1234)
np.random.seed(1234)


class ConvNetRunner:
    def __init__(self, args, train_file, test_file):
        args['train_file'] = train_file
        self.train_file = train_file
        self.test_file = test_file
        self.run_name = args.run_name + '_' + train_file.split('.')[0] + '_' + str(time.time()).split('.')[0]
        self.current_run_basepath = args.network_metrics_basepath + '/' + self.run_name + '/'
        self.learning_rate = args.learning_rate
        self.epochs = args.epochs
        self.test_net = args.test_net
        self.train_net = args.train_net
        self.batch_size = args.batch_size
        self.num_classes = args.num_classes
        if args.data_source == 'mit':
            self.data_read_path = args.mit_data_save_path
        elif args.data_source == 'coswara':
            self.data_read_path = args.coswara_data_save_path
        else:  # Keep coswara as default for now
            self.data_read_path = args.coswara_data_save_path
        self.is_cuda_available = torch.cuda.is_available()
        self.display_interval = args.display_interval
        self.logger = None

        self.network_metrics_basepath = args.network_metrics_basepath
        self.tensorboard_summary_path = self.current_run_basepath + args.tensorboard_summary_path
        self.network_save_path = self.current_run_basepath + args.network_save_path

        self.network_restore_path = args.network_restore_path

        self.device = torch.device("cuda" if self.is_cuda_available else "cpu")
        self.network_save_interval = args.network_save_interval
        self.normalise = args.normalise_while_training
        self.dropout = args.dropout
        self.threshold = args.threshold
        self.debug_filename = self.current_run_basepath + '/' + args.debug_filename

        paths = [self.network_save_path, self.tensorboard_summary_path]
        file_utils.create_dirs(paths)

        self.network = ConvNet().to(self.device)
        self.pos_weight = None
        self.loss_function = None
        self.learning_rate_decay = args.learning_rate_decay

        self.optimiser = optim.Adam(self.network.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimiser, gamma=self.learning_rate_decay)

        self._min, self._max = float('inf'), -float('inf')

        if self.train_net:
            wnb.init(project=args.project_name, config=args, save_code=True, name=self.run_name,
                     entity="shreeshanwnb", reinit=True)
            wnb.watch(self.network)  # , log='all', log_freq=3
            self.network.train()
            self.logger = Logger(name=self.run_name, log_path=self.network_save_path).get_logger()
            self.logger.info("********* DATA FILE: " + train_file + " *********")
            self.logger.info(str(id(self.logger)))
        if self.test_net:
            self.logger.info('Loading Network')
            self.network.load_state_dict(torch.load(self.network_restore_path, map_location=self.device))
            self.network.eval()
            self.logger.info('\n\n\n********************************************************')
            self.logger.info(f'Testing Model - {self.network_restore_path}')
            self.logger.info('********************************************************')

        self.logger.info(f"Network Architecture:\n,{self.network}")

        self.batch_loss, self.batch_accuracy, self.uar = [], [], []
        self.logger.info(f'Configs used:\n{json.dumps(args, indent=4)}')

    def data_reader(self, data_filepath, data_files, train, should_batch=True, shuffle=True,
                    infer=False):
        input_data, labels = [], []

        def split_data(combined_data):
            return combined_data[0], combined_data[1]

        if infer:
            pass
        else:
            for file in data_files:
                self.logger.info('Reading input file ' + file)
                in_data = read_pkl(data_filepath + file)
                in_data, out_data = split_data(in_data)
                input_data.extend(in_data), labels.extend(out_data)

            split_type = None
            if train:
                split_type = 'train'
                for x in input_data:
                    self._min = min(np.min(x), self._min)
                    self._max = max(np.max(x), self._max)
                self._mean, self._std = np.mean(input_data), np.std(input_data)

                data = [(x, y) for x, y in zip(input_data, labels)]
                random.shuffle(data)
                input_data, labels = np.array([x[0] for x in data]), [x[1] for x in data]

                # Initialize pos_weight based on training data
                self.pos_weight = len([x for x in labels if x == 0]) / 1 if sum(labels) == 0 else len(
                        [x for x in labels if x == 1])
                self.logger.info(f'Pos weight for the train data - {self.pos_weight}')
                wnb.config.update({'pos_weight': self.pos_weight})
            else:
                split_type = 'test'

            self.logger.info(f'Total data {str(len(input_data))}')
            wnb.config.update({split_type + '_data_len': len(input_data)})

            self.logger.info(f'Event rate {str(sum(labels) / len(labels))}')
            wnb.config.update({split_type + '_event_rate': sum(labels) / len(labels)})
            wnb.config.update(
                    {split_type + '_ones_count': sum(labels), split_type + '_zeros_count': len(labels) - sum(labels)})

            self.logger.info(
                    f'Input data shape:{np.array(input_data).shape} | Output data shape:{np.array(labels).shape}')
            wnb.config.update({split_type + '_input_data_shape': np.array(input_data).shape})

            self.logger.info(f'Min max values {self._min, self._max}')
            self.logger.info(f'Std values {self._std}')
            self.logger.info(f'Mean values {self._mean}')
            wnb.config.update({split_type + '_min_val': self._min, split_type + '_max_val': self._max,
                               split_type + '_mean': self._mean, split_type + '_std': self._std})

            # Normalizing `input data` on train dataset's min and max values
            if self.normalise:
                input_data = (input_data - self._min) / (self._max - self._min)
                input_data = (input_data - self._mean) / self._std

            if should_batch:
                batched_input = [input_data[pos:pos + self.batch_size] for pos in
                                 range(0, len(input_data), self.batch_size)]
                batched_labels = [labels[pos:pos + self.batch_size] for pos in range(0, len(labels), self.batch_size)]
                return batched_input, batched_labels
            else:
                return input_data, labels

    # def train(self):
    #     pass

    def train(self):
        train_data, train_labels = self.data_reader(self.data_read_path, [self.train_file], shuffle=True,
                                                    train=True)
        test_data, test_labels = self.data_reader(self.data_read_path, [self.test_file], shuffle=False,
                                                  train=False)

        # For the purposes of assigning pos weight on the fly we are initializing the cost function here
        self.loss_function = nn.BCEWithLogitsLoss(pos_weight=to_tensor(self.pos_weight, device=self.device))

        total_step = len(train_data)
        for epoch in range(1, self.epochs):
            self.network.train()
            self.batch_loss, self.batch_accuracy, self.batch_uar, self.batch_f1, self.batch_precision, \
            self.batch_recall, self.batch_auc, train_predictions, train_logits, audio_for_tensorboard_train = [], [], [], [], [], [], [], [], [], None
            for i, (audio_data, label) in enumerate(
                    zip(train_data, train_labels)):
                self.optimiser.zero_grad()
                label = to_tensor(label, device=self.device).float()
                audio_data = to_tensor(audio_data, device=self.device)
                predictions = self.network(audio_data).squeeze(1)
                train_logits.extend(predictions)
                loss = self.loss_function(predictions, label)
                predictions = nn.Sigmoid()(predictions)
                train_predictions.extend(predictions)
                loss.backward()
                self.optimiser.step()
                self.batch_loss.append(to_numpy(loss))

                batch_metrics = accuracy_fn(predictions, label, self.threshold)
                batch_metrics['loss'] = to_numpy(loss)
                self.batch_accuracy.append(to_numpy(batch_metrics['accuracy']))
                self.batch_uar.append(batch_metrics['uar'])
                self.batch_f1.append(batch_metrics['f1'])
                self.batch_precision.append(batch_metrics['precision'])
                self.batch_recall.append(batch_metrics['recall'])
                self.batch_auc.append(batch_metrics['auc'])

                wnb.log(batch_metrics)

                if i % self.display_interval == 0:
                    self.logger.info(
                            f"Epoch: {epoch}/{self.epochs} | Step: {i}/{total_step} | Loss: {'%.5f' % loss} | "
                            f"Accuracy: {'%.5f' % batch_metrics['accuracy']} | UAR: {'%.5f' % batch_metrics['uar']}| "
                            f"F1:{'%.5f' % batch_metrics['f1']} | Precision: {'%.5f' % batch_metrics['precision']} | "
                            f"Recall: {'%.5f' % batch_metrics['recall']} | AUC: {'%.5f' % batch_metrics['auc']}")

            # log_learnable_parameter(self.writer, epoch, to_tensor(train_logits, device=self.device),
            #                         name='train_logits')
            # log_learnable_parameter(self.writer, epoch, to_tensor(train_predictions, device=self.device),
            #                         name='train_activated')

            # Decay learning rate
            self.scheduler.step(epoch=epoch)
            wnb.log({"LR": self.optimiser.state_dict()['param_groups'][0]['lr']})
            # log_summary(self.writer, epoch, accuracy=np.mean(self.batch_accuracy),
            #             loss=np.mean(self.batch_loss),
            #             uar=np.mean(self.batch_uar), precision=np.mean(self.batch_precision),
            #             recall=np.mean(self.batch_recall),
            #             auc=np.mean(self.batch_auc), lr=self.optimiser.state_dict()['param_groups'][0]['lr'],
            #             type='Train')
            self.logger.info('***** Overall Train Metrics ***** ')
            self.logger.info(
                    f"Loss: {'%.5f' % np.mean(self.batch_loss)} | Accuracy: {'%.5f' % np.mean(self.batch_accuracy)} "
                    f"| UAR: {'%.5f' % np.mean(self.batch_uar)} | F1:{'%.5f' % np.mean(self.batch_f1)} "
                    f"| Precision:{'%.5f' % np.mean(self.batch_precision)} | Recall:{'%.5f' % np.mean(self.batch_recall)} "
                    f"| AUC:{'%.5f' % np.mean(self.batch_auc)}")

            # test data
            self.run_for_epoch(epoch, test_data, test_labels, type='Test')

            if epoch % self.network_save_interval == 0:
                save_path = self.network_save_path + '/' + self.run_name + '_' + str(epoch) + '.pt'
                torch.save(self.network.state_dict(), save_path)
                self.logger.info(f'Network successfully saved: {save_path}')

    def run_for_epoch(self, epoch, x, y, type):
        self.network.eval()
        # for m in self.network.modules():
        #     if isinstance(m, nn.BatchNorm2d):
        #         m.track_running_stats = False
        predictions_dict = {"tp": [], "fp": [], "tn": [], "fn": []}
        logits, predictions = [], []
        self.test_batch_loss, self.test_batch_accuracy, self.test_batch_uar, self.test_batch_ua, self.test_batch_f1, \
        self.test_batch_precision, self.test_batch_recall, self.test_batch_auc, audio_for_tensorboard_test = [], [], \
                                                                                                             [], [], [], [], [], [], None
        with torch.no_grad():
            for i, (audio_data, label) in enumerate(zip(x, y)):
                label = to_tensor(label, device=self.device).float()
                audio_data = to_tensor(audio_data, device=self.device)
                test_predictions = self.network(audio_data).squeeze(1)
                logits.extend(to_numpy(test_predictions))
                test_loss = self.loss_function(test_predictions, label)
                test_predictions = nn.Sigmoid()(test_predictions)
                predictions.append(to_numpy(test_predictions))
                test_batch_metrics = accuracy_fn(test_predictions, label, self.threshold)
                self.test_batch_loss.append(to_numpy(test_loss))
                self.test_batch_accuracy.append(to_numpy(test_batch_metrics['accuracy']))
                self.test_batch_uar.append(test_batch_metrics['uar'])
                self.test_batch_f1.append(test_batch_metrics['f1'])
                self.test_batch_precision.append(test_batch_metrics['precision'])
                self.test_batch_recall.append(test_batch_metrics['recall'])
                self.test_batch_auc.append(test_batch_metrics['auc'])
                tp, fp, tn, fn = custom_confusion_matrix(test_predictions, label, threshold=self.threshold)
                predictions_dict['tp'].extend(tp)
                predictions_dict['fp'].extend(fp)
                predictions_dict['tn'].extend(tn)
                predictions_dict['fn'].extend(fn)

        predictions = [element for sublist in predictions for element in sublist]
        self.logger.info(f'***** {type} Metrics ***** ')
        self.logger.info(
                f"Loss: {'%.5f' % np.mean(self.test_batch_loss)} | Accuracy: {'%.5f' % np.mean(self.test_batch_accuracy)} "
                f"| UAR: {'%.5f' % np.mean(self.test_batch_uar)}| F1:{'%.5f' % np.mean(self.test_batch_f1)} "
                f"| Precision:{'%.5f' % np.mean(self.test_batch_precision)} "
                f"| Recall:{'%.5f' % np.mean(self.test_batch_recall)} | AUC:{'%.5f' % np.mean(self.test_batch_auc)}")
        epoch_test_batch_metrics = {"test_loss": np.mean(self.test_batch_loss),
                                    "test_accuracy": np.mean(self.test_batch_accuracy),
                                    "test_uar": np.mean(self.test_batch_uar),
                                    "test_f1": np.mean(self.test_batch_f1),
                                    "test_precision": np.mean(self.test_batch_precision),
                                    "test_recall": np.mean(self.test_batch_recall),
                                    "test_auc": np.mean(self.test_batch_auc)}
        wnb.log(epoch_test_batch_metrics)
        wnb.log({"test_cf": wnb.sklearn.plot_confusion_matrix(y_true=[label for sublist in y for label in sublist],
                                                              y_pred=np.where(predictions > self.threshold, 1, 0),
                                                              labels=['Negative', 'Positive'])})
        # log_summary(self.writer, epoch, accuracy=np.mean(self.test_batch_accuracy),
        #             loss=np.mean(self.test_batch_loss),
        #             uar=np.mean(self.test_batch_uar), precision=np.mean(self.test_batch_precision),
        #             recall=np.mean(self.test_batch_recall), auc=np.mean(self.test_batch_auc),
        #             lr=self.optimiser.state_dict()['param_groups'][0]['lr'],
        #             type=type)
        # log_conf_matrix(self.writer, epoch, predictions_dict=predictions_dict, type=type)
        #
        # log_learnable_parameter(self.writer, epoch, to_tensor(logits, device=self.device),
        #                         name=f'{type}_logits')
        # log_learnable_parameter(self.writer, epoch, to_tensor(predictions, device=self.device),
        #                         name=f'{type}_predictions')

        write_to_npy(filename=self.debug_filename, predictions=predictions, labels=y, epoch=epoch, accuracy=np.mean(
                self.test_batch_accuracy), loss=np.mean(self.test_batch_loss), uar=np.mean(self.test_batch_uar),
                     precision=np.mean(self.test_batch_precision),
                     recall=np.mean(self.test_batch_recall), auc=np.mean(self.test_batch_auc),
                     lr=self.optimiser.state_dict()['param_groups'][0]['lr'], predictions_dict=predictions_dict,
                     type=type)

    def test(self):
        test_data, test_labels = self.data_reader(self.data_read_path + 'test_challenge_data.npy',
                                                  self.data_read_path + 'test_challenge_labels.npy',
                                                  shuffle=False, train=False)
        test_predictions = self.network(test_data).squeeze(1)
        test_predictions = nn.Sigmoid()(test_predictions)
        test_accuracy, test_uar = accuracy_fn(test_predictions, test_labels, self.threshold)
        self.logger.info(f"Accuracy: {test_accuracy} | UAR: {test_uar}")
        self.logger.info(f"Accuracy: {test_accuracy} | UAR: {test_uar}")

    def infer(self, data_file):
        test_data = self.data_reader(data_file, shuffle=False, train=False, infer=True)
        test_predictions = self.network(test_data).squeeze(1)
        test_predictions = nn.Sigmoid()(test_predictions)
        return test_predictions
