# -*- coding: utf-8 -*-
"""
@created on: 3/1/21,
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

from covid_19.networks.conv_ae import ConvAutoEncoder
from covid_19.utils import file_utils
from covid_19.utils.data_utils import read_pkl
from covid_19.utils.logger import Logger
from covid_19.utils.network_utils import accuracy_fn, log_summary, custom_confusion_matrix, \
    log_conf_matrix, write_to_npy, to_tensor, to_numpy, log_learnable_parameter
import pickle

import wandb as wnb
from sklearn import svm

# setting seed
torch.manual_seed(1234)
np.random.seed(1234)


class ConvAutoencoderRunner:
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

        self.network = ConvAutoEncoder().to(self.device)
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

    def mask_preds_for_one_class(self, predictions):
        # 1 --> inliers, -1 --> outliers
        # in our case, inliers are non covid samples. i.e label 0.
        # outliers are covid samples. i.e label 1.
        return [1 if x == -1 else 0 for x in predictions]

    def train(self):

        train_data, train_labels = self.data_reader(self.data_read_path, [self.train_file], shuffle=True,
                                                    train=True)
        test_data, test_labels = self.data_reader(self.data_read_path, [self.test_file], shuffle=False,
                                                  train=False)

        # For the purposes of assigning pos weight on the fly we are initializing the cost function here
        # self.loss_function = nn.BCEWithLogitsLoss(pos_weight=to_tensor(self.pos_weight, device=self.device))
        self.loss_function = nn.MSELoss()

        for epoch in range(1, self.epochs):
            self.oneclass_svm = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
            self.network.train()
            self.latent_features, train_predictions = [], []

            for i, (audio_data, label) in enumerate(
                    zip(train_data, train_labels)):
                self.optimiser.zero_grad()

                audio_data = to_tensor(audio_data, device=self.device)
                predictions, latent = self.network(audio_data)
                predictions = predictions.squeeze(1)
                self.latent_features.extend(to_numpy(latent.squeeze(1)))
                loss = self.loss_function(predictions, audio_data)
                loss.backward()
                self.optimiser.step()
                self.batch_loss.append(to_numpy(loss))

            oneclass_predictions = self.oneclass_svm.fit_predict(self.latent_features)
            masked_predictions = self.mask_preds_for_one_class(oneclass_predictions)
            train_metrics = accuracy_fn(to_tensor(masked_predictions),
                                        to_tensor([element for sublist in train_labels for element in sublist]),
                                        threshold=self.threshold)
            wnb.log(train_metrics)
            wnb.log({'reconstruction_loss': np.mean(self.batch_loss)})

            # Decay learning rate
            self.scheduler.step(epoch=epoch)
            wnb.log({"LR": self.optimiser.state_dict()['param_groups'][0]['lr']})

            self.logger.info('***** Overall Train Metrics ***** ')
            self.logger.info(
                    f"Loss: {'%.5f' % np.mean(self.batch_loss)} | Accuracy: {'%.5f' % train_metrics['accuracy']} "
                    f"| UAR: {'%.5f' % train_metrics['uar']} | F1:{'%.5f' % train_metrics['f1']} "
                    f"| Precision:{'%.5f' % train_metrics['precision']} | Recall:{'%.5f' % train_metrics['recall']} "
                    f"| AUC:{'%.5f' % train_metrics['auc']}")

            # test data
            self.run_for_epoch(epoch, test_data, test_labels, type='Test')

            self.oneclass_svm = None  # Clearing the model for every epoch

            if epoch % self.network_save_interval == 0:
                save_path = self.network_save_path + '/' + self.run_name + '_' + str(epoch) + '.pt'
                save_path_oneclass_svm = self.network_save_path + '/oneclass_svm' + str(epoch) + '.pkl'
                torch.save(self.network.state_dict(), save_path)
                pickle.dump(self.oneclass_svm, open(save_path_oneclass_svm, 'wb'))
                self.logger.info(f'Network successfully saved: {save_path}')

    def run_for_epoch(self, epoch, x, y, type):
        self.network.eval()
        # for m in self.network.modules():
        #     if isinstance(m, nn.BatchNorm2d):
        #         m.track_running_stats = False
        self.test_batch_loss, predictions, latent_features = [], [], []

        with torch.no_grad():
            for i, (audio_data, label) in enumerate(zip(x, y)):
                audio_data = to_tensor(audio_data, device=self.device)
                test_predictions, test_latent = self.network(audio_data)
                test_predictions = test_predictions.squeeze(1)
                latent_features.extend(to_numpy(test_latent.squeeze(1)))
                test_loss = self.loss_function(test_predictions, audio_data)
                self.test_batch_loss.append(to_numpy(test_loss))

        oneclass_predictions = self.oneclass_svm.predict(latent_features)
        masked_predictions = self.mask_preds_for_one_class(oneclass_predictions)
        test_metrics = accuracy_fn(to_tensor(masked_predictions),
                                   to_tensor([element for sublist in y for element in sublist]),
                                   threshold=self.threshold)
        test_metrics = {'test_' + k: v for k, v in test_metrics.items()}
        self.logger.info(f'***** {type} Metrics ***** ')
        self.logger.info(
                f"Loss: {'%.5f' % np.mean(self.test_batch_loss)} | Accuracy: {'%.5f' % test_metrics['test_accuracy']} "
                f"| UAR: {'%.5f' % test_metrics['test_uar']}| F1:{'%.5f' % test_metrics['test_f1']} "
                f"| Precision:{'%.5f' % test_metrics['test_precision']} "
                f"| Recall:{'%.5f' % test_metrics['test_recall']} | AUC:{'%.5f' % test_metrics['test_auc']}")
        wnb.log(test_metrics)

        write_to_npy(filename=self.debug_filename, predictions=predictions, labels=y, epoch=epoch,
                     accuracy=test_metrics['test_accuracy'], loss=np.mean(self.test_batch_loss),
                     uar=test_metrics['test_auc'],
                     precision=test_metrics['test_precision'],
                     recall=test_metrics['test_recall'], auc=test_metrics['test_auc'],
                     lr=self.optimiser.state_dict()['param_groups'][0]['lr'],
                     type=type)
        if epoch + 1 == self.epochs:
            wnb.log({"test_cf": wnb.sklearn.plot_confusion_matrix(y_true=[label for sublist in y for label in sublist],
                                                                  y_pred=masked_predictions,
                                                                  labels=['Negative', 'Positive'])})

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
