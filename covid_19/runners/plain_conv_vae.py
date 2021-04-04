# -*- coding: utf-8 -*-
"""
@created on: 4/3/21,
@author: Shreesha N,
@version: v0.0.1
@system name: badgod
Description:

..todo::

"""
import librosa
import librosa.display
import matplotlib.pyplot as plt
import cv2

import json
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from covid_19.networks.conv_vae import ConvVariationalAutoEncoder
from covid_19.utils import file_utils
from covid_19.utils.data_utils import read_pkl
from covid_19.utils.logger import Logger
from covid_19.utils.network_utils import to_tensor, to_numpy, accuracy_fn

import wandb as wnb

# setting seed
torch.manual_seed(1234)
np.random.seed(1234)


class PlainConvVariationalAutoencoderRunner:
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
        self.randomize_data = args.randomize_data
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

        self.network = ConvVariationalAutoEncoder().to(self.device)
        self.loss_function = nn.MSELoss(reduction='none')
        self.learning_rate_decay = args.learning_rate_decay

        self.optimiser = optim.Adam(self.network.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimiser, gamma=self.learning_rate_decay)

        self._min, self._max = float('inf'), -float('inf')

        if self.train_net:
            wnb.init(project=args.project_name, config=args, save_code=True, name=self.run_name,
                     entity="shreeshanwnb", reinit=True, tags=args.wnb_tag)  # , mode='disabled'
            wnb.watch(self.network)  # , log='all', log_freq=3
            self.network.train()
            self.logger = Logger(name=self.run_name, log_path=self.network_save_path).get_logger()
            self.logger.info("********* DATA FILE: " + train_file + " *********")
            self.logger.info(str(id(self.logger)))
        if self.test_net:
            wnb.init(project=args.project_name, config=args, save_code=True, name=self.run_name,
                     entity="shreeshanwnb", reinit=True, tags=args.wnb_tag, mode='disabled')  #
            wnb.watch(self.network)  # , log='all', log_freq=3
            self.logger = Logger(name=self.run_name, log_path=self.network_save_path).get_logger()
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
                    infer=False, only_negative_samples=True):
        input_data, labels = [], []

        def split_data_only_negative(combined_data):
            # pass only negative samples
            idx = [e for e, x in enumerate(combined_data[1]) if x == 0]  #
            return np.array(combined_data[0])[[idx]], np.array(combined_data[1])[[idx]]

        def split_data(combined_data):
            return np.array(combined_data[0]), np.array(combined_data[1])

        if infer:
            for file in data_files:
                self.logger.info('Reading input file ' + file)
                in_data = read_pkl(data_filepath + file)
                if only_negative_samples:
                    in_data, out_data = split_data_only_negative(in_data)
                else:
                    in_data, out_data = split_data(in_data)
                input_data.extend(in_data), labels.extend(out_data)
        else:
            for file in data_files:
                self.logger.info('Reading input file ' + file)
                in_data = read_pkl(data_filepath + file)
                if only_negative_samples:
                    in_data, out_data = split_data_only_negative(in_data)
                else:
                    in_data, out_data = split_data(in_data)
                input_data.extend(in_data), labels.extend(out_data)

        if train:
            split_type = 'train'
            for x in input_data:
                self._min = min(np.min(x), self._min)
                self._max = max(np.max(x), self._max)
            self._mean, self._std = np.mean(input_data), np.std(input_data)

            self.logger.info('Raw train data values')
            self.logger.info(f'Raw Train data Min max values {self._min, self._max}')
            self.logger.info(f'Raw Train data Std values {self._std}')
            self.logger.info(f'Raw Train data Mean values {self._mean}')
            wnb.config.update(
                    {'raw_' + split_type + '_min_val': str(self._min),
                     'raw_' + split_type + '_max_val': str(self._max),
                     'raw_' + split_type + '_mean': str(self._mean),
                     'raw_' + split_type + '_std': str(self._std)})

            if self.randomize_data:
                data = [(x, y) for x, y in zip(input_data, labels)]
                random.shuffle(data)
                input_data, labels = np.array([x[0] for x in data]), [x[1] for x in data]
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

        # Normalizing `input data` on train dataset's min and max values
        if self.normalise:
            input_data = (np.array(input_data) - self._min) / (self._max - self._min)
            # input_data = (input_data - self._mean) / self._std

            self.logger.info(f'Normalized {split_type} data values')
            self.logger.info(
                    f'Normalized {split_type} data Min max values {np.min(input_data), np.max(input_data)}')
            self.logger.info(f'Normalized {split_type} data Std values {np.std(input_data)}')
            self.logger.info(f'Normalized {split_type} data Mean values {np.mean(input_data)}')
            wnb.config.update({'normalized_' + split_type + '_min_val': str(np.min(input_data)),
                               'normalized_' + split_type + '_max_val': str(np.max(input_data)),
                               'normalized_' + split_type + '_mean': str(np.mean(input_data)),
                               'normalized_' + split_type + '_std': str(np.std(input_data))})

        if should_batch:
            batched_input = [input_data[pos:pos + self.batch_size] for pos in
                             range(0, len(input_data), self.batch_size)]
            batched_labels = [labels[pos:pos + self.batch_size] for pos in range(0, len(labels), self.batch_size)]
            return batched_input, batched_labels
        else:
            return input_data, labels

    def train(self):

        train_data, train_labels = self.data_reader(self.data_read_path, [self.train_file], shuffle=True,
                                                    train=True, only_negative_samples=True)
        test_data, test_labels = self.data_reader(self.data_read_path, [self.test_file], shuffle=False,
                                                  train=False, only_negative_samples=True)

        # Temporary analysis purpose
        self.flat_train_data = [element for sublist in train_data for element in sublist]
        self.flat_train_labels = [element for sublist in train_labels for element in sublist]

        self.flat_test_data = [element for sublist in test_data for element in sublist]
        self.flat_test_labels = [element for sublist in test_labels for element in sublist]

        for epoch in range(1, self.epochs):
            self.network.train()
            self.latent_features, train_reconstructed, train_losses, self.batch_loss, self.batch_kld = [], [], [], [], []

            for i, (audio_data, label) in enumerate(
                    zip(train_data, train_labels)):
                self.optimiser.zero_grad()

                audio_data = to_tensor(audio_data, device=self.device)
                predictions, mu, log_var = self.network(audio_data)
                predictions = predictions.squeeze(1)
                train_reconstructed.extend(to_numpy(predictions))
                squared_loss = self.loss_function(predictions, audio_data)
                kld_loss = torch.mean(0.5 * torch.sum(log_var.exp() - log_var - 1 + mu.pow(2)))
                mse_loss = torch.mean(squared_loss, dim=[1, 2])  # Loss per sample in the batch
                train_losses.extend(to_numpy(mse_loss))
                torch.mean(mse_loss).add(kld_loss).backward()
                self.optimiser.step()
                self.batch_loss.append(to_numpy(torch.mean(mse_loss)))
                self.batch_kld.append(to_numpy(kld_loss))

            self.logger.info('***** Overall Train Metrics ***** ')
            self.logger.info(
                    f"Epoch: {epoch} | Loss: {'%.5f' % np.mean(self.batch_loss)} |"
                    f" KLD: {'%.5f' % np.mean(self.batch_kld)}")

            wnb.log({'train_reconstruction_loss': np.mean(self.batch_loss)})
            wnb.log({'train_kld': np.mean(self.batch_kld)})

            # test data
            self.run_for_epoch(epoch, test_data, test_labels, type='Test')

            if epoch % self.network_save_interval == 0:
                save_path = self.network_save_path + '/' + self.run_name + '_' + str(epoch) + '.pt'
                torch.save(self.network.state_dict(), save_path)
                self.logger.info(f'Network successfully saved: {save_path}')

                # Log 10 images from each class
                one_images, tr_ones_losses = self.merge(class_label=1, labels=self.flat_train_labels,
                                                        data=self.flat_train_data,
                                                        reconstructed_data=train_reconstructed, losses=train_losses)
                zero_images, tr_zero_losses = self.merge(class_label=0, labels=self.flat_train_labels,
                                                         data=self.flat_train_data,
                                                         reconstructed_data=train_reconstructed,
                                                         losses=train_losses)

                wnb.log({"Train negative samples reconstruction": [wnb.Image(img, caption=str(l)) for img, l in
                                                                   zip(zero_images, tr_zero_losses)]})
                wnb.log({"Train positive samples reconstruction": [wnb.Image(img, caption=str(l)) for img, l in
                                                                   zip(one_images, tr_ones_losses)]})

    def merge(self, class_label, labels, data, reconstructed_data, losses):
        input_idx = [i for i, x in enumerate(labels) if x == class_label][:10]
        required_losses = np.array(losses)[input_idx]
        required_input = np.array(data)[input_idx]
        required_reconstructed = np.array(reconstructed_data)[[input_idx]]
        merged_images = []
        for tr, re in zip(required_input, required_reconstructed):
            plt.figure(figsize=(3, 2))
            librosa.display.specshow(tr)
            plt.savefig('tr.jpg')

            plt.clf()
            librosa.display.specshow(re)
            plt.savefig('re.jpg')
            plt.close('all')
            merged_im = np.concatenate((cv2.cvtColor(cv2.imread('tr.jpg'), cv2.COLOR_BGR2RGB),
                                        cv2.cvtColor(cv2.imread('re.jpg'), cv2.COLOR_BGR2RGB)),
                                       axis=1)
            merged_images.append(merged_im)
        return merged_images, required_losses

    def run_for_epoch(self, epoch, x, y, type):
        self.network.eval()
        # for m in self.network.modules():
        #     if isinstance(m, nn.BatchNorm2d):
        #         m.track_running_stats = False
        self.test_batch_loss, self.test_batch_kld, test_reconstructed, latent_features, losses = [], [], [], [], []

        with torch.no_grad():
            for i, (audio_data, label) in enumerate(zip(x, y)):
                audio_data = to_tensor(audio_data, device=self.device)
                test_predictions, test_mu, test_log_var = self.network(audio_data)
                test_predictions = test_predictions.squeeze(1)
                test_reconstructed.extend(to_numpy(test_predictions))
                test_squared_loss = self.loss_function(test_predictions, audio_data)
                test_mse_loss = torch.mean(test_squared_loss, dim=[1, 2]) # Loss per sample in the batch
                test_batch_kld = torch.mean(0.5 * torch.sum(test_log_var.exp() - test_log_var - 1 + test_mu.pow(2)))
                losses.extend(to_numpy(test_mse_loss))
                self.test_batch_loss.append(to_numpy(torch.mean(test_mse_loss)))
                self.test_batch_kld.append(to_numpy(test_batch_kld))
        wnb.log({"test_reconstruction_loss": np.mean(self.test_batch_loss)})
        wnb.log({"test_kld_loss": np.mean(self.test_batch_kld)})

        self.logger.info(f'***** {type} Metrics ***** ')
        self.logger.info(
                f"Loss: {'%.5f' % np.mean(self.test_batch_loss)} |"
                f" KLD: {'%.5f' % np.mean(self.test_batch_kld)}")

        if epoch % self.network_save_interval == 0:
            one_images, one_losses = self.merge(class_label=1, labels=self.flat_test_labels, data=self.flat_test_data,
                                                reconstructed_data=test_reconstructed, losses=losses)
            zero_images, zero_losses = self.merge(class_label=0, labels=self.flat_test_labels, data=self.flat_test_data,
                                                  reconstructed_data=test_reconstructed, losses=losses)
            wnb.log({"Test positive samples reconstruction": [wnb.Image(img, caption=str(l)) for img, l in
                                                              zip(one_images, one_losses)]})
            wnb.log({"Test negative samples reconstruction": [wnb.Image(img, caption=str(l)) for img, l in
                                                              zip(zero_images, zero_losses)]})

    def mask_preds_for_one_class(self, predictions):
        # 1 --> inliers, -1 --> outliers
        # in our case, inliers are non covid samples. i.e label 0.
        # outliers are covid samples. i.e label 1.
        return [1 if x == -1 else 0 for x in predictions]

    def infer(self):
        # from sklearn import svm
        # from sklearn.metrics import confusion_matrix
        # import pickle
        # self._min, self._max = -80.0, 3.8146973e-06
        # train_data, train_labels = self.data_reader(self.data_read_path, [self.train_file],
        #                                             shuffle=False,
        #                                             train=True)
        #
        # test_data, test_labels = self.data_reader(self.data_read_path, [self.test_file],
        #                                           shuffle=False,
        #                                           train=False)
        # train_latent_features, test_latent_features = [], []
        # with torch.no_grad():
        #     for i, (audio_data, label) in enumerate(zip(train_data, train_labels)):
        #         audio_data = to_tensor(audio_data, device=self.device)
        #         _, train_latent = self.network(audio_data)
        #         train_latent_features.extend(to_numpy(train_latent.squeeze(1)))
        # pickle.dump(train_latent_features,
        #             open('forced_train_latent.npy', 'wb'))
        #
        # oneclass_svm = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
        # oneclass_svm.fit(train_latent_features)
        # oneclass_predictions = oneclass_svm.predict(train_latent_features)
        # masked_predictions = self.mask_preds_for_one_class(oneclass_predictions)
        # train_metrics = accuracy_fn(to_tensor(masked_predictions),
        #                             to_tensor([element for sublist in train_labels for element in sublist]),
        #                             threshold=self.threshold)
        # train_metrics = {'train_' + k: v for k, v in train_metrics.items()}
        # self.logger.info(f'***** Train Metrics ***** ')
        # self.logger.info(
        #         f"Accuracy: {'%.5f' % train_metrics['train_accuracy']} "
        #         f"| UAR: {'%.5f' % train_metrics['train_uar']}| F1:{'%.5f' % train_metrics['train_f1']} "
        #         f"| Precision:{'%.5f' % train_metrics['train_precision']} "
        #         f"| Recall:{'%.5f' % train_metrics['train_recall']} | AUC:{'%.5f' % train_metrics['train_auc']}")
        # self.logger.info('Train Confusion matrix - \n' + str(
        #         confusion_matrix([element for sublist in train_labels for element in sublist], masked_predictions)))
        #
        # # Test
        # with torch.no_grad():
        #     for i, (audio_data, label) in enumerate(zip(test_data, test_labels)):
        #         audio_data = to_tensor(audio_data, device=self.device)
        #         _, test_latent = self.network(audio_data)
        #         test_latent_features.extend(to_numpy(test_latent.squeeze(1)))
        # pickle.dump(test_latent_features,
        #             open('forced_test_latent.npy', 'wb'))
        #
        # oneclass_predictions = oneclass_svm.predict(test_latent_features)
        # masked_predictions = self.mask_preds_for_one_class(oneclass_predictions)
        # test_metrics = accuracy_fn(to_tensor(masked_predictions),
        #                            to_tensor([element for sublist in test_labels for element in sublist]),
        #                            threshold=self.threshold)
        # test_metrics = {'test_' + k: v for k, v in test_metrics.items()}
        # self.logger.info(f'***** Test Metrics ***** ')
        # self.logger.info(
        #         f"Accuracy: {'%.5f' % test_metrics['test_accuracy']} "
        #         f"| UAR: {'%.5f' % test_metrics['test_uar']}| F1:{'%.5f' % test_metrics['test_f1']} "
        #         f"| Precision:{'%.5f' % test_metrics['test_precision']} "
        #         f"| Recall:{'%.5f' % test_metrics['test_recall']} | AUC:{'%.5f' % test_metrics['test_auc']}")
        # self.logger.info('Test Confusion matrix - \n' + str(
        #         confusion_matrix([element for sublist in test_labels for element in sublist], masked_predictions)))

        # ------------------------------------------------------------------------------------------------------------------------
        from sklearn import svm
        from sklearn.ensemble import IsolationForest
        from sklearn.metrics import confusion_matrix
        import pickle as pk
        train_labels, test_labels = pk.load(open(
                '/Users/badgod/badgod_documents/Datasets/covid19/processed_data/coswara_train_data_fbank_cough-shallow_labels.pkl',
                'rb')), pk.load(open(
                '/Users/badgod/badgod_documents/Datasets/covid19/processed_data/coswara_test_data_fbank_cough-shallow_labels.pkl',
                'rb'))
        train_latent_features, test_latent_features = pk.load(
                open('/Users/badgod/badgod_documents/Datasets/covid19/processed_data/forced_train_latent.npy',
                     'rb')), pk.load(
                open('/Users/badgod/badgod_documents/Datasets/covid19/processed_data/forced_test_latent.npy', 'rb'))
        # for x, y in zip(train_latent_features, train_labels):
        #     if y == 0:
        #         print('Mean: ', np.mean(x), ' Std: ', np.std(x), ' | Label: ', y)
        # for x, y in zip(train_latent_features, train_labels):
        #     if y == 1:
        #         print('Mean: ', np.mean(x), ' Std: ', np.std(x), ' | Label: ', y)
        #
        # exit()
        self.logger.info(
                'Total train data len: ' + str(len(train_labels)) + ' | Positive samples: ' + str(sum(train_labels)))
        self.logger.info(
                'Total test data len: ' + str(len(test_labels)) + ' | Positive samples: ' + str(sum(test_labels)))
        # oneclass_svm = svm.OneClassSVM(kernel="rbf")
        oneclass_svm = IsolationForest(random_state=0)
        oneclass_svm.fit(train_latent_features)
        oneclass_predictions = oneclass_svm.predict(train_latent_features)
        masked_predictions = self.mask_preds_for_one_class(oneclass_predictions)
        train_metrics = accuracy_fn(to_tensor(masked_predictions), to_tensor(train_labels), threshold=self.threshold)
        train_metrics = {'train_' + k: v for k, v in train_metrics.items()}
        self.logger.info(f'***** Train Metrics ***** ')
        self.logger.info(
                f"Accuracy: {'%.5f' % train_metrics['train_accuracy']} "
                f"| UAR: {'%.5f' % train_metrics['train_uar']}| F1:{'%.5f' % train_metrics['train_f1']} "
                f"| Precision:{'%.5f' % train_metrics['train_precision']} "
                f"| Recall:{'%.5f' % train_metrics['train_recall']} | AUC:{'%.5f' % train_metrics['train_auc']}")
        self.logger.info('Train Confusion matrix - \n' + str(confusion_matrix(train_labels, masked_predictions)))
        # Test
        oneclass_predictions = oneclass_svm.predict(test_latent_features)
        masked_predictions = self.mask_preds_for_one_class(oneclass_predictions)
        test_metrics = accuracy_fn(to_tensor(masked_predictions), to_tensor(test_labels), threshold=self.threshold)
        test_metrics = {'test_' + k: v for k, v in test_metrics.items()}
        self.logger.info(f'***** Test Metrics ***** ')
        self.logger.info(
                f"Accuracy: {'%.5f' % test_metrics['test_accuracy']} "
                f"| UAR: {'%.5f' % test_metrics['test_uar']}| F1:{'%.5f' % test_metrics['test_f1']} "
                f"| Precision:{'%.5f' % test_metrics['test_precision']} "
                f"| Recall:{'%.5f' % test_metrics['test_recall']} | AUC:{'%.5f' % test_metrics['test_auc']}")
        self.logger.info('Test Confusion matrix - \n' + str(confusion_matrix(test_labels, masked_predictions)))
