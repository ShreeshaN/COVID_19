# -*- coding: utf-8 -*-
"""
@created on: 3/8/21,
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

from covid_19.networks.conv_ae import ConvAutoEncoder
from covid_19.utils import file_utils
from covid_19.utils.data_utils import read_pkl
from covid_19.utils.logger import Logger
from covid_19.utils.network_utils import to_tensor, to_numpy

import wandb as wnb

# setting seed
torch.manual_seed(1234)
np.random.seed(1234)


class PlainConvAutoencoderRunner:
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
        self.loss_function = nn.MSELoss(reduction='none')
        self.learning_rate_decay = args.learning_rate_decay

        self.optimiser = optim.Adam(self.network.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimiser, gamma=self.learning_rate_decay)

        self._min, self._max = float('inf'), -float('inf')

        if self.train_net:
            wnb.init(project=args.project_name, config=args, save_code=True, name=self.run_name,
                     entity="shreeshanwnb", reinit=True)  # , mode='disabled'
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

            # Normalizing `input data` on train dataset's min and max values
            if self.normalise:
                input_data = (input_data - self._min) / (self._max - self._min)
                input_data = (input_data - self._mean) / self._std

            self.logger.info(f'Min max values {self._min, self._max}')
            self.logger.info(f'Std values {self._std}')
            self.logger.info(f'Mean values {self._mean}')
            wnb.config.update({split_type + '_min_val': self._min, split_type + '_max_val': self._max,
                               split_type + '_mean': self._mean, split_type + '_std': self._std})

            if should_batch:
                batched_input = [input_data[pos:pos + self.batch_size] for pos in
                                 range(0, len(input_data), self.batch_size)]
                batched_labels = [labels[pos:pos + self.batch_size] for pos in range(0, len(labels), self.batch_size)]
                return batched_input, batched_labels
            else:
                return input_data, labels

    def train(self):

        train_data, train_labels = self.data_reader(self.data_read_path, [self.train_file], shuffle=True,
                                                    train=True)
        test_data, test_labels = self.data_reader(self.data_read_path, [self.test_file], shuffle=False,
                                                  train=False)

        # Temporary analysis purpose
        self.flat_train_data = [element for sublist in train_data for element in sublist]
        self.flat_train_labels = [element for sublist in train_labels for element in sublist]

        self.flat_test_data = [element for sublist in test_data for element in sublist]
        self.flat_test_labels = [element for sublist in test_labels for element in sublist]

        for epoch in range(1, self.epochs):
            self.network.train()
            self.latent_features, train_reconstructed, train_losses, self.batch_loss = [], [], [], []

            for i, (audio_data, label) in enumerate(
                    zip(train_data, train_labels)):
                self.optimiser.zero_grad()

                audio_data = to_tensor(audio_data, device=self.device)
                predictions, latent = self.network(audio_data)
                predictions = predictions.squeeze(1)
                train_reconstructed.extend(to_numpy(predictions))
                loss = self.loss_function(predictions, audio_data)
                train_losses.extend(to_numpy(torch.mean(loss, dim=[1, 2])))
                torch.mean(loss).backward()
                self.optimiser.step()
                self.batch_loss.append(to_numpy(torch.mean(loss)))

            self.logger.info('***** Overall Train Metrics ***** ')
            self.logger.info(
                    f"Epoch: {epoch} | Loss: {'%.5f' % np.mean(self.batch_loss)}")

            wnb.log({'train_reconstruction_loss': np.mean(self.batch_loss)})

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
        self.test_batch_loss, test_reconstructed, latent_features, losses = [], [], [], []

        with torch.no_grad():
            for i, (audio_data, label) in enumerate(zip(x, y)):
                audio_data = to_tensor(audio_data, device=self.device)
                test_predictions, test_latent = self.network(audio_data)
                test_predictions = test_predictions.squeeze(1)
                test_reconstructed.extend(to_numpy(test_predictions))
                test_loss = self.loss_function(test_predictions, audio_data)
                losses.extend(to_numpy(torch.mean(test_loss, dim=[1, 2])))
                self.test_batch_loss.append(to_numpy(torch.mean(test_loss)))

        wnb.log({"test_reconstruction_loss": np.mean(self.test_batch_loss)})
        self.logger.info(f'***** {type} Metrics ***** ')
        self.logger.info(
                f"Loss: {'%.5f' % np.mean(self.test_batch_loss)}")

        if epoch % self.network_save_interval == 0:
            one_images, one_losses = self.merge(class_label=1, labels=self.flat_test_labels, data=self.flat_test_data,
                                                reconstructed_data=test_reconstructed, losses=losses)
            zero_images, zero_losses = self.merge(class_label=0, labels=self.flat_test_labels, data=self.flat_test_data,
                                                  reconstructed_data=test_reconstructed, losses=losses)
            wnb.log({"Test positive samples reconstruction": [wnb.Image(img, caption=str(l)) for img, l in
                                                              zip(one_images, one_losses)]})
            wnb.log({"Test negative samples reconstruction": [wnb.Image(img, caption=str(l)) for img, l in
                                                              zip(zero_images, zero_losses)]})
