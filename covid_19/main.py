# -*- coding: utf-8 -*-
"""
@created on: 02/16/20,
@author: Shreesha N,
@version: v0.0.1
@system name: badgod
Description:

..todo::

"""

import argparse
import json

import torch

from covid_19.utils.class_utils import AttributeDict


def parse():
    parser = argparse.ArgumentParser(description="covid19_configs")
    parser.add_argument('--train_net', type=bool)
    parser.add_argument('--test_net', type=bool)
    parser.add_argument('--configs_file', type=str)
    parser.add_argument('--network', type=str, choices=['convnet'])
    args = parser.parse_args()
    return args


def run(args):
    train_data_files, test_data_files = ["mit_train_data_fbank_aaaa.pkl", "mit_train_data_fbank_alphabet.pkl",
                                         "mit_train_data_fbank_cough.pkl", "mit_train_data_fbank_count.pkl",
                                         "mit_train_data_fbank_eeee.pkl", "mit_train_data_fbank_oooo.pkl",
                                         "mit_train_data_fbank_story.pkl", "mit_train_data_mfcc_aaaa.pkl",
                                         "mit_train_data_mfcc_alphabet.pkl", "mit_train_data_mfcc_cough.pkl",
                                         "mit_train_data_mfcc_count.pkl", "mit_train_data_mfcc_eeee.pkl",
                                         "mit_train_data_mfcc_oooo.pkl", "mit_train_data_mfcc_story.pkl"], [
                                            "mit_test_data_fbank_aaaa.pkl", "mit_test_data_fbank_alphabet.pkl",
                                            "mit_test_data_fbank_cough.pkl",
                                            "mit_test_data_fbank_count.pkl", "mit_test_data_fbank_eeee.pkl",
                                            "mit_test_data_fbank_oooo.pkl",
                                            "mit_test_data_fbank_story.pkl", "mit_test_data_mfcc_aaaa.pkl",
                                            "mit_test_data_mfcc_alphabet.pkl",
                                            "mit_test_data_mfcc_cough.pkl", "mit_test_data_mfcc_count.pkl",
                                            "mit_test_data_mfcc_eeee.pkl",
                                            "mit_test_data_mfcc_oooo.pkl", "mit_test_data_mfcc_story.pkl"]
    for train_file, test_file in zip(train_data_files, test_data_files):
        if args.network == 'convnet':
            from covid_19.runners.convnet_runner import ConvNetRunner
            network = ConvNetRunner(args=args, train_file=train_file, test_file=test_file)
        else:  # Default network
            from covid_19.runners.convnet_runner import ConvNetRunner
            network = ConvNetRunner(args=args, train_file=train_file, test_file=test_file)

        if args.train_net:
            network.train()

        if args.test_net:
            network.infer()


if __name__ == '__main__':

    args = parse().__dict__
    configs = json.load(open(args['configs_file']))
    if configs['use_gpu']:
        if not torch.cuda.is_available():
            print('GPU is not available. The program is exiting . . . ')
            exit()
        else:
            print('GPU device found: ', torch.cuda.get_device_name(0))
    configs = {**configs, **args}
    configs = AttributeDict(configs)
    run(configs)
