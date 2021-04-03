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
    parser.add_argument('--network', type=str, choices=['convnet', 'convAE', 'plainAE', 'plainVAE'])
    args = parser.parse_args()
    return args


def extract_and_add_keys_from_filename(filename, args):
    data_source = filename.split("_")[0]
    data_processing_method = filename.split("_")[3]
    data_variant = filename.split("_")[-1]
    args['data_processing_method'] = data_processing_method
    args['data_variant'] = data_variant
    args['data_source'] = data_source


def run(args):
    # MIT
    # train_data_files, test_data_files = ["mit_train_data_fbank_aaaa.pkl", "mit_train_data_fbank_alphabet.pkl",
    #                                      "mit_train_data_fbank_cough.pkl", "mit_train_data_fbank_count.pkl",
    #                                      "mit_train_data_fbank_eeee.pkl", "mit_train_data_fbank_oooo.pkl",
    #                                      "mit_train_data_fbank_story.pkl", "mit_train_data_mfcc_aaaa.pkl",
    #                                      "mit_train_data_mfcc_alphabet.pkl", "mit_train_data_mfcc_cough.pkl",
    #                                      "mit_train_data_mfcc_count.pkl", "mit_train_data_mfcc_eeee.pkl",
    #                                      "mit_train_data_mfcc_oooo.pkl", "mit_train_data_mfcc_story.pkl"
    #                                      ],
    #                                      [
    #                                         "mit_test_data_fbank_aaaa.pkl", "mit_test_data_fbank_alphabet.pkl",
    #                                         "mit_test_data_fbank_cough.pkl",
    #                                         "mit_test_data_fbank_count.pkl", "mit_test_data_fbank_eeee.pkl",
    #                                         "mit_test_data_fbank_oooo.pkl",
    #                                         "mit_test_data_fbank_story.pkl", "mit_test_data_mfcc_aaaa.pkl",
    #                                         "mit_test_data_mfcc_alphabet.pkl",
    #                                         "mit_test_data_mfcc_cough.pkl", "mit_test_data_mfcc_count.pkl",
    #                                         "mit_test_data_mfcc_eeee.pkl",
    #                                         "mit_test_data_mfcc_oooo.pkl", "mit_test_data_mfcc_story.pkl"
    #                                         ]

    # COSWARA
    # train_data_files, test_data_files = ["coswara_train_data_fbank_counting-normal.pkl",
    #                                      "coswara_train_data_fbank_vowel-a.pkl",
    #                                      "coswara_train_data_fbank_vowel-e.pkl",
    #                                      "coswara_train_data_fbank_vowel-o.pkl",
    #                                      "coswara_train_data_mfcc_breathing-deep.pkl",
    #                                      "coswara_train_data_mfcc_breathing-shallow.pkl",
    #                                      "coswara_train_data_mfcc_cough-heavy.pkl",
    #                                      "coswara_train_data_mfcc_cough-shallow.pkl",
    #                                      "coswara_train_data_mfcc_counting-fast.pkl",
    #                                      "coswara_train_data_mfcc_counting-normal.pkl",
    #                                      "coswara_train_data_mfcc_vowel-a.pkl",
    #                                      "coswara_train_data_mfcc_vowel-e.pkl",
    #                                      "coswara_train_data_mfcc_vowel-o.pkl",
    #                                      "coswara_train_data_fbank_breathing-deep.pkl",
    #                                      "coswara_train_data_fbank_breathing-shallow.pkl",
    #                                      "coswara_train_data_fbank_cough-heavy.pkl",
    #                                      "coswara_train_data_fbank_cough-shallow.pkl",
    #                                      "coswara_train_data_fbank_counting-fast.pkl"], [
    #                                         "coswara_test_data_fbank_counting-normal.pkl",
    #                                         "coswara_test_data_fbank_vowel-a.pkl",
    #                                         "coswara_test_data_fbank_vowel-e.pkl",
    #                                         "coswara_test_data_fbank_vowel-o.pkl",
    #                                         "coswara_test_data_mfcc_breathing-deep.pkl",
    #                                         "coswara_test_data_mfcc_breathing-shallow.pkl",
    #                                         "coswara_test_data_mfcc_cough-heavy.pkl",
    #                                         "coswara_test_data_mfcc_cough-shallow.pkl",
    #                                         "coswara_test_data_mfcc_counting-fast.pkl",
    #                                         "coswara_test_data_mfcc_counting-normal.pkl",
    #                                         "coswara_test_data_mfcc_vowel-a.pkl",
    #                                         "coswara_test_data_mfcc_vowel-e.pkl",
    #                                         "coswara_test_data_mfcc_vowel-o.pkl",
    #                                         "coswara_test_data_fbank_breathing-deep.pkl",
    #                                         "coswara_test_data_fbank_breathing-shallow.pkl",
    #                                         "coswara_test_data_fbank_cough-heavy.pkl",
    #                                         "coswara_test_data_fbank_cough-shallow.pkl",
    #                                         "coswara_test_data_fbank_counting-fast.pkl"]

    # train_data_files = ["mit_test_data_fbank_breathing.pkl"
    #     , "mit_train_data_fbank_cough.pkl",
    #                     "mit_train_data_fbank_count.pkl", "mit_train_data_mfcc_breathing.pkl",
    #                     "mit_train_data_mfcc_cough.pkl",
    #                     "mit_train_data_mfcc_count.pkl"
    #                     ]
    #
    # test_data_files = ["mit_test_data_fbank_breathing.pkl"
    #     , "mit_test_data_fbank_cough.pkl",
    #                    "mit_test_data_fbank_count.pkl", "mit_test_data_mfcc_breathing.pkl",
    #                    "mit_test_data_mfcc_cough.pkl",
    #                    "mit_test_data_mfcc_count.pkl"
    #                    ]

    train_data_files, test_data_files = ["coswara_train_data_fbank_cough-shallow.pkl"], [
        "coswara_test_data_fbank_cough-shallow.pkl"]

    for train_file, test_file in zip(train_data_files, test_data_files):
        # Process file name and extract keys
        # 1. Extract runtime data source
        # 2. Extract data processing method
        # 3. Extract data variant
        # 4. add all these to args
        extract_and_add_keys_from_filename(train_file, args)
        if args.network == 'convnet':
            from covid_19.runners.convnet_runner import ConvNetRunner
            network = ConvNetRunner(args=args, train_file=train_file, test_file=test_file)
        elif args.network == 'convAE':
            from covid_19.runners.forced_autoencoder import ConvAutoencoderRunner
            network = ConvAutoencoderRunner(args=args, train_file=train_file, test_file=test_file)
        elif args.network == 'plainAE':
            from covid_19.runners.plain_conv_ae import PlainConvAutoencoderRunner
            network = PlainConvAutoencoderRunner(args=args, train_file=train_file, test_file=test_file)
        elif args.network == 'plainVAE':
            from covid_19.runners.plain_conv_vae import PlainConvVariationalAutoencoderRunner
            network = PlainConvVariationalAutoencoderRunner(args=args, train_file=train_file, test_file=test_file)
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
