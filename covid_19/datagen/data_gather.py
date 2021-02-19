# -*- coding: utf-8 -*-
"""
@created on: 2/18/21,
@author: Shreesha N,
@version: v0.0.1
@system name: badgod
Description:

..todo::

"""

import json
import argparse
import pandas as pd
import glob
import os
from tqdm import tqdm
import pickle

from covid_19.utils.data_utils import stratified_train_test_split
from covid_19.utils.class_utils import AttributeDict
from covid_19.datagen.audio_feature_extractors import preprocess_data


class DataGather:
    def __init__(self, args):
        self.coswara_datapath = args.coswara_datapath
        self.mit_datapath = args.mit_datapath
        self.mit_audiopath = args.mit_audiopath
        self.normalize_while_creating = args.normalise_while_creating
        self.sample_size_in_seconds = args.sample_size_in_seconds
        self.sampling_rate = args.sampling_rate
        self.overlap = args.overlap
        self.data_processing_method = args.data_processing_method
        self.mit_variations = args.mit_variations

    def mit_datagather(self):
        train_data = pd.read_csv(self.mit_audiopath + 'train.csv')
        train_data.set_index('id', inplace=True)
        train_data = train_data.to_dict()
        test_data = pd.read_csv(self.mit_audiopath + 'test.csv')
        test_data.set_index('id', inplace=True)
        test_data = test_data.to_dict()

        def process(folder_names, save_name, data_structure, audio_variation):
            for folder_name in tqdm(folder_names, total=len(folder_names)):
                final_path = self.mit_audiopath + '/' + folder_name + '/' + self.data_processing_method + '.pkl'
                if not os.path.exists(final_path):
                    print(folder_name, 'does not exist')
                    continue
                folder_data = pickle.load(
                        open(final_path, 'rb'))
                if audio_variation in folder_data.keys():
                    data_structure[0].append(folder_data[audio_variation])
                    data_structure[1].append(train_data[folder_name])
                else:
                    print('Variation ', audio_variation, 'not present in ', folder_name)
                    continue
            pickle.dump(data_structure, open(self.mit_audiopath + '/' + save_name, 'wb'))

        for variation in tqdm(self.mit_variations, total=len(self.mit_variations)):
            data = [[], []]
            process(train_data.keys(), 'mit_train_data_' + variation.split('.')[0] + '.pkl', data, variation)
            data.clear()
            process(test_data.keys(), 'mit_test_data_' + variation.split('.')[0] + '.pkl', data, variation)

    def gather(self):
        self.mit_datagather()
        self.coswara_datagather()

    def run(self):
        self.gather()


def parse():
    parser = argparse.ArgumentParser(description="covid19_data_processor")
    parser.add_argument('--configs_file', type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse().__dict__
    configs = json.load(open(args['configs_file']))
    configs = {**configs, **args}
    configs = AttributeDict(configs)
    processor = DataGather(configs)
    print(json.dumps(configs, indent=4))
    processor.run()
