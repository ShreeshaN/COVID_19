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
        self.coswara_variations = args.coswara_variations

    def mit_datagather(self):
        def transform(df):
            df['diagnosis'] = df['diagnosis'].map({'Yes': 1, 'No': 0})
            df.set_index('id', inplace=True)
            dict_data = df.to_dict()
            return dict_data['diagnosis']

        train_data = pd.read_csv(self.mit_audiopath + 'train_data.csv')
        train_data = transform(train_data)

        test_data = pd.read_csv(self.mit_audiopath + 'test_data.csv')
        test_data = transform(test_data)

        def process(data_dict, save_name, data_structure, audio_variation):
            folder_names = data_dict.keys()
            for e, folder_name in tqdm(enumerate(folder_names), total=len(folder_names)):
                final_path = self.mit_audiopath + '/' + folder_name + '/' + self.data_processing_method + '.pkl'
                if not os.path.exists(final_path):
                    print(folder_name, 'does not exist')
                    continue
                folder_data = pickle.load(
                        open(final_path, 'rb'))
                if audio_variation in folder_data.keys():
                    for i in range(len(folder_data[audio_variation])):
                        data_structure[0].append(folder_data[audio_variation][i])
                        data_structure[1].append(data_dict[folder_name])
                else:
                    print('Variation ', audio_variation, 'not present in ', folder_name, '-', e)
                    continue
            pickle.dump(data_structure, open(self.mit_audiopath + '/browns_experiment/' + save_name, 'wb'))

        for variation in tqdm(self.mit_variations, total=len(self.mit_variations)):
            print('**************************** Starting', variation, '****************************')
            data = [[], []]
            process(train_data,
                    'mit_train_data_' + self.data_processing_method + '_' + variation.split('.')[0] + '.pkl',
                    data, variation)
            data = [[], []]
            process(test_data, 'mit_test_data_' + self.data_processing_method + '_' + variation.split('.')[0] + '.pkl',
                    data, variation)

    def coswara_datagather(self):
        def transform(df):
            df = df[['id', 'merged_label']]
            df['id'] = df['id'].map(wav_folders)
            df.set_index('id', inplace=True)
            dict_data = df.to_dict()
            return dict_data['merged_label']

        def process(data_dict, save_name, data_structure, audio_variation):
            folder_names = data_dict.keys()
            for e, folder_name in tqdm(enumerate(folder_names), total=len(folder_names)):
                final_path = self.coswara_datapath + '/' + folder_name + '/' + self.data_processing_method + '.pkl'
                if not os.path.exists(final_path):
                    print(folder_name, 'does not exist')
                    continue
                folder_data = pickle.load(
                        open(final_path, 'rb'))
                if audio_variation in folder_data.keys():
                    for i in range(len(folder_data[audio_variation])):
                        data_structure[0].append(folder_data[audio_variation][i])
                        data_structure[1].append(data_dict[folder_name])
                else:
                    print('Variation ', audio_variation, 'not present in ', folder_name, '-', e)
                    continue
            pickle.dump(data_structure, open(self.coswara_datapath + '/browns_experiment/' + save_name, 'wb'))

        wav_folders = []
        folders_with_date = glob.glob(self.coswara_datapath + '/*')
        folders_with_date = [x for x in folders_with_date if os.path.isdir(x)]
        for folder_with_date in folders_with_date:
            wav_folders.extend(['/'.join([folder_with_date.split('/')[-1], x]) for x in os.listdir(folder_with_date) if
                                os.path.isdir(folder_with_date + '/' + x)])

        wav_folders = {x.split('/')[1]: x for x in wav_folders}

        train_data = pd.read_csv(self.coswara_datapath + 'train_data.csv')
        train_data = transform(train_data)

        test_data = pd.read_csv(self.coswara_datapath + 'test_data.csv')
        test_data = transform(test_data)

        for variation in tqdm(self.coswara_variations, total=len(self.coswara_variations)):
            print('**************************** Starting', variation, '****************************')
            data = [[], []]
            process(train_data,
                    'coswara_train_data_' + self.data_processing_method + '_' + variation.split('.')[0] + '.pkl', data,
                    variation)
            data = [[], []]
            process(test_data,
                    'coswara_test_data_' + self.data_processing_method + '_' + variation.split('.')[0] + '.pkl', data,
                    variation)

        pass

    def gather(self):
        for processor_ in [self.mit_datagather, self.coswara_datagather]:
            print('Processing ', processor_)
            processor_()

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
