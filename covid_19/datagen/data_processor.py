# -*- coding: utf-8 -*-
"""
@created on: 2/11/21,
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

from covid_19.utils.data_utils import stratified_train_test_split
from covid_19.utils.class_utils import AttributeDict
from covid_19.datagen.audio_feature_extractors import preprocess_data


class DataProcessor:
    def __init__(self, args):
        self.coswara_datapath = args.coswara_datapath
        self.mit_datapath = args.mit_datapath
        self.mit_audiopath = args.mit_audiopath
        self.normalize_while_creating = args.normalise_while_creating
        self.sample_size_in_seconds = args.sample_size_in_seconds
        self.sampling_rate = args.sampling_rate
        self.overlap = args.overlap
        self.data_processing_method = args.data_processing_method

    def coswara_processor(self):
        wav_folders = []
        folders_with_date = glob.glob(self.coswara_datapath + '/*')
        for folder_with_date in folders_with_date:
            wav_folders.extend(['/'.join([folder_with_date.split('/')[-1], x]) for x in os.listdir(folder_with_date)])
        preprocess_data(self.coswara_datapath, wav_folders, self.normalize_while_creating,
                        self.sample_size_in_seconds, self.sampling_rate, self.overlap, self.data_processing_method)

        pass

    def mit_processor(self):
        columns_to_consider = ['id', 'diagnosis']
        data = pd.read_csv(self.mit_datapath)[columns_to_consider]
        data = data.dropna()
        X, y = data['id'].values, data['diagnosis'].map({'Yes': 1, 'No': 0}).values
        # train_index, test_index = stratified_train_test_split(X, y, test_size=0.3,
        #                                                       random_state=10)
        # data[data.index.isin(train_index)].to_csv(self.mit_audiopath + '/train_data.csv')
        # data[data.index.isin(test_index)].to_csv(self.mit_audiopath + '/test_data.csv')
        # train_X, test_X, train_y, test_y = X[train_index], X[test_index], y[train_index], y[test_index]
        preprocess_data(self.mit_audiopath, X, self.normalize_while_creating,
                        self.sample_size_in_seconds, self.sampling_rate, self.overlap, self.data_processing_method)

    def run(self):
        # self.mit_processor()
        self.coswara_processor()


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
    processor = DataProcessor(configs)
    print(json.dumps(configs, indent=4))
    processor.run()
