# -*- coding: utf-8 -*-
"""
@created on: 2/15/21,
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


class DataAnalyser:
    def __init__(self, args):
        self.coswara_datapath = args.coswara_datapath
        self.mit_datapath = args.mit_datapath
        self.mit_audiopath = args.mit_audiopath
        self.normalize_while_creating = args.normalise_while_creating
        self.sample_size_in_seconds = args.sample_size_in_seconds
        self.sampling_rate = args.sampling_rate
        self.overlap = args.overlap
        self.data_processing_method = args.data_processing_method


    def analyze(self):
        self.mit_datapath


    def run(self):
        self.analyze()


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
    processor = DataAnalyser(configs)
    print(json.dumps(configs, indent=4))
    processor.run()
