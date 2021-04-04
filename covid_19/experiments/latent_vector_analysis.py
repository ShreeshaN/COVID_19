# -*- coding: utf-8 -*-
"""
@created on: 4/4/21,
@author: Shreesha N,
@version: v0.0.1
@system name: badgod
Description:

..todo::

"""
import pickle as pk
import numpy as np

train_labels, test_labels = pk.load(open(
        '/Users/badgod/badgod_documents/Datasets/covid19/processed_data/coswara_train_data_fbank_cough-shallow_labels.pkl',
        'rb')), pk.load(open(
        '/Users/badgod/badgod_documents/Datasets/covid19/processed_data/coswara_test_data_fbank_cough-shallow_labels.pkl',
        'rb'))
train_latent_features, test_latent_features = pk.load(
        open('/Users/badgod/badgod_documents/Datasets/covid19/processed_data/vae_forced_train_latent.npy',
             'rb')), pk.load(
        open('/Users/badgod/badgod_documents/Datasets/covid19/processed_data/vae_forced_test_latent.npy', 'rb'))

print(
        'Total train data len: ' + str(len(train_labels)) + ' | Positive samples: ' + str(sum(train_labels)))
print(
        'Total test data len: ' + str(len(test_labels)) + ' | Positive samples: ' + str(sum(test_labels)))


