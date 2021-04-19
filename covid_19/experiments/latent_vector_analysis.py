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

train_labels, test_labels = np.array(pk.load(open(
        '/Users/badgod/badgod_documents/Datasets/covid19/processed_data/coswara_train_data_fbank_cough-shallow_labels.pkl',
        'rb'))), np.array(pk.load(open(
        '/Users/badgod/badgod_documents/Datasets/covid19/processed_data/coswara_test_data_fbank_cough-shallow_labels.pkl',
        'rb')))
train_latent_features, test_latent_features = np.array(pk.load(
        open('/Users/badgod/badgod_documents/Datasets/covid19/processed_data/ae_forced_train_latent.npy',
             'rb'))), np.array(pk.load(
        open('/Users/badgod/badgod_documents/Datasets/covid19/processed_data/ae_forced_test_latent.npy', 'rb')))

print(
        'Total train data len: ' + str(len(train_labels)) + ' | Positive samples: ' + str(sum(train_labels)))
print(
        'Total test data len: ' + str(len(test_labels)) + ' | Positive samples: ' + str(sum(test_labels)))

ones_idx = [i for i, x in enumerate(train_labels) if x == 1]
zeros_idx = [i for i, x in enumerate(train_labels) if x == 0]

print('Train ones')
print(np.mean(train_latent_features[ones_idx]))
print(np.std(train_latent_features[ones_idx]))

print('Train zeros')
print(np.mean(train_latent_features[zeros_idx]))
print(np.std(train_latent_features[zeros_idx]))

ones_idx = [i for i, x in enumerate(test_labels) if x == 1]
zeros_idx = [i for i, x in enumerate(test_labels) if x == 0]

print('Test ones')
print(np.mean(test_latent_features[ones_idx]))
print(np.std(test_latent_features[ones_idx]))

print('Test zeros')
print(np.mean(test_latent_features[zeros_idx]))
print(np.std(test_latent_features[zeros_idx]))
