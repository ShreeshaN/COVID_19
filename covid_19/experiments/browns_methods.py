# -*- coding: utf-8 -*-
"""
@created on: 5/8/21,
@author: Shreesha N,
@version: v0.0.1
@system name: badgod
Description:

..todo::

"""

from sklearn.decomposition import PCA
# from sklearn import svm
# from sklearn.metrics import confusion_matrix
import pickle as pk
import numpy as np

threshold = 0.5

base_path = '/home/snarasimhamurthy/COVID_DATASET/Coswara-Data/Extracted_data/browns_experiment/'

train_browns, test_browns = pk.load(open(base_path + '/coswara_train_data_brown_cough-shallow.pkl', 'rb')), \
                            pk.load(open(base_path + '/coswara_test_data_brown_cough-shallow.pkl', 'rb'))

train_vggish, test_vggish = pk.load(open(base_path + '/coswara_train_data_vggish_cough-shallow.pkl', 'rb')), \
                            pk.load(open(base_path + '/coswara_test_data_vggish_cough-shallow.pkl', 'rb'))

train_brown_data, train_brown_labels = train_browns[0], train_browns[1]
test_brown_data, test_brown_labels = test_browns[0], test_browns[1]
train_vggish_data, train_vggish_labels = np.squeeze(train_vggish[0], axis=1), train_vggish[1]
test_vggish_data, test_vggish_labels = np.squeeze(test_vggish[0], axis=1), test_vggish[1]

train_data = np.concatenate((train_brown_data, train_vggish_data), axis=1)
test_data = np.concatenate((test_brown_data, test_vggish_data), axis=1)

for components in [10, 30, 50, 70, 90, 100]:
    print('Components ', components)
    pca = PCA(n_components=100)
    pca_train_data = pca.fit_transform(train_data)
    for component in range(1, 11):
        print('Variance explained by', component, sum(pca.explained_variance_ratio_[:component]))
    print('********************************************************************************')
    print('********************************************************************************')
