# -*- coding: utf-8 -*-
"""
@created on: 4/4/21,
@author: Shreesha N,
@version: v0.0.1
@system name: badgod
Description:

..todo::

"""

from sklearn import svm
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix
from sklearn.covariance import EllipticEnvelope
import pickle as pk
import numpy as np

from covid_19.utils.network_utils import to_tensor, accuracy_fn

threshold = 0.5


def mask_preds_for_one_class(predictions):
    # 1 --> inliers, -1 --> outliers
    # in our case, inliers are non covid samples. i.e label 0.
    # outliers are covid samples. i.e label 1.
    return [1 if x == -1 else 0 for x in predictions]


train_labels, test_labels = pk.load(open(
        '/Users/badgod/badgod_documents/Datasets/covid19/processed_data/coswara_train_data_fbank_cough-shallow_labels.pkl',
        'rb')), pk.load(open(
        '/Users/badgod/badgod_documents/Datasets/covid19/processed_data/coswara_test_data_fbank_cough-shallow_labels.pkl',
        'rb'))
train_latent_features, test_latent_features = np.array(pk.load(
        open('/Users/badgod/badgod_documents/Datasets/covid19/processed_data/vae_forced_train_latent_contrastive.npy',
             'rb'))), np.array(pk.load(
        open('/Users/badgod/badgod_documents/Datasets/covid19/processed_data/vae_forced_test_latent_contrastive.npy',
             'rb')))
print(
        'Total train data len: ' + str(len(train_labels)) + ' | Positive samples: ' + str(sum(train_labels)))
print(
        'Total test data len: ' + str(len(test_labels)) + ' | Positive samples: ' + str(sum(test_labels)))
# exit()

ones_idx = [i for i, x in enumerate(train_labels) if x == 1]
zeros_idx = [i for i, x in enumerate(train_labels) if x == 0]
print(train_latent_features[ones_idx].mean(), train_latent_features[ones_idx].std())
print(train_latent_features[zeros_idx].mean(), train_latent_features[zeros_idx].std())
# exit()
# model = svm.OneClassSVM(kernel="poly")
# oneclass_svm = IsolationForest(random_state=0)
model = EllipticEnvelope()
model.fit(train_latent_features)
oneclass_predictions = model.predict(train_latent_features)
masked_predictions = mask_preds_for_one_class(oneclass_predictions)
train_metrics = accuracy_fn(to_tensor(masked_predictions), to_tensor(train_labels), threshold=threshold)
train_metrics = {'train_' + k: v for k, v in train_metrics.items()}
print(f'***** Train Metrics ***** ')
print(
        f"Accuracy: {'%.5f' % train_metrics['train_accuracy']} "
        f"| UAR: {'%.5f' % train_metrics['train_uar']}| F1:{'%.5f' % train_metrics['train_f1']} "
        f"| Precision:{'%.5f' % train_metrics['train_precision']} "
        f"| Recall:{'%.5f' % train_metrics['train_recall']} | AUC:{'%.5f' % train_metrics['train_auc']}")
print('Train Confusion matrix - \n' + str(confusion_matrix(train_labels, masked_predictions)))

# Test
oneclass_predictions = model.predict(test_latent_features)
masked_predictions = mask_preds_for_one_class(oneclass_predictions)
test_metrics = accuracy_fn(to_tensor(masked_predictions), to_tensor(test_labels), threshold=threshold)
test_metrics = {'test_' + k: v for k, v in test_metrics.items()}
print(f'***** Test Metrics ***** ')
print(
        f"Accuracy: {'%.5f' % test_metrics['test_accuracy']} "
        f"| UAR: {'%.5f' % test_metrics['test_uar']}| F1:{'%.5f' % test_metrics['test_f1']} "
        f"| Precision:{'%.5f' % test_metrics['test_precision']} "
        f"| Recall:{'%.5f' % test_metrics['test_recall']} | AUC:{'%.5f' % test_metrics['test_auc']}")
print('Test Confusion matrix - \n' + str(confusion_matrix(test_labels, masked_predictions)))
