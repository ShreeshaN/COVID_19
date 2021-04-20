# -*- coding: utf-8 -*-
"""
@created on: 4/19/21,
@author: Shreesha N,
@version: v0.0.1
@system name: badgod
Description:

..todo::

"""

# from sklearn import svm
# from sklearn.metrics import confusion_matrix
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.manifold import TSNE
# from sklearn.svm import SVC
# import pickle as pk
# import numpy as np
# import math
# from sklearn.metrics import recall_score, precision_recall_fscore_support, roc_curve, auc
#
#
# def mask_preds_for_one_class(predictions):
#     # 1 --> inliers, -1 --> outliers
#     # in our case, inliers are non covid samples. i.e label 0.
#     # outliers are covid samples. i.e label 1.
#     return [1 if x == -1 else 0 for x in predictions]
#
#
# def data_read(data_path):
#     def split_data_only_negative(combined_data):
#         # pass only negative samples
#         idx = [e for e, x in enumerate(combined_data[1]) if x == 0]  #
#         return np.array(combined_data[0])[[idx]], np.array(combined_data[1])[[idx]]
#
#     def split_data(combined_data):
#         return np.array(combined_data[0]).mean(axis=2), np.array(combined_data[1])
#
#     data = pk.load(open(data_path, 'rb'))
#     data, labels = split_data(data)
#     return data, labels
#
#
# def accuracy_fn(preds, labels, threshold):
#     # todo: Remove F1, Precision, Recall from return , or change all caller method dynamics
#
#     metrics = dict()
#     predictions = (np.array(preds) > threshold).astype(int)
#     precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
#     metrics['precision'] = precision
#     metrics['recall'] = recall
#     metrics['f1'] = f1
#
#     accuracy = np.sum(predictions == labels) / float(len(labels))
#     metrics['accuracy'] = accuracy
#
#     uar = recall_score(labels, predictions, average='macro')
#     metrics['uar'] = uar
#
#     false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, preds)
#     auc_score = auc(false_positive_rate, true_positive_rate)
#     metrics['auc'] = 0 if math.isnan(auc_score) else auc_score
#
#     return metrics
#
#
# threshold = 0.5
# # labels_path = '/Users/badgod/badgod_documents/Datasets/covid19/processed_data/'
# # data_path = '/Users/badgod/badgod_documents/Datasets/covid19/processed_data/'
# labels_path = '/home/snarasimhamurthy/COVID_DATASET/Coswara-Data/Extracted_data/'
# data_path = '/home/snarasimhamurthy/COVID_DATASET/Coswara-Data/Extracted_data/'
#
# train_features, train_labels = data_read(data_path + 'coswara_train_data_fbank_cough-shallow.pkl')
# test_features, test_labels = data_read(data_path + 'coswara_test_data_fbank_cough-shallow.pkl')
#
# print('Total train data len: ' + str(len(train_labels)) + ' | Positive samples: ' + str(sum(train_labels)))
# print('Total test data len: ' + str(len(test_labels)) + ' | Positive samples: ' + str(sum(test_labels)))
# print('Train Features shape ', train_features.shape)
# print('Test Features shape ', test_features.shape)
#
# for kernel_ in ["linear", "poly", "rbf", "sigmoid"]:
#     print('***********************************', kernel_, '***********************************')
#     model = svm.OneClassSVM(kernel=kernel_)
#     model.fit(train_features)
#     oneclass_predictions = model.predict(train_features)
#     masked_predictions = mask_preds_for_one_class(oneclass_predictions)
#     train_metrics = accuracy_fn(masked_predictions, train_labels, threshold=threshold)
#     train_metrics = {'train_' + k: v for k, v in train_metrics.items()}
#     print(f'***** Train Metrics ***** ')
#     print(
#             f"Accuracy: {'%.5f' % train_metrics['train_accuracy']} "
#             f"| UAR: {'%.5f' % train_metrics['train_uar']}| F1:{'%.5f' % train_metrics['train_f1']} "
#             f"| Precision:{'%.5f' % train_metrics['train_precision']} "
#             f"| Recall:{'%.5f' % train_metrics['train_recall']} | AUC:{'%.5f' % train_metrics['train_auc']}")
#     print('Train Confusion matrix - \n' + str(confusion_matrix(train_labels, masked_predictions)))
#
#     # Test
#     oneclass_predictions = model.predict(test_features)
#     masked_predictions = mask_preds_for_one_class(oneclass_predictions)
#     test_metrics = accuracy_fn(masked_predictions, test_labels, threshold=threshold)
#     test_metrics = {'test_' + k: v for k, v in test_metrics.items()}
#     print(f'***** Test Metrics ***** ')
#     print(
#             f"Accuracy: {'%.5f' % test_metrics['test_accuracy']} "
#             f"| UAR: {'%.5f' % test_metrics['test_uar']}| F1:{'%.5f' % test_metrics['test_f1']} "
#             f"| Precision:{'%.5f' % test_metrics['test_precision']} "
#             f"| Recall:{'%.5f' % test_metrics['test_recall']} | AUC:{'%.5f' % test_metrics['test_auc']}")
#     print('Test Confusion matrix - \n' + str(confusion_matrix(test_labels, masked_predictions)))


# Binary classification

from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.svm import SVC
import pickle as pk
import numpy as np
import math
from sklearn.metrics import recall_score, precision_recall_fscore_support, roc_curve, auc
import random
from sklearn.model_selection import KFold, StratifiedKFold

random.seed(1)


def data_read(data_path):
    def split_data_only_negative(combined_data):
        # pass only negative samples
        idx = [e for e, x in enumerate(combined_data[1]) if x == 0]  #
        return np.array(combined_data[0])[[idx]], np.array(combined_data[1])[[idx]]

    def split_data(combined_data):
        # zero_idx = [i for i, x in enumerate(combined_data[1]) if x == 0]
        # ones_idx = [i for i, x in enumerate(combined_data[1]) if x == 1]
        # ones_len = len(ones_idx)
        # zero_idx = random.sample(zero_idx, ones_len)
        # combined_data[0] = np.array(combined_data[0])
        # combined_data[1] = np.array(combined_data[1])
        # data = np.concatenate((combined_data[0][ones_idx], combined_data[0][zero_idx])).mean(axis=2)
        # labels = np.concatenate((combined_data[1][ones_idx], combined_data[1][zero_idx]))
        # return data, labels  # np.array(combined_data[0]).mean(axis=2), np.array(combined_data[1])
        return np.array(combined_data[0]).mean(axis=2), np.array(combined_data[1])

    data = pk.load(open(data_path, 'rb'))
    data, labels = split_data(data)
    return data, labels


def accuracy_fn(preds, labels, threshold):
    # todo: Remove F1, Precision, Recall from return , or change all caller method dynamics

    metrics = dict()
    predictions = (np.array(preds) > threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1'] = f1

    accuracy = np.sum(predictions == labels) / float(len(labels))
    metrics['accuracy'] = accuracy

    uar = recall_score(labels, predictions, average='macro')
    metrics['uar'] = uar

    false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, preds)
    auc_score = auc(false_positive_rate, true_positive_rate)
    metrics['auc'] = 0 if math.isnan(auc_score) else auc_score

    return metrics


threshold = 0.5
# labels_path = '/Users/badgod/badgod_documents/Datasets/covid19/processed_data/'
# data_path = '/Users/badgod/badgod_documents/Datasets/covid19/processed_data/'
labels_path = '/home/snarasimhamurthy/COVID_DATASET/Coswara-Data/Extracted_data/'
data_path = '/home/snarasimhamurthy/COVID_DATASET/Coswara-Data/Extracted_data/'

train_features, train_labels = data_read(data_path + 'coswara_train_data_fbank_cough-shallow.pkl')
test_features, test_labels = data_read(data_path + 'coswara_test_data_fbank_cough-shallow.pkl')

print('Total train data len: ' + str(len(train_labels)) + ' | Positive samples: ' + str(sum(train_labels)))
print('Total test data len: ' + str(len(test_labels)) + ' | Positive samples: ' + str(sum(test_labels)))
print('Train Features shape ', train_features.shape)
print('Test Features shape ', test_features.shape)

for kernel_ in ["poly", "rbf"]:
    print('***********************************', kernel_, '***********************************')
    k = 5
    kf = StratifiedKFold(n_splits=k, random_state=None, shuffle=False)
    model = svm.SVC(kernel=kernel_, gamma='auto')
    for e, (train_idx, test_idx) in enumerate(kf.split(train_features, train_labels)):
        print(' ---------- KFOLD ', e)
        tr_features, tr_labels = train_features[train_idx], train_labels[train_idx]
        te_features, te_labels = train_features[test_idx], train_labels[test_idx]
        model.fit(tr_features, tr_labels)
        predictions = model.predict(tr_features)
        train_metrics = accuracy_fn(predictions, tr_labels, threshold=threshold)
        train_metrics = {'train_' + k: v for k, v in train_metrics.items()}
        print(f'***** Train Metrics ***** ')
        print(
                f"Accuracy: {'%.5f' % train_metrics['train_accuracy']} "
                f"| UAR: {'%.5f' % train_metrics['train_uar']}| F1:{'%.5f' % train_metrics['train_f1']} "
                f"| Precision:{'%.5f' % train_metrics['train_precision']} "
                f"| Recall:{'%.5f' % train_metrics['train_recall']} | AUC:{'%.5f' % train_metrics['train_auc']}")
        print('Train Confusion matrix - \n' + str(confusion_matrix(tr_labels, predictions)))

        val_predictions = model.predict(te_features)
        val_metrics = accuracy_fn(val_predictions, te_labels, threshold=threshold)
        val_metrics = {'val_' + k: v for k, v in val_metrics.items()}
        print(f'***** Val Metrics ***** ')
        print(
                f"Accuracy: {'%.5f' % val_metrics['val_accuracy']} "
                f"| UAR: {'%.5f' % val_metrics['val_uar']}| F1:{'%.5f' % val_metrics['val_f1']} "
                f"| Precision:{'%.5f' % val_metrics['val_precision']} "
                f"| Recall:{'%.5f' % val_metrics['val_recall']} | AUC:{'%.5f' % val_metrics['val_auc']}")
        print('Train Confusion matrix - \n' + str(confusion_matrix(te_labels, val_predictions)))

    print('Final test results')
    # Test
    predictions = model.predict(test_features)
    test_metrics = accuracy_fn(predictions, test_labels, threshold=threshold)
    test_metrics = {'test_' + k: v for k, v in test_metrics.items()}
    print(f'***** Test Metrics ***** ')
    print(
            f"Accuracy: {'%.5f' % test_metrics['test_accuracy']} "
            f"| UAR: {'%.5f' % test_metrics['test_uar']}| F1:{'%.5f' % test_metrics['test_f1']} "
            f"| Precision:{'%.5f' % test_metrics['test_precision']} "
            f"| Recall:{'%.5f' % test_metrics['test_recall']} | AUC:{'%.5f' % test_metrics['test_auc']}")
    print('Test Confusion matrix - \n' + str(confusion_matrix(test_labels, predictions)))
