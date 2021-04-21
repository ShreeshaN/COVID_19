# -*- coding: utf-8 -*-
"""
@created on: 4/21/21,
@author: Shreesha N,
@version: v0.0.1
@system name: badgod
Description:

..todo::

"""

# -*- coding: utf-8 -*-
"""
@created on: 5/2/20,
@author: Shreesha N,
@version: v0.0.1
@system name: badgod
Description:

..todo::

"""

from sklearn.manifold import TSNE
import numpy as np
import cv2
import pickle as pk
import pandas as pd


def norm(data):
    min_val = data.min()
    max_val = data.max()
    data = (data - min_val) / (max_val - min_val)
    return np.mean(data, axis=(2, 3))


def read_data(csv_path):
    images = []
    d = pd.read_csv(csv_path)
    for image_path in d['spectrogram_path'].values:
        img = cv2.imread(image_path)
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        img = np.mean(img, axis=(1, 2))
        images.append(img)

    return np.array(images), d['labels'].values


def get_labels(csv_path):
    d = pd.read_csv(csv_path)
    return d['labels'].values


train_labels, test_labels = pk.load(open(
        '/Users/badgod/badgod_documents/Datasets/covid19/processed_data/coswara_train_data_fbank_cough-shallow_labels.pkl',
        'rb')), pk.load(open(
        '/Users/badgod/badgod_documents/Datasets/covid19/processed_data/coswara_test_data_fbank_cough-shallow_labels.pkl',
        'rb'))
train_data, test_data = np.array(pk.load(
        open('/Users/badgod/badgod_documents/Datasets/covid19/processed_data/ae_contrastive_train_latent.npy',
             'rb'))), np.array(pk.load(
        open('/Users/badgod/badgod_documents/Datasets/covid19/processed_data/ae_contrastive_test_latent.npy',
             'rb')))

tsne = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=1000, n_jobs=-1)
train_tsne_results = tsne.fit_transform(train_data)
tsne = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=1000, n_jobs=-1)
test_tsne_results = tsne.fit_transform(test_data)

np.save(
        "/Users/badgod/badgod_documents/Datasets/covid19/processed_data/train_tsne_ae_contrastive.npy",
        train_tsne_results)
np.save(
        "/Users/badgod/badgod_documents/Datasets/covid19/processed_data/test_tsne_ae_contrastive.npy",
        test_tsne_results)

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd
import plotly.express as px


def two_d():
    train_data = np.load("/Users/badgod/badgod_documents/Alco_audio/server_data/2d/train_tsne_2d.npy",
                         allow_pickle=True)
    train_labels = np.load(
            "/Users/badgod/badgod_documents/Alco_audio/server_data/2d/train_challenge_with_d1_labels.npy",
            allow_pickle=True)
    test_data = np.load("/Users/badgod/badgod_documents/Alco_audio/server_data/2d/test_tsne_2d.npy", allow_pickle=True)
    test_labels = np.load("/Users/badgod/badgod_documents/Alco_audio/server_data/2d/test_challenge_labels.npy",
                          allow_pickle=True)

    train_df = pd.DataFrame({'x1': train_tsne_results[:, 0], 'x2': train_tsne_results[:, 1], 'y': train_labels})
    test_df = pd.DataFrame({'x1': test_tsne_results[:, 0], 'x2': test_tsne_results[:, 1], 'y': test_labels})

    sns.scatterplot('x1', 'x2', hue='y', palette=sns.color_palette('hls', 2), data=train_df, legend='full', alpha=0.3)
    plt.savefig('/Users/badgod/badgod_documents/Alco_audio/server_data/2d/train_2d_tsne_norm_on_timeseries.png')
    plt.show()
    plt.close()

    sns.scatterplot('x1', 'x2', hue='y', palette=sns.color_palette('hls', 2), data=test_df, legend='full', alpha=0.3)
    plt.savefig('/Users/badgod/badgod_documents/Alco_audio/server_data/2d/test_2d_tsne_norm_on_timeseries.png')
    plt.show()
    plt.close()


def three_d():
    sns.set()
    train_data = np.load("/Users/badgod/badgod_documents/Alco_audio/server_data/2d/cae_tsne_3d/train_tsne_3d.npy",
                         allow_pickle=True)
    train_labels = np.load(
            "/Users/badgod/badgod_documents/Alco_audio/server_data/2d/train_challenge_with_d1_labels.npy",
            allow_pickle=True)
    dev_data = np.load("/Users/badgod/badgod_documents/Alco_audio/server_data/2d/cae_tsne_3d/dev_tsne_3d.npy",
                       allow_pickle=True)
    dev_labels = np.load("/Users/badgod/badgod_documents/Alco_audio/server_data/2d/dev_challenge_with_d1_labels.npy",
                         allow_pickle=True)
    test_data = np.load("/Users/badgod/badgod_documents/Alco_audio/server_data/2d/cae_tsne_3d/test_tsne_3d.npy",
                        allow_pickle=True)
    test_labels = np.load("/Users/badgod/badgod_documents/Alco_audio/server_data/2d/test_challenge_labels.npy",
                          allow_pickle=True)

    train_df = pd.DataFrame({'x1': train_data[:, 0], 'x2': train_data[:, 1], 'x3': train_data[:, 2], 'y': train_labels})
    dev_df = pd.DataFrame({'x1': dev_data[:, 0], 'x2': dev_data[:, 1], 'x3': dev_data[:, 2], 'y': dev_labels})
    test_df = pd.DataFrame({'x1': test_data[:, 0], 'x2': test_data[:, 1], 'x3': test_data[:, 2], 'y': test_labels})

    def three_d_plot(df, type):
        ax = plt.figure(figsize=(16, 10)).gca(projection='3d')
        scatter = ax.scatter(
                xs=df['x1'],
                ys=df['x2'],
                zs=df['x3'],
                c=df["y"],
                cmap='PiYG'
        )
        # produce a legend with the unique colors from the scatter
        legend1 = ax.legend(*scatter.legend_elements(),
                            loc="upper right", title="Classes")
        ax.add_artist(legend1)
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('x3')
        plt.savefig(f'/Users/badgod/badgod_documents/Alco_audio/server_data/2d/cae_tsne_3d/{type}_tsne_3d_plot.png')
        plt.show()

    three_d_plot(train_df, type='train')
    three_d_plot(dev_df, type='dev')
    three_d_plot(test_df, type='test')


def plotly_3d():
    def plotly_3d_plot(df, name):
        fig = px.scatter_3d(df, x='x1', y='x2', z='x3',
                            color='y', symbol='y')
        fig.update_layout(title=name)
        fig.show()

    path = '/Users/badgod/badgod_documents/Datasets/covid19/processed_data/'
    train_data = np.load(path + "train_tsne_ae_contrastive.npy", allow_pickle=True)
    test_data = np.load(path + "test_tsne_ae_contrastive.npy", allow_pickle=True)
    train_labels, test_labels = pk.load(open(path + 'coswara_train_data_fbank_cough-shallow_labels.pkl', 'rb')), \
                                pk.load(open(path + 'coswara_test_data_fbank_cough-shallow_labels.pkl', 'rb'))

    train_df = pd.DataFrame({'x1': train_data[:, 0], 'x2': train_data[:, 1], 'x3': train_data[:, 2], 'y': train_labels})
    test_df = pd.DataFrame({'x1': test_data[:, 0], 'x2': test_data[:, 1], 'x3': test_data[:, 2], 'y': test_labels})

    plotly_3d_plot(train_df, 'Train')
    plotly_3d_plot(test_df, 'Test')


def plotly_2d():
    def plotly_2d_plot(df):
        fig = px.scatter(df, x='x1', y='x2',
                         color='y')
        fig.show()

    import plotly.express as px
    train_data = np.load(
            "/Users/badgod/badgod_documents/Alco_audio/server_data/2d/tsne_mel_filters_3d_without_db/train_tsne_with_d1_images.npy",
            allow_pickle=True)
    train_labels = get_labels(
            "/Users/badgod/badgod_documents/Alco_audio/small_data/40_mels/train_challenge_with_d1_mel_images_data_melfilter_specs.csv")
    dev_data = np.load(
            "/Users/badgod/badgod_documents/Alco_audio/server_data/2d/tsne_mel_filters_3d_without_db/dev_tsne_with_d1_images.npy",
            allow_pickle=True)
    dev_labels = get_labels(
            "/Users/badgod/badgod_documents/Alco_audio/small_data/40_mels/dev_challenge_with_d1_mel_images_data_melfilter_specs.csv")
    test_data = np.load(
            "/Users/badgod/badgod_documents/Alco_audio/server_data/2d/tsne_mel_filters_3d_without_db/test_tsne_with_d1_images.npy",
            allow_pickle=True)
    test_labels = get_labels(
            "/Users/badgod/badgod_documents/Alco_audio/small_data/40_mels/test_challenge_with_d1_mel_images_data_melfilter_specs.csv")

    train_df = pd.DataFrame({'x1': train_data[:, 0], 'x2': train_data[:, 1], 'y': train_labels})
    dev_df = pd.DataFrame({'x1': dev_data[:, 0], 'x2': dev_data[:, 1], 'y': dev_labels})
    test_df = pd.DataFrame({'x1': test_data[:, 0], 'x2': test_data[:, 1], 'y': test_labels})

    plotly_2d_plot(train_df)
    plotly_2d_plot(dev_df)
    plotly_2d_plot(test_df)


plotly_3d()
# plotly_2d()
# three_d()
