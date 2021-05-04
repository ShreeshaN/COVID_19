# -*- coding: utf-8 -*-
"""
@created on: 2/17/20,
@author: Shreesha N,
@version: v0.0.1
@system name: badgod
Description:

..todo::

"""
import os
import subprocess
import glob

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import pysptk
from math import pi
import torch
import wavio
from joblib import Parallel, delayed
from pyannote.audio.utils.signal import Binarize
from pyts.image import GramianAngularField
from tqdm import tqdm
from collections import defaultdict
from scipy.fftpack import fft, hilbert
import urllib
from covid_19.datagen.vggish import vggish_input
from covid_19.datagen.vggish import vggish_params
from covid_19.datagen.vggish import vggish_slim
import tensorflow as tf
import sys
import pathlib
from covid_19.utils.file_utils import delete_file
import json

sys.path.append(str(pathlib.Path(__file__).parent.absolute()) + '/vggish')
tf.compat.v1.disable_v2_behavior()

SR = 22050
FRAME_LEN = int(SR / 10)  # 100 ms
HOP = int(FRAME_LEN / 2)  # 50% overlap, meaning 5ms hop length
MFCC_dim = 13  # the MFCC dimension
SR_VGG = 16000


# Vggish
def download(url, dst_dir):
    """Download file.
    If the file not exist then download it.
    Args:url: Web location of the file.
    Returns: path to downloaded file.
    """
    filename = url.split('/')[-1]
    filepath = os.path.join(dst_dir, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' %
                             (filename,
                              float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(url, filepath, _progress)
        statinfo = os.stat(filepath)
        print('Successfully downloaded:', filename, statinfo.st_size, 'bytes.')
    return filepath


def sta_fun_2(npdata):  # 1D np array
    """Extract various statistical features from the numpy array provided as input.
    :param np_data: the numpy array to extract the features from
    :type np_data: numpy.ndarray
    :return: The extracted features as a vector
    :rtype: numpy.ndarray
    """

    # perform a sanity check
    if npdata is None:
        raise ValueError("Input array cannot be None")

    # perform the feature extraction
    Mean = np.mean(npdata, axis=0)
    Std = np.std(npdata, axis=0)

    # finally return the features in a concatenated array (as a vector)
    return np.concatenate((Mean, Std), axis=0).reshape(1, -1)


print("\nTesting your install of VGGish\n")
# Paths to downloaded VGGish files.
checkpoint_path = str(pathlib.Path(__file__).parent.absolute()) + "/vggish/vggish_model.ckpt"

if not os.path.exists(checkpoint_path):  # automatically download the checkpoint if not exist.
    url = 'https://storage.googleapis.com/audioset/vggish_model.ckpt'
    download(url, str(pathlib.Path(__file__).parent.absolute()) + '/vggish')

sess = tf.compat.v1.Session()
vggish_slim.define_vggish_slim()
vggish_slim.load_vggish_slim_checkpoint(sess, checkpoint_path)
features_tensor = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
embedding_tensor = sess.graph.get_tensor_by_name(
        vggish_params.OUTPUT_TENSOR_NAME
)


def tensorflow_close():
    if sess is not None:
        sess.close()


def vggish_features(signal):
    input_batch = vggish_input.waveform_to_examples(
            signal, SR_VGG
    )  # ?x96x64 --> ?x128
    [features] = sess.run(
            [embedding_tensor], feed_dict={features_tensor: input_batch}
    )
    features = sta_fun_2(features)
    return features


def mfcc_features(audio, sampling_rate, normalise=False):
    mfcc = librosa.feature.mfcc(y=audio, n_mfcc=40, sr=sampling_rate)
    if normalise:
        mfcc_norm = np.mean(mfcc.T, axis=0)
        return mfcc_norm
    else:
        return mfcc


def mel_filters(audio, sampling_rate, normalise=False):
    mel_spec = librosa.feature.melspectrogram(y=audio, n_mels=40, sr=sampling_rate)
    if normalise:
        return np.mean(librosa.power_to_db(mel_spec, ref=np.max).T)
    else:
        return librosa.power_to_db(mel_spec, ref=np.max)


def cut_audio(audio, sampling_rate, sample_size_in_seconds, overlap):
    """
    Method to split a audio signal into pieces based on `sample_size_in_seconds` and `overlap` parameters
    :param audio: The main audio signal to be split
    :param sampling_rate: The rate at which audio is sampled
    :param sample_size_in_seconds: number of seconds in each split
    :param overlap: in seconds, how much of overlap is required within splits
    :return: List of splits
    """
    if overlap >= sample_size_in_seconds:
        raise Exception("Please maintain this condition: sample_size_in_seconds > overlap")

    def add_to_audio_list(y):
        if len(y) / sampling_rate < sample_size_in_seconds:
            raise Exception(
                    f'Length of audio lesser than `sampling size in seconds` - {len(y) / sampling_rate} seconds, required {sample_size_in_seconds} seconds')
        y = y[:required_length]
        audio_list.append(y)

    audio_list = []
    required_length = sample_size_in_seconds * sampling_rate
    audio_in_seconds = len(audio) // sampling_rate

    # Check if the main audio file is larger than the required number of seconds
    if audio_in_seconds >= sample_size_in_seconds:
        start = 0
        end = sample_size_in_seconds
        left_out = None

        # Until highest multiple of sample_size_in_seconds is reached, ofcourse, wrt audio_in_seconds, run this loop
        while end <= audio_in_seconds:
            index_at_start, index_at_end = start * sampling_rate, end * sampling_rate
            one_audio_sample = audio[index_at_start:index_at_end]
            add_to_audio_list(one_audio_sample)
            left_out = audio_in_seconds - end
            start = (start - overlap) + sample_size_in_seconds
            end = (end - overlap) + sample_size_in_seconds

        # Whatever is left out after the iteration, just include that to the final list.
        # Eg: if 3 seconds is left out and sample_size_in_seconds is 5 seconds, then cut the last 5 seconds of the audio
        # and append to final list.
        if left_out > 0:
            one_audio_sample = audio[-sample_size_in_seconds * sampling_rate:]
            add_to_audio_list(one_audio_sample)
    # Else, just repeat the required number of seconds at the end. The repeated audio is taken from the start
    else:
        less_by = sample_size_in_seconds - audio_in_seconds
        excess_needed = less_by * sampling_rate
        one_audio_sample = np.append(audio, audio[-excess_needed:])

        # This condition is for samples which are too small and need to be repeated
        # multiple times to satisfy the `sample_size_in_seconds` parameter
        while len(one_audio_sample) < (sampling_rate * sample_size_in_seconds):
            one_audio_sample = np.hstack((one_audio_sample, one_audio_sample))
        add_to_audio_list(one_audio_sample)
    return audio_list


def envelope(y, rate, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate / 10), min_periods=1, center=True).mean()
    for e, mean in enumerate(y_mean):
        # print('Mean - ', e, int(e/rate) ,mean) if e%500==0 else None
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask


def get_shimmer_jitter_from_opensmile(audio, index, sr):
    wavio.write(f'temp_{str(index)}.wav', audio, sr, sampwidth=3)
    subprocess.call(
            ["SMILExtract", "-C", os.environ['OPENSMILE_CONFIG_DIR'] + "/IS10_paraling.conf", "-I",
             f"temp_{str(index)}.wav", "-O",
             f"temp_{str(index)}.arff"])
    # Read file and extract shimmer and jitter features from the generated arff file
    file = open(f"temp_{str(index)}.arff", "r")
    data = file.readlines()

    # First 3 values are title, empty line and name | Last 5 values are numeric data,
    # and bunch of empty lines and unwanted text
    # headers = data[3:-5]

    headers = data[3:data.index('@data\n')]
    headers = headers[:headers.index('@attribute class numeric\n')]

    # Last line of data is where the actual numeric data is. It is in comma separated string format. After splitting,
    # remove the first value which is name and the last value which is class
    numeric_data = data[-1].split(',')[1:-1]

    assert len(headers) == len(numeric_data), "Features generated from opensmile are not matching with its headers"

    # data_needed = {x.strip(): float(numeric_data[e]) for e, x in enumerate(headers) if 'jitter' in x or 'shimmer' in x}
    data_needed = [float(numeric_data[e]) for e, x in enumerate(headers) if 'jitter' in x or 'shimmer' in x]

    # clean up all files
    delete_file(f'temp_{str(index)}.wav')
    delete_file(f'temp_{str(index)}.arff')

    return data_needed


def sta_fun(np_data):
    """Extract various statistical features from the numpy array provided as input.
    :param np_data: the numpy array to extract the features from
    :type np_data: numpy.ndarray
    :return: The extracted features as a vector
    :rtype: numpy.ndarray
    """

    # perform a sanity check
    if np_data is None:
        raise ValueError("Input array cannot be None")

    # perform the feature extraction
    dat_min = np.min(np_data)
    dat_max = np.max(np_data)
    dat_mean = np.mean(np_data)
    dat_rms = np.sqrt(np.sum(np.square(np_data)) / len(np_data))
    dat_median = np.median(np_data)
    dat_qrl1 = np.percentile(np_data, 25)
    dat_qrl3 = np.percentile(np_data, 75)
    dat_lower_q = np.quantile(np_data, 0.25, interpolation="lower")
    dat_higher_q = np.quantile(np_data, 0.75, interpolation="higher")
    dat_iqrl = dat_higher_q - dat_lower_q
    dat_std = np.std(np_data)
    s = pd.Series(np_data)
    dat_skew = s.skew()
    dat_kurt = s.kurt()

    # finally return the features in a concatenated array (as a vector)
    return np.array([dat_mean, dat_min, dat_max, dat_std, dat_rms,
                     dat_median, dat_qrl1, dat_qrl3, dat_iqrl, dat_skew, dat_kurt])


def get_period(signal, signal_sr):
    """Extract the period from the the provided signal
    :param signal: the signal to extract the period from
    :type signal: numpy.ndarray
    :param signal_sr: the sampling rate of the input signal
    :type signal_sr: integer
    :return: a vector containing the signal period
    :rtype: numpy.ndarray
    """

    # perform a sanity check
    if signal is None:
        raise ValueError("Input signal cannot be None")

    # transform the signal to the hilbert space
    hy = hilbert(signal)

    ey = np.sqrt(signal ** 2 + hy ** 2)
    min_time = 1.0 / signal_sr
    tot_time = len(ey) * min_time
    pow_ft = np.abs(fft(ey))
    peak_freq = pow_ft[3: int(len(pow_ft) / 2)]
    peak_freq_pos = peak_freq.argmax()
    peak_freq_val = 2 * pi * (peak_freq_pos + 2) / tot_time
    period = 2 * pi / peak_freq_val

    return np.array([period])


def extract_signal_features(signal, signal_sr):
    """Extract part of handcrafted features from the input signal.
    :param signal: the signal the extract features from
    :type signal: numpy.ndarray
    :param signal_sr: the sample rate of the signal
    :type signal_sr: integer
    :return: the populated feature vector
    :rtype: numpy.ndarray
    """

    # normalise the sound signal before processing
    signal = signal / np.max(np.abs(signal))
    mean_ = np.float(np.mean(signal))
    mean_ = 0 if np.isnan(mean_) else mean_
    signal = np.nan_to_num(signal, nan=mean_, posinf=mean_, neginf=mean_)
    # trim the signal to the appropriate length
    trimmed_signal, idc = librosa.effects.trim(signal, frame_length=FRAME_LEN, hop_length=HOP)
    # extract the signal duration
    signal_duration = librosa.get_duration(y=signal, sr=signal_sr)
    # use librosa to track the beats
    tempo, beats = librosa.beat.beat_track(y=signal, sr=signal_sr)
    # find the onset strength of the trimmed signal
    o_env = librosa.onset.onset_strength(signal, sr=signal_sr)
    # find the frames of the onset
    onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=signal_sr)
    # keep only the first onset frame
    onsets = onset_frames.shape[0]
    # decompose the signal into its magnitude and the phase components such that signal = mag * phase
    mag, phase = librosa.magphase(librosa.stft(signal, n_fft=FRAME_LEN, hop_length=HOP))
    # extract the rms from the magnitude component
    rms = librosa.feature.rms(y=signal)[0]
    # extract the spectral centroid of the magnitude
    cent = librosa.feature.spectral_centroid(S=mag)[0]
    # extract the spectral rolloff point from the magnitude
    rolloff = librosa.feature.spectral_rolloff(S=mag, sr=signal_sr)[0]
    # extract the zero crossing rate from the trimmed signal using the predefined frame and hop lengths
    zcr = librosa.feature.zero_crossing_rate(signal, frame_length=FRAME_LEN, hop_length=HOP)[0]

    # pack the extracted features into the feature vector to be returned
    signal_features = np.concatenate(
            (
                np.array([signal_duration, tempo, onsets]),
                get_period(signal, signal_sr=signal_sr),
                sta_fun(rms),
                sta_fun(cent),
                sta_fun(rolloff),
                sta_fun(zcr),
            ),
            axis=0,
    )

    # finally, return the gathered features and the trimmed signal
    return signal_features, signal


def extract_mfcc(signal, signal_sr=SR, n_fft=FRAME_LEN, hop_length=HOP, n_mfcc=MFCC_dim):
    """Extracts the Mel-frequency cepstral coefficients (MFCC) from the provided signal
    :param signal: the signal to extract the mfcc from
    :type signal: numpy.ndarray
    :param signal_sr: the signal sample rate
    :type signal_sr: integer
    :param n_fft: the fft window size
    :type n_fft: integer
    :param hop_length: the hop length
    :type hop_length: integer
    :param n_mfcc: the dimension of the mfcc
    :type n_mfcc: integer
    :return: the populated feature vector
    :rtype: numpy.ndarray
    """
    # compute the mfcc of the input signal
    mfcc = librosa.feature.mfcc(
            y=signal, sr=signal_sr, n_fft=n_fft, hop_length=hop_length, n_mfcc=n_mfcc, dct_type=3
    )

    # extract the first and second order deltas from the retrieved mfcc's
    mfcc_delta = librosa.feature.delta(mfcc, order=1)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    # create the mfcc array
    mfccs = []

    # populate it using the extracted features
    for i in range(n_mfcc):
        mfccs.extend(sta_fun(mfcc[i, :]))
    for i in range(n_mfcc):
        mfccs.extend(sta_fun(mfcc_delta[i, :]))
    for i in range(n_mfcc):
        mfccs.extend(sta_fun(mfcc_delta2[i, :]))

    # finally return the coefficients
    return mfccs


def brown_data_features(signal, sampling_rate):
    # extract the signal features
    signal_features, trimmed_signal = extract_signal_features(signal, sampling_rate)

    # extract the mfcc's from the trimmed signal and get the statistical feature.
    mfccs = extract_mfcc(trimmed_signal)

    return np.concatenate((signal_features, mfccs), axis=0)


def read_audio_n_process(file, base_path, sampling_rate, sample_size_in_seconds, overlap, normalise, method):
    """
    This method is called by the preprocess data method
    :param file:
    :param base_path:
    :param sampling_rate:
    :param sample_size_in_seconds:
    :param overlap:
    :param normalise:
    :return:
    """
    data = defaultdict(list)
    filepath = base_path + file
    if os.path.exists(filepath):
        filenames = glob.glob(filepath + '/*.wav')
        for audio_file in filenames:
            try:
                audio, _ = librosa.load(audio_file, sr=sampling_rate)
            except Exception as e:
                print(e)
                continue
            if librosa.get_duration(audio, sr=sampling_rate) < 2:
                continue
            chunks = cut_audio(audio, sampling_rate=sampling_rate, sample_size_in_seconds=sample_size_in_seconds,
                               overlap=overlap)
            for chunk in chunks:
                mean_ = np.float(np.mean(chunk))
                mean_ = 0 if np.isnan(mean_) else mean_
                chunk = np.nan_to_num(chunk, nan=mean_, posinf=mean_, neginf=mean_)
                if method == 'fbank':
                    # zero_crossing = librosa.feature.zero_crossing_rate(chunk)
                    # f0 = pysptk.swipe(chunk.astype(np.float64), fs=sampling_rate, hopsize=510, min=60, max=240,
                    #                   otype="f0").reshape(1,
                    #                                       -1)
                    # pitch = pysptk.swipe(chunk.astype(np.float64), fs=sampling_rate, hopsize=510, min=60, max=240,
                    #                      otype="pitch").reshape(
                    #         1, -1)
                    # f0_pitch_multiplier = 1

                    # f0 = np.reshape(f0[:, :features.shape[1] * f0_pitch_multiplier], newshape=(f0_pitch_multiplier, -1))
                    # pitch = np.reshape(pitch[:, :features.shape[1] * f0_pitch_multiplier],
                    #                    newshape=(f0_pitch_multiplier, -1))
                    # shimmer_jitter = get_shimmer_jitter_from_opensmile(chunk, time.time(), sr)
                    # shimmer_jitter = np.tile(shimmer_jitter, math.ceil(features.shape[-1] / len(shimmer_jitter)))[
                    #                  :features.shape[
                    #                      -1]]  # Repeating the values to match the features length of filterbanks
                    # shimmer_jitter = np.reshape(shimmer_jitter, newshape=(1, -1))
                    # features = np.concatenate((features, zero_crossing, f0, pitch), axis=0)  # shimmer_jitter
                    features = mel_filters(chunk, sampling_rate, normalise)
                elif method == 'mfcc':
                    features = mfcc_features(chunk, sampling_rate, normalise)
                elif method == 'raw':
                    features = chunk
                elif method == 'brown':
                    features = brown_data_features(chunk, sampling_rate)
                elif method == 'vggish':
                    features = vggish_features(chunk)
                else:
                    raise Exception(
                            'Specify a method to use for pre processing raw audio signal. '
                            'Available options - {fbank, mfcc, gaf, raw, brown, vggish}')
                data[audio_file.split('/')[-1]].append(features)
        pickle.dump(dict(data), open(filepath + '/' + method + '.pkl', 'wb'))
    else:
        print('File not found ', filepath)


def read_audio_n_generate_vggish(file, base_path, sampling_rate, sample_size_in_seconds, overlap, normalise, method):
    """
    This method is called by the preprocess data method
    :param file:
    :param base_path:
    :param sampling_rate:
    :param sample_size_in_seconds:
    :param overlap:
    :param normalise:
    :return:
    """
    data = defaultdict(list)
    filepath = base_path + file
    if os.path.exists(filepath):
        filenames = glob.glob(filepath + '/*.wav')
        for audio_file in filenames:
            try:
                audio, _ = librosa.load(audio_file, sr=sampling_rate)
            except Exception as e:
                print(e)
                continue
            if librosa.get_duration(audio, sr=sampling_rate) < 2:
                continue
            chunks = cut_audio(audio, sampling_rate=sampling_rate,
                               sample_size_in_seconds=sample_size_in_seconds,
                               overlap=overlap)
            chunks = np.array(chunks)
            features = vggish_features(chunks)
            data[audio_file.split('/')[-1]].append(features)
        pickle.dump(dict(data), open(filepath + '/' + method + '.pkl', 'wb'))
    else:
        print('File not found ', filepath)


def preprocess_data(base_path, files, normalise, sample_size_in_seconds, sampling_rate, overlap, method):
    # Parallel(n_jobs=os.cpu_count(), backend='multiprocessing')(
    #         delayed(read_audio_n_process)(file, base_path, sampling_rate, sample_size_in_seconds, overlap,
    #                                       normalise, method) for file in
    #         tqdm(files, total=len(files)))
    for e, file in tqdm(enumerate(files), total=len(files)):
        print('Processing ', e, '/', len(files))
        read_audio_n_process(file, base_path, sampling_rate, sample_size_in_seconds, overlap, normalise, method)
    # read_audio_n_generate_vggish(file, base_path, sampling_rate, sample_size_in_seconds, overlap, normalise, method)
    tensorflow_close()

    # for per_file_data in aggregated_data:
    #     # per_file_data[1] are labels for the audio file.
    #     # Might be an array as one audio file can be split into many pieces based on sample_size_in_seconds parameter
    #     for i, label in enumerate(per_file_data[1]):
    #         # per_file_data[0] is array of audio samples based on sample_size_in_seconds parameter
    #         data.append(per_file_data[0][i])
    #         raw.append(per_file_data[2][i])
    #         out_labels.append(label)
    # return data, out_labels, raw

    ############################## TESTING ##############################
    # file = '/Users/badgod/Downloads/musicradar-303-style-acid-samples/High Arps/132bpm/AM_HiTeeb[A]_132D.wav'
    # note, sr = librosa.load(file)
    # print(note.shape)
    # list_y = get_audio_list(note)
    # print([librosa.output.write_wav(
    #         "/Users/badgod/Downloads/musicradar-303-style-acid-samples/High Arps/132bpm/" + str(i) + ".wav", x, 22050) for
    #     i, x
    #     in enumerate(list_y)])

    # def mfcc():
    #     # this is mel filters + dct
    #     file_name = '/Users/badgod/Downloads/AC_12Str85F-01.mp3'
    #     audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    #
    #     print("audio, sample_rate", audio.shape, sample_rate)
    #     mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    #     print(mfccs.shape)
    #     print(np.max(mfccs), np.min(mfccs))
    #     # exit()
    #     mfccsscaled = np.mean(mfccs.T, axis=0)
    #     print(mfccsscaled.shape)
    #
    #     plt.figure(figsize=(12, 4))
    #     # plt.plot(audio)
    #     # plt.plot(mfccsscaled)
    #     librosa.display.specshow(mfccs, sr=sample_rate, x_axis='time')
    #     plt.show()

    #

    # mfcc()

    # def mel_filters_x():
    #     file_name = '/Users/badgod/badgod_documents/Projects/Alco_audio/data/ALC/DATA/audio_2.wav'
    #     audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    #
    #     print("audio, sample_rate", audio.shape, sample_rate)
    #     # plt.plot(range(len(audio)), audio)
    #     # plt.savefig('/Users/badgod/badgod_documents/Projects/Alco_audio/raw_signal.jpg')
    #     # plt.show()
    #     # exit()
    #     logmel = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=40)
    #     print("melspectrogram ", logmel.shape)
    #     # exit()
    #     print(np.min(logmel), np.max(logmel))
    #     S_dB = librosa.power_to_db(logmel, ref=np.max)
    #     print(np.min(S_dB), np.max(S_dB))
    #     print(S_dB[0].shape)
    #     print(S_dB.shape)
    #     print(S_dB.mean())
    #     # S_dB = S_dB / 255
    #     print(S_dB.mean())
    #     # exit()
    #     # S_dB = np.mean(S_dB.T, axis=0)
    #     # print(S_dB.shape)
    #
    #     # plt.figure(figsize=(12, 8))
    #     # plt.plot(audio)
    #     # plt.plot(mfccsscaled)
    #     # librosa.display.specshow(logmel, sr=sample_rate, x_axis='time')
    #     librosa.display.specshow(S_dB, sr=sample_rate)
    #     plt.xlabel('Time')
    #     plt.ylabel('Mels')
    #     plt.savefig("/Users/badgod/badgod_documents/Projects/Alco_audio/test_40mels.jpg")
    #
    #     # plt.plot(S_dB)
    #     plt.show()
    #
    #     plt.close()
    #
    #
    # mel_filters_x()
    #
    #
    # import matplotlib.pyplot as plt
    # import numpy as np
    #
    # data = np.load("/Users/badgod/badgod_documents/Alco_audio/small_data/40_mels/train_challenge_data.npy",
    #                allow_pickle=True)
    # print(data.shape)
    #
    # data_means = np.array([x.mean() for x in data])
    #
    # plt.hist(data_means)
    # plt.show()
    ############################## TESTING ##############################
