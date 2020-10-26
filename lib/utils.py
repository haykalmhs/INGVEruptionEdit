# -*- coding: utf-8 -*-
import math
import numpy as np
import pandas as pd
from scipy import fftpack  # Fourier
from sklearn.metrics import mean_squared_error as mse
from scipy.ndimage import maximum_filter1d
from librosa.feature import mfcc, spectral_contrast
from tqdm import tqdm
import pywt

PATH_DATA = "../data/"


def load_segment(id):
    """Returns the data about a specific segment_id"""
    try:
        return pd.read_csv(f"{PATH_DATA}train/{id}.csv")
    except FileNotFoundError:
        return pd.read_csv(f"{PATH_DATA}test/{id}.csv")


def get_index(target="train"):
    """Returns the list of segments of a set (train or test)"""
    file = "train" if target == "train" else "sample_submission"
    data = pd.read_csv(f"{PATH_DATA}{file}.csv")
    return data["segment_id"].values


def maddest(d, axis=None):
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)


def denoise_signal(x, wavelet='db4', level=1):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1/0.6745) * maddest(coeff[-level])
    uthresh = sigma * np.sqrt(2*np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh,
                                mode='hard') for i in coeff[1:])
    return pywt.waverec(coeff, wavelet, mode='per')


def get_features(sig, sensor_id):
    fourier = fftpack.fft(sig.values)
    real, imag = np.real(fourier), np.imag(fourier)

    # Temporal data
    features = {}
    features[f"{sensor_id}_mean"] = [sig.mean()]
    features[f"{sensor_id}_var"] = [sig.var()]
    features[f"{sensor_id}_skew"] = [sig.skew()]
    features[f"{sensor_id}_delta"] = [sig.max() - sig.min()]
    features[f"{sensor_id}_mad"] = [sig.mad()]
    features[f"{sensor_id}_kurtosis"] = [sig.kurtosis()]
    features[f"{sensor_id}_sem"] = [sig.sem()]
    features[f"{sensor_id}_q5"] = [np.quantile(sig, 0.05)]
    features[f"{sensor_id}_q25"] = [np.quantile(sig, 0.25)]
    features[f"{sensor_id}_q75"] = [np.quantile(sig, 0.75)]
    features[f"{sensor_id}_q95"] = [np.quantile(sig, 0.95)]
    grad_rol_max = [maximum_filter1d(np.gradient(np.abs(sig.values)), 50)]
    delta = np.max(grad_rol_max) - np.min(grad_rol_max)
    features[f"{sensor_id}_grmax_delta"] = delta

    # Frequencial
    features[f"{sensor_id}_real_mean"] = [real.mean()]
    features[f"{sensor_id}_real_var"] = [real.var()]
    features[f"{sensor_id}_real_delta"] = [real.max() - real.min()]

    features[f"{sensor_id}_imag_mean"] = [imag.mean()]
    features[f"{sensor_id}_imag_var"] = [imag.var()]
    features[f"{sensor_id}_imag_delta"] = [imag.max() - imag.min()]

    # Mel-frequency cepstral coefficients
    try:
        mfcc_ = mfcc(sig.values)
        mfcc_mean = mfcc_.mean(axis=1)
        for i in range(20):
            features[f"{sensor_id}_mfcc_mean_{i}"] = mfcc_mean[i]
    except:
        pass

    # Contrast spectral
    try:
        sc = spectral_contrast(sig.values).mean(axis=1)
        for i in range(7):
            features[f"{sensor_id}_lib_spec_cont_{i}"] = sc[i]
    except:
        pass

    return pd.DataFrame.from_dict(features)


def rmse(y_true, y_pred):
    return math.sqrt(mse(y_true, y_pred))


def preprocess_data(target="train"):
    data_set = []
    for seg in tqdm(get_index(target)):
        train_row = [pd.DataFrame.from_dict({"segment_id": [seg]})]
        data = load_segment(seg)
        for i in range(10):
            sensor_id = f"sensor_{i+1}"
            train_row.append(get_features(data[sensor_id], sensor_id))
        train_row = pd.concat(train_row, axis=1)
        data_set.append(train_row)

    data_set = pd.concat(data_set).reset_index()
    data_set.fillna(-1, inplace=True)
    data_set.drop(['index'], axis=1, inplace=True)
    return data_set
