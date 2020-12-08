# -*- coding: utf-8 -*-

"""Contains the functions preprocessing the data, or grabing the data"""

import numpy as np
import pandas as pd
from scipy import fftpack  # Fourier
from scipy.ndimage import maximum_filter1d
from librosa.feature import mfcc, spectral_contrast, zero_crossing_rate
from tqdm import tqdm
from tsfresh.feature_extraction import feature_calculators as fc
import pywt
from bokeh.plotting import figure, show

PATH_DATA = "../data/"
COLORS = ["#263554", "#D85604", "#E88D14"]


def plot(data, col):
    fig = figure(title=f"Evolution de {col}", x_range=(0, len(data)),
                 plot_width=800, plot_height=300)
    fig.xaxis.axis_label = 'Time (s)'
    fig.grid.visible = False
    fig.line(x="index", y=col, source=data)
    show(fig)


def load_segment(segment_id):
    """Returns the data about a specific segment_id"""
    try:
        return pd.read_csv(f"{PATH_DATA}train/{segment_id}.csv")
    except FileNotFoundError:
        return pd.read_csv(f"{PATH_DATA}test/{segment_id}.csv")


def get_index(target="train"):
    """Returns the list of segments of a set (train or test)"""
    file = "train" if target == "train" else "sample_submission"
    data = pd.read_csv(f"{PATH_DATA}{file}.csv")
    return data["segment_id"].values


def maddest(serie, axis=None):
    """Returns the mean of de deviation of a serie"""
    return np.mean(np.absolute(serie - np.mean(serie, axis)), axis)


def denoise_signal_simple(x, wavelet='db4', level=1):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    coeff[1:] = (pywt.threshold(i, value=10,
                                mode='hard') for i in coeff[1:])
    return pywt.waverec(coeff, wavelet, mode='per')


def get_features(sig, sensor_id):
    """Analysis of a signal. Grabs temporal and frequential features.
    Returns a pandas dataframe"""

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

    features[f"{sensor_id}_nb_peak"] = fc.number_peaks(sig.values, 2)
    features[f"{sensor_id}_median_roll_std"] = np.median(
        pd.Series(sig).rolling(50).std().dropna().values)
    features[f"{sensor_id}_autocorr5"] = fc.autocorrelation(sig, 5)

    # Added 16
    features[f"{sensor_id}_nb_peak_3"] = fc.number_peaks(sig.values, 3)
    features[f"{sensor_id}_absquant95"] = np.quantile(np.abs(sig), 0.95)

    try:
        # Mel-frequency cepstral coefficients
        mfcc_mean = mfcc(sig.values).mean(axis=1)
        for i in range(20):
            features[f"{sensor_id}_mfcc_mean_{i}"] = mfcc_mean[i]
        # Contrast spectral
        spec_contrast = spectral_contrast(sig.values).mean(axis=1)
        for i in range(7):
            features[f"{sensor_id}_lib_spec_cont_{i}"] = spec_contrast[i]
        features[f"{sensor_id}_zero_cross"] = zero_crossing_rate(sig)[0].mean()
        # Added 16
        features[f"{sensor_id}_percentile_roll20_std_50"] = np.percentile(
            sig.rolling(20).std().dropna().values, 50)

    except:
        pass

# =============================================================================
    # fftrhann20000 = np.sum(np.abs(np.fft.fft(np.hanning(len(z))*z)[:20000]))
    # fftrhann20000_denoise = np.sum(np.abs(np.fft.fft(np.hanning(len(z))*den_sample)[:20000]))
    # fftrhann20000_diff_rate = (fftrhann20000 - fftrhann20000_denoise)/fftrhann20000
    # X['LGBM_fftrhann20000_diff_rate'] = fftrhann20000_diff_rate
# =============================================================================
    return pd.DataFrame.from_dict(features)


def preprocess_data(target="train"):
    """Generates a dataframe containing all the features of a
    dataset (train or test)"""
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


def lr_decay(current_iter):
    return max(1e-1, 0.29 * np.power(.9985, current_iter))
