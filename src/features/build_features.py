import pickle
import numpy as np
import pandas as pd
from pyts.utils import windowed_view
from scipy.signal import stft
from pyts.classification import LearningShapelets


def get_shapelet_distances(
    df,
    raw_signals,
    y,
    num_shapelets,
    load=False,
    save_path="dataset/shaplet_transform_clf.pkl",
):
    """
    Get distances between time series and shapelets.

    Parameters:
    df (pd.DataFrame): DataFrame to store shapelet distances.
    raw_signals (np.ndarray): Raw signals.
    y (np.ndarray): Target labels.
    num_shapelets (int): Number of shapelets.
    load (bool): Load classifier from file.
    save_path (str): Path to save the classifier.

    Returns:
    df: DataFrame with added shapelet distance features.
    clf: Shapelet classifier.
    """
    print("Finding shapelets. It might take some time")
    if load:
        with open(save_path, "rb") as f:
            clf = pickle.load(f)
    else:
        clf = LearningShapelets(random_state=42, tol=0.01, class_weight="balanced")
        clf.fit(raw_signals, y)
        with open(save_path, "wb") as f:
            pickle.dump(clf, f)

    coefs = clf.coef_[0]
    sorted_coefs = np.argsort(coefs)[::-1][:num_shapelets]

    for i, ind in enumerate(sorted_coefs):
        shapelets = np.asarray([clf.shapelets_[0, ind]])

        shapelet_size = shapelets.shape[1]
        X_window = windowed_view(raw_signals, window_size=shapelet_size, window_step=1)
        X_dist = np.mean((X_window[:, :, None] - shapelets[None, :]) ** 2, axis=3).min(
            axis=1
        )
        df["shapelet_" + str(i)] = X_dist.ravel()

    return df, clf


def get_stft_features(signal, fs, nperseg):
    """
    Compute Short-Time Fourier Transform (STFT) features for a signal.

    Parameters:
    signal (np.ndarray): Input signal.
    fs (float): Sampling frequency.
    nperseg (int): Length of each segment.

    Returns:
    freq (np.ndarray): Frequency array.
    time (np.ndarray): Time array.
    zxx_mag (list): Magnitude of STFT coefficients.
    """
    freq, time, zxx = stft(signal, fs, nperseg=nperseg)
    zxx_mag = np.abs(zxx).flatten().tolist()
    return freq, time, zxx_mag


def get_stft_X(df, signals, fs=1, nperseg=20, no_traditional=False):
    """
    Compute STFT features for a set of signals.

    Parameters:
    df (pd.DataFrame): DataFrame containing traditional features.
    signals (np.ndarray): Array of signals.
    fs (float): Sampling frequency.
    nperseg (int): Length of each segment.
    no_traditional (bool): Flag to exclude traditional features.

    Returns:
    X (np.ndarray): Feature matrix.
    """
    X = []
    i = 0
    for idx, row in df.iterrows():
        new_features = row.values.tolist() if not no_traditional else []
        signal = signals[i]
        freq, time, stft_features = get_stft_features(signal, fs, nperseg)
        new_features.extend(stft_features)
        X.append(new_features)
        i += 1
    return np.array(X)
