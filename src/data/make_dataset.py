import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def prepare_data(path):
    """
    Load and preprocess the dataset.

    Parameters:
    path (str): The path to the dataset file.

    Returns:
    new_data: Preprocessed dataset.
    """
    data = pd.read_excel(path, sheet_name=0)
    signal_type = "Maximum 2.0 ERG Response"
    new_data = pd.DataFrame()
    for i in range(7):
        row = data.iloc[i, :].values
        new_data[row[0]] = row[1:]
    new_data["signal_type"] = signal_type
    new_data["signal"] = [data[col].values[9:210].tolist() for col in data.columns[1:]]
    new_data["Diagnosis"] = new_data["Diagnosis"].map({"healthy": 0, "unhealthy": 1})
    new_data = new_data.dropna(axis=0)
    return new_data


def get_Xy(df, train_cols, target_col, test_ratio):
    """
    Split dataset into train and test sets.

    Parameters:
    df: The dataset to be split.
    train_cols (list): Columns used for training.
    target_col (str): Column representing target labels.
    test_ratio (float): Ratio of test data.

    Returns:
    train_signals: Train signals.
    test_signals: Test signals.
    X_train: Train features.
    X_test: Test features.
    y_train: Train labels.
    y_test: Test labels.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        df, df[target_col], test_size=test_ratio, random_state=42
    )
    train_signals = np.array(X_train["signal"].values.tolist())
    test_signals = np.array(X_test["signal"].values.tolist())
    X_train, X_test = X_train[train_cols], X_test[train_cols]

    return train_signals, test_signals, X_train, X_test, y_train, y_test
