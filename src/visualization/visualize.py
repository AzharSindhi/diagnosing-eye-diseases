import matplotlib.pyplot as plt
import numpy as np
from pyts.classification import LearningShapelets
from pyts.utils import windowed_view


def plot_learning_shapelets(X, y):
    """
    Plot learned shapelets and their distances.

    Parameters:
        X (numpy.ndarray): Input time series data.
        y (numpy.ndarray): Corresponding target labels.

    Returns:
        None
    """
    # Load the data set and fit the classifier
    print("data shape:", X.shape)
    clf = LearningShapelets(random_state=42, tol=0.01)
    clf.fit(X, y)

    # Select two shapelets
    shapelets = np.asarray([clf.shapelets_[0, -9], clf.shapelets_[0, -12]])

    # Derive the distances between the time series and the shapelets
    shapelet_size = shapelets.shape[1]
    print("shapelet size:", shapelet_size)
    X_window = windowed_view(X, window_size=shapelet_size, window_step=1)
    X_dist = np.mean((X_window[:, :, None] - shapelets[None, :]) ** 2, axis=3).min(
        axis=1
    )
    print("distance shape;", X_dist.shape)
    plt.figure(figsize=(14, 4))

    # Plot the two shapelets
    plt.subplot(1, 2, 1)
    plt.plot(shapelets[0])
    plt.plot(shapelets[1])
    plt.title("Two learned shapelets", fontsize=14)

    # Plot the distances
    plt.subplot(1, 2, 2)
    plt.hist(X_dist[y == 1, 0], bins=50, color="r", label="Class 1 shapelet 1")
    plt.hist(X_dist[y == 2, 0], bins=50, color="b", label="Class 2 shapelet 1")
    plt.title("Distances between the time series and both shapelets", fontsize=14)
    plt.legend()
    plt.savefig("figs/shapelet.png")


def example_signals(new_data):
    """
    Plot example healthy and unhealthy signals.

    Parameters:
        new_data (pandas.DataFrame): DataFrame containing signals and diagnoses.

    Returns:
        None
    """
    unhealthy_signals = new_data[new_data["Diagnosis"] == 1]["signal"].values
    healthy_signals = new_data[new_data["Diagnosis"] == 0]["signal"].values
    np.random.shuffle(unhealthy_signals)
    np.random.shuffle(healthy_signals)
    fig, ax = plt.subplots(2, 4)
    for k, signal in enumerate(healthy_signals[:4]):
        ax[0][k].plot(signal, color="b")
    for k, signal in enumerate(unhealthy_signals[:4]):
        ax[1][k].plot(signal, color="r")


def plot_count(new_data):
    """
    Plot counts of healthy and unhealthy diagnoses.

    Parameters:
        new_data (pandas.DataFrame): DataFrame containing diagnoses.

    Returns:
        None
    """
    new_data["Diagnosis"].map({1: "unhealthy", 0: "healthy"}).value_counts().plot.bar()
