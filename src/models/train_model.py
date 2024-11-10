from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from src.features.build_features import get_stft_X
from src.features.build_features import get_shapelet_distances
from src.models.predict_model import test_model
import numpy as np


def train_kfold(model, X_train, y_train, kfolds):
    """
    Train a model using k-fold cross-validation and return the trained model and mean average precision.

    Parameters:
    model : sklearn.base.BaseEstimator
        The model to be trained.
    X_train : array-like
        Training feature matrix.
    y_train : array-like
        True labels for the training set.
    kfolds : int
        Number of folds for cross-validation.

    Returns:
    model : sklearn.base.BaseEstimator
        Trained model.
    mean_precision : float
        Mean average precision score.
    """
    skf = StratifiedKFold(n_splits=kfolds, shuffle=True, random_state=42)

    average_precisions = []

    for fold, (tr, te) in enumerate(skf.split(X_train, y_train)):
        X_tr, X_te = X_train[tr], X_train[te]
        y_tr, y_te = y_train[tr], y_train[te]

        model.fit(X_tr, y_tr)
        y_pr = model.predict_proba(X_te)[:, 1]
        aps = average_precision_score(y_te, y_pr)
        average_precisions.append(aps)

    return model, np.mean(average_precisions)


def train_traditional(Xtrain, Xtest, ytrain, ytest):
    """
    Train a traditional model on the data and evaluate its performance.

    Parameters:
    Xtrain : pd.DataFrame
        Training feature matrix.
    Xtest : pd.DataFrame
        Testing feature matrix.
    ytrain : pd.Series
        True labels for the training set.
    ytest : pd.Series
        True labels for the testing set.

    Returns:
    mean_precision : float
        Mean average precision score.
    """
    model = RandomForestClassifier(n_jobs=-1, random_state=42)
    model, mean_precision = train_kfold(model, Xtrain.values, ytrain.values, 10)

    # test model
    test_model(model, Xtest.values, ytest.values)
    return mean_precision


def train_shapelet(
    Xtrain, Xtest, ytrain, ytest, train_signals, test_signals, num_shapelets
):
    """
    Train a shapelet-based model on the data and evaluate its performance.

    Parameters:
    Xtrain : pd.DataFrame
        Training feature matrix.
    Xtest : pd.DataFrame
        Testing feature matrix.
    ytrain : pd.Series
        True labels for the training set.
    ytest : pd.Series
        True labels for the testing set.
    train_signals : np.ndarray
        Training signals.
    test_signals : np.ndarray
        Testing signals.
    num_shapelets : int
        Number of shapelets to consider.

    Returns:
    mean_precision : float
        Mean average precision score.
    """
    # get shapelet and their distances
    df_train, _ = get_shapelet_distances(
        Xtrain, train_signals, ytrain, num_shapelets, load=False
    )
    df_test, _ = get_shapelet_distances(
        Xtest, test_signals, ytest, num_shapelets, load=True
    )
    print("training with columns:", df_train.columns)
    model = RandomForestClassifier(n_jobs=-1, random_state=42)
    model, mean_precision = train_kfold(model, df_train.values, ytrain.values, 10)

    # test model
    test_model(model, df_test.values, ytest)
    return mean_precision
