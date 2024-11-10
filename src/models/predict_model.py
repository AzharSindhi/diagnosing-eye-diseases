from sklearn.metrics import classification_report


def test_model(model, X_test, y_test):
    """
    Test a trained model and print classification report.

    Parameters:
    model : sklearn.base.BaseEstimator
        The trained model to be tested.
    X_test : array-like or pd.DataFrame
        Testing feature matrix.
    y_test : array-like
        True labels for the testing set.

    Returns:
    None
    """
    y_pred = model.predict(X_test)
    class_names = ["healthy", "unhealthy"]
    print(classification_report(y_test, y_pred, target_names=class_names), "\n")
