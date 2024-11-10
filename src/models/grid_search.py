from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def run_grid_search(X, y):
    """
    Perform randomized grid search for different classifiers.

    Parameters:
    X (np.ndarray): Feature matrix.
    y (np.ndarray): Target labels.

    Returns:
    None
    """
    # Define the parameter grid for each model
    decision_tree_params = {
        "max_depth": [None, 3, 5, 7],
        "min_samples_split": [2, 5, 10],
    }

    random_forest_params = {
        "n_estimators": [10, 50, 100],
        "max_depth": [None, 3, 5, 7],
        "min_samples_split": [2, 5, 10],
    }

    logistic_regression_params = {
        "C": [0.1, 1.0, 10.0],
        "solver": ["liblinear", "lbfgs"],
    }

    svm_params = {"C": [0.1, 1.0, 10.0], "kernel": ["linear", "rbf"]}

    # Perform randomized grid search for each model
    models = [
        ("Decision Tree", DecisionTreeClassifier(), decision_tree_params),
        ("Random Forest", RandomForestClassifier(), random_forest_params),
        ("Logistic Regression", LogisticRegression(), logistic_regression_params),
        ("SVM", SVC(), svm_params),
    ]

    for model_name, model, params in models:
        print(f"Performing randomized grid search for {model_name}...")
        random_search = RandomizedSearchCV(
            model, params, n_jobs=2, n_iter=20, cv=3, random_state=42
        )
        random_search.fit(X, y)
        best_params = random_search.best_params_
        print(f"Best parameters for {model_name}:")
        for param, value in best_params.items():
            print(f"{param}: {value}")
        print()
