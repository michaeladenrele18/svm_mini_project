import numpy as np

from model.svm import SVM
from sklearn.model_selection import GridSearchCV


def load_dataset(path):
    """
    Load a CSV dataset where the last column is the +1/-1 label.
    """
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    X = data[:, :-1]
    y = data[:, -1]
    return X, y


def make_param_grid(kernel, X_train):
    """
    Build the hyperparameter grid for a given kernel.

    For RBF, gamma is searched on a log scale centred on sklearn's
    "scale" heuristic: gamma = 1 / (n_features * Var(X)).
    For the polynomial kernel, degree d and offset c are searched jointly with C.
    """
    C_values = [0.1, 1, 10, 100]

    if kernel == "rbf":
        gamma_scale = 1.0 / (X_train.shape[1] * X_train.var())
        gammas = gamma_scale * np.logspace(-2, 2, 5)
        return {"C": C_values, "gamma": gammas}

    if kernel == "poly":
        return {"C": C_values, "degree": [2, 3], "coef0": [0.0, 1.0]}

    return {"C": C_values}


def evaluate(name, kernel, train_path, test_path, cv=5):
    X_train, y_train = load_dataset(train_path)
    X_test, y_test = load_dataset(test_path)

    param_grid = make_param_grid(kernel, X_train)

    search = GridSearchCV(
        SVM(kernel=kernel), param_grid, cv=cv, scoring="accuracy"
    )
    search.fit(X_train, y_train)

    # search.predict uses the best model, refit on the full training set
    correct = int(np.sum(search.predict(X_test) == y_test))
    accuracy = correct / len(y_test)
    n_sv = len(search.best_estimator_.support_vectors_)
    print(
        f"{name}: best params = {search.best_params_}, "
        f"CV accuracy = {search.best_score_:.4f}, "
        f"test accuracy = {accuracy:.4f} ({correct}/{len(y_test)}), "
        f"support vectors = {n_sv}/{len(y_train)}"
    )


if __name__ == "__main__":
    evaluate(
        "Linearly separable (linear kernel)",
        "linear",
        "data/datasets/synthetic/linear_separable_train.csv",
        "data/datasets/synthetic/linear_separable_test.csv",
    )

    evaluate(
        "Overlapping classes (linear kernel)",
        "linear",
        "data/datasets/synthetic/overlapping_train.csv",
        "data/datasets/synthetic/overlapping_test.csv",
    )

    evaluate(
        "Nonlinear circles (linear kernel)",
        "linear",
        "data/datasets/synthetic/nonlinear_circles_train.csv",
        "data/datasets/synthetic/nonlinear_circles_test.csv",
    )

    evaluate(
        "Nonlinear circles (RBF kernel)",
        "rbf",
        "data/datasets/synthetic/nonlinear_circles_train.csv",
        "data/datasets/synthetic/nonlinear_circles_test.csv",
    )

    evaluate(
        "Nonlinear circles (polynomial kernel)",
        "poly",
        "data/datasets/synthetic/nonlinear_circles_train.csv",
        "data/datasets/synthetic/nonlinear_circles_test.csv",
    )

    evaluate(
        "Breast cancer (linear kernel)",
        "linear",
        "data/datasets/breast_cancer_train.csv",
        "data/datasets/breast_cancer_test.csv",
    )

    evaluate(
        "Breast cancer (RBF kernel)",
        "rbf",
        "data/datasets/breast_cancer_train.csv",
        "data/datasets/breast_cancer_test.csv",
    )

    evaluate(
        "Breast cancer (polynomial kernel)",
        "poly",
        "data/datasets/breast_cancer_train.csv",
        "data/datasets/breast_cancer_test.csv",
    )
