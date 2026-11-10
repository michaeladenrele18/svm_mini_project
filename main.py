import numpy as np

from model.svm import SVM
from kernels.kernels import rbf_kernel, linear_kernel


def load_dataset(path):
    """
    Load a CSV dataset where the last column is the +1/-1 label.
    """
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    X = data[:, :-1]
    y = data[:, -1]
    return X, y


def evaluate(name, kernel, train_path, test_path, C=1.0):
    X_train, y_train = load_dataset(train_path)
    X_test, y_test = load_dataset(test_path)

    clf = SVM(C=C, kernel=kernel)
    clf.fit(X_train, y_train)

    accuracy = np.mean(clf.predict(X_test) == y_test)
    print(f"{name}: accuracy = {accuracy:.3f}")


if __name__ == "__main__":
    evaluate(
        "Linearly separable (linear kernel)",
        linear_kernel,
        "data/datasets/synthetic/linear_separable_train.csv",
        "data/datasets/synthetic/linear_separable_test.csv",
    )

    evaluate(
        "Overlapping classes (linear kernel)",
        linear_kernel,
        "data/datasets/synthetic/overlapping_train.csv",
        "data/datasets/synthetic/overlapping_test.csv",
    )

    evaluate(
        "Nonlinear circles (linear kernel)",
        linear_kernel,
        "data/datasets/synthetic/nonlinear_circles_train.csv",
        "data/datasets/synthetic/nonlinear_circles_test.csv",
    )

    evaluate(
        "Nonlinear circles (RBF kernel)",
        rbf_kernel,
        "data/datasets/synthetic/nonlinear_circles_train.csv",
        "data/datasets/synthetic/nonlinear_circles_test.csv",
    )

    evaluate(
        "Nonlinear circles (polynomial kernel, degree=2)",
        linear_kernel,
        "data/datasets/synthetic/nonlinear_circles_train.csv",
        "data/datasets/synthetic/nonlinear_circles_test.csv",
    )

    evaluate(
        "Breast cancer (linear kernel)",
        linear_kernel,
        "data/datasets/breast_cancer_train.csv",
        "data/datasets/breast_cancer_test.csv",
    )
