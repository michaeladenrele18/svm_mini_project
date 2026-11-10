
import numpy as np


def linear_kernel(X1, X2):
    """
    Linear kernel: K(x, z) = x . z

    Parameters:
        X1: (n_samples_1, n_features)
        X2: (n_samples_2, n_features)

    Returns:
        K: (n_samples_1, n_samples_2)
    """
    return X1 @ X2.T

# create and add kernel functions below


def rbf_kernel(X, Z=None, gamma=1.0):
    X = np.asarray(X, dtype=float)
    if Z is None:
        Z = X
    else:
        Z = np.asarray(Z, dtype=float)

    distances = (
        np.sum(X**2, axis=1).reshape(-1, 1)
        + np.sum(Z**2, axis=1).reshape(1, -1)
        - 2 * X @ Z.T
    )

    distances = np.maximum(distances, 0)
    K = np.exp(-gamma * distances)
    return K
