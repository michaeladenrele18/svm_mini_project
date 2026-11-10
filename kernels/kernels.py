
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
