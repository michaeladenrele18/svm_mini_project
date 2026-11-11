
import numpy as np
from cvxopt import matrix, solvers
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array
from sklearn.svm import SVC

from functools import partial

from kernels.kernels import linear_kernel, rbf_kernel, polynomial_kernel

KERNELS = {"linear": linear_kernel, "rbf": rbf_kernel, "poly": polynomial_kernel}


def compute_gram_matrix(X, kernel=linear_kernel):
    """
    Compute the Gram matrix for a given kernel.

    Parameters:
        X: Training feature matrix (n_samples, n_features)
        kernel: Kernel function K(X1, X2) -> (len(X1), len(X2))

    Returns:
        K: Gram matrix (n_samples, n_samples)
    """
    K = kernel(X, X)

    return K


def compute_P(X, y, kernel=linear_kernel):
    """
    Construct the quadratic matrix P for the dual SVM.

    P[i, j] = y[i] * y[j] * kernel(X[i], X[j])

    Parameters:
        X: Training feature matrix (n_samples, n_features)
        y: Class labels (n_samples,), containing -1 or +1
        kernel: Kernel function K(X1, X2) -> (len(X1), len(X2))

    Returns:
        P: Quadratic matrix (n_samples, n_samples)
    """

    K = compute_gram_matrix(X, kernel)

    # Outer product of class labels
    yyT = np.outer(y, y)

    # Element-wise multiplication
    P = yyT * K

    return P


def compute_q(X):
    """
    Construct the linear vector q for the dual SVM.

    q = [-1, -1, ..., -1]

    Parameters:
        X: Training feature matrix (n_samples, n_features)

    Returns:
        q: Linear vector (n_samples,)
    """

    n_samples = X.shape[0]  # gets the number of training observations

    # create a vector of -1's with length equal to the number of training observations
    q = -np.ones(n_samples)

    return q


def compute_inequality_constraints(X, C):
    """
    Construct G and h for the constraints:

        0 <= alpha_i <= C

    Parameters:
        X: Training feature matrix
        C: Regularization parameter (positive)

    Returns:
        G: Inequality matrix (2*n_samples, n_samples)
        h: Inequality vector (2*n_samples,)
    """

    if C <= 0:
        raise ValueError("C must be positive.")

    n_samples = X.shape[0]  # gets the number of training observations

    I = np.eye(n_samples)   # identity matrix of size n_samples

    # stack -I and I vertically to form the inequality matrix
    G = np.vstack((-I, I))

    h = np.hstack((
        np.zeros(n_samples),
        C * np.ones(n_samples)
    ))

    return G, h


def compute_equality_constraints(y):
    """
    Construct the equality constraint:

        sum(alpha_i * y_i) = 0

    Parameters:
        y: Training labels (n_samples,)

    Returns:
        A: Equality matrix (1, n_samples)
        b: Equality vector (1,)
    """

    A = np.asarray(y, dtype=float).reshape(1, -1)

    b = np.array([0.0])

    return A, b


def solve_dual_svm(X, y, C=1.0, kernel=linear_kernel):
    """
    Solve the soft-margin SVM dual QP.

    Returns:
        alpha: Optimal Lagrange multipliers
    """

    # Step 1: Construct the objective
    P = compute_P(X, y, kernel)
    q = compute_q(X)

    # Step 2: Construct the constraints
    G, h = compute_inequality_constraints(X, C)
    A, b = compute_equality_constraints(y)

    # Step 3: Convert to CVXOPT matrices CVXOPT expects its own matrix objects rather than ordinary NumPy arrays. The "d" specifies double-precision floating-point values.
    P = matrix(P, tc="d")
    q = matrix(q, tc="d")
    G = matrix(G, tc="d")
    h = matrix(h, tc="d")
    A = matrix(A, tc="d")
    b = matrix(b, tc="d")

    # Step 4: Solve the quadratic program
    solvers.options["show_progress"] = False

    solution = solvers.qp(P, q, G, h, A, b)

    if solution["status"] != "optimal":
        raise RuntimeError(
            f"QP solver failed: {solution['status']}"
        )

    # Step 5: Extract optimal multipliers
    alpha = np.array(solution["x"]).flatten()

    return alpha


def identify_support_vectors(X, y, alpha, tol=1e-5):
    """
    Identify support vectors from optimal multipliers.

    Parameters:
        X: Training features (n_samples, n_features)
        y: Training labels (n_samples,)
        alpha: Optimal Lagrange multipliers (n_samples,)
        tol: Numerical tolerance

    Returns:
        support_vectors: Feature vectors of support vectors
        support_labels: Labels of support vectors
        support_alpha: Nonzero multipliers
    """

    # Identify nonzero multipliers
    mask = alpha > tol

    # Select corresponding observations
    support_vectors = X[mask]
    support_labels = y[mask]
    support_alpha = alpha[mask]

    return support_vectors, support_labels, support_alpha


def compute_bias(X, y, alpha, C, kernel=linear_kernel, tol=1e-5):
    """
    Calculate the bias using margin support vectors,
    via the kernel trick.

    b = mean( y_i - sum_j(alpha_j * y_j * K(x_j, x_i)) )
    over margin support vectors i (0 < alpha_i < C).

    Parameters:
        X: Training features (n_samples, n_features)
        y: Training labels (n_samples,)
        alpha: Optimal Lagrange multipliers
        C: Regularization parameter
        kernel: Kernel function K(X1, X2) -> (len(X1), len(X2))
        tol: Numerical tolerance

    Returns:
        b: SVM bias (scalar)
    """

    sv_mask = alpha > tol
    margin_mask = (alpha > tol) & (alpha < C - tol)

    if not np.any(margin_mask):
        raise ValueError(
            "No margin support vectors found. "
            "A KKT-based bias fallback is required."
        )

    # Kernel values between margin support vectors and all support vectors
    K_margin = kernel(X[margin_mask], X[sv_mask])

    # Decision scores (without bias) for the margin support vectors
    decision = K_margin @ (alpha[sv_mask] * y[sv_mask])

    # Average to reduce numerical error
    b = np.mean(y[margin_mask] - decision)

    return float(b)


def decision_function(X, support_vectors, support_labels, support_alpha, b, kernel=linear_kernel):
    """
    Calculate decision scores via the kernel trick.

    f(x) = sum(alpha_i * y_i * K(sv_i, x)) + b

    Parameters:
        X: Observations (n_samples, n_features)
        support_vectors: Support vector features
        support_labels: Support vector labels
        support_alpha: Support vector multipliers
        b: Bias (scalar)
        kernel: Kernel function K(X1, X2) -> (len(X1), len(X2))

    Returns:
        scores: Decision scores (n_samples,)
    """

    K = kernel(X, support_vectors)

    scores = K @ (support_alpha * support_labels) + b

    return scores


def predict(X, support_vectors, support_labels, support_alpha, b, kernel=linear_kernel):
    """
    Predict binary class labels using the kernel trick.

    Parameters:
        X: Observations (n_samples, n_features)
        support_vectors: Support vector features
        support_labels: Support vector labels
        support_alpha: Support vector multipliers
        b: Bias (scalar)
        kernel: Kernel function K(X1, X2) -> (len(X1), len(X2))

    Returns:
        predictions: Predicted labels (-1 or +1)
    """

    scores = decision_function(
        X, support_vectors, support_labels, support_alpha, b, kernel
    )

    predictions = np.where(scores >= 0, 1, -1)

    return predictions


class SVM(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, kernel="linear", gamma=1.0, degree=2, coef0=1.0):
        """
        Initialize a soft-margin SVM.

        Following sklearn conventions, __init__ only stores the
        hyperparameters so that clone()/GridSearchCV work correctly.
        Validation happens in fit().

        Parameters:
            C: Regularization parameter (positive)
            kernel: "linear", "rbf", "poly", or a callable K(X1, X2) -> (len(X1), len(X2))
            gamma: RBF kernel width parameter (only used when kernel="rbf")
            degree: Polynomial degree d (only used when kernel="poly")
            coef0: Polynomial offset c (only used when kernel="poly")
        """
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0

    def _get_kernel(self):
        """
        Resolve self.kernel into a callable K(X1, X2), binding any
        kernel hyperparameters (e.g. gamma).
        """
        if isinstance(self.kernel, str):
            if self.kernel not in KERNELS:
                raise ValueError(
                    f"Unknown kernel '{self.kernel}'. "
                    f"Choose from {list(KERNELS)} or pass a callable."
                )
            k = KERNELS[self.kernel]
        elif callable(self.kernel):
            k = self.kernel
        else:
            raise ValueError("kernel must be a string or a callable.")

        if k is rbf_kernel:
            return partial(rbf_kernel, gamma=self.gamma)
        if k is polynomial_kernel:
            return partial(polynomial_kernel, degree=self.degree, coef0=self.coef0)
        return k

    def fit(self, X, y):
        """
        Train the SVM using dual quadratic optimization.

        Parameters:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)

        Returns:
            self: Trained SVM model
        """
        if self.C <= 0:
            raise ValueError("C must be positive.")

        if self.gamma <= 0:
            raise ValueError("gamma must be positive.")

        if self.kernel == "poly" and (int(self.degree) != self.degree or self.degree < 1):
            raise ValueError("degree must be a positive integer.")

        if self.coef0 < 0:
            raise ValueError("coef0 must be non-negative.")

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        # Validate input
        if X.ndim != 2:
            raise ValueError("X must be a 2D feature matrix.")

        if y.ndim != 1 or len(y) != len(X):
            raise ValueError(
                "y must contain one label per observation."
            )

        if not np.all(np.isin(y, [-1, 1])):
            raise ValueError("Labels must be -1 or +1.")

        if len(np.unique(y)) != 2:
            raise ValueError(
                "Training data must contain both classes."
            )

        self.classes_ = np.array([-1, 1])
        self.n_features_in_ = X.shape[1]

        # Step 0: Resolve the kernel function (with its hyperparameters)
        self.kernel_ = self._get_kernel()

        # Step 1: Solve the dual optimization problem
        self.alpha_ = solve_dual_svm(X, y, self.C, self.kernel_)

        # Step 2: Identify support vectors
        (
            self.support_vectors_,
            self.support_labels_,
            self.support_alpha_
        ) = identify_support_vectors(X, y, self.alpha_)

        # Step 3: Calculate bias
        self.b_ = compute_bias(
            X, y, self.alpha_, self.C, self.kernel_
        )

        return self

    def decision_function(self, X):
        """
        Calculate decision scores.

        f(x) = sum(alpha_i * y_i * K(sv_i, x)) + b

        Parameters:
            X: Observations (n_samples, n_features)

        Returns:
            scores: Decision scores
        """
        check_is_fitted(self, ["support_vectors_", "b_"])

        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D feature matrix.")

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "Number of features does not match training data."
            )

        return decision_function(
            X,
            self.support_vectors_,
            self.support_labels_,
            self.support_alpha_,
            self.b_,
            self.kernel_
        )

    def predict(self, X):
        """
        Predict binary class labels (-1 or +1).

        Parameters:
            X: Observations (n_samples, n_features)

        Returns:
            predictions: Predicted class labels
        """
        scores = self.decision_function(X)

        predictions = np.where(scores >= 0, 1, -1)

        return predictions
