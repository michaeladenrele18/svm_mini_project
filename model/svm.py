
import numpy as np
from cvxopt import matrix, solvers

def compute_gram_matrix(X):
    """
    Compute the linear Gram matrix.

    Parameters:
        X: Training feature matrix (n_samples, n_features)

    Returns:
        K: Gram matrix (n_samples, n_samples)
    """
    K = X @ X.T

    return K


def compute_P(X, y):
    """
    Construct the quadratic matrix P for the dual SVM.

    P[i, j] = y[i] * y[j] * (X[i] @ X[j])

    Parameters:
        X: Training feature matrix (n_samples, n_features)
        y: Class labels (n_samples,), containing -1 or +1

    Returns:
        P: Quadratic matrix (n_samples, n_samples)
    """

    K = compute_gram_matrix(X)

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

    n_samples = X.shape[0] # gets the number of training observations

    q = -np.ones(n_samples) # create a vector of -1's with length equal to the number of training observations

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

    n_samples = X.shape[0] # gets the number of training observations

    I = np.eye(n_samples)   # identity matrix of size n_samples

    G = np.vstack((-I, I))   # stack -I and I vertically to form the inequality matrix

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


def solve_dual_svm(X, y, C=1.0):
    """
    Solve the soft-margin SVM dual QP.

    Returns:
        alpha: Optimal Lagrange multipliers
    """

    # Step 1: Construct the objective
    P = compute_P(X, y)
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



def compute_weights(X, y, alpha):
    """
    Calculate the linear SVM weight vector.

    w = sum(alpha_i * y_i * x_i)

    Parameters:
        X: Training features (n_samples, n_features)
        y: Training labels (n_samples,)
        alpha: Optimal Lagrange multipliers (n_samples,)

    Returns:
        w: Weight vector (n_features,)
    """

    # Multiply each alpha by its class label
    weighted_labels = alpha * y

    # Sum the weighted training feature vectors
    w = X.T @ weighted_labels

    return w


def compute_bias(X, y, alpha, w, C, tol=1e-5):
    """
    Calculate the bias for a linear SVM using
    margin support vectors.

    Parameters:
        X: Training features (n_samples, n_features)
        y: Training labels (n_samples,)
        alpha: Optimal Lagrange multipliers
        w: Linear SVM weight vector
        C: Regularization parameter
        tol: Numerical tolerance

    Returns:
        b: SVM bias (scalar)
    """

    # Identify margin support vectors: 0 < alpha_i < C
    mask = (alpha > tol) & (alpha < C - tol)

    if not np.any(mask):
        raise ValueError(
            "No margin support vectors found. "
            "A KKT-based bias fallback is required."
        )

    # Select eligible observations
    X_margin = X[mask]
    y_margin = y[mask]

    # Calculate bias from each margin support vector
    bias_values = y_margin - X_margin @ w

    # Average to reduce numerical error
    b = np.mean(bias_values)

    return float(b)


def decision_function(X, w, b):
    """
    Calculate decision scores for a linear SVM.

    Parameters:
        X: Observations (n_samples, n_features)
        w: Weight vector (n_features,)
        b: Bias (scalar)

    Returns:
        scores: Decision scores (n_samples,)
    """

    # Calculate f(X) = Xw + b
    scores = X @ w + b

    return scores


def predict(X, w, b):
    """
    Predict binary class labels using a linear SVM.

    Parameters:
        X: Observations (n_samples, n_features)
        w: Weight vector (n_features,)
        b: Bias (scalar)

    Returns:
        predictions: Predicted labels (-1 or +1)
    """

    # Calculate decision scores
    scores = decision_function(X, w, b)

    # Convert scores into class labels
    predictions = np.where(scores >= 0, 1, -1)

    return predictions


class SVM:
    def __init__(self, C=1.0):
        """
        Initialize a linear soft-margin SVM.

        Parameters:
            C: Regularization parameter
        """
        if C <= 0:
            raise ValueError("C must be positive.")

        self.C = C

        # Parameters learned during training
        self.alpha = None
        self.w = None
        self.b = None

        # Support vector information
        self.support_vectors = None
        self.support_labels = None
        self.support_alpha = None

    def fit(self, X, y):
        """
        Train the SVM using dual quadratic optimization.

        Parameters:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)

        Returns:
            self: Trained SVM model
        """
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

        # Step 1: Solve the dual optimization problem
        self.alpha = solve_dual_svm(X, y, self.C)

        # Step 2: Identify support vectors
        (
            self.support_vectors,
            self.support_labels,
            self.support_alpha
        ) = identify_support_vectors(X, y, self.alpha)

        # Step 3: Calculate weight vector
        self.w = compute_weights(X, y, self.alpha)

        # Step 4: Calculate bias
        self.b = compute_bias(
            X, y, self.alpha, self.w, self.C
        )

        return self

    def decision_function(self, X):
        """
        Calculate decision scores.

        f(X) = Xw + b

        Parameters:
            X: Observations (n_samples, n_features)

        Returns:
            scores: Decision scores
        """
        if self.w is None or self.b is None:
            raise RuntimeError(
                "The model must be fitted before prediction."
            )

        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D feature matrix.")

        if X.shape[1] != len(self.w):
            raise ValueError(
                "Number of features does not match training data."
            )

        return decision_function(X, self.w, self.b)

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
