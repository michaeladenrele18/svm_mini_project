
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
