# HW12 - Soft-margin linear SVM on the South African heart data (C = 2)
# Use the last 16 rows. Features: ldl, adiposity, typea, obesity, alcohol,
# age. Label: chd. Solve the dual with box constraints 0 <= a_i <= C:
#   A: the dual variables alpha
#   B: the weight vector w = sum(a_i y_i x_i)
#   C: the intercept w0, averaged over margin support vectors (0 < a < C)
#   D: sort points into cases by y_i (w'x_i + w0): (i) > 1, (ii) = 1,
#      (iii) in (0, 1), (iv) < 0
#   E: the support vector indices
#   F: distances, margin, predicted labels and training accuracy
#   G: predict the class of an all-ones sample

import numpy as np
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import minimize

data = np.loadtxt(
    'hard-margin-svm/SAheart.data', delimiter=",", skiprows=1, usecols=[3, 4, 6, 7, 8, 9, 10])
data = data[-16:, :]
X = data[:, :6]
Y = data[:, 6]

Yb = np.zeros(len(Y))
Yb[Y == 1] = 1
Yb[Y == 0] = -1
Yb2d = np.expand_dims(Yb, axis=1)
Q = (X * Yb2d) @ (X * Yb2d).T


def objective(a):
    a2d = np.expand_dims(a, axis=1)
    return 0.5 * a2d.T @ Q @ a2d - np.sum(a)


bounds = Bounds(lb=np.zeros(len(Y)), ub=2*np.ones(len(Y)))
linear_constraint = LinearConstraint(Yb, lb=0, ub=0)
dual = minimize(objective, x0=np.ones(len(Y)), bounds=bounds,
                constraints=linear_constraint)
print(dual)

# A
alphas = dual.x
print(f"Alphas = {alphas}")

# B
w = X.T @ (Yb * alphas)
print(f"w = {w}")

# C
support = np.where(alphas > 1e-6)[0]
margin_support = np.where((alphas > 1e-6) & (alphas < 2 - 1e-6))[0]
w0 = np.mean(Yb[margin_support] - X[margin_support] @ w)
print(f"w0 = {w0}")

# D
signed_margin = Yb * (X @ w + w0)
case_i = np.where(signed_margin > 1)[0]
case_ii = np.where(np.abs(signed_margin - 1) < 1e-6)[0]
case_iii = np.where((signed_margin > 0) & (signed_margin < 1))[0]
case_iv = np.where(signed_margin < 0)[0]

print(f"Case (i):   {case_i}")
print(f"Case (ii):  {case_ii}")
print(f"Case (iii): {case_iii}")
print(f"Case (iv):  {case_iv}")

# E
print(f"Support vector indices = {support}")

# F
distances = (X @ w + w0) / np.sqrt(np.sum(w ** 2))
signed_class_distances = distances * Yb
pred_Yb = np.sign(X @ w + w0)
pred_Y = np.where(pred_Yb == 1, 1, 0)
accuracy = np.mean(pred_Y == Y)
margin = 1 / np.sqrt(np.sum(w ** 2))

print(f"Distances = {distances}")
print(f"Class-signed distances = {signed_class_distances}")
print(f"Margin = {margin}")
print(f"Predicted labels = {pred_Y}")
print(f"True labels = {Y}")
print(f"Accuracy = {accuracy}")

# G
x_new = np.ones(X.shape[1])
pred_new_Yb = np.sign(x_new @ w + w0)
pred_new_Y = 1 if pred_new_Yb == 1 else 0
if pred_new_Y == 1:
    pred_label = "Heart Disease Present"
else:
    pred_label = "Heart Disease not Present"
print(f"Prediction for all features = 1: {pred_label}")
