# HW11 Problem 2 - Hard-margin linear SVM on the South African heart data
# Use the first 20 rows. Features: ldl, adiposity, typea, obesity, alcohol,
# age. Label: chd. Solve the SVM dual QP with scipy, recover w, w0, the
# support vectors, distances, accuracy and margin, and classify an all-ones
# sample. The second half repeats this with sklearn's SVC for comparison.

from sklearn import svm
import numpy as np
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import minimize

data = np.loadtxt(
    'SAheart.data', delimiter=",", skiprows=1, usecols=[3, 4, 6, 7, 8, 9, 10])
data = data[:20, :]
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


bounds = Bounds(lb=np.zeros(len(Y)))
linear_constraint = LinearConstraint(Yb, lb=0, ub=0)
dual = minimize(objective, x0=np.ones(len(Y)), bounds=bounds,
                constraints=linear_constraint)
print(dual)

alphas = dual.x
print(f"Alphas = {alphas}")

w = X.T @ (Yb * alphas)
print(f"w = {w}")

support = np.where(alphas > 1e-6)[0]
w0 = np.mean(Yb[support] - X[support] @ w)
print(f"w0 = {w0}")

print(f"Support vector indices = {support}")

distances = (X @ w + w0) / np.sqrt(np.sum(w ** 2))
signed_class_distances = distances * Yb

print(f"Distances = {distances}")
print(f"Class-signed distances = {signed_class_distances}")

pred_Yb = np.sign(X @ w + w0)
pred_Y = np.where(pred_Yb == 1, 1, 0)

accuracy = np.mean(pred_Y == Y)
print(f"Accuracy = {accuracy}")

margin = 1 / np.sqrt(np.sum(w ** 2))
print(f"Margin = {margin}")

x_new = np.ones(X.shape[1])
pred_new_Yb = np.sign(x_new @ w + w0)
pred_new_Y = 1 if pred_new_Yb == 1 else 0
if pred_new_Y == 1:
    pred_label = "Heart Disease Present"
else:
    pred_label = "Heart Disease not Present"
print(f"Prediction for all features = 1: {pred_label}")


# below is the sklearn implementation
model = svm.SVC(kernel='linear')
model.fit(X, Y)

alphas2 = np.zeros(len(Y))
alphas2[model.support_] = np.abs(model.dual_coef_[0])

print("Alphas =", alphas2)

w = model.coef_.flatten()
print("w =", w)

w0 = model.intercept_[0]
print("w0 =", w0)

print("Support vector indices =", model.support_)

distances = (X @ w + w0) / np.linalg.norm(w)
print("Distances =", distances)

Y_pred = model.predict(X)
accuracy = np.mean(Y_pred == Y)
print("Accuracy =", accuracy)

margin = 1 / np.linalg.norm(w)
print("Margin =", margin)

x_new = np.ones(X.shape[1])
pred = model.predict([x_new])[0]


if pred == 1:
    print("Predicted class: Heart disease present")
else:
    print("Predicted class: Heart disease not present")
