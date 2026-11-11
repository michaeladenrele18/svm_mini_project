# svm_mini_project

A soft-margin SVM implemented from scratch (dual QP via CVXOPT), with support
for linear, polynomial, and RBF kernels.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## How to run

```bash
python3 main.py
```

This trains the SVM on the synthetic and breast cancer datasets under
`data/datasets/` and prints test accuracy for each kernel.

## Project structure

- `model/svm.py` — the `SVM` class (dual QP solver, support vectors, bias,
  prediction via the kernel trick).
- `kernels/kernels.py` — kernel functions (`linear_kernel`,
  `polynomial_kernel`, `rbf_kernel`) passed into `SVM(kernel=...)`.
- `data/` — datasets, MATLAB preprocessing scripts, and reference figures.
- `main.py` — example script that trains/evaluates the SVM on each dataset.

## Using a different kernel

```python
from model.svm import SVM
from kernels.kernels import rbf_kernel

clf = SVM(C=1.0, kernel=lambda X1, X2: rbf_kernel(X1, X2, gamma=0.5))
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
```
