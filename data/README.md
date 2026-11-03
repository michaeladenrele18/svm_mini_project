# Data Team — MATH 582 SVM Mini Project

This folder contains the datasets, MATLAB scripts, and figures prepared by the Data Team for the SVM Mini Project.

## Folder Contents

### `datasets/`
Contains the processed Breast Cancer Wisconsin dataset and three synthetic binary classification datasets.

- **Breast Cancer:** Cleaned, standardized, and divided into training and testing sets (456 training, 113 testing observations).
- **Linearly Separable:** Synthetic data for testing linear SVM classification.
- **Overlapping Classes:** Synthetic data for evaluating soft-margin SVMs.
- **Concentric Circles:** Synthetic data for evaluating nonlinear kernel methods.

All datasets are provided as CSV files with class labels of +1 and -1.

### `matlab/`
Contains MATLAB scripts used to clean and prepare the datasets, generate synthetic data, and evaluate classifier performance.

A MATLAB linear SVM was also used as a reference benchmark to verify the data preparation and evaluation procedures.

### `figures/`
Contains visualizations of the synthetic datasets and reference classification results.

## Notes

- The Breast Cancer training and testing CSV files are already standardized.
- Use the provided training/testing splits for consistent model comparisons.
- The MATLAB reference SVM is for benchmarking only; it is not the final dual SVM implementation.
