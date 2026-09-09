# QRT Data Challenge

## Overview

This project was developed as part of the QRT Data Challenge. The objective is to predict whether an allocation's future return will be positive, with binary classification accuracy used as the evaluation metric.

The repository contains a single notebook covering the full workflow, from initial data exploration and validation design to model comparison, final training and submission.

## Approach

The project follows a timestamp-aware modeling approach designed to limit temporal leakage and obtain a more realistic estimate of out-of-sample performance.

The main steps were:

- Exploring and validating the structure of the dataset
- Reproducing the official LightGBM benchmark
- Designing purged expanding timestamp folds for model selection
- Engineering timestamp-relative and confidence-based features
- Adding fold-safe allocation target encoding
- Comparing several regression and classification approaches
- Calibrating the classification threshold using out-of-fold predictions
- Testing an ensemble before selecting the final model

## Models

I experimented with several approaches:

- Ridge and Logistic Regression
- LightGBM
- XGBoost
- CatBoost
- Probability-based model ensemble

Both regression and classification formulations were tested where relevant.

CatBoost classification provided the strongest and most consistent results under the final validation framework. The ensemble was ultimately rejected because it did not improve out-of-fold accuracy over CatBoost alone.

## Result

**Public leaderboard accuracy: 0.5150**

The final submission uses a CatBoost classifier with a probability threshold of `0.492`.

Validation results and the public leaderboard result are reported separately in the notebook.

## Notebook

The complete workflow, experiments and results can be found in:

[qrt_data_challenge_final.ipynb](qrt_data_challenge_final.ipynb)

The notebook is organized chronologically so that the reasoning behind each modeling decision can be followed from the initial benchmark to the final submission.

## Data

The original QRT challenge datasets are not included in this repository.

To reproduce the project, place the challenge CSV files in the repository root or in a local `data/` directory:

- `X_train_*.csv`
- `X_test_*.csv`
- `y_train_*.csv`
- `sample_submission_*.csv`

These files are excluded from Git.

## Key Takeaways

The most interesting part of this project was not simply comparing increasingly complex models, but understanding which changes genuinely improved out-of-sample performance.

In particular, the project highlighted the importance of validation design, careful feature construction and controlled experimentation. Several ideas that appeared promising did not improve validation performance, including the final ensemble, and were therefore not retained.

The final pipeline reflects that process: keeping the improvements supported by the validation results while avoiding additional complexity when it did not provide a measurable benefit.