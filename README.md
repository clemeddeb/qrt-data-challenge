# QRT Data Challenge

## Overview
This repository contains my work for the QRT Data Challenge, focused on predicting the direction of future returns from financial and allocation data.

The project covers the full modelling process, from exploratory data analysis and validation design to model comparison, feature engineering and final prediction.

## Approach
I explored several modelling approaches, including:
- Logistic regression and simple baselines
- LightGBM
- XGBoost
- CatBoost
- Feature engineering and target encoding
- Purged validation to reduce temporal leakage
- Threshold calibration
- Ensemble methods

The final pipeline was based on CatBoost after comparing the different approaches through controlled experiments.

## Result
Final leaderboard score: **0.5098**

## What I learned
The most interesting part of the challenge was not simply improving the score, but understanding which modelling choices actually generalized.

Several ideas that appeared promising did not improve validation performance, while choices in validation design, feature construction and threshold calibration sometimes mattered more than increasing model complexity.

## Repository structure
- `qrt_challenge_pipeline_final.ipynb` — final end-to-end pipeline
- `qrt_challenge_accuracy_research.ipynb` — experimental research and model comparison
- `ACCURACY_RESEARCH_NOTEBOOK_REPORT.md` — summary of research findings
- `ACCURACY_IMPROVEMENT_ROADMAP.md` — ideas and experiments explored during development

## Data
The original challenge datasets are not included in this repository.
