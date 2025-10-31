# Analysis of model.ipynb

This document summarizes the data processing, modeling steps, evaluation, and reproducibility notes from `model.ipynb` in this repository. It is written to be reproducible and concise so you can quickly understand or rerun the pipeline.

## Overview
- Goal: Predict whether a patient is readmitted within 30 days (`readmitted == '<30'`) using tabular features from the provided dataset `data.csv`.
- Approach: Data cleaning and imputation, feature preprocessing (iterative imputation + scaling for numerics, most-frequent imputation + one-hot for categoricals), train multiple fast classifiers (RandomForest, XGBoost, LinearSVC), calibrate probabilities, pick thresholds via calibration set (maximize F1), evaluate on test set, and inspect feature importance via SHAP for XGBoost.

## Environment / Dependencies
Run in a Python 3.10+ virtual environment. The notebook installs the following (used versions not pinned):

- scikit-learn
- xgboost
- shap
- matplotlib
- seaborn
- pandas
- numpy

Install with pip (from the notebook):

%pip install scikit-learn xgboost shap matplotlib seaborn pandas numpy

## Data loading
- The notebook reads `data.csv` into a DataFrame `df`.
- It prints shape information and explores missingness with `df.info()` and `df.isnull().sum()`.

## Key cleaning & imputation steps
1. weight column:
   - `weight` contains categorical bins like `[75-100)` and `?` for missing values.
   - A mapping from bins to numeric midpoints is defined (`midpoints`) and used to replace known bins with their numeric midpoints.
   - Missing values represented by `?` are filled with a frequency-weighted mean computed from observed bin frequencies.

2. payer_code:
   - Missing values (`?`, empty strings, literal 'nan') are treated as missing.
   - Imputation is done by sampling from the observed distribution of payer codes (proportional sampling) with a fixed RNG seed for reproducibility.

3. medical_specialty:
   - Missing `?` values are imputed using a RandomForest classifier trained on a set of predictors (age, gender, weight, admission/discharge IDs, time_in_hospital, numeric procedure counts, diagnosis codes, payer_code, etc.).
   - Categorical features are label-encoded before training. The RandomForest predicts missing specialties, and predictions are assigned back to the DataFrame.

4. Generic categorical target imputation helper (`impute_categorical_targets`):
   - A reusable function that imputes multiple categorical target columns (e.g., `diag_1`, `diag_2`, `diag_3`, `max_glu_serum`, `A1Cresult`, ...) using a RandomForest.
   - For features: label-encodes object columns, imputes remaining missing values with the most frequent value.
   - For each target, trains a RandomForest on non-missing rows and predicts missing rows.

## Feature engineering & preprocessing
- Target: binary y = 1 if `readmitted == '<30'`, else 0.
- Drop id columns: `encounter_id`, `patient_nbr`.
- Numerical pipeline:
  - IterativeImputer with BayesianRidge (max_iter=5) → StandardScaler
- Categorical pipeline:
  - SimpleImputer(strategy='most_frequent') → OneHotEncoder(handle_unknown='ignore')
- ColumnTransformer assembles numeric and categorical pipelines; remainder dropped.
- Transformed data may be sparse; notebook converts to dense arrays if sparse.
- Feature names obtained via `preproc.get_feature_names_out()` with cleaning to remove characters not suitable for SHAP (replace [, ], < with underscore, remove double underscores).

## Train / test split and calibration
- Stratified train/test split: test_size=0.20, random_state=42.
- A calibration holdout is carved from training data (10% by `CALIB_HOLDOUT`) to calibrate probabilities and tune thresholds.
- Models trained on the remaining training subset:
  - RandomForestClassifier (class_weight='balanced')
  - XGBoost (if installed) with `scale_pos_weight` computed from class imbalance
  - LinearSVC (linear SVM) with class_weight='balanced'
- Calibration with `CalibratedClassifierCV(..., method='sigmoid', cv='prefit')` fitted on calibration set.
- Threshold selection: `find_best_threshold_by_f1` searches thresholds (0.01 to 0.99) selecting the threshold maximizing F1 on the calibration set.

## Evaluation
- Metrics reported: Brier score, ROC-AUC, Accuracy, Precision, Recall, F1, Confusion Matrix.
- Plots produced for ROC, calibration curve, and confusion matrix heatmap.
- Evaluation performed both at fixed thresholds (notably 0.12–0.14 used in the notebook) and at best thresholds found via calibration.
- Summary table compares models and highlights best-by-F1 and best-by-ROC.

## SHAP analysis
- SHAP TreeExplainer runs on a subset (up to 1000 rows) of test data.
- Ensures feature names are cleaned for SHAP compatibility.
- Uses `shap.summary_plot` for bar and beeswarm plots for XGBoost model.
- Plots show global feature importance and effect directions.

## Quick reproduction steps
1. Create and activate a Python virtual environment (e.g., `python -m venv env` and activate).
2. Install dependencies (see `Environment / Dependencies`).
3. Place `data.csv` in the repository root (or update the path in the notebook).
4. Open and run `model.ipynb` in a Jupyter environment. Run cells sequentially.

Notes:
- XGBoost is optional; the notebook runs without it but some analyses (SHAP on XGBoost) will be skipped if XGBoost is not available.
- Watch memory when converting large sparse matrices to dense arrays for preprocessing.
- Random seeds are set in key places for reproducibility (RANDOM_STATE=42, RNG for payer_code sampling).

## Recommendations / Next steps
- Add unit tests for the imputation helper functions to ensure consistent behavior.
- Pin dependency versions (create `requirements.txt`) for reproducibility.
- Consider using category encoders (e.g., TargetEncoder) for high-cardinality categorical variables rather than one-hot encoding.
- Cache preprocessed arrays to avoid re-running expensive `fit_transform` steps during experimentation.

