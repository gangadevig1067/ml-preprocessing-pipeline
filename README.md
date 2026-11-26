# Task 03: Leak-Free Feature Engineering & Preprocessing Pipeline

## Overview
This repository contains a modular, reusable machine learning preprocessing pipeline built with `scikit-learn`. It demonstrates best practices for preparing raw, mixed-type tabular data (numerical and categorical features with missing values) while strictly preventing **data leakage**.

## Key Implementation Steps
1. **Exploratory & Correlation Analysis:** Inspected missing values across features and generated a correlation heatmap to identify relationships among numerical variables.
2. **Leak-Free Train/Test Split:** Split the dataset into training (80%) and testing (20%) sets *prior* to applying any imputation, scaling, or encoding transformations.
3. **Numerical Pipeline (`Pipeline`):**
   - Imputed missing numerical values using `SimpleImputer(strategy='median')` to remain robust against outliers.
   - Standardized feature scales using `StandardScaler()`.
4. **Categorical Pipeline (`Pipeline`):**
   - Imputed missing categorical entries using `SimpleImputer(strategy='most_frequent')`.
   - Encoded categorical features into binary vectors using `OneHotEncoder(handle_unknown='ignore')` to safely handle unseen categories at inference time.
5. **Unified `ColumnTransformer`:** Combined both pipelines into a single transformer fitted strictly on `X_train` (`fit_transform`) and applied to `X_test` (`transform`).

## Files
- `Task_03_Preprocessing_Pipeline.ipynb`: Complete Jupyter Notebook containing exploratory analysis, correlation matrix, `ColumnTransformer` implementation, and end-to-end model verification.

## Tech Stack
- Python 3
- Pandas & NumPy
- Scikit-Learn (`Pipeline`, `ColumnTransformer`, `SimpleImputer`, `StandardScaler`, `OneHotEncoder`)
- Matplotlib & Seaborn
