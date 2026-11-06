"""
=============================================================================
Module: data_preprocessing.py
Project: Student Performance Analysis and Prediction System Using Data Mining
=============================================================================
This module handles all data loading, cleaning, transformation, and feature
engineering tasks required for both exploratory analysis and model training.

Academic Context (Data Mining Syllabus):
- Data Cleaning: Handling missing values and removing duplicate records.
- Data Transformation: Categorical encoding, standardization, and target discretization.
- Feature Selection: Choosing relevant predictors while strictly preventing target leakage.
=============================================================================
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Default feature configuration
# Selected attributes as per project requirements and academic significance
# ---------------------------------------------------------------------------
NUMERICAL_FEATURES = [
    'age',         # Student's age (numeric: 15 to 22)
    'studytime',   # Weekly study time (1: <2 hrs, 2: 2-5 hrs, 3: 5-10 hrs, 4: >10 hrs)
    'failures',    # Number of past class failures (0 to 3)
    'absences',    # Number of school absences (0 to 93)
    'G1',          # First period examination grade (0 to 20)
    'G2'           # Second period examination grade (0 to 20)
]

CATEGORICAL_FEATURES = [
    'sex',         # Student gender ('F' - female, 'M' - male)
    'address',     # Home address type ('U' - urban, 'R' - rural)
    'schoolsup',   # Extra educational support ('yes' or 'no')
    'famsup',      # Family educational support ('yes' or 'no')
    'paid',        # Extra paid classes within the course subject ('yes' or 'no')
    'activities',  # Extra-curricular activities ('yes' or 'no')
    'higher',      # Wants to take higher education ('yes' or 'no')
    'internet'     # Internet access at home ('yes' or 'no')
]

FEATURE_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = 'passed'  # Binary target: 1 = Pass (G3 >= 10), 0 = Fail (G3 < 10)


def resolve_data_path(file_path=None):
    """
    Resolve the absolute path to student-mat.csv reliably across environments.
    """
    if file_path and os.path.exists(file_path):
        return file_path

    # Try common relative paths
    candidates = [
        os.path.join(os.path.dirname(__file__), '..', 'data', 'student-mat.csv'),
        os.path.join(os.path.dirname(__file__), 'data', 'student-mat.csv'),
        os.path.join(os.getcwd(), 'data', 'student-mat.csv'),
        'data/student-mat.csv',
        'student-mat.csv'
    ]
    for candidate in candidates:
        abs_cand = os.path.abspath(candidate)
        if os.path.exists(abs_cand):
            return abs_cand

    raise FileNotFoundError(
        "Could not locate 'student-mat.csv'. Please place it inside the 'data/' directory."
    )


def load_dataset(file_path=None, verbose=True):
    """
    Load the UCI Student Performance dataset from CSV.
    The UCI dataset uses semicolons (;) as column delimiters.

    Creates the binary target column:
        - passed = 1 (Pass: G3 >= 10)
        - passed = 0 (Fail: G3 < 10)

    Returns:
        pd.DataFrame: Loaded and verified dataframe.
    """
    resolved_path = resolve_data_path(file_path)
    if verbose:
        print(f"[INFO] Loading dataset from: {resolved_path}")

    # Read semicolon-delimited CSV
    df = pd.read_csv(resolved_path, sep=';')

    if verbose:
        print(f"[INFO] Initial dataset dimensions: {df.shape[0]} rows, {df.shape[1]} columns")

    # Step 4 & 5: Missing value audit
    null_counts = df.isnull().sum()
    total_missing = null_counts.sum()
    if verbose:
        print(f"[INFO] Total missing values detected: {total_missing}")
        if total_missing > 0:
            print("[INFO] Missing values per column:")
            print(null_counts[null_counts > 0])

    # Step 6: Duplicate detection and removal
    initial_len = len(df)
    df = df.drop_duplicates()
    removed_duplicates = initial_len - len(df)
    if verbose:
        print(f"[INFO] Duplicate records removed: {removed_duplicates}")

    # Step 10: Create Pass/Fail target column based on final grade G3
    # G3 is on a 0-20 scale in Portuguese secondary schools; >= 10 indicates passing.
    df['passed'] = (df['G3'] >= 10).astype(int)

    if verbose:
        pass_count = (df['passed'] == 1).sum()
        fail_count = (df['passed'] == 0).sum()
        print(f"[INFO] Binary target created: {pass_count} Passed (>=10), {fail_count} Failed (<10)")

    return df


def get_feature_lists():
    """
    Returns tuple of (FEATURE_COLUMNS, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMN).
    Ensures that G3 is explicitly NOT in FEATURE_COLUMNS to prevent target leakage.
    """
    return FEATURE_COLUMNS, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMN


def build_preprocessor(numerical_cols=None, categorical_cols=None):
    """
    Constructs a Scikit-learn ColumnTransformer for preprocessing:
    - Numerical: Imputes missing values with median.
    - Categorical: Imputes missing values with most frequent value, then applies One-Hot Encoding.

    Returns:
        ColumnTransformer: Preprocessing transformer ready to be used in a Pipeline.
    """
    if numerical_cols is None:
        numerical_cols = NUMERICAL_FEATURES
    if categorical_cols is None:
        categorical_cols = CATEGORICAL_FEATURES

    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])

    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, numerical_cols),
            ('cat', cat_transformer, categorical_cols)
        ]
    )

    return preprocessor


def prepare_data_splits(df, test_size=0.20, random_state=42):
    """
    Separates predictors and target, and splits into train and test sets.

    Crucial Data Mining Concept:
    - Target Leakage Prevention: G3 is excluded from X.
    - G1 and G2 are retained as historical exam predictors.
    - Stratified splitting guarantees identical class distribution in train and test sets.

    Returns:
        X_train, X_test, y_train, y_test: Train and test sets.
    """
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"[INFO] Data split completed: {len(X_train)} training records (80%), {len(X_test)} testing records (20%)")
    print(f"[INFO] Train class distribution: {dict(y_train.value_counts())}")
    print(f"[INFO] Test class distribution:  {dict(y_test.value_counts())}")

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    print("=" * 70)
    print("RUNNING: data_preprocessing.py (Standalone Verification)")
    print("=" * 70)

    dataset = load_dataset()
    print("\n--- First 5 Records (Selected Columns) ---")
    print(dataset[['school', 'sex', 'age', 'studytime', 'failures', 'absences', 'G1', 'G2', 'G3', 'passed']].head())

    print("\n--- Numerical Attributes Summary ---")
    print(dataset[NUMERICAL_FEATURES].describe().round(2))

    print("\n--- Categorical Attributes Distribution ---")
    for cat in CATEGORICAL_FEATURES:
        print(f"{cat}: {dict(dataset[cat].value_counts())}")

    X_train, X_test, y_train, y_test = prepare_data_splits(dataset)
    print("\n[SUCCESS] Data preprocessing module executed without errors.")
    print("=" * 70)

