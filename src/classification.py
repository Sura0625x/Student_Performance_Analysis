"""
=============================================================================
Module: classification.py
Project: Student Performance Analysis and Prediction System Using Data Mining
=============================================================================
This module builds, trains, and serializes the Decision Tree Classifier pipeline.

Academic Concepts Demonstrated:
1. Supervised Learning: Predicting student Pass/Fail status from historical records.
2. Target Leakage Prevention: G3 is strictly excluded from feature inputs.
3. Information Gain (Entropy): Splitting criterion used by ID3/C4.5 decision trees.
4. Preprocessing Pipeline: Combining imputers, one-hot encoders, and classifier into an end-to-end pipeline.
5. Reproducibility: Setting a fixed random seed (random_state=42).
=============================================================================
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline

from data_preprocessing import (
    load_dataset,
    prepare_data_splits,
    build_preprocessor,
    FEATURE_COLUMNS,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN
)


def get_model_path(custom_path=None):
    """Resolve destination path for saving/loading the trained model pipeline."""
    if custom_path:
        os.makedirs(os.path.dirname(os.path.abspath(custom_path)), exist_ok=True)
        return custom_path
    default_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'trained_model.pkl')
    os.makedirs(os.path.dirname(os.path.abspath(default_path)), exist_ok=True)
    return os.path.abspath(default_path)


def build_decision_tree_pipeline(max_depth=4, criterion='entropy', random_state=42):
    """
    Constructs an end-to-end Scikit-learn Pipeline containing:
    1. Preprocessor: ColumnTransformer (Imputation + One-Hot Encoding)
    2. Classifier: DecisionTreeClassifier with Information Gain ('entropy')

    Using a Scikit-learn Pipeline prevents data leakage because all transformers
    are fitted strictly on training data during pipeline.fit().
    """
    preprocessor = build_preprocessor(NUMERICAL_FEATURES, CATEGORICAL_FEATURES)

    classifier = DecisionTreeClassifier(
        criterion=criterion,      # Information Gain / Entropy criterion
        max_depth=max_depth,      # Constrained depth prevents overfitting & aids visualization
        min_samples_split=10,     # Minimum samples required to split an internal node
        min_samples_leaf=5,       # Minimum samples required to be at a leaf node
        random_state=random_state # Deterministic training for academic evaluation
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])

    return pipeline


def train_and_save_model(df=None, model_path=None, test_size=0.20, random_state=42, max_depth=4):
    """
    Loads data, performs stratified train-test split, trains pipeline,
    and serializes the pipeline using Joblib.

    Returns:
        tuple: (trained_pipeline, X_train, X_test, y_train, y_test, model_path)
    """
    if df is None:
        df = load_dataset(verbose=False)

    # Stratified split: preserves Pass/Fail ratio across train and test sets
    X_train, X_test, y_train, y_test = prepare_data_splits(
        df, test_size=test_size, random_state=random_state
    )

    print("[INFO] Building Decision Tree pipeline (Criterion: Entropy / Information Gain)...")
    pipeline = build_decision_tree_pipeline(max_depth=max_depth, random_state=random_state)

    print("[INFO] Fitting pipeline on training data...")
    pipeline.fit(X_train, y_train)

    # Persist the trained pipeline to disk
    save_path = get_model_path(model_path)
    joblib.dump(pipeline, save_path)
    print(f"[SUCCESS] Trained model pipeline saved to: {save_path}")

    return pipeline, X_train, X_test, y_train, y_test, save_path


def load_trained_model(model_path=None):
    """
    Loads the serialized pipeline from disk.
    If the file does not exist, automatically trains and persists a new one.
    """
    path = get_model_path(model_path)
    if os.path.exists(path):
        return joblib.load(path)
    print(f"[WARNING] Model file not found at {path}. Training new model...")
    pipeline, _, _, _, _, _ = train_and_save_model(model_path=path)
    return pipeline


def predict_single_student(student_dict, model_pipeline=None):
    """
    Predicts Pass/Fail outcome for a single student record.

    Parameters:
        student_dict (dict): Dictionary containing feature values.
        model_pipeline (Pipeline): Optional pre-loaded pipeline.

    Returns:
        dict: {
            'prediction': 'Pass' or 'Fail',
            'pass_probability': float (0.0 to 1.0),
            'fail_probability': float (0.0 to 1.0),
            'explanation': str
        }
    """
    if model_pipeline is None:
        model_pipeline = load_trained_model()

    # Convert single dictionary to 1-row DataFrame with expected columns
    input_df = pd.DataFrame([student_dict])

    # Ensure all required columns exist, fill missing with sensible defaults
    for col in FEATURE_COLUMNS:
        if col not in input_df.columns:
            if col in NUMERICAL_FEATURES:
                input_df[col] = 0
            else:
                input_df[col] = 'no'

    # Filter to exact feature columns
    input_df = input_df[FEATURE_COLUMNS]

    # Predict class (1 = Pass, 0 = Fail) and class probabilities
    pred_class = int(model_pipeline.predict(input_df)[0])
    probabilities = model_pipeline.predict_proba(input_df)[0]
    classes = list(model_pipeline.classes_)

    # Extract probability for Pass (class 1) and Fail (class 0)
    pass_prob = float(probabilities[classes.index(1)]) if 1 in classes else 0.0
    fail_prob = float(probabilities[classes.index(0)]) if 0 in classes else 0.0

    prediction_label = "Pass" if pred_class == 1 else "Fail"

    # Human-understandable reasoning for academic presentation
    g2_val = student_dict.get('G2', 0)
    g1_val = student_dict.get('G1', 0)
    failures_val = student_dict.get('failures', 0)
    study_val = student_dict.get('studytime', 2)

    reasons = []
    if g2_val >= 10:
        reasons.append(f"Strong second-period exam score (G2 = {g2_val}/20)")
    else:
        reasons.append(f"Sub-threshold second-period score (G2 = {g2_val}/20)")

    if g1_val >= 10:
        reasons.append(f"Solid foundational score in G1 ({g1_val}/20)")
    else:
        reasons.append(f"Low initial score in G1 ({g1_val}/20)")

    if failures_val > 0:
        reasons.append(f"History of {failures_val} past class failure(s)")
    else:
        reasons.append("No record of past failures")

    if study_val >= 3:
        reasons.append("High dedicated weekly study time (>5 hrs)")

    explanation = (
        f"The Decision Tree predicts '{prediction_label}' with {pass_prob if pred_class==1 else fail_prob:.1%} confidence. "
        f"Key determining factors: {'; '.join(reasons)}."
    )

    return {
        'prediction': prediction_label,
        'pass_probability': pass_prob,
        'fail_probability': fail_prob,
        'confidence': max(pass_prob, fail_prob),
        'explanation': explanation
    }


if __name__ == '__main__':
    print("=" * 70)
    print("RUNNING: classification.py (Decision Tree Model Training)")
    print("=" * 70)

    pipeline, X_train, X_test, y_train, y_test, model_path = train_and_save_model()

    # Test prediction on a sample student
    sample_student = {
        'age': 16,
        'sex': 'F',
        'address': 'U',
        'studytime': 3,
        'failures': 0,
        'schoolsup': 'no',
        'famsup': 'yes',
        'paid': 'no',
        'activities': 'yes',
        'higher': 'yes',
        'internet': 'yes',
        'absences': 2,
        'G1': 14,
        'G2': 15
    }
    result = predict_single_student(sample_student, pipeline)
    print("\n--- Sample Prediction Verification ---")
    print(f"Sample Student Input: {sample_student}")
    print(f"Prediction: {result['prediction']}")
    print(f"Pass Probability: {result['pass_probability']:.2%}")
    print(f"Explanation: {result['explanation']}")
    print("\n[SUCCESS] Classification module executed successfully.")
    print("=" * 70)

