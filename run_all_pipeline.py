"""
=============================================================================
Script: run_all_pipeline.py
Project: Student Performance Analysis and Prediction System Using Data Mining
=============================================================================
Master pipeline execution script.
Executes all project stages sequentially:
1. Data Preprocessing & Validation
2. Exploratory Data Analysis (generates 7 high-res plots)
3. Unsupervised K-Means Clustering & Dynamic Profiling
4. Supervised Decision Tree Training (Pipeline serialization)
5. Comprehensive Model Evaluation & Report Generation
=============================================================================
"""

import os
import sys
import time

# Ensure src is in system path
sys.path.insert(0, os.path.abspath('src'))

from data_preprocessing import load_dataset, prepare_data_splits
from exploratory_analysis import generate_all_eda_plots
from clustering import compute_elbow_method, perform_kmeans_clustering
from classification import train_and_save_model, predict_single_student
from evaluation import evaluate_model


def run_full_pipeline():
    start_time = time.time()
    print("=" * 80)
    print("🎓 STUDENT PERFORMANCE ANALYSIS & PREDICTION SYSTEM - MASTER PIPELINE")
    print("=" * 80)

    # -----------------------------------------------------------------------
    # STAGE 1: Data Ingestion and Preprocessing
    # -----------------------------------------------------------------------
    print("\n[STAGE 1/5] Ingesting and Preprocessing Dataset...")
    df = load_dataset(verbose=True)
    print(f"  ✓ Records loaded: {len(df)}")
    print(f"  ✓ Pass/Fail distribution: Pass={int((df['passed']==1).sum())}, Fail={int((df['passed']==0).sum())}")

    # -----------------------------------------------------------------------
    # STAGE 2: Exploratory Data Analysis (EDA)
    # -----------------------------------------------------------------------
    print("\n[STAGE 2/5] Performing Exploratory Data Analysis...")
    eda_plots = generate_all_eda_plots(df)
    print(f"  ✓ Successfully exported {len(eda_plots)} visualizations to 'outputs/graphs/'")

    # -----------------------------------------------------------------------
    # STAGE 3: Unsupervised K-Means Clustering
    # -----------------------------------------------------------------------
    print("\n[STAGE 3/5] Executing K-Means Clustering (Unsupervised Learning)...")
    k_vals, inertias, elbow_path = compute_elbow_method(df, save=True)
    print(f"  ✓ Elbow method evaluated for k=1..8 (Optimal elbow at k=3)")
    
    clustering_res = perform_kmeans_clustering(df, n_clusters=3, save=True)
    summary_df = clustering_res['cluster_summary']
    print("  ✓ Discovered Student Cohort Profiles:")
    for cid, label in clustering_res['cluster_labels_map'].items():
        row = summary_df.loc[cid]
        print(f"    • Cluster {cid} [{label}]: n={int(row['student_count'])}, Mean G3={row['G3']:.2f}/20, Mean Failures={row['failures']:.2f}")

    # -----------------------------------------------------------------------
    # STAGE 4: Supervised Decision Tree Training
    # -----------------------------------------------------------------------
    print("\n[STAGE 4/5] Training Decision Tree Classifier (Supervised Learning)...")
    pipeline, X_train, X_test, y_train, y_test, model_path = train_and_save_model(df)
    print(f"  ✓ Training completed on {len(X_train)} samples; tested on {len(X_test)} samples")
    print(f"  ✓ Model pipeline serialized to: {model_path}")

    # -----------------------------------------------------------------------
    # STAGE 5: Model Evaluation & Reporting
    # -----------------------------------------------------------------------
    print("\n[STAGE 5/5] Evaluating Classifier on Test Holdout (n=79)...")
    eval_res = evaluate_model(pipeline, X_test, y_test, save_files=True)
    metrics = eval_res['metrics']
    print(f"  ✓ Model Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  ✓ Model Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"  ✓ Model Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"  ✓ Model F1-Score:  {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
    print(f"  ✓ Evaluation report saved to: {eval_res['report_file_path']}")

    # -----------------------------------------------------------------------
    # Verification: Sample Inference
    # -----------------------------------------------------------------------
    print("\n[VERIFICATION] Testing Real-Time Inference on Sample Students...")
    sample_pass = {
        'age': 16, 'sex': 'F', 'address': 'U', 'studytime': 3, 'failures': 0,
        'schoolsup': 'no', 'famsup': 'yes', 'paid': 'no', 'activities': 'yes',
        'higher': 'yes', 'internet': 'yes', 'absences': 2, 'G1': 14, 'G2': 15
    }
    sample_fail = {
        'age': 18, 'sex': 'M', 'address': 'R', 'studytime': 1, 'failures': 2,
        'schoolsup': 'yes', 'famsup': 'no', 'paid': 'no', 'activities': 'no',
        'higher': 'no', 'internet': 'no', 'absences': 18, 'G1': 6, 'G2': 7
    }
    res_pass = predict_single_student(sample_pass, pipeline)
    res_fail = predict_single_student(sample_fail, pipeline)
    print(f"  • Student A (High Marks): Predicted {res_pass['prediction']} ({res_pass['pass_probability']:.1%} confidence)")
    print(f"  • Student B (Low Marks):  Predicted {res_fail['prediction']} ({res_fail['fail_probability']:.1%} confidence)")

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"🎉 MASTER PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS!")
    print("=" * 80)
    print("To launch the interactive dashboard, run:")
    print("  streamlit run app.py\n")


if __name__ == '__main__':
    run_full_pipeline()

