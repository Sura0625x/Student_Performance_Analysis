"""
=============================================================================
Module: evaluation.py
Project: Student Performance Analysis and Prediction System Using Data Mining
=============================================================================
This module evaluates the Decision Tree Classifier, computes metrics,
generates evaluation visualizations, and exports an academic evaluation report.
=============================================================================
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.tree import plot_tree


def get_graphs_dir(output_dir=None):
    """Resolve directory path for saving graphs."""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    default_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'graphs')
    os.makedirs(default_dir, exist_ok=True)
    return os.path.abspath(default_dir)


def get_reports_dir(output_dir=None):
    """Resolve directory path for saving reports."""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    default_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'reports')
    os.makedirs(default_dir, exist_ok=True)
    return os.path.abspath(default_dir)


def apply_dark_theme(fig, ax):
    """Applies a modern deep dark theme matching the Streamlit dashboard."""
    fig.patch.set_facecolor('#0d1527')
    axes = ax if isinstance(ax, np.ndarray) else [ax]
    for a in np.array(axes).flat:
        a.set_facecolor('#111c33')
        a.tick_params(colors='#94a3b8', labelsize=10)
        a.xaxis.label.set_color('#cbd5e1')
        a.yaxis.label.set_color('#cbd5e1')
        a.title.set_color('#f8fafc')
        a.grid(True, color='#1e293b', linestyle='--', alpha=0.6)
        for spine in a.spines.values():
            spine.set_color('#1e293b')
        legend = a.get_legend()
        if legend:
            legend.get_frame().set_facecolor('#111c33')
            legend.get_frame().set_edgecolor('#334155')
            for text in legend.get_texts():
                text.set_color('#f8fafc')
            if legend.get_title():
                legend.get_title().set_color('#f8fafc')


def plot_confusion_matrix_heatmap(y_true, y_pred, output_dir=None, save=True, dark_mode=False, **kwargs):
    """
    Generates an annotated heatmap of the Confusion Matrix.
    Classes: 0 = Fail, 1 = Pass.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    labels = [
        [f"TN = {tn}\n(True Fail)", f"FP = {fp}\n(False Alarm)"],
        [f"FN = {fn}\n(Missed Pass)", f"TP = {tp}\n(True Pass)"]
    ]

    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = "Blues" if not dark_mode else "Blues_r"
    sns.heatmap(cm, annot=labels, fmt="", cmap="Blues", cbar=True,
                xticklabels=['Predicted Fail (0)', 'Predicted Pass (1)'],
                yticklabels=['Actual Fail (0)', 'Actual Pass (1)'],
                annot_kws={"size": 11, "fontweight": "bold"}, ax=ax)

    ax.set_title("Confusion Matrix: Student Pass/Fail Prediction", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Predicted Class Label", fontsize=11)
    ax.set_ylabel("True / Actual Class Label", fontsize=11)
    
    if dark_mode:
        apply_dark_theme(fig, ax)
        
    plt.tight_layout()

    if save:
        path = os.path.join(get_graphs_dir(output_dir), 'confusion_matrix.png')
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        return path
    return fig


def plot_metrics_comparison(metrics_dict, output_dir=None, save=True, dark_mode=False, **kwargs):
    """
    Generates a professional bar chart comparing Accuracy, Precision, Recall, and F1-Score.
    """
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    metric_values = [
        metrics_dict['accuracy'] * 100,
        metrics_dict['precision'] * 100,
        metrics_dict['recall'] * 100,
        metrics_dict['f1_score'] * 100
    ]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    if dark_mode:
        colors = ['#38bdf8', '#34d399', '#a855f7', '#f59e0b']
    else:
        colors = ['#1f77b4', '#388ecc', '#6baed6', '#9ecae1']
        
    bars = ax.bar(metric_names, metric_values, color=colors, edgecolor='#1e293b' if dark_mode else 'black', width=0.5)

    text_color = '#f8fafc' if dark_mode else 'black'
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.2f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold', color=text_color)

    ax.set_title("Decision Tree Model Performance Metrics", fontsize=14, fontweight='bold', pad=12)
    ax.set_ylabel("Metric Score (%)", fontsize=11)
    ax.set_ylim(0, 115)
    ax.axhline(80, color='#10b981', linestyle=':', label='80% Academic Benchmark')
    ax.legend(loc='lower right')
    
    if dark_mode:
        apply_dark_theme(fig, ax)
        
    plt.tight_layout()

    if save:
        path = os.path.join(get_graphs_dir(output_dir), 'metrics_comparison.png')
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        return path
    return fig


def plot_decision_tree_graph(pipeline, output_dir=None, save=True, dark_mode=False, **kwargs):
    """
    Renders and exports a visual representation of the trained Decision Tree.
    """
    classifier = pipeline.named_steps['classifier']
    preprocessor = pipeline.named_steps['preprocessor']

    try:
        feature_names = preprocessor.get_feature_names_out()
        cleaned_names = [f.replace('num__', '').replace('cat__', '') for f in feature_names]
    except Exception:
        cleaned_names = None

    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(
        classifier,
        feature_names=cleaned_names,
        class_names=['Fail', 'Pass'],
        filled=True,
        rounded=True,
        fontsize=9,
        ax=ax,
        precision=2
    )

    ax.set_title("Trained Decision Tree Classifier (Entropy / Information Gain Splits)",
                 fontsize=16, fontweight='bold', pad=15)
                 
    if dark_mode:
        apply_dark_theme(fig, ax)
        
    plt.tight_layout()

    if save:
        path = os.path.join(get_graphs_dir(output_dir), 'tree_visualization.png')
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        return path
    return fig


def evaluate_model(pipeline, X_test, y_test, output_dir=None, save_files=True, **kwargs):
    """
    Evaluates the trained pipeline on test records, produces metrics, generates graphs,
    and writes a comprehensive evaluation report.
    """
    y_pred = pipeline.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    cm = confusion_matrix(y_test, y_pred)
    report_text = classification_report(y_test, y_pred, target_names=['Fail (0)', 'Pass (1)'])
    report_dict = classification_report(y_test, y_pred, target_names=['Fail (0)', 'Pass (1)'], output_dict=True)

    metrics_dict = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'confusion_matrix': cm.tolist(),
        'test_samples': len(y_test)
    }

    cm_path = None
    metrics_path = None
    tree_path = None
    report_file_path = None

    if save_files:
        cm_path = plot_confusion_matrix_heatmap(y_test, y_pred, output_dir, save=True)
        metrics_path = plot_metrics_comparison(metrics_dict, output_dir, save=True)
        tree_path = plot_decision_tree_graph(pipeline, output_dir, save=True)

        tn, fp, fn, tp = cm.ravel()
        rep_dir = get_reports_dir(output_dir)
        report_file_path = os.path.join(rep_dir, 'model_evaluation_report.txt')

        with open(report_file_path, 'w', encoding='utf-8') as f:
            f.write("=" * 75 + "\n")
            f.write("STUDENT PERFORMANCE PREDICTION SYSTEM - MODEL EVALUATION REPORT\n")
            f.write("=" * 75 + "\n\n")
            f.write(f"Algorithm:           Decision Tree Classifier\n")
            f.write(f"Splitting Criterion: Entropy (Information Gain)\n")
            f.write(f"Test Set Size:       {len(y_test)} students (20% stratified hold-out)\n\n")
            f.write("-" * 75 + "\n")
            f.write("1. SUMMARY METRICS\n")
            f.write("-" * 75 + "\n")
            f.write(f"Accuracy:  {acc:.4f} ({acc * 100:.2f}%)\n")
            f.write(f"Precision: {prec:.4f} ({prec * 100:.2f}%)\n")
            f.write(f"Recall:    {rec:.4f} ({rec * 100:.2f}%)\n")
            f.write(f"F1-Score:  {f1:.4f} ({f1 * 100:.2f}%)\n\n")
            f.write("-" * 75 + "\n")
            f.write("2. CONFUSION MATRIX BREAKDOWN\n")
            f.write("-" * 75 + "\n")
            f.write(f"True Negatives  (TN - Correctly predicted Fail): {tn}\n")
            f.write(f"False Positives (FP - Incorrectly predicted Pass): {fp}\n")
            f.write(f"False Negatives (FN - Incorrectly predicted Fail): {fn}\n")
            f.write(f"True Positives  (TP - Correctly predicted Pass): {tp}\n\n")
            f.write("-" * 75 + "\n")
            f.write("3. DETAILED SKLEARN CLASSIFICATION REPORT\n")
            f.write("-" * 75 + "\n")
            f.write(report_text + "\n\n")
            f.write("-" * 75 + "\n")
            f.write("4. ACADEMIC INTERPRETATION OF METRICS\n")
            f.write("-" * 75 + "\n")
            f.write(get_metrics_explanation_text())
            f.write("=" * 75 + "\n")

    return {
        'metrics': metrics_dict,
        'confusion_matrix': cm,
        'classification_report_text': report_text,
        'classification_report_dict': report_dict,
        'confusion_matrix_plot': cm_path,
        'metrics_plot': metrics_path,
        'tree_plot': tree_path,
        'report_file_path': report_file_path
    }


def get_metrics_explanation_text():
    """
    Returns an academic, plain-language explanation of each evaluation metric.
    """
    return (
        "• Accuracy: The overall percentage of correct predictions (both Pass and Fail)\n"
        "  out of all tested students. Formula: (TP + TN) / (TP + TN + FP + FN).\n\n"
        "• Precision: Out of all students predicted as 'Pass', what fraction actually passed?\n"
        "  High precision means very few false alarms. Formula: TP / (TP + FP).\n\n"
        "• Recall (Sensitivity): Out of all students who actually passed, what fraction did\n"
        "  the model successfully identify? Formula: TP / (TP + FN).\n\n"
        "• F1-Score: The harmonic mean of Precision and Recall. It provides a balanced\n"
        "  evaluation especially when classes are unevenly distributed.\n"
        "  Formula: 2 * (Precision * Recall) / (Precision + Recall).\n\n"
        "• Information Gain (Entropy): Measures the reduction in uncertainty/impurity\n"
        "  achieved when splitting students based on an attribute (e.g., G2 >= 10).\n"
    )
