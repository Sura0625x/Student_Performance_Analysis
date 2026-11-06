"""
=============================================================================
Module: clustering.py
Project: Student Performance Analysis and Prediction System Using Data Mining
=============================================================================
This module performs unsupervised K-Means clustering to discover natural
student groupings based on their academic behavior and historical performance.
=============================================================================
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

CLUSTERING_FEATURES = ['studytime', 'failures', 'absences', 'G1', 'G2']


def get_graphs_dir(output_dir=None):
    """Resolve directory path for saving graphs."""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    default_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'graphs')
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


def compute_elbow_method(df, features=None, k_range=range(1, 9), output_dir=None, save=True, dark_mode=False, **kwargs):
    """
    Computes Within-Cluster Sum of Squares (WCSS / Inertia) across k values
    and plots the Elbow curve.
    """
    if features is None:
        features = [f for f in CLUSTERING_FEATURES if f in df.columns]
        if len(features) < 2:
            features = df.select_dtypes(include=[np.number]).columns.tolist()[:5]

    X = df[features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    k_values = list(k_range)
    inertias = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)

    fig, ax = plt.subplots(figsize=(8, 5))
    line_color = '#38bdf8' if dark_mode else '#1f77b4'
    ax.plot(k_values, inertias, marker='o', linewidth=2.5, markersize=8, color=line_color)
    
    # Highlight the recommended elbow point (k=3)
    if 3 in k_values:
        idx_3 = k_values.index(3)
        ax.scatter([3], [inertias[idx_3]], color='#f43f5e', s=160, zorder=5,
                   label=f'Optimal Elbow (k=3, WCSS={inertias[idx_3]:.1f})')
        ax.annotate('Recommended Elbow (k=3)\n(Best balance of cohesion & interpretability)',
                    xy=(3, inertias[idx_3]), xytext=(3.5, inertias[idx_3] + 150),
                    arrowprops=dict(facecolor='#f43f5e', shrink=0.08, width=1.5, headwidth=8),
                    fontweight='bold', fontsize=9, color='#f8fafc' if dark_mode else 'black')

    ax.set_title("Elbow Method for Optimal Number of Student Clusters", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Number of Clusters (k)", fontsize=11)
    ax.set_ylabel("Within-Cluster Sum of Squares (Inertia)", fontsize=11)
    ax.set_xticks(k_values)
    ax.legend(frameon=True)
    
    if dark_mode:
        apply_dark_theme(fig, ax)
        
    plt.tight_layout()

    if save:
        path = os.path.join(get_graphs_dir(output_dir), 'elbow_curve.png')
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        return k_values, inertias, path
    return k_values, inertias, fig


def perform_kmeans_clustering(df, n_clusters=3, features=None, output_dir=None, save=True, dark_mode=False, **kwargs):
    """
    Fits K-Means clustering, assigns cluster IDs, computes profiles,
    and dynamically interprets clusters based on their actual average grades.
    """
    if features is None:
        features = [f for f in CLUSTERING_FEATURES if f in df.columns]
        if len(features) < 2:
            features = df.select_dtypes(include=[np.number]).columns.tolist()[:5]

    X = df[features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train KMeans with fixed seed for academic reproducibility
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_ids = kmeans.fit_predict(X_scaled)

    df_clustered = df.copy()
    df_clustered['cluster_raw'] = cluster_ids

    summary_cols = [c for c in ['G1', 'G2', 'G3', 'studytime', 'failures', 'absences', 'passed'] if c in df_clustered.columns]
    cluster_stats = df_clustered.groupby('cluster_raw')[summary_cols].mean()
    cluster_counts = df_clustered['cluster_raw'].value_counts()
    cluster_stats['student_count'] = cluster_counts
    cluster_stats['student_pct'] = (cluster_counts / len(df_clustered) * 100).round(1)

    # Dynamic and objective interpretation based on ACTUAL average G3 / G2
    ranking_metric = 'G3' if 'G3' in cluster_stats.columns else ('G2' if 'G2' in cluster_stats.columns else summary_cols[0])
    sorted_clusters = cluster_stats.sort_values(by=ranking_metric, ascending=False).index.tolist()

    archetypes = [
        ("High Academic Achievers", "High grades across G1/G2/G3, low failure rate, consistent study habits."),
        ("Moderate / Average Performers", "Passing grades near class average, moderate study time, average attendance."),
        ("At-Risk / Needs Support", "Low exam scores, elevated prior failures, requiring early academic intervention.")
    ]

    cluster_labels_map = {}
    cluster_descriptions = {}
    for rank, cluster_id in enumerate(sorted_clusters):
        if rank < len(archetypes):
            label, desc = archetypes[rank]
        else:
            label, desc = f"Cohort {cluster_id + 1}", "Specialized performance group."
        cluster_labels_map[cluster_id] = label
        cluster_descriptions[cluster_id] = desc

    df_clustered['cluster_name'] = df_clustered['cluster_raw'].map(cluster_labels_map)
    cluster_stats['assigned_label'] = cluster_stats.index.map(cluster_labels_map)

    # Visualize clusters using scatter plot
    fig, ax = plt.subplots(figsize=(9, 6))
    
    palette_map = {
        "High Academic Achievers": '#34d399',       # Vivid Emerald Green
        "Moderate / Average Performers": '#38bdf8', # Vivid Sky Blue
        "At-Risk / Needs Support": '#f43f5e'        # Vivid Rose Red
    }
    
    # Select plot axes
    x_axis = 'G1' if 'G1' in features else features[0]
    y_axis = 'G2' if 'G2' in features else (features[1] if len(features) > 1 else features[0])

    for cluster_name, group in df_clustered.groupby('cluster_name'):
        color = palette_map.get(cluster_name, '#a855f7')
        ax.scatter(group[x_axis], group[y_axis], label=f"{cluster_name} (n={len(group)})",
                   color=color, alpha=0.8, s=65, edgecolors='none')

    # Project cluster centers
    x_idx = features.index(x_axis)
    y_idx = features.index(y_axis)
    centers_orig = scaler.inverse_transform(kmeans.cluster_centers_)
    
    centroid_color = '#ffffff' if dark_mode else '#000000'
    ax.scatter(centers_orig[:, x_idx], centers_orig[:, y_idx],
               s=240, color=centroid_color, marker='X', linewidths=2.5,
               edgecolors='#0f172a' if dark_mode else '#ffffff',
               label='Cluster Centroids', zorder=10)

    ax.set_title(f"K-Means Student Segments ({x_axis} vs. {y_axis} Performance)",
                 fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel(f"{x_axis} Dimension", fontsize=11)
    ax.set_ylabel(f"{y_axis} Dimension", fontsize=11)
    ax.axhline(10, color='gray', linestyle=':', alpha=0.6)
    ax.axvline(10, color='gray', linestyle=':', alpha=0.6)
    ax.legend(frameon=True, loc='upper left')
    
    if dark_mode:
        apply_dark_theme(fig, ax)
        
    plt.tight_layout()

    plot_result = None
    if save:
        plot_result = os.path.join(get_graphs_dir(output_dir), 'cluster_distribution.png')
        fig.savefig(plot_result, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
    else:
        plot_result = fig

    return {
        'clustered_df': df_clustered,
        'cluster_summary': cluster_stats,
        'cluster_labels_map': cluster_labels_map,
        'cluster_descriptions': cluster_descriptions,
        'kmeans_model': kmeans,
        'scaler': scaler,
        'plot': plot_result
    }
