"""
=============================================================================
Module: exploratory_analysis.py
Project: Student Performance Analysis and Prediction System Using Data Mining
=============================================================================
This module generates exploratory data analysis (EDA) visualizations to
uncover patterns, distributions, and correlations in student academic data.

Includes support for both publication-quality light themes and sleek,
modern dark themes for the web dashboard.
=============================================================================
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Apply baseline font settings
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 150


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


def plot_g3_distribution(df, output_dir=None, save=True, dark_mode=False):
    """
    1. Distribution of final student grades (G3).
    Displays histogram with Kernel Density Estimation (KDE) and passing cutoff line.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    bar_color = '#38bdf8' if dark_mode else '#1f77b4'
    sns.histplot(df['G3'], bins=20, kde=True, color=bar_color, edgecolor='#0f172a' if dark_mode else 'black', alpha=0.75, ax=ax)
    
    # Passing threshold line (grade >= 10)
    ax.axvline(10, color='#f43f5e', linestyle='--', linewidth=2, label='Passing Threshold (10/20)')
    mean_val = df['G3'].mean()
    median_val = df['G3'].median()
    ax.axvline(mean_val, color='#10b981', linestyle=':', linewidth=2, label=f'Mean Grade ({mean_val:.2f})')
    ax.axvline(median_val, color='#f59e0b', linestyle='-.', linewidth=2, label=f'Median Grade ({median_val:.1f})')
    
    ax.set_title("Distribution of Final Academic Grades (G3)", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Final Grade G3 (Scale 0 - 20)", fontsize=11)
    ax.set_ylabel("Student Frequency", fontsize=11)
    ax.legend(frameon=True)
    
    if dark_mode:
        apply_dark_theme(fig, ax)
        
    plt.tight_layout()

    if save:
        path = os.path.join(get_graphs_dir(output_dir), 'g3_distribution.png')
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        return path
    return fig


def plot_pass_fail_distribution(df, output_dir=None, save=True, dark_mode=False):
    """
    2. Number and proportion of students who passed and failed.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    pass_counts = df['passed'].map({1: 'Passed (G3 ≥ 10)', 0: 'Failed (G3 < 10)'}).value_counts()
    
    if dark_mode:
        colors = ['#38bdf8', '#f43f5e'] if pass_counts.index[0].startswith('Passed') else ['#f43f5e', '#38bdf8']
    else:
        colors = ['#1f77b4', '#d62728'] if pass_counts.index[0].startswith('Passed') else ['#d62728', '#1f77b4']
    
    bars = ax.bar(pass_counts.index, pass_counts.values, color=colors, width=0.5, edgecolor='#1e293b' if dark_mode else 'black')
    total = len(df)
    
    text_color = '#f8fafc' if dark_mode else 'black'
    for bar in bars:
        height = bar.get_height()
        percentage = (height / total) * 100
        ax.annotate(f'{height}\n({percentage:.1f}%)',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold', color=text_color)
        
    ax.set_title("Student Academic Outcome (Pass vs. Fail)", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Academic Outcome Category", fontsize=11)
    ax.set_ylabel("Number of Students", fontsize=11)
    ax.set_ylim(0, max(pass_counts.values) * 1.20)
    
    if dark_mode:
        apply_dark_theme(fig, ax)
        
    plt.tight_layout()

    if save:
        path = os.path.join(get_graphs_dir(output_dir), 'pass_fail_count.png')
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        return path
    return fig


def plot_study_time_vs_grades(df, output_dir=None, save=True, dark_mode=False):
    """
    3. Relationship between study time and final grades.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    study_labels = {1: '< 2 hours', 2: '2 - 5 hours', 3: '5 - 10 hours', 4: '> 10 hours'}
    df_plot = df.copy()
    df_plot['study_label'] = df_plot['studytime'].map(study_labels)

    order = ['< 2 hours', '2 - 5 hours', '5 - 10 hours', '> 10 hours']
    
    if dark_mode:
        palette = ['#60a5fa', '#38bdf8', '#22d3ee', '#34d399']
    else:
        palette = ['#aec7e8', '#7bafde', '#388ecc', '#08519c']

    sns.boxplot(x='study_label', y='G3', data=df_plot, order=order, hue='study_label', palette=palette, ax=ax, width=0.5, legend=False)
    sns.stripplot(x='study_label', y='G3', data=df_plot, order=order, color='#f8fafc' if dark_mode else 'black', alpha=0.35, jitter=0.2, size=4, ax=ax)
    
    ax.axhline(10, color='#f43f5e', linestyle='--', linewidth=1.5, label='Pass Threshold (10)')
    ax.set_title("Impact of Weekly Study Time on Final Grades (G3)", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Weekly Study Time Category", fontsize=11)
    ax.set_ylabel("Final Grade G3 (0 - 20)", fontsize=11)
    ax.legend(loc='lower right')
    
    if dark_mode:
        apply_dark_theme(fig, ax)
        
    plt.tight_layout()

    if save:
        path = os.path.join(get_graphs_dir(output_dir), 'studytime_vs_g3.png')
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        return path
    return fig


def plot_absences_vs_grades(df, output_dir=None, save=True, dark_mode=False):
    """
    4. Relationship between absences and final grades.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    df_plot = df.copy()
    df_plot['Outcome'] = df_plot['passed'].map({1: 'Pass', 0: 'Fail'})

    pass_color = '#38bdf8' if dark_mode else '#1f77b4'
    fail_color = '#f43f5e' if dark_mode else '#d62728'

    sns.scatterplot(
        data=df_plot, x='absences', y='G3', hue='Outcome',
        palette={'Pass': pass_color, 'Fail': fail_color},
        alpha=0.8, s=55, ax=ax
    )
    sns.regplot(data=df_plot, x='absences', y='G3', scatter=False, ax=ax,
                color='#94a3b8' if dark_mode else '#333333',
                line_kws={'linestyle': '--', 'linewidth': 1.5})
    
    ax.axhline(10, color='#10b981', linestyle=':', linewidth=1.5, label='Pass Cutoff (10)')
    ax.set_title("School Absences vs. Final Grade (G3)", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Total School Absences (Days)", fontsize=11)
    ax.set_ylabel("Final Grade G3 (0 - 20)", fontsize=11)
    ax.legend(title='Student Status', loc='upper right')
    
    if dark_mode:
        apply_dark_theme(fig, ax)
        
    plt.tight_layout()

    if save:
        path = os.path.join(get_graphs_dir(output_dir), 'absences_vs_g3.png')
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        return path
    return fig


def plot_previous_vs_final_grades(df, output_dir=None, save=True, dark_mode=False):
    """
    5. Relationship between previous grades (G1, G2) and final grades (G3).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    corr_g1 = df['G1'].corr(df['G3']) if 'G1' in df.columns and 'G3' in df.columns else 0.8
    corr_g2 = df['G2'].corr(df['G3']) if 'G2' in df.columns and 'G3' in df.columns else 0.9

    g1_color = '#38bdf8' if dark_mode else '#1f77b4'
    g2_color = '#34d399' if dark_mode else '#2ca02c'
    trend_color = '#f43f5e' if dark_mode else '#d62728'

    # G1 vs G3
    sns.regplot(data=df, x='G1', y='G3', ax=axes[0],
                scatter_kws={'alpha': 0.65, 'color': g1_color},
                line_kws={'color': trend_color, 'linewidth': 2})
    axes[0].set_title(f"Period 1 Grade (G1) vs. Final Grade (G3)\nPearson Correlation: r = {corr_g1:.2f}",
                      fontsize=12, fontweight='bold')
    axes[0].set_xlabel("First Period Grade G1 (0 - 20)", fontsize=11)
    axes[0].set_ylabel("Final Grade G3 (0 - 20)", fontsize=11)
    axes[0].axhline(10, color='gray', linestyle=':', alpha=0.7)
    axes[0].axvline(10, color='gray', linestyle=':', alpha=0.7)

    # G2 vs G3
    sns.regplot(data=df, x='G2', y='G3', ax=axes[1],
                scatter_kws={'alpha': 0.65, 'color': g2_color},
                line_kws={'color': trend_color, 'linewidth': 2})
    axes[1].set_title(f"Period 2 Grade (G2) vs. Final Grade (G3)\nPearson Correlation: r = {corr_g2:.2f}",
                      fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Second Period Grade G2 (0 - 20)", fontsize=11)
    axes[1].set_ylabel("Final Grade G3 (0 - 20)", fontsize=11)
    axes[1].axhline(10, color='gray', linestyle=':', alpha=0.7)
    axes[1].axvline(10, color='gray', linestyle=':', alpha=0.7)

    if dark_mode:
        apply_dark_theme(fig, axes)

    plt.suptitle("Predictive Value of Earlier Examination Results on Final Performance",
                 fontsize=14, fontweight='bold', y=1.03, color='#f8fafc' if dark_mode else 'black')
    plt.tight_layout()

    if save:
        path = os.path.join(get_graphs_dir(output_dir), 'g1_g2_vs_g3.png')
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        return path
    return fig


def plot_correlation_heatmap(df, output_dir=None, save=True, dark_mode=False):
    """
    6. Correlation heatmap of key numerical attributes.
    """
    candidate_cols = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2', 'G3']
    num_cols = [c for c in candidate_cols if c in df.columns]
    if len(num_cols) < 3:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    corr_matrix = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(11, 8.5))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    cmap = sns.diverging_palette(240, 10, as_cmap=True) if not dark_mode else "coolwarm"
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmin=-0.6, vmax=1.0, center=0,
                annot=True, fmt=".2f", linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    
    ax.set_title("Correlation Matrix of Student Academic & Behavioral Attributes",
                 fontsize=14, fontweight='bold', pad=15)
                 
    if dark_mode:
        apply_dark_theme(fig, ax)
        
    plt.tight_layout()

    if save:
        path = os.path.join(get_graphs_dir(output_dir), 'correlation_heatmap.png')
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        return path
    return fig


def plot_studytime_performance_comparison(df, output_dir=None, save=True, dark_mode=False):
    """
    7. Comparison of student performance across different study-time groups.
    """
    study_labels = {1: '<2 hrs', 2: '2-5 hrs', 3: '5-10 hrs', 4: '>10 hrs'}
    df_plot = df.copy()
    if 'studytime' in df_plot.columns:
        df_plot['Study Tier'] = df_plot['studytime'].map(study_labels)
    else:
        df_plot['Study Tier'] = 'General'

    valid_cols = [c for c in ['G1', 'G2', 'G3'] if c in df_plot.columns]
    if not valid_cols:
        valid_cols = ['passed']

    order = ['<2 hrs', '2-5 hrs', '5-10 hrs', '>10 hrs']
    present_order = [o for o in order if o in df_plot['Study Tier'].values]
    if not present_order:
        present_order = df_plot['Study Tier'].unique().tolist()

    means = df_plot.groupby('Study Tier')[valid_cols].mean().reindex(present_order)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(means.index))
    width = 0.25 if len(valid_cols) == 3 else 0.35

    palette = ['#60a5fa', '#38bdf8', '#0284c7'] if dark_mode else ['#9ecae1', '#4292c6', '#08519c']

    for i, col in enumerate(valid_cols):
        offset = (i - (len(valid_cols) - 1) / 2) * width
        bars = ax.bar(x + offset, means[col], width, label=col, color=palette[i % len(palette)], edgecolor='#1e293b' if dark_mode else 'black')
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9,
                        color='#f8fafc' if dark_mode else 'black')

    ax.set_title("Average Examination Performance Across Weekly Study Time Groups",
                 fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Weekly Study Time Category", fontsize=11)
    ax.set_ylabel("Average Score (Scale 0 - 20)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(means.index, fontsize=10)
    ax.set_ylim(0, 16)
    ax.axhline(10, color='#f43f5e', linestyle='--', linewidth=1.5, label='Pass Threshold (10)')
    ax.legend(loc='upper left', frameon=True)
    
    if dark_mode:
        apply_dark_theme(fig, ax)
        
    plt.tight_layout()

    if save:
        path = os.path.join(get_graphs_dir(output_dir), 'studytime_performance_comparison.png')
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        return path
    return fig


def generate_all_eda_plots(df=None, output_dir=None):
    """
    Generates and saves all 7 EDA visualizations.
    """
    if df is None:
        from data_preprocessing import load_dataset
        df = load_dataset(verbose=False)

    out = get_graphs_dir(output_dir)
    print(f"[INFO] Generating EDA plots in: {out}")

    saved_plots = {
        'g3_distribution': plot_g3_distribution(df, out, save=True),
        'pass_fail_count': plot_pass_fail_distribution(df, out, save=True),
        'studytime_vs_g3': plot_study_time_vs_grades(df, out, save=True),
        'absences_vs_g3': plot_absences_vs_grades(df, out, save=True),
        'g1_g2_vs_g3': plot_previous_vs_final_grades(df, out, save=True),
        'correlation_heatmap': plot_correlation_heatmap(df, out, save=True),
        'studytime_comparison': plot_studytime_performance_comparison(df, out, save=True)
    }

    for name, path in saved_plots.items():
        print(f"  ✓ Saved: {os.path.basename(path)}")

    return saved_plots
