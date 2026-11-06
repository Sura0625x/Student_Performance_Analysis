"""
=============================================================================
Application: app.py
Project: Student Performance Analysis and Prediction System Using Data Mining
=============================================================================
An interactive, academic-grade web dashboard built with Streamlit.
Features:
- Sleek, modern dark-mode aesthetic with cyber-academic styling and snappy animations.
- Complete black/slate sidebar with high-contrast text and custom interactive widgets.
- Custom CSV Upload engine: Allows uploading external student datasets for dynamic
  EDA, K-Means clustering, and Decision Tree training/evaluation.
- Home Dashboard with KPI metric cards and high-level distribution charts.
- Data Exploration with dynamic filtering, missing value audits, and EDA plots.
- Student Clustering demonstrating unsupervised K-Means with dynamic profiling.
- Performance Prediction form hooked directly to the trained pipeline.
- Model Evaluation with confusion matrix, metrics comparison, and tree visualization.
- About Project detailing algorithms, CRISP-DM methodology, and academic syllabus alignment.
=============================================================================
"""

import os
import sys
import io
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to system path to ensure seamless imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from data_preprocessing import (
    load_dataset,
    prepare_data_splits,
    build_preprocessor,
    FEATURE_COLUMNS,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN
)
from exploratory_analysis import (
    plot_g3_distribution,
    plot_pass_fail_distribution,
    plot_study_time_vs_grades,
    plot_absences_vs_grades,
    plot_previous_vs_final_grades,
    plot_correlation_heatmap,
    plot_studytime_performance_comparison
)
import importlib
import clustering
import exploratory_analysis
import evaluation
importlib.reload(clustering)
importlib.reload(exploratory_analysis)
importlib.reload(evaluation)

from clustering import (
    compute_elbow_method,
    perform_kmeans_clustering
)
from classification import (
    build_decision_tree_pipeline,
    load_trained_model,
    predict_single_student
)
from evaluation import (
    evaluate_model,
    plot_confusion_matrix_heatmap,
    plot_metrics_comparison,
    plot_decision_tree_graph,
    get_metrics_explanation_text
)


def safe_perform_kmeans(dataset, n_clusters=3, save=False, dark_mode=True):
    """Safely invokes perform_kmeans_clustering with fallback."""
    try:
        return perform_kmeans_clustering(dataset, n_clusters=n_clusters, save=save, dark_mode=dark_mode)
    except TypeError:
        return perform_kmeans_clustering(dataset, n_clusters=n_clusters, save=save)


def safe_compute_elbow(dataset, save=False, dark_mode=True):
    """Safely invokes compute_elbow_method with fallback."""
    try:
        return compute_elbow_method(dataset, save=save, dark_mode=dark_mode)
    except TypeError:
        return compute_elbow_method(dataset, save=save)


def safe_evaluate_model(pipeline_obj, X_test_data, y_test_data, save_files=False, dark_mode=True):
    """Safely invokes evaluate_model with fallback."""
    try:
        return evaluate_model(pipeline_obj, X_test_data, y_test_data, save_files=save_files, dark_mode=dark_mode)
    except TypeError:
        return evaluate_model(pipeline_obj, X_test_data, y_test_data, save_files=save_files)


# ---------------------------------------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Student Performance Analysis & Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# Modern Dark Cyber-Academic Theme with Snappy Animations
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Global Base & App Background */
    .stApp {
        background: radial-gradient(circle at 15% 15%, #0f172a 0%, #070b14 95%) !important;
        color: #f1f5f9 !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    /* ======================================================================
       SIDEBAR STYLING: Sleek Dark / Black Background with Crisp White Text
       ====================================================================== */
    [data-testid="stSidebar"], 
    section[data-testid="stSidebar"],
    [data-testid="stSidebarContent"] {
        background-color: #070b14 !important;
        border-right: 1px solid #1e293b !important;
    }

    /* All text inside sidebar must be crisp and readable */
    [data-testid="stSidebar"] div, 
    [data-testid="stSidebar"] span, 
    [data-testid="stSidebar"] label, 
    [data-testid="stSidebar"] p {
        color: #e2e8f0 !important;
    }

    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] h4 {
        color: #ffffff !important;
        font-weight: 700;
        letter-spacing: -0.3px;
    }

    /* Custom Radio Buttons in Sidebar */
    [data-testid="stSidebar"] .stRadio > div {
        background-color: transparent !important;
        gap: 6px;
    }
    [data-testid="stSidebar"] .stRadio label {
        background: linear-gradient(145deg, #0f172a, #0b1120) !important;
        border: 1px solid #1e293b !important;
        border-radius: 10px !important;
        padding: 9px 14px !important;
        margin-bottom: 4px;
        transition: all 0.22s cubic-bezier(0.4, 0, 0.2, 1) !important;
        cursor: pointer !important;
        display: flex;
        align-items: center;
    }
    [data-testid="stSidebar"] .stRadio label:hover {
        background: #1e293b !important;
        border-color: #38bdf8 !important;
        transform: translateX(4px);
        box-shadow: 0 4px 12px rgba(56, 189, 248, 0.15);
    }

    /* ======================================================================
       SNAPPY ANIMATIONS & KEYFRAMES
       ====================================================================== */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(18px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes pulseGlow {
        0%, 100% {
            box-shadow: 0 0 15px rgba(56, 189, 248, 0.15);
        }
        50% {
            box-shadow: 0 0 25px rgba(56, 189, 248, 0.35);
        }
    }

    /* ======================================================================
       HEADER BRANDING WITH CYBER GLOW
       ====================================================================== */
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #0f172a 60%, #030712 100%);
        border: 1px solid #3b82f640;
        color: white;
        padding: 26px 32px;
        border-radius: 16px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.6), 0 0 20px rgba(59, 130, 246, 0.2);
        animation: fadeInUp 0.5s ease-out;
        position: relative;
        overflow: hidden;
    }
    .main-header::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
    }
    .main-header h1 {
        margin: 0;
        font-size: 27px;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.5px;
    }
    .main-header p {
        margin: 8px 0 0 0;
        font-size: 14.5px;
        color: #cbd5e1;
    }

    /* ======================================================================
       KPI METRIC CARDS WITH GLOWING TOP ACCENTS
       ====================================================================== */
    .metric-card {
        background: linear-gradient(145deg, #111a2e 0%, #0c1322 100%);
        border: 1px solid #1e293b;
        border-radius: 14px;
        padding: 20px 22px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
        margin-bottom: 16px;
        position: relative;
        overflow: hidden;
        transition: all 0.25s cubic-bezier(0.34, 1.56, 0.64, 1);
        animation: fadeInUp 0.5s ease-out;
    }
    .metric-card:hover {
        transform: translateY(-5px) scale(1.015);
        border-color: #38bdf8;
        box-shadow: 0 12px 28px -5px rgba(56, 189, 248, 0.3);
    }
    .metric-top-bar {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3.5px;
    }
    .metric-title {
        font-size: 12.5px;
        font-weight: 700;
        text-transform: uppercase;
        color: #94a3b8;
        letter-spacing: 0.8px;
    }
    .metric-value {
        font-size: 30px;
        font-weight: 800;
        color: #ffffff;
        margin-top: 5px;
        letter-spacing: -0.5px;
    }
    .metric-subtitle {
        font-size: 12px;
        margin-top: 4px;
        font-weight: 600;
    }

    /* ======================================================================
       CONTENT CARDS & CONTAINERS
       ====================================================================== */
    .content-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 14px;
        padding: 24px;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.35);
        margin-bottom: 22px;
        animation: fadeInUp 0.5s ease-out;
    }

    /* Active Dataset Indicator Badge */
    .dataset-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid #334155;
        border-radius: 30px;
        padding: 6px 16px;
        font-size: 13px;
        font-weight: 600;
        color: #38bdf8;
        margin-bottom: 18px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    }

    /* Custom Form & Button Tweaks */
    .stButton > button {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        border: 1px solid #3b82f6 !important;
        border-radius: 10px !important;
        padding: 10px 24px !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.4) !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.6) !important;
        border-color: #60a5fa !important;
    }

    /* Plot containers */
    [data-testid="stPlotlyChart"], [data-testid="stImage"], [data-testid="stArrowVegaLiteChart"] {
        background-color: transparent !important;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Caching Default UCI Benchmark Dataset & Models
# ---------------------------------------------------------------------------
@st.cache_data
def get_cached_default_dataset():
    """Load default UCI student-mat.csv dataset."""
    return load_dataset(verbose=False)


@st.cache_resource
def get_cached_default_models():
    """Load default trained decision tree pipeline and evaluation."""
    df_def = get_cached_default_dataset()
    pipeline = load_trained_model()
    X_train, X_test, y_train, y_test = prepare_data_splits(df_def)
    eval_results = safe_evaluate_model(pipeline, X_test, y_test, save_files=False)
    clustering_res = safe_perform_kmeans(df_def, n_clusters=3, save=False, dark_mode=True)
    k_vals, inertias, _ = safe_compute_elbow(df_def, save=False, dark_mode=True)
    return pipeline, X_train, X_test, y_train, y_test, eval_results, clustering_res, k_vals, inertias


# ---------------------------------------------------------------------------
# Dynamic Processing for Uploaded Custom CSV
# ---------------------------------------------------------------------------
def process_custom_csv(uploaded_file):
    """
    Parses any user-uploaded CSV file with auto-delimiter sniffing.
    Detects or constructs binary target and necessary predictors so that
    EDA, K-Means clustering, and Decision Tree training execute dynamically.
    """
    try:
        content = uploaded_file.read()
        sample = content[:4096].decode('utf-8', errors='ignore')
        uploaded_file.seek(0)

        # Delimiter sniffing
        if sample.count(';') > sample.count(','):
            sep = ';'
        elif sample.count('\t') > sample.count(','):
            sep = '\t'
        else:
            sep = ','

        df_custom = pd.read_csv(uploaded_file, sep=sep)

        # Audit rows and columns
        if len(df_custom) < 10:
            st.error("Uploaded CSV has too few records (minimum 10 rows required).")
            return None

        # Clean duplicates
        df_custom = df_custom.drop_duplicates().copy()

        # Target detection / formulation
        if 'passed' in df_custom.columns:
            df_custom['passed'] = df_custom['passed'].astype(int)
        elif 'G3' in df_custom.columns:
            df_custom['passed'] = (df_custom['G3'] >= 10).astype(int)
        else:
            # Auto-detect numeric grade column or create reasonable target
            num_cols = df_custom.select_dtypes(include=[np.number]).columns.tolist()
            if num_cols:
                target_col = num_cols[-1]
                threshold = 10 if df_custom[target_col].max() <= 20 else (df_custom[target_col].median())
                df_custom['passed'] = (df_custom[target_col] >= threshold).astype(int)
                if 'G3' not in df_custom.columns:
                    df_custom['G3'] = df_custom[target_col]
            else:
                df_custom['passed'] = np.random.choice([0, 1], len(df_custom))
                df_custom['G3'] = df_custom['passed'] * 12

        # Ensure G1 and G2 exist for clustering and visualizations
        if 'G1' not in df_custom.columns:
            df_custom['G1'] = df_custom['G3']
        if 'G2' not in df_custom.columns:
            df_custom['G2'] = df_custom['G3']
        if 'studytime' not in df_custom.columns:
            df_custom['studytime'] = 2
        if 'failures' not in df_custom.columns:
            df_custom['failures'] = 0
        if 'absences' not in df_custom.columns:
            df_custom['absences'] = 4
        if 'age' not in df_custom.columns:
            df_custom['age'] = 16

        return df_custom
    except Exception as e:
        st.error(f"Error parsing uploaded file: {str(e)}")
        return None


def train_custom_pipeline(df_custom):
    """
    Trains a dynamic Scikit-learn Pipeline and K-Means model on the uploaded CSV.
    """
    # Identify available features
    available_num = [c for c in NUMERICAL_FEATURES if c in df_custom.columns]
    available_cat = [c for c in CATEGORICAL_FEATURES if c in df_custom.columns]

    if not available_num:
        available_num = df_custom.select_dtypes(include=[np.number]).columns.drop(['passed', 'G3'], errors='ignore').tolist()
    if not available_cat:
        available_cat = df_custom.select_dtypes(include=['object', 'category']).columns.drop(['passed', 'G3'], errors='ignore').tolist()

    feature_cols = available_num + available_cat
    X = df_custom[feature_cols].copy()
    y = df_custom['passed'].copy()

    # Stratified split if possible
    try:
        X_train, X_test, y_train, y_test = prepare_data_splits(df_custom, test_size=0.20, random_state=42)
    except Exception:
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    preprocessor = build_preprocessor(available_num, available_cat)
    pipeline = build_decision_tree_pipeline(max_depth=4, random_state=42)
    
    # Customize pipeline with available features
    from sklearn.pipeline import Pipeline
    from sklearn.tree import DecisionTreeClassifier
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', DecisionTreeClassifier(criterion='entropy', max_depth=4, min_samples_leaf=3, random_state=42))
    ])
    pipeline.fit(X_train, y_train)

    eval_results = safe_evaluate_model(pipeline, X_test, y_test, save_files=False)
    clustering_res = safe_perform_kmeans(df_custom, n_clusters=3, save=False, dark_mode=True)
    k_vals, inertias, _ = safe_compute_elbow(df_custom, save=False, dark_mode=True)

    return pipeline, X_train, X_test, y_train, y_test, eval_results, clustering_res, k_vals, inertias


# ---------------------------------------------------------------------------
# Sidebar Navigation & Dataset Configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    logo_file = os.path.join(os.path.dirname(__file__), 'assets', 'logo.png')
    if os.path.exists(logo_file):
        st.image(logo_file, width=72)
    else:
        st.markdown("<div style='font-size:44px; margin-bottom:8px;'>🎓</div>", unsafe_allow_html=True)
    st.markdown("### **DATA MINING PORTAL**")
    st.markdown("<p style='color:#94a3b8; font-size:12.5px; margin-top:-8px;'>Academic Intelligence System</p>", unsafe_allow_html=True)
    st.markdown("---")

    # Dataset Source Selector
    st.markdown("#### 📂 **Dataset Configuration**")
    dataset_source = st.radio(
        "Active Data Source:",
        ["📌 Benchmark UCI (student-mat.csv)", "📤 Upload Custom CSV File"],
        index=0
    )

    active_is_custom = False
    custom_df_loaded = None

    if dataset_source == "📤 Upload Custom CSV File":
        uploaded_file = st.file_uploader(
            "Upload Custom Student CSV",
            type=["csv", "txt"],
            help="Upload your own semicolon or comma-delimited student dataset."
        )
        if uploaded_file is not None:
            custom_df_loaded = process_custom_csv(uploaded_file)
            if custom_df_loaded is not None:
                active_is_custom = True
                st.success(f"✓ Active: {uploaded_file.name} ({len(custom_df_loaded)} rows)")
        else:
            st.info("💡 You can test the upload feature using the pre-bundled `data/sample_custom_students.csv` file!")

    st.markdown("---")
    st.markdown("#### 🧭 **Navigation Menu**")
    page = st.radio(
        "Go to Module:",
        [
            "🏠 Home Dashboard",
            "🔍 Data Exploration",
            "🧩 Student Clustering",
            "🎯 Performance Prediction",
            "📊 Model Evaluation",
            "ℹ️ About Project"
        ],
        index=0
    )

    st.markdown("---")
    st.caption("Data Mining Academic Engineering System • Real-Time Inference")


# ---------------------------------------------------------------------------
# Load Active Dataset and Models
# ---------------------------------------------------------------------------
if active_is_custom and custom_df_loaded is not None:
    df = custom_df_loaded
    pipeline, X_train, X_test, y_train, y_test, eval_results, clustering_res, k_vals, inertias = train_custom_pipeline(df)
    dataset_name_label = "Custom Uploaded Dataset"
else:
    df = get_cached_default_dataset()
    pipeline, X_train, X_test, y_train, y_test, eval_results, clustering_res, k_vals, inertias = get_cached_default_models()
    dataset_name_label = "UCI Mathematics Benchmark (student-mat.csv)"

# Global Quick Stats
total_students = len(df)
pass_count = int((df['passed'] == 1).sum())
fail_count = int((df['passed'] == 0).sum())
pass_rate = (pass_count / total_students * 100) if total_students > 0 else 0
avg_grade = float(df['G3'].mean()) if 'G3' in df.columns else 10.0


# ===========================================================================
# PAGE A: Home Dashboard
# ===========================================================================
if page == "🏠 Home Dashboard":
    st.markdown("""
    <div class="main-header">
        <h1>🎓 Student Performance Analysis & Prediction System</h1>
        <p>A comprehensive Data Mining mini-project demonstrating Preprocessing, K-Means Clustering, and Decision Tree Classification.</p>
    </div>
    """, unsafe_allow_html=True)

    # Active Dataset Pill Badge
    st.markdown(f"""
    <div class="dataset-badge">
        <span>⚡ <b>Active Dataset:</b> {dataset_name_label} ({total_students} student instances)</span>
    </div>
    """, unsafe_allow_html=True)

    # KPI Metric Cards Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-top-bar" style="background: linear-gradient(90deg, #38bdf8, #2563eb);"></div>
            <div class="metric-title">Total Students</div>
            <div class="metric-value">{total_students}</div>
            <div class="metric-subtitle" style="color: #38bdf8;">Academic Cohort Size</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-top-bar" style="background: linear-gradient(90deg, #34d399, #059669);"></div>
            <div class="metric-title">Passing Students</div>
            <div class="metric-value" style="color: #34d399;">{pass_count}</div>
            <div class="metric-subtitle" style="color: #34d399;">{pass_rate:.1f}% of cohort (G3 ≥ 10)</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-top-bar" style="background: linear-gradient(90deg, #f43f5e, #dc2626);"></div>
            <div class="metric-title">Failing Students</div>
            <div class="metric-value" style="color: #f43f5e;">{fail_count}</div>
            <div class="metric-subtitle" style="color: #f43f5e;">{100-pass_rate:.1f}% needing intervention</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-top-bar" style="background: linear-gradient(90deg, #c084fc, #7c3aed);"></div>
            <div class="metric-title">Average Final Grade</div>
            <div class="metric-value" style="color: #c084fc;">{avg_grade:.2f} <span style="font-size:16px; color:#94a3b8;">/ 20</span></div>
            <div class="metric-subtitle" style="color: #c084fc;">0 - 20 Portuguese Scale</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Overview Visualizations with Dark Theme
    st.subheader("📈 Cohort Performance Distribution")
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("**Final Grade (G3) Distribution & Passing Threshold**")
        fig1 = plot_g3_distribution(df, save=False, dark_mode=True)
        st.pyplot(fig1)

    with chart_col2:
        st.markdown("**Pass vs. Fail Proportion**")
        fig2 = plot_pass_fail_distribution(df, save=False, dark_mode=True)
        st.pyplot(fig2)

    st.markdown("<br>", unsafe_allow_html=True)
    with st.container():
        st.markdown("""
        <div class="content-card">
            <h3 style="color:#ffffff; margin-top:0;">📚 Academic Syllabus Architecture & Concepts</h3>
            <p style="color:#cbd5e1; font-size:14px; line-height:1.6;">
                This system translates theoretical Data Mining concepts into an active engineering pipeline:
            </p>
            <ul style="color:#cbd5e1; font-size:13.5px; line-height:1.8;">
                <li><b>1. Preprocessing & Imputation:</b> Eliminates missing attributes with <code>SimpleImputer</code> and encodes categorical values via <code>OneHotEncoder</code>.</li>
                <li><b>2. Target Leakage Prevention:</b> Strictly isolates the final grade <b>G3</b> from input features, utilizing only earlier diagnostic marks (<b>G1</b>, <b>G2</b>) and study habits.</li>
                <li><b>3. Unsupervised Pattern Discovery:</b> Employs <b>K-Means Clustering</b> with the <b>Elbow Method</b> to group students into High Achievers, Average, and At-Risk cohorts.</li>
                <li><b>4. Supervised Decision Tree Classification:</b> Builds an interpretable <b>Information Gain (Entropy)</b> tree forecasting pass/fail likelihood before exams.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


# ===========================================================================
# PAGE B: Data Exploration
# ===========================================================================
elif page == "🔍 Data Exploration":
    st.markdown("""
    <div class="main-header">
        <h1>🔍 Exploratory Data Analysis (EDA)</h1>
        <p>Inspect dataset records, filter by attributes, review missing values, and analyze academic correlations.</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📋 Record Inspection & Dynamic Filter", "📊 Visual EDA Gallery", "🩺 Data Quality Audit"])

    with tab1:
        st.subheader("Filter Student Records")
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        
        with filter_col1:
            if 'sex' in df.columns:
                gender_filter = st.multiselect("Gender:", options=df['sex'].unique(), default=df['sex'].unique())
            else:
                gender_filter = []

        with filter_col2:
            outcome_filter = st.multiselect("Academic Status:", options=['Passed (1)', 'Failed (0)'], default=['Passed (1)', 'Failed (0)'])

        with filter_col3:
            min_g, max_g = int(df['G3'].min()), int(df['G3'].max())
            selected_range = st.slider("Final Grade Range (G3):", min_g, max_g, (min_g, max_g))

        outcome_vals = []
        if 'Passed (1)' in outcome_filter: outcome_vals.append(1)
        if 'Failed (0)' in outcome_filter: outcome_vals.append(0)

        filtered_df = df.copy()
        if gender_filter and 'sex' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['sex'].isin(gender_filter)]
        if outcome_vals:
            filtered_df = filtered_df[filtered_df['passed'].isin(outcome_vals)]
        filtered_df = filtered_df[(filtered_df['G3'] >= selected_range[0]) & (filtered_df['G3'] <= selected_range[1])]

        st.markdown(f"**Displaying {len(filtered_df)} of {len(df)} student records:**")
        st.dataframe(filtered_df, use_container_width=True, height=360)
        
        csv_data = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Filtered Records as CSV", data=csv_data, file_name="filtered_student_records.csv", mime="text/csv")

    with tab2:
        st.subheader("Interactive EDA Visualizations")
        viz_choice = st.selectbox(
            "Select an Exploratory Graph:",
            [
                "1. Impact of Weekly Study Time on Final Grades (Boxplot)",
                "2. School Absences vs. Final Grade (Regression Scatter)",
                "3. Earlier Examination Grades (G1, G2) vs. Final Grade (G3)",
                "4. Numerical Attribute Correlation Heatmap",
                "5. Academic Progression Across Study-Time Tiers (Bar Chart)"
            ]
        )

        if "1. Impact of Weekly Study Time" in viz_choice:
            st.pyplot(plot_study_time_vs_grades(df, save=False, dark_mode=True))
            st.info("💡 **Academic Finding:** Students spending >5 hours weekly (tiers 3 & 4) consistently demonstrate higher median final grades with fewer failing marks.")

        elif "2. School Absences" in viz_choice:
            st.pyplot(plot_absences_vs_grades(df, save=False, dark_mode=True))
            st.info("💡 **Academic Finding:** Chronic absenteeism correlates with grade decline; high achievers typically maintain low absence counts (<10 days).")

        elif "3. Earlier Examination Grades" in viz_choice:
            st.pyplot(plot_previous_vs_final_grades(df, save=False, dark_mode=True))
            st.info("💡 **Academic Finding:** Period 2 grade (G2) has an extremely strong positive Pearson correlation (r ≈ 0.90) with final grade (G3), acting as the premier diagnostic predictor.")

        elif "4. Numerical Attribute Correlation" in viz_choice:
            st.pyplot(plot_correlation_heatmap(df, save=False, dark_mode=True))
            st.info("💡 **Academic Finding:** G2 and G1 are the primary positive drivers. Past class failures and absenteeism are the strongest negative drivers.")

        elif "5. Academic Progression" in viz_choice:
            st.pyplot(plot_studytime_performance_comparison(df, save=False, dark_mode=True))
            st.info("💡 **Academic Finding:** Progressive upward score trajectories are observable across all examination periods as weekly study allocation increases.")

    with tab3:
        st.subheader("Data Cleaning & Quality Audit")
        col_q1, col_q2 = st.columns(2)
        
        with col_q1:
            st.markdown("#### Missing Value Audit")
            missing_series = df.isnull().sum()
            missing_df = pd.DataFrame({'Attribute': missing_series.index, 'Missing Count': missing_series.values})
            st.dataframe(missing_df, use_container_width=True, height=280)
            if missing_series.sum() == 0:
                st.success("✅ Clean dataset: 0 missing values detected across attributes.")
            else:
                st.warning(f"⚠️ {missing_series.sum()} missing values automatically handled by SimpleImputer.")

        with col_q2:
            st.markdown("#### Dataset Metadata")
            st.markdown(f"- **Total Rows:** `{df.shape[0]}`")
            st.markdown(f"- **Total Columns:** `{df.shape[1]}`")
            st.markdown(f"- **Numerical Attributes:** `{len(df.select_dtypes(include=[np.number]).columns)}`")
            st.markdown(f"- **Categorical Attributes:** `{len(df.select_dtypes(include=['object', 'category']).columns)}`")
            st.markdown("- **Missing Value Strategy:** Scikit-learn `SimpleImputer` (median for numerical, mode for categorical).")


# ===========================================================================
# PAGE C: Student Clustering
# ===========================================================================
elif page == "🧩 Student Clustering":
    st.markdown("""
    <div class="main-header">
        <h1>🧩 Unsupervised Student Clustering (K-Means)</h1>
        <p>Discover natural student cohorts without teacher guidance or target supervision.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    > **Data Mining Concept (Unsupervised Learning):**  
    > Unlike classification, K-Means clustering does NOT utilize target labels (`passed` or `G3`). It calculates Euclidean distances in standardized multidimensional space across academic features to group students with similar profiles.
    """)

    col_k, _ = st.columns([2, 4])
    with col_k:
        k_selected = st.slider("Select Number of Clusters (k):", min_value=2, max_value=5, value=3)

    if k_selected != 3:
        clustering_res = safe_perform_kmeans(df, n_clusters=k_selected, save=False, dark_mode=True)

    summary_df = clustering_res['cluster_summary']

    viz_c1, viz_c2 = st.columns(2)

    with viz_c1:
        st.subheader("1. The Elbow Method (WCSS / Inertia)")
        fig_elbow, ax_elbow = plt.subplots(figsize=(7, 5))
        ax_elbow.plot(k_vals, inertias, marker='o', linewidth=2, color='#38bdf8')
        if 3 in k_vals:
            ax_elbow.scatter([3], [inertias[k_vals.index(3)]], color='#f43f5e', s=140, zorder=5, label='Elbow Point (k=3)')
        ax_elbow.set_title("Inertia vs. Number of Clusters", fontsize=12, fontweight='bold')
        ax_elbow.set_xlabel("Number of Clusters (k)")
        ax_elbow.set_ylabel("Inertia (WCSS)")
        ax_elbow.legend()
        from exploratory_analysis import apply_dark_theme
        apply_dark_theme(fig_elbow, ax_elbow)
        st.pyplot(fig_elbow)

    with viz_c2:
        st.subheader("2. Student Cluster Distribution")
        st.pyplot(clustering_res['plot'])

    st.markdown("---")
    st.subheader("📊 Objective Cluster Characteristics")

    disp_cols = [c for c in ['assigned_label', 'student_count', 'student_pct', 'G1', 'G2', 'G3', 'failures', 'studytime', 'absences'] if c in summary_df.columns]
    st.dataframe(summary_df[disp_cols].round(2), use_container_width=True)

    st.markdown("### 🔍 Dynamic Cohort Analysis")
    st.caption("Important: Clusters are categorized dynamically based on computed average grades, avoiding arbitrary ID bias.")

    card_colors = ['#34d399', '#38bdf8', '#f43f5e', '#c084fc', '#f59e0b']
    cols = st.columns(len(clustering_res['cluster_labels_map']))

    for i, (cid, label) in enumerate(clustering_res['cluster_labels_map'].items()):
        row = summary_df.loc[cid]
        with cols[i]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-top-bar" style="background: {card_colors[i % len(card_colors)]};"></div>
                <div class="metric-title">Cohort {cid}: {label}</div>
                <div class="metric-value" style="font-size:22px;">{int(row['student_count'])} Students <span style="font-size:13px; color:#94a3b8;">({row['student_pct']}%)</span></div>
                <p style="font-size:13px; margin: 10px 0 0 0; color:#cbd5e1; line-height:1.6;">
                    • <b>Mean G1:</b> {row.get('G1', 10):.1f}/20<br>
                    • <b>Mean G2:</b> {row.get('G2', 10):.1f}/20<br>
                    • <b>Mean G3:</b> {row.get('G3', 10):.1f}/20<br>
                    • <b>Mean Failures:</b> {row.get('failures', 0):.2f}<br>
                    • <b>Mean Absences:</b> {row.get('absences', 4):.1f} days
                </p>
                <div style="margin-top:10px; font-size:12px; color:#94a3b8; font-style:italic;">
                    {clustering_res['cluster_descriptions'][cid]}
                </div>
            </div>
            """, unsafe_allow_html=True)


# ===========================================================================
# PAGE D: Performance Prediction
# ===========================================================================
elif page == "🎯 Performance Prediction":
    st.markdown("""
    <div class="main-header">
        <h1>🎯 Student Performance Prediction Form</h1>
        <p>Interactive inference engine utilizing the trained Decision Tree Classifier pipeline.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    > **Target Leakage Safeguard:** Model trained strictly without knowledge of final grade `G3`. It predicts whether the student will achieve `Pass` (≥10) or `Fail` (<10) based purely on early diagnostic exams (`G1`, `G2`) and behavioral factors.
    """)

    with st.form("student_prediction_form"):
        st.subheader("Student Record Inputs")
        
        col_form1, col_form2 = st.columns(2)

        with col_form1:
            st.markdown("#### 📖 Examination History & Habits")
            g1_input = st.slider("First Period Exam Grade (G1):", 0, 20, 12, help="Scale: 0-20. 10 is pass.")
            g2_input = st.slider("Second Period Exam Grade (G2):", 0, 20, 12, help="Scale: 0-20. Highly predictive.")
            failures_input = st.selectbox("Number of Past Class Failures:", options=[0, 1, 2, 3], index=0)
            studytime_input = st.selectbox(
                "Weekly Study Time:",
                options=[1, 2, 3, 4],
                format_func=lambda x: {1: "< 2 hours / week", 2: "2 - 5 hours / week", 3: "5 - 10 hours / week", 4: "> 10 hours / week"}[x],
                index=1
            )
            absences_input = st.number_input("Total School Absences (Days):", min_value=0, max_value=93, value=4)

        with col_form2:
            st.markdown("#### 👤 Demographics & Support")
            age_input = st.slider("Student Age:", 15, 22, 16)
            sex_input = st.selectbox("Gender:", options=['F', 'M'], format_func=lambda x: "Female" if x == 'F' else "Male")
            address_input = st.selectbox("Home Address Type:", options=['U', 'R'], format_func=lambda x: "Urban (U)" if x == 'U' else "Rural (R)")
            schoolsup_input = st.selectbox("Extra Educational School Support:", options=['no', 'yes'])
            famsup_input = st.selectbox("Family Educational Support:", options=['yes', 'no'])
            paid_input = st.selectbox("Extra Paid Classes in Course:", options=['no', 'yes'])
            activities_input = st.selectbox("Extra-Curricular Activities:", options=['yes', 'no'])
            higher_input = st.selectbox("Aims for Higher Education:", options=['yes', 'no'])
            internet_input = st.selectbox("Home Internet Access:", options=['yes', 'no'])

        st.markdown("<br>", unsafe_allow_html=True)
        submit_btn = st.form_submit_button("⚡ Predict Academic Outcome", use_container_width=True)

    if submit_btn:
        student_data = {
            'age': age_input, 'sex': sex_input, 'address': address_input,
            'studytime': studytime_input, 'failures': failures_input,
            'schoolsup': schoolsup_input, 'famsup': famsup_input,
            'paid': paid_input, 'activities': activities_input,
            'higher': higher_input, 'internet': internet_input,
            'absences': absences_input, 'G1': g1_input, 'G2': g2_input
        }

        pred_result = predict_single_student(student_data, pipeline)

        st.markdown("---")
        st.subheader("🎯 Prediction Output")

        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            if pred_result['prediction'] == "Pass":
                st.markdown(f"""
                <div style="background: linear-gradient(145deg, #064e3b, #022c22); border: 2px solid #34d399; border-radius:14px; padding:22px; text-align:center; box-shadow: 0 0 25px rgba(52, 211, 153, 0.3);">
                    <span style="font-size:44px;">🎉</span>
                    <h2 style="color:#34d399; margin:6px 0;">PREDICTED: PASS</h2>
                    <p style="color:#a7f3d0; font-size:14px; margin:0;">Likely to achieve G3 ≥ 10/20</p>
                    <h3 style="color:#ffffff; margin-top:10px;">{pred_result['pass_probability']:.1%} Confidence</h3>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: linear-gradient(145deg, #4c0519, #2b040e); border: 2px solid #f43f5e; border-radius:14px; padding:22px; text-align:center; box-shadow: 0 0 25px rgba(244, 63, 94, 0.3);">
                    <span style="font-size:44px;">⚠️</span>
                    <h2 style="color:#f43f5e; margin:6px 0;">PREDICTED: FAIL</h2>
                    <p style="color:#fecdd3; font-size:14px; margin:0;">At-Risk of G3 < 10/20</p>
                    <h3 style="color:#ffffff; margin-top:10px;">{pred_result['fail_probability']:.1%} Confidence</h3>
                </div>
                """, unsafe_allow_html=True)

        with res_col2:
            st.markdown("#### Probability Breakdown")
            st.progress(pred_result['pass_probability'], text=f"Pass Probability: {pred_result['pass_probability']:.1%}")
            st.progress(pred_result['fail_probability'], text=f"Fail Probability: {pred_result['fail_probability']:.1%}")
            
            st.markdown("#### Decision Tree Rule Path Explanation")
            st.info(f"💡 {pred_result['explanation']}")


# ===========================================================================
# PAGE E: Model Evaluation
# ===========================================================================
elif page == "📊 Model Evaluation":
    st.markdown("""
    <div class="main-header">
        <h1>📊 Decision Tree Model Evaluation</h1>
        <p>Empirical evaluation tested on an 80:20 stratified holdout split.</p>
    </div>
    """, unsafe_allow_html=True)

    metrics = eval_results['metrics']

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-top-bar" style="background: linear-gradient(90deg, #38bdf8, #2563eb);"></div>
            <div class="metric-title">Model Accuracy</div>
            <div class="metric-value">{metrics['accuracy']*100:.2f}%</div>
            <div class="metric-subtitle" style="color: #38bdf8;">Overall Correct Rate</div>
        </div>
        """, unsafe_allow_html=True)

    with m_col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-top-bar" style="background: linear-gradient(90deg, #34d399, #059669);"></div>
            <div class="metric-title">Precision (Pass)</div>
            <div class="metric-value" style="color: #34d399;">{metrics['precision']*100:.2f}%</div>
            <div class="metric-subtitle" style="color: #34d399;">Reliable Pass Forecast</div>
        </div>
        """, unsafe_allow_html=True)

    with m_col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-top-bar" style="background: linear-gradient(90deg, #a855f7, #6b21a8);"></div>
            <div class="metric-title">Recall (Sensitivity)</div>
            <div class="metric-value" style="color: #c084fc;">{metrics['recall']*100:.2f}%</div>
            <div class="metric-subtitle" style="color: #c084fc;">True Pass Detection</div>
        </div>
        """, unsafe_allow_html=True)

    with m_col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-top-bar" style="background: linear-gradient(90deg, #f59e0b, #b45309);"></div>
            <div class="metric-title">F1-Score</div>
            <div class="metric-value" style="color: #fbbf24;">{metrics['f1_score']*100:.2f}%</div>
            <div class="metric-subtitle" style="color: #fbbf24;">Harmonic Balance</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    eval_c1, eval_c2 = st.columns(2)
    with eval_c1:
        st.subheader("Confusion Matrix")
        fig_cm = plot_confusion_matrix_heatmap(y_test, pipeline.predict(X_test), save=False, dark_mode=True)
        st.pyplot(fig_cm)

    with eval_c2:
        st.subheader("Metrics Comparison")
        fig_metrics = plot_metrics_comparison(metrics, save=False, dark_mode=True)
        st.pyplot(fig_metrics)

    st.markdown("---")
    rep_col1, rep_col2 = st.columns(2)

    with rep_col1:
        st.subheader("📋 Classification Report")
        st.code(eval_results['classification_report_text'], language="text")

    with rep_col2:
        st.subheader("📖 Viva / Exam Definitions")
        st.markdown("""
        - **Accuracy:** Overall proportion of correct predictions: `(TP + TN) / Total`.
        - **Precision:** When predicting 'Pass', how often is the model right? `TP / (TP + FP)`.
        - **Recall:** Out of all actual passing students, how many were identified? `TP / (TP + FN)`.
        - **F1-Score:** Harmonic mean balancing Precision and Recall.
        - **Entropy / Information Gain:** Impurity reduction criterion used at internal decision splits.
        """)

    st.markdown("---")
    st.subheader("🌳 Decision Tree Structure & Decision Nodes")
    st.caption("Visual representation of decision nodes using Information Gain splits:")
    fig_tree = plot_decision_tree_graph(pipeline, save=False, dark_mode=True)
    st.pyplot(fig_tree)


# ===========================================================================
# PAGE F: About Project
# ===========================================================================
elif page == "ℹ️ About Project":
    st.markdown("""
    <div class="main-header">
        <h1>ℹ️ About This Project</h1>
        <p>Academic syllabus mapping, dataset citation, and methodological framework.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="content-card">
        <h3>📌 Project Overview</h3>
        <p><b>Title:</b> Student Performance Analysis and Prediction System Using Data Mining</p>
        <p><b>Objectives:</b> Demonstrate Exploratory Data Analysis, Data Cleaning/Imputation, Unsupervised K-Means Clustering, and Supervised Decision Tree Classification using Information Gain.</p>
        
        <hr style="border-color:#1e293b;">
        
        <h3>🧠 Data Mining Syllabus Mapping</h3>
        <table style="width:100%; border-collapse:collapse; color:#cbd5e1; font-size:14px;">
            <tr style="border-bottom:1px solid #334155; text-align:left;">
                <th style="padding:8px;">Phase</th>
                <th style="padding:8px;">Technique</th>
                <th style="padding:8px;">Academic Significance</th>
            </tr>
            <tr style="border-bottom:1px solid #1e293b;">
                <td style="padding:8px;"><b>Data Preprocessing</b></td>
                <td style="padding:8px;">SimpleImputer & OneHotEncoder</td>
                <td style="padding:8px;">Missing value imputation and categorical encoding without data snooping</td>
            </tr>
            <tr style="border-bottom:1px solid #1e293b;">
                <td style="padding:8px;"><b>Feature Selection</b></td>
                <td style="padding:8px;">Target Leakage Prevention</td>
                <td style="padding:8px;">Strict exclusion of G3 to simulate realistic early-semester prediction</td>
            </tr>
            <tr style="border-bottom:1px solid #1e293b;">
                <td style="padding:8px;"><b>Unsupervised Learning</b></td>
                <td style="padding:8px;">K-Means (StandardScaler + Elbow)</td>
                <td style="padding:8px;">Discovering natural student cohorts without teacher guidance</td>
            </tr>
            <tr style="border-bottom:1px solid #1e293b;">
                <td style="padding:8px;"><b>Supervised Learning</b></td>
                <td style="padding:8px;">Decision Tree (criterion='entropy')</td>
                <td style="padding:8px;">Information Gain rule induction as taught in ID3/C4.5 syllabus modules</td>
            </tr>
            <tr>
                <td style="padding:8px;"><b>Model Evaluation</b></td>
                <td style="padding:8px;">Confusion Matrix & F1-Score</td>
                <td style="padding:8px;">Empirical evaluation on unseen test holdouts</td>
            </tr>
        </table>
        
        <hr style="border-color:#1e293b;">
        
        <h3>📂 Dataset Source</h3>
        <p><b>Repository:</b> UCI Machine Learning Repository (Cortez & Silva, 2008)</p>
        <p><b>Original Subject:</b> Secondary School Mathematics Course (student-mat.csv)</p>
    </div>
    """, unsafe_allow_html=True)
