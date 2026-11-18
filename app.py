import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from src.profiler import (
    get_dataset_overview,
    get_column_diagnostics,
    detect_outliers_iqr,
)
from src.cleaner import clean_dataset

st.set_page_config(
    page_title="Automated Data Cleaning & EDA Profiler",
    page_icon="📊",
    layout="wide",
)


def generate_sample_data() -> pd.DataFrame:
    """Creates a realistic messy dataset for instant demo purposes."""
    np.random.seed(42)
    data = {
        "Customer_ID": [101, 102, 103, 104, 105, 105, 107, 108, 109, 110],
        "Age": [25, 34, np.nan, 45, 29, 29, 120, 38, np.nan, 31],
        "Annual_Income": [52000, 68000, 74000, np.nan, 61000, 61000, 82000, 490000, 55000, 63000],
        "City": ["Bengaluru", "Mysuru", "Bengaluru", None, "Mangaluru", "Mangaluru", "Hubballi", "Bengaluru", None, "Mysuru"],
        "Spending_Score": [65, 78, 82, 45, 90, 90, 12, 88, 70, np.nan],
    }
    return pd.DataFrame(data)


st.title("📊 Automated Data Cleaning & EDA Profiler")
st.markdown(
    "Upload any **CSV** or **Excel** file to inspect data quality, visualize distributions, "
    "clean anomalies, and export both the cleaned file and reproducible Python code."
)

# --- Sidebar: File Upload & Cleaning Rules ---
st.sidebar.header("1. Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])
use_sample = st.sidebar.checkbox("Use built-in messy sample dataset", value=(uploaded_file is None))

df = None
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
elif use_sample:
    df = generate_sample_data()

if df is not None:
    st.sidebar.markdown("---")
    st.sidebar.header("2. Configure Cleaning Pipeline")
    drop_duplicates = st.sidebar.checkbox("Remove duplicate rows", value=True)
    drop_high_missing = st.sidebar.checkbox("Drop columns with >50% missing values", value=False)
    numeric_strategy = st.sidebar.selectbox(
        "Numeric Missing Values",
        options=["median", "mean", "drop", "none"],
        index=0,
    )
    categorical_strategy = st.sidebar.selectbox(
        "Categorical Missing Values",
        options=["mode", "unknown", "drop", "none"],
        index=0,
    )
    cap_outliers = st.sidebar.checkbox("Cap outliers using IQR bounds", value=False)

    # --- Top KPI Metrics ---
    overview = get_dataset_overview(df)
    st.subheader("Dataset Health Overview")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Rows", overview["rows"])
    col2.metric("Columns", overview["columns"])
    col3.metric("Missing Cells", f"{overview['missing_cells']} ({overview['missing_pct']}%)")
    col4.metric("Duplicate Rows", overview["duplicate_rows"])
    col5.metric("Numeric / Categorical", f"{overview['numeric_cols']} / {overview['categorical_cols']}")

    # --- Main Navigation Tabs ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Data Diagnostics",
        "📈 Visual EDA",
        "🧹 Automated Cleaning & Export",
        "🐍 Generated Python Script",
    ])

    # TAB 1: Data Diagnostics
    with tab1:
        st.subheader("Raw Data Preview")
        st.dataframe(df.head(20), use_container_width=True)

        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("Column Diagnostics")
            st.dataframe(get_column_diagnostics(df), use_container_width=True)

        with col_right:
            st.subheader("Outlier Detection (IQR Method)")
            outlier_df = detect_outliers_iqr(df)
            if not outlier_df.empty:
                st.dataframe(outlier_df, use_container_width=True)
            else:
                st.info("No numeric columns available for outlier detection.")

        st.subheader("Summary Statistics")
        st.dataframe(df.describe(include="all").T, use_container_width=True)

    # TAB 2: Visual EDA
    with tab2:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if numeric_cols:
            st.subheader("Distribution & Box Plot Analysis")
            selected_num = st.selectbox("Select a numeric column", numeric_cols)
            c1, c2 = st.columns(2)
            with c1:
                fig_hist = px.histogram(
                    df, x=selected_num, marginal="rug", title=f"Distribution of {selected_num}"
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            with c2:
                fig_box = px.box(df, y=selected_num, title=f"Box Plot of {selected_num}")
                st.plotly_chart(fig_box, use_container_width=True)

            if len(numeric_cols) >= 2:
                st.subheader("Correlation Heatmap")
                corr_matrix = df[numeric_cols].corr()
                fig_corr = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    color_continuous_scale="RdBu_r",
                    title="Feature Correlation Matrix",
                )
                st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Upload a dataset with numeric columns to view histograms and correlation heatmaps.")

        st.subheader("Missing Values by Column")
        missing_series = df.isna().sum()
        missing_series = missing_series[missing_series > 0]
        if not missing_series.empty:
            fig_miss = px.bar(
                x=missing_series.index,
                y=missing_series.values,
                labels={"x": "Column", "y": "Missing Count"},
                title="Missing Value Counts",
            )
            st.plotly_chart(fig_miss, use_container_width=True)
        else:
            st.success("Zero missing values detected in the raw dataset!")

    # TAB 3: Automated Cleaning & Export
    with tab3:
        cleaned_df, generated_script = clean_dataset(
            df=df,
            drop_duplicates=drop_duplicates,
            numeric_strategy=numeric_strategy,
            categorical_strategy=categorical_strategy,
            cap_outliers=cap_outliers,
            drop_high_missing_cols=drop_high_missing,
        )

        cleaned_overview = get_dataset_overview(cleaned_df)
        st.subheader("Before vs. After Cleaning")
        m1, m2, m3 = st.columns(3)
        m1.metric(
            "Rows",
            cleaned_overview["rows"],
            delta=cleaned_overview["rows"] - overview["rows"],
        )
        m2.metric(
            "Missing Cells",
            cleaned_overview["missing_cells"],
            delta=cleaned_overview["missing_cells"] - overview["missing_cells"],
            delta_color="inverse",
        )
        m3.metric(
            "Duplicate Rows",
            cleaned_overview["duplicate_rows"],
            delta=cleaned_overview["duplicate_rows"] - overview["duplicate_rows"],
            delta_color="inverse",
        )

        st.dataframe(cleaned_df.head(20), use_container_width=True)

        csv_bytes = cleaned_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Cleaned Dataset (CSV)",
            data=csv_bytes,
            file_name="cleaned_dataset.csv",
            mime="text/csv",
        )

    # TAB 4: Generated Python Script
    with tab4:
        st.subheader("Reproducible Python Cleaning Script")
        st.markdown("Copy or download this standalone script to reproduce the exact cleaning pipeline locally.")
        st.code(generated_script, language="python")
        st.download_button(
            label="⬇️ Download Cleaning Script (.py)",
            data=generated_script.encode("utf-8"),
            file_name="clean_pipeline.py",
            mime="text/x-python",
        )
else:
    st.info("👈 Upload a CSV or Excel file from the sidebar, or enable the sample dataset to get started.")