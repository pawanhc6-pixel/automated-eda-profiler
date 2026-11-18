import pandas as pd
import numpy as np


def get_dataset_overview(df: pd.DataFrame) -> dict:
    """Returns high-level metrics about the dataset."""
    total_cells = df.shape[0] * df.shape[1]
    total_missing = int(df.isna().sum().sum())
    missing_pct = round((total_missing / total_cells) * 100, 2) if total_cells > 0 else 0.0

    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing_cells": total_missing,
        "missing_pct": missing_pct,
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_cols": len(df.select_dtypes(include=[np.number]).columns),
        "categorical_cols": len(df.select_dtypes(exclude=[np.number]).columns),
    }


def get_column_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    """Generates per-column diagnostics including data types, missing counts, and unique values."""
    diagnostics = []
    for col in df.columns:
        missing_count = int(df[col].isna().sum())
        missing_pct = round((missing_count / len(df)) * 100, 2) if len(df) > 0 else 0.0
        unique_count = int(df[col].nunique(dropna=True))

        diagnostics.append({
            "Column": col,
            "Data Type": str(df[col].dtype),
            "Missing Count": missing_count,
            "Missing (%)": missing_pct,
            "Unique Values": unique_count,
        })

    return pd.DataFrame(diagnostics)


def detect_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
    """Detects outliers in numeric columns using the Interquartile Range (IQR) method."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outlier_summary = []

    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = series[(series < lower_bound) | (series > upper_bound)]
        outlier_count = len(outliers)
        outlier_pct = round((outlier_count / len(series)) * 100, 2)

        outlier_summary.append({
            "Column": col,
            "Q1 (25%)": round(q1, 3),
            "Q3 (75%)": round(q3, 3),
            "Lower Bound": round(lower_bound, 3),
            "Upper Bound": round(upper_bound, 3),
            "Outlier Count": outlier_count,
            "Outlier (%)": outlier_pct,
        })

    return pd.DataFrame(outlier_summary)