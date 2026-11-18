import pandas as pd
import numpy as np


def clean_dataset(
    df: pd.DataFrame,
    drop_duplicates: bool = True,
    numeric_strategy: str = "median",
    categorical_strategy: str = "mode",
    cap_outliers: bool = False,
    drop_high_missing_cols: bool = False,
    missing_threshold: float = 0.5,
) -> tuple[pd.DataFrame, str]:
    """
    Cleans the DataFrame based on user-selected rules and returns:
    1. The cleaned DataFrame
    2. A standalone Python script reproducing the exact cleaning steps
    """
    cleaned_df = df.copy()
    code_lines = [
        "import pandas as pd",
        "import numpy as np",
        "",
        "# Load your dataset",
        "# df = pd.read_csv('your_dataset.csv')",
        "",
    ]

    # 1. Drop columns with excessive missing values
    if drop_high_missing_cols:
        cols_to_drop = [
            col for col in cleaned_df.columns
            if cleaned_df[col].isna().mean() > missing_threshold
        ]
        if cols_to_drop:
            cleaned_df.drop(columns=cols_to_drop, inplace=True)
            code_lines.append(f"# Drop columns with > {int(missing_threshold * 100)}% missing values")
            code_lines.append(f"df.drop(columns={cols_to_drop}, inplace=True)\n")

    # 2. Remove duplicate rows
    if drop_duplicates:
        before = len(cleaned_df)
        cleaned_df.drop_duplicates(inplace=True)
        if len(cleaned_df) < before:
            code_lines.append("# Remove duplicate rows")
            code_lines.append("df.drop_duplicates(inplace=True)\n")

    # 3. Handle missing values in numeric columns
    num_cols = cleaned_df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols and numeric_strategy != "none":
        code_lines.append(f"# Impute numeric missing values using {numeric_strategy}")
        if numeric_strategy == "mean":
            for col in num_cols:
                if cleaned_df[col].isna().any():
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            code_lines.append("for col in df.select_dtypes(include=[np.number]).columns:")
            code_lines.append("    df[col] = df[col].fillna(df[col].mean())\n")

        elif numeric_strategy == "median":
            for col in num_cols:
                if cleaned_df[col].isna().any():
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
            code_lines.append("for col in df.select_dtypes(include=[np.number]).columns:")
            code_lines.append("    df[col] = df[col].fillna(df[col].median())\n")

        elif numeric_strategy == "drop":
            cleaned_df.dropna(subset=num_cols, inplace=True)
            code_lines.append(f"df.dropna(subset={num_cols}, inplace=True)\n")

    # 4. Handle missing values in categorical columns
    cat_cols = cleaned_df.select_dtypes(exclude=[np.number]).columns.tolist()
    if cat_cols and categorical_strategy != "none":
        code_lines.append(f"# Impute categorical missing values using {categorical_strategy}")
        if categorical_strategy == "mode":
            for col in cat_cols:
                if cleaned_df[col].isna().any():
                    mode_val = cleaned_df[col].mode()
                    fill_val = mode_val[0] if not mode_val.empty else "Unknown"
                    cleaned_df[col] = cleaned_df[col].fillna(fill_val)
            code_lines.append("for col in df.select_dtypes(exclude=[np.number]).columns:")
            code_lines.append("    mode_val = df[col].mode()")
            code_lines.append("    df[col] = df[col].fillna(mode_val[0] if not mode_val.empty else 'Unknown')\n")

        elif categorical_strategy == "unknown":
            for col in cat_cols:
                if cleaned_df[col].isna().any():
                    cleaned_df[col] = cleaned_df[col].fillna("Unknown")
            code_lines.append("for col in df.select_dtypes(exclude=[np.number]).columns:")
            code_lines.append("    df[col] = df[col].fillna('Unknown')\n")

        elif categorical_strategy == "drop":
            cleaned_df.dropna(subset=cat_cols, inplace=True)
            code_lines.append(f"df.dropna(subset={cat_cols}, inplace=True)\n")

    # 5. Cap outliers using IQR (Winsorization)
    if cap_outliers and num_cols:
        code_lines.append("# Cap outliers using 1.5 * IQR bounds")
        code_lines.append("for col in df.select_dtypes(include=[np.number]).columns:")
        code_lines.append("    q1 = df[col].quantile(0.25)")
        code_lines.append("    q3 = df[col].quantile(0.75)")
        code_lines.append("    iqr = q3 - q1")
        code_lines.append("    df[col] = df[col].clip(lower=q1 - 1.5 * iqr, upper=q3 + 1.5 * iqr)\n")

        for col in num_cols:
            q1 = cleaned_df[col].quantile(0.25)
            q3 = cleaned_df[col].quantile(0.75)
            iqr = q3 - q1
            cleaned_df[col] = cleaned_df[col].clip(lower=q1 - 1.5 * iqr, upper=q3 + 1.5 * iqr)

    code_lines.append("# Export cleaned dataset")
    code_lines.append("df.to_csv('cleaned_dataset.csv', index=False)")

    return cleaned_df, "\n".join(code_lines)