# 📊 Automated Data Cleaning & EDA Profiler

An interactive web application built with **Python**, **Streamlit**, **Pandas**, and **Plotly** that automates Exploratory Data Analysis (EDA), diagnoses data quality issues, cleans messy tabular datasets, and generates reproducible Python cleaning scripts.

---

## ✨ Key Features

- **Instant Dataset Health Metrics:** Detects total rows, columns, missing value percentages, duplicate records, and data type distributions.
- **Automated Outlier Diagnostics:** Identifies statistical outliers across numeric features using the Interquartile Range (`1.5 * IQR`) method.
- **Interactive Visual EDA:** Generates dynamic histograms, box plots, missing-value bar charts, and correlation heatmaps using Plotly.
- **Configurable Cleaning Pipeline:**
  - Duplicate removal
  - High-missingness column pruning (`>50%` nulls)
  - Numeric imputation (`Median`, `Mean`, or `Drop`)
  - Categorical imputation (`Mode`, `'Unknown'`, or `Drop`)
  - IQR-based outlier capping (Winsorization)
- **One-Click Exports:** Download the cleaned dataset as a `.csv` file alongside a standalone `.py` script that reproduces every transformation.

---

## 🏗️ Project Architecture

```text
automated-eda-profiler/
├── src/
│   ├── __init__.py
│   ├── profiler.py       # Statistical summaries, missing diagnostics, and IQR outlier detection
│   └── cleaner.py        # Data transformation pipeline and dynamic Python code generator
├── app.py                # Streamlit dashboard and Plotly visualizations
├── requirements.txt      # Pinned dependencies
├── .gitignore
└── README.md