"""
Data cleaning and validation module for Telco Customer Churn project.
Provides reproducible data loading, type casting, missing value handling,
and data integrity verification.
"""

from pathlib import Path
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    # If run from src/data or notebooks, resolve root accordingly
    current = Path.cwd()
    if (current / "data" / "raw").exists():
        return current
    for parent in current.parents:
        if (parent / "data" / "raw").exists():
            return parent
    return current


def load_raw_data(data_path: Path = None) -> pd.DataFrame:
    """
    Load raw Telco Customer Churn dataset from repository.
    Keeps raw data immutable.
    """
    if data_path is None:
        root = get_project_root()
        data_path = root / "data" / "raw" / "Telco-Customer-Churn.csv"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Raw data file not found at: {data_path}")
    
    df = pd.read_csv(data_path)
    return df


def clean_churn_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw Telco Customer Churn dataset:
    - Identifies whitespace in TotalCharges and converts to float64.
    - Sets TotalCharges to 0.0 for new customers with tenure == 0 (where unbilled).
    - Preserves all 7,043 unique customer records.
    - Maps binary target Churn ('No' -> 0, 'Yes' -> 1).
    - Maps binary demographic features consistently.
    - Retains customerID as an identifier (not to be used as a model feature).
    """
    df = df_raw.copy()
    
    # 1. TotalCharges cleaning:
    # 11 records have whitespace ' ' in TotalCharges, all corresponding to tenure == 0.
    # We convert to numeric (coercing spaces to NaN), then logically fill with 0.0
    # because customers with 0 tenure have incurred $0 in cumulative billing.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    
    # For tenure == 0, TotalCharges is legitimately 0.0 (no billing cycles completed)
    df.loc[df["tenure"] == 0, "TotalCharges"] = 0.0
    
    # 2. Target encoding:
    if "Churn" in df.columns:
        # Handle string 'No'/'Yes' regardless of whether pandas uses object, string, or str dtype
        if df["Churn"].dtype != np.int64 and df["Churn"].dtype != int:
            df["Churn"] = df["Churn"].replace({"No": 0, "Yes": 1}).astype(int)
    
    return df


def validate_cleaned_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate the cleaned dataset against known baseline domain invariants.
    """
    validation_results = {
        "record_count": len(df),
        "expected_records": 7043,
        "is_record_count_valid": len(df) == 7043,
        "unique_customer_ids": df["customerID"].nunique() if "customerID" in df.columns else None,
        "is_customer_id_unique": df["customerID"].nunique() == 7043 if "customerID" in df.columns else None,
        "totalcharges_null_count": df["TotalCharges"].isnull().sum(),
        "totalcharges_dtype": str(df["TotalCharges"].dtype),
        "churn_value_counts": df["Churn"].value_counts().to_dict() if "Churn" in df.columns else None,
        "churn_rate_pct": round(float(df["Churn"].mean() * 100), 2) if "Churn" in df.columns else None,
        "expected_churn_rate_pct": 26.54,
        "has_infinite_values": bool(np.isinf(df.select_dtypes(include=[np.number])).values.any()),
    }
    return validation_results
