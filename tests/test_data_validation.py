"""
Unit tests for Telco Customer Churn data preparation, cleaning,
duplicate record validation, and feature engineering.
"""

import unittest
from pathlib import Path
import pandas as pd
import numpy as np

from src.data.cleaning import load_raw_data, clean_churn_data, validate_cleaned_dataset, get_project_root
from src.features.engineering import engineer_features


class TestDataPreparation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = get_project_root()
        cls.df_raw = load_raw_data()
        cls.df_clean = clean_churn_data(cls.df_raw)
        cls.df_featured = engineer_features(cls.df_clean)

    def test_raw_dataset_integrity(self):
        """Verify baseline properties of the raw dataset."""
        self.assertEqual(len(self.df_raw), 7043, "Raw dataset must contain exactly 7043 rows.")
        self.assertEqual(self.df_raw.shape[1], 21, "Raw dataset must contain 21 columns.")
        self.assertEqual(self.df_raw["customerID"].nunique(), 7043, "All 7043 customer IDs must be unique.")
        
        # Verify exactly 11 whitespace records in TotalCharges
        ws_count = (self.df_raw["TotalCharges"].str.strip() == "").sum()
        self.assertEqual(ws_count, 11, "Raw data must have exactly 11 whitespace values in TotalCharges.")

    def test_cleaned_dataset_properties(self):
        """Verify that cleaning fixes types, handles nulls, and preserves records."""
        self.assertEqual(len(self.df_clean), 7043, "Cleaning must preserve all 7043 customer records.")
        self.assertEqual(self.df_clean["customerID"].nunique(), 7043, "Primary key customerID must remain unique.")
        self.assertTrue(pd.api.types.is_numeric_dtype(self.df_clean["TotalCharges"]), "TotalCharges must be numeric.")
        self.assertEqual(self.df_clean["TotalCharges"].isnull().sum(), 0, "No missing values allowed in TotalCharges.")
        
        # Test tenure == 0 customers have TotalCharges == 0.0
        tenure_zero = self.df_clean[self.df_clean["tenure"] == 0]
        self.assertEqual(len(tenure_zero), 11, "There must be exactly 11 customers with tenure == 0.")
        self.assertTrue((tenure_zero["TotalCharges"] == 0.0).all(), "Tenure 0 customers must have TotalCharges 0.0.")

    def test_target_variable_distribution(self):
        """Verify target column encoding and class balance."""
        self.assertTrue(set(self.df_clean["Churn"].unique()).issubset({0, 1}), "Churn must be binary {0, 1}.")
        val_counts = self.df_clean["Churn"].value_counts().to_dict()
        self.assertEqual(val_counts[0], 5174, "Non-churn count must be exactly 5174.")
        self.assertEqual(val_counts[1], 1869, "Churn count must be exactly 1869.")
        churn_rate = self.df_clean["Churn"].mean() * 100
        self.assertAlmostEqual(churn_rate, 26.54, places=2, msg="Churn rate must match baseline 26.54%.")

    def test_duplicate_records_not_deleted(self):
        """Verify duplicate feature records represent real distinct customers."""
        # 22 duplicate feature rows (excluding customerID)
        feature_dups = self.df_clean.drop(columns=["customerID"]).duplicated().sum()
        self.assertEqual(feature_dups, 22, "Expected 22 feature duplicate records with TotalCharges present.")
        
        # When customerID is included, duplicates must be 0
        self.assertEqual(self.df_clean.duplicated().sum(), 0, "Zero duplicates when customerID is included.")

    def test_feature_engineering_integrity(self):
        """Verify engineered features are non-leaking and correctly structured."""
        self.assertEqual(len(self.df_featured), 7043, "Featured dataset must preserve 7043 rows.")
        
        expected_cols = [
            "tenure_group", "total_services", "has_internet", "is_fiber_optic",
            "security_services_count", "streaming_services_count",
            "is_automatic_payment", "is_long_term_contract", "avg_monthly_charges"
        ]
        for col in expected_cols:
            self.assertIn(col, self.df_featured.columns, f"Engineered column {col} must exist.")
            self.assertEqual(self.df_featured[col].isnull().sum(), 0, f"Column {col} must have 0 nulls.")

    def test_validation_function(self):
        """Test validation summary dictionary output."""
        res = validate_cleaned_dataset(self.df_clean)
        self.assertTrue(res["is_record_count_valid"])
        self.assertTrue(res["is_customer_id_unique"])
        self.assertEqual(res["totalcharges_null_count"], 0)
        self.assertFalse(res["has_infinite_values"])


if __name__ == "__main__":
    unittest.main()
