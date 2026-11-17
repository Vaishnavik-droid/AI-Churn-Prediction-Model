"""
Feature engineering module for Telco Customer Churn project.
Provides business-grounded, reproducible feature extraction without target leakage.
"""

import pandas as pd
import numpy as np


def engineer_features(df_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Generate business-oriented derived features from cleaned telco data:
    1. Tenure lifecycle cohorts (tenure_group)
    2. Service adoption count (total_services)
    3. Internet presence and technology flags (has_internet, is_fiber_optic)
    4. Security bundle count (security_services_count)
    5. Streaming bundle count (streaming_services_count)
    6. Payment automation flag (is_automatic_payment)
    7. Long-term contract flag (is_long_term_contract)
    8. Historical charge interaction: avg_monthly_charges and charge_ratio
    
    All features avoid target leakage: Churn is never used to construct inputs.
    """
    df = df_clean.copy()
    
    # 1. Tenure Cohorts (Lifecycle stages)
    # Bins: [0, 12, 24, 48, 60, 72]
    tenure_bins = [-1, 12, 24, 48, 60, 72]
    tenure_labels = ["0-12m", "13-24m", "25-48m", "49-60m", "61-72m"]
    df["tenure_group"] = pd.cut(df["tenure"], bins=tenure_bins, labels=tenure_labels)
    
    # 2. Service Adoption Counts
    # Active services: PhoneService, MultipleLines, Internet, OnlineSecurity,
    # OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
    service_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    # In raw data, values can be 'Yes', 'No', 'No internet service'
    security_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
    streaming_cols = ["StreamingTV", "StreamingMovies"]
    
    # Count of security services
    df["security_services_count"] = (df[security_cols] == "Yes").sum(axis=1)
    
    # Count of streaming services
    df["streaming_services_count"] = (df[streaming_cols] == "Yes").sum(axis=1)
    
    # Total count of all active services (including phone and internet)
    phone_count = (df["PhoneService"] == "Yes").astype(int) + (df["MultipleLines"] == "Yes").astype(int)
    internet_count = (df["InternetService"].isin(["DSL", "Fiber optic"])).astype(int)
    add_on_count = (df[service_cols] == "Yes").sum(axis=1)
    df["total_services"] = phone_count + internet_count + add_on_count
    
    # 3. Internet Service Technology Flags
    df["has_internet"] = (df["InternetService"] != "No").astype(int)
    df["is_fiber_optic"] = (df["InternetService"] == "Fiber optic").astype(int)
    
    # 4. Payment Automation Flag
    # Bank transfer and Credit card are automatic; Electronic check and Mailed check are manual
    auto_payments = ["Bank transfer (automatic)", "Credit card (automatic)"]
    df["is_automatic_payment"] = df["PaymentMethod"].isin(auto_payments).astype(int)
    
    # 5. Contract Stability Flag
    df["is_long_term_contract"] = df["Contract"].isin(["One year", "Two year"]).astype(int)
    
    # 6. Charge Dynamics & Price Escalation Ratios
    # Average monthly charge over customer tenure (tenure + 1 to prevent div by 0)
    df["avg_monthly_charges"] = df["TotalCharges"] / (df["tenure"] + 1)
    
    # Ratio of current monthly charge to average monthly charge
    # Values > 1.0 indicate customer's charges have recently increased
    df["charge_escalation_ratio"] = df["MonthlyCharges"] / (df["avg_monthly_charges"] + 1e-5)
    
    return df
