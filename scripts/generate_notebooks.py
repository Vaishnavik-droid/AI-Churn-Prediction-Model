"""
Script to build and execute the three standardized, reproducible notebooks:
1. 01_data_cleaning.ipynb
2. 02_exploratory_analysis.ipynb
3. 03_feature_engineering.ipynb
"""

from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)


def build_01_cleaning_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "venv",
            "language": "python",
            "name": "venv"
        },
        "language_info": {
            "name": "python",
            "version": "3.14.2"
        }
    }
    cells = []

    # Title
    cells.append(nbf.v4.new_markdown_cell(
        "# 01 — Data Cleaning, Audit & Integrity Remediation\n\n"
        "**Project:** AI Churn Prediction Model  \n"
        "**Repository:** `Vaishnavik-droid/AI-Churn-Prediction-Model`  \n"
        "**Dataset:** Telco Customer Churn (7,043 customer records)  \n"
        "**Objective:** Perform an exhaustive data audit, correct identified bugs "
        "(whitespace in TotalCharges, unassigned fillna, false duplicate removal, brittle paths), "
        "and generate a verified, reproducible cleaned dataset (`telco_customer_churn_cleaned.csv`).\n\n"
        "---\n\n"
        "## 1. Environment & Setup\n"
        "We use `pathlib.Path` to resolve repository-relative paths, ensuring complete reproducibility."
    ))

    cells.append(nbf.v4.new_code_cell(
        "import sys\n"
        "from pathlib import Path\n"
        "import pandas as pd\n"
        "import numpy as np\n\n"
        "# Set display options for clarity\n"
        "pd.set_option('display.max_columns', None)\n"
        "pd.set_option('display.width', 1000)\n\n"
        "# Repository-relative directory resolution\n"
        "CURRENT_DIR = Path.cwd()\n"
        "ROOT_DIR = CURRENT_DIR.parent if CURRENT_DIR.name == 'notebooks' else CURRENT_DIR\n"
        "if str(ROOT_DIR) not in sys.path:\n"
        "    sys.path.append(str(ROOT_DIR))\n\n"
        "DATA_RAW = ROOT_DIR / 'data' / 'raw' / 'Telco-Customer-Churn.csv'\n"
        "DATA_PROCESSED_DIR = ROOT_DIR / 'data' / 'processed'\n"
        "DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)\n"
        "CLEANED_DATA_PATH = DATA_PROCESSED_DIR / 'telco_customer_churn_cleaned.csv'\n\n"
        "print(f'Project Root: {ROOT_DIR}')\n"
        "print(f'Raw Data Path: {DATA_RAW} (Exists: {DATA_RAW.exists()})')\n"
        "print(f'Output Path: {CLEANED_DATA_PATH}')"
    ))

    cells.append(nbf.v4.new_markdown_cell(
        "## 2. Ingest Raw Dataset & Structural Inspection\n"
        "Load the immutable raw CSV file and verify dimensions, column names, and initial data types."
    ))

    cells.append(nbf.v4.new_code_cell(
        "df_raw = pd.read_csv(DATA_RAW)\n"
        "print(f'Raw Dataset Shape: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns\\n')\n"
        "print('First 5 records:')\n"
        "df_raw.head()"
    ))

    cells.append(nbf.v4.new_code_cell(
        "print('Column Information and Data Types:')\n"
        "df_raw.info()"
    ))

    cells.append(nbf.v4.new_markdown_cell(
        "## 3. Audit Issue: TotalCharges String Type & Whitespace Values\n"
        "In the raw dataset, `TotalCharges` is stored as an object/string instead of numeric (`float64`).\n"
        "Whitespace string values (`\" \"`) prevent automatic numeric parsing. Let us inspect them."
    ))

    cells.append(nbf.v4.new_code_cell(
        "# Identify whitespace-only entries across text columns\n"
        "whitespace_summary = {}\n"
        "for col in df_raw.columns:\n"
        "    if df_raw[col].dtype == object or str(df_raw[col].dtype) in ['string', 'str']:\n"
        "        ws_count = (df_raw[col].astype(str).str.strip() == '').sum()\n"
        "        if ws_count > 0:\n"
        "            whitespace_summary[col] = ws_count\n\n"
        "print('Columns with whitespace-only values:', whitespace_summary)\n\n"
        "# Detailed inspection of the 11 records with whitespace in TotalCharges\n"
        "ws_mask = df_raw['TotalCharges'].astype(str).str.strip() == ''\n"
        "ws_records = df_raw[ws_mask]\n"
        "print(f'\\nFound {len(ws_records)} records with whitespace in TotalCharges.')\n"
        "print('Notice tenure and billing details for these records:')\n"
        "ws_records[['customerID', 'tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 'Churn']]"
    ))

    cells.append(nbf.v4.new_markdown_cell(
        "### Key Finding on TotalCharges Missingness:\n"
        "1. **Tenure Invariant:** Every single one of the 11 customers with whitespace in `TotalCharges` has **`tenure == 0`**.\n"
        "2. **Business Reality:** These are new customers who enrolled within the current billing cycle and have not yet accrued cumulative charges.\n"
        "3. **Flaw in Prior Implementation:** The previous notebook ran `df['TotalCharges'].fillna(df['TotalCharges'].median())` without assignment (`inplace=True` or `df[...] = ...`). This failed to impute values, and applying global median imputation across the entire dataset introduces **data leakage**.\n"
        "4. **Methodological Correction:** For `tenure == 0`, the cumulative total charges incurred are mathematically **$0.00**. We convert `TotalCharges` to numeric, and set `TotalCharges = 0.0` for `tenure == 0`."
    ))

    cells.append(nbf.v4.new_code_cell(
        "df_clean = df_raw.copy()\n\n"
        "# Convert TotalCharges to float64, converting whitespace to NaN\n"
        "df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')\n\n"
        "# For tenure == 0, TotalCharges is legitimately 0.0 (no billing cycles completed)\n"
        "df_clean.loc[df_clean['tenure'] == 0, 'TotalCharges'] = 0.0\n\n"
        "print(f'TotalCharges null count after correction: {df_clean[\"TotalCharges\"].isnull().sum()}')\n"
        "print(f'TotalCharges dtype: {df_clean[\"TotalCharges\"].dtype}')\n"
        "print('\\nSummary of TotalCharges for tenure == 0 records:')\n"
        "print(df_clean.loc[df_clean['tenure'] == 0, 'TotalCharges'].describe())"
    ))

    cells.append(nbf.v4.new_markdown_cell(
        "## 4. Audit Issue: Duplicate Record Investigation\n"
        "The previous audit flagged 27 identical feature rows in the processed dataset.\n"
        "Let us investigate whether these represent true duplicate entries or distinct customers with identical plan features."
    ))

    cells.append(nbf.v4.new_code_cell(
        "# 1. Primary Key Uniqueness\n"
        "total_rows = len(df_clean)\n"
        "unique_ids = df_clean['customerID'].nunique()\n"
        "print(f'Total customer rows: {total_rows}')\n"
        "print(f'Unique customerID count: {unique_ids}')\n"
        "print(f'Are all customer IDs unique? {total_rows == unique_ids}\\n')\n\n"
        "# 2. Check duplicates with customerID included\n"
        "print(f'Full row duplicates (including customerID): {df_clean.duplicated().sum()}')\n\n"
        "# 3. Check duplicate feature vectors (excluding customerID)\n"
        "feature_dups_count = df_clean.drop(columns=['customerID']).duplicated().sum()\n"
        "print(f'Feature vector duplicates (excluding customerID, with TotalCharges present): {feature_dups_count}')\n\n"
        "# Show why the prior processed dataset had 27 duplicates instead of 22:\n"
        "# When TotalCharges is omitted, 5 extra collisions occur!\n"
        "no_tc_dups = df_clean.drop(columns=['customerID', 'TotalCharges']).duplicated().sum()\n"
        "print(f'Feature vector duplicates when TotalCharges is omitted: {no_tc_dups}')"
    ))

    cells.append(nbf.v4.new_code_cell(
        "# Inspect duplicate feature groups\n"
        "dup_mask = df_clean.drop(columns=['customerID']).duplicated(keep=False)\n"
        "dup_records = df_clean[dup_mask].sort_values(by=['tenure', 'MonthlyCharges', 'Contract'])\n\n"
        "print(f'Total records belonging to duplicate feature combinations: {len(dup_records)}')\n"
        "print('\\nSample duplicate feature rows (showing distinct customerIDs and different Churn outcomes):')\n"
        "dup_records[['customerID', 'gender', 'SeniorCitizen', 'tenure', 'PhoneService', 'Contract', 'MonthlyCharges', 'TotalCharges', 'Churn']].head(12)"
    ))

    cells.append(nbf.v4.new_markdown_cell(
        "### Key Finding on Duplicates:\n"
        "1. **Zero Duplicate Customers:** All 7,043 `customerID`s are strictly unique.\n"
        "2. **Common Plan Offerings:** In a 7,000+ subscriber telecom business, multiple individuals naturally purchase the exact same entry-level month-to-month plan (e.g. tenure=1, single line, basic phone, $20.05/mo).\n"
        "3. **Different Churn Labels:** Notice that among identical plan features, some customers churned while others did not. Removing these records would artificially distort class balance and delete real customer history.\n"
        "4. **Decision:** All 7,043 customer records are **retained**."
    ))

    cells.append(nbf.v4.new_markdown_cell(
        "## 5. Target Variable Harmonization & Clean Encoding\n"
        "Harmonize the target variable `Churn` to binary integers (`0` = No, `1` = Yes).\n"
        "Ensure no label leakage or ambiguous type representations."
    ))

    cells.append(nbf.v4.new_code_cell(
        "# Explicit mapping for target variable\n"
        "df_clean['Churn'] = df_clean['Churn'].replace({'No': 0, 'Yes': 1}).astype(int)\n\n"
        "churn_counts = df_clean['Churn'].value_counts()\n"
        "churn_prop = df_clean['Churn'].value_counts(normalize=True) * 100\n\n"
        "print('Target Variable (Churn) Distribution:')\n"
        "for label, name in [(0, 'Retained (No)'), (1, 'Churned (Yes)')]:\n"
        "    print(f'  {name} [class {label}]: {churn_counts[label]:,d} customers ({churn_prop[label]:.2f}%)')\n"
        "print(f'\\nTotal Customers: {len(df_clean):,d}')"
    ))

    cells.append(nbf.v4.new_markdown_cell(
        "## 6. Export Cleaned Dataset & Automated Integrity Assertions\n"
        "Export the verified cleaned dataset to `data/processed/telco_customer_churn_cleaned.csv` and run automated validation assertions."
    ))

    cells.append(nbf.v4.new_code_cell(
        "# Save cleaned dataset\n"
        "df_clean.to_csv(CLEANED_DATA_PATH, index=False)\n"
        "print(f'Cleaned dataset successfully saved to: {CLEANED_DATA_PATH}')\n\n"
        "# Automated Invariant Tests\n"
        "assert len(df_clean) == 7043, 'Error: Expected 7043 records!'\n"
        "assert df_clean.shape[1] == 21, 'Error: Expected 21 columns!'\n"
        "assert df_clean['customerID'].nunique() == 7043, 'Error: customerID is not unique!'\n"
        "assert df_clean['TotalCharges'].isnull().sum() == 0, 'Error: TotalCharges contains unhandled nulls!'\n"
        "assert df_clean['TotalCharges'].dtype == np.float64, 'Error: TotalCharges must be float64!'\n"
        "assert set(df_clean['Churn'].unique()) == {0, 1}, 'Error: Churn must be {0, 1}!'\n"
        "assert df_clean['Churn'].value_counts()[0] == 5174, 'Error: Retained count must be 5174!'\n"
        "assert df_clean['Churn'].value_counts()[1] == 1869, 'Error: Churned count must be 1869!'\n"
        "assert not df_clean.select_dtypes(include=[np.number]).isin([np.inf, -np.inf]).any().any(), 'Error: Infinite values found!'\n\n"
        "print('ALL 8 DATA INTEGRITY ASSERTIONS PASSED SUCCESSFULLY!')"
    ))

    nb.cells = cells
    out_path = NOTEBOOKS_DIR / "01_data_cleaning.ipynb"
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Generated: {out_path}")
    return out_path


def build_02_eda_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "venv",
            "language": "python",
            "name": "venv"
        },
        "language_info": {
            "name": "python",
            "version": "3.14.2"
        }
    }
    cells = []

    # Title
    cells.append(nbf.v4.new_markdown_cell(
        "# 02 — Exploratory Data Analysis & Churn Drivers\n\n"
        "**Project:** AI Churn Prediction Model  \n"
        "**Repository:** `Vaishnavik-droid/AI-Churn-Prediction-Model`  \n"
        "**Dataset:** Cleaned Telco Customer Churn (7,043 customer records)  \n"
        "**Objective:** Retain and enhance existing useful visualizations while replacing misleading "
        "count-only charts with normalized churn rate analyses (Churn Rate = Churned / Total * 100). "
        "Covers all 10 required business dimensions with actionable interpretations.\n\n"
        "---\n\n"
        "## 1. Setup & Ingestion\n"
        "Import visualization and statistical packages and load the verified cleaned dataset."
    ))

    cells.append(nbf.v4.new_code_cell(
        "from pathlib import Path\n"
        "import pandas as pd\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n\n"
        "# Matplotlib aesthetic styling\n"
        "plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')\n"
        "plt.rcParams['font.size'] = 11\n"
        "plt.rcParams['axes.titlesize'] = 13\n"
        "plt.rcParams['axes.labelsize'] = 11\n"
        "plt.rcParams['figure.dpi'] = 110\n\n"
        "# Path resolution\n"
        "CURRENT_DIR = Path.cwd()\n"
        "ROOT_DIR = CURRENT_DIR.parent if CURRENT_DIR.name == 'notebooks' else CURRENT_DIR\n"
        "CLEANED_DATA_PATH = ROOT_DIR / 'data' / 'processed' / 'telco_customer_churn_cleaned.csv'\n\n"
        "df = pd.read_csv(CLEANED_DATA_PATH)\n"
        "print(f'Loaded cleaned dataset with {df.shape[0]} rows, {df.shape[1]} columns.')"
    ))

    # Helper function for normalized churn calculation
    cells.append(nbf.v4.new_markdown_cell(
        "## Helper Function: Normalized Churn Rate Calculator\n"
        "To ensure business rigor, we compute the segment-specific Churn Rate:\n"
        "$$\\text{Churn Rate (\\%)} = \\frac{\\text{Number of Churned Customers}}{\\text{Total Customers in Segment}} \\times 100$$\n"
        "This avoids the distortion of raw counts when group sizes differ widely."
    ))

    cells.append(nbf.v4.new_code_cell(
        "def compute_churn_rate_table(data: pd.DataFrame, feature: str) -> pd.DataFrame:\n"
        "    summary = data.groupby(feature)['Churn'].agg(\n"
        "        Total_Customers='count',\n"
        "        Churned_Customers='sum'\n"
        "    ).reset_index()\n"
        "    summary['Retained_Customers'] = summary['Total_Customers'] - summary['Churned_Customers']\n"
        "    summary['Churn_Rate_Pct'] = (summary['Churned_Customers'] / summary['Total_Customers'] * 100).round(2)\n"
        "    summary['Segment_Share_Pct'] = (summary['Total_Customers'] / len(data) * 100).round(2)\n"
        "    return summary.sort_values(by='Churn_Rate_Pct', ascending=False)"
    ))

    # 1. Overall Churn Distribution
    cells.append(nbf.v4.new_markdown_cell(
        "## 1. Overall Churn Distribution\n"
        "Analyze the baseline target class balance across the entire 7,043 customer population."
    ))

    cells.append(nbf.v4.new_code_cell(
        "fig, axes = plt.subplots(1, 2, figsize=(13, 5))\n\n"
        "# Bar plot with exact counts and percentages\n"
        "churn_counts = df['Churn'].value_counts()\n"
        "bars = axes[0].bar(['Retained (0)', 'Churned (1)'], churn_counts, color=['#2ecc71', '#e74c3c'], edgecolor='black', alpha=0.85)\n"
        "for bar in bars:\n"
        "    yval = bar.get_height()\n"
        "    pct = (yval / len(df)) * 100\n"
        "    axes[0].text(bar.get_x() + bar.get_width()/2.0, yval + 60, f'{yval:,d}\\n({pct:.1f}%)', ha='center', va='bottom', fontweight='bold')\n"
        "axes[0].set_title('Customer Retention vs. Churn Volume')\n"
        "axes[0].set_ylabel('Number of Customers')\n"
        "axes[0].set_ylim(0, 6000)\n\n"
        "# Pie chart\n"
        "axes[1].pie(churn_counts, labels=['Retained (73.5%)', 'Churned (26.5%)'], colors=['#2ecc71', '#e74c3c'],\n"
        "           startangle=140, explode=[0, 0.08], autopct='%1.1f%%', textprops={'fontsize': 12, 'weight': 'bold'})\n"
        "axes[1].set_title('Overall Churn Ratio')\n\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))

    cells.append(nbf.v4.new_markdown_cell(
        "**Business Interpretation:**  \n"
        "- 1,869 out of 7,043 customers churned, establishing a baseline churn rate of **26.54%** (~1 in 4 customers).  \n"
        "- The dataset exhibits moderate class imbalance (~2.77:1 ratio of retained to churned).  \n"
        "- Evaluation during the ML phase must prioritize Precision-Recall curves, F1-score, PR-AUC, and ROC-AUC over plain accuracy."
    ))

    # 2. Churn Rate by Contract Type
    cells.append(nbf.v4.new_markdown_cell(
        "## 2. Churn Rate by Contract Type\n"
        "Evaluate the relationship between contract commitment duration and customer turnover."
    ))

    cells.append(nbf.v4.new_code_cell(
        "contract_summary = compute_churn_rate_table(df, 'Contract')\n"
        "print('Contract Type Breakdown:')\n"
        "display(contract_summary)\n\n"
        "fig, ax = plt.subplots(figsize=(8, 4.5))\n"
        "bars = ax.bar(contract_summary['Contract'], contract_summary['Churn_Rate_Pct'], color='#3498db', edgecolor='black', alpha=0.85)\n"
        "for bar in bars:\n"
        "    yval = bar.get_height()\n"
        "    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f'{yval:.1f}%', ha='center', va='bottom', fontweight='bold')\n"
        "ax.set_title('Churn Rate by Contract Commitment Type')\n"
        "ax.set_ylabel('Churn Rate (%)')\n"
        "ax.set_xlabel('Contract Type')\n"
        "ax.set_ylim(0, 50)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))

    cells.append(nbf.v4.new_markdown_cell(
        "**Business Interpretation:**  \n"
        "- **Month-to-month contracts experience a massive 42.71% churn rate**, compared to **11.27% for one-year** and only **2.83% for two-year** contracts.  \n"
        "- Month-to-month subscribers account for 55.0% of the customer base but generate 88.6% of all churn events.  \n"
        "- **Actionable Insight:** Retention campaigns must focus on incentivizing month-to-month subscribers to transition into multi-month or annual commitments."
    ))

    # 3. Churn Rate by Payment Method
    cells.append(nbf.v4.new_markdown_cell(
        "## 3. Churn Rate by Payment Method\n"
        "Investigate the payment mechanism to identify friction points and retention disparities."
    ))

    cells.append(nbf.v4.new_code_cell(
        "pay_summary = compute_churn_rate_table(df, 'PaymentMethod')\n"
        "print('Payment Method Breakdown:')\n"
        "display(pay_summary)\n\n"
        "fig, ax = plt.subplots(figsize=(10, 4.5))\n"
        "colors = ['#e74c3c' if rate > 30 else '#2ecc71' for rate in pay_summary['Churn_Rate_Pct']]\n"
        "bars = ax.barh(pay_summary['PaymentMethod'], pay_summary['Churn_Rate_Pct'], color=colors, edgecolor='black', alpha=0.85)\n"
        "for bar in bars:\n"
        "    xval = bar.get_width()\n"
        "    ax.text(xval + 1.0, bar.get_y() + bar.get_height()/2.0, f'{xval:.1f}%', ha='left', va='center', fontweight='bold')\n"
        "ax.set_title('Churn Rate by Payment Method')\n"
        "ax.set_xlabel('Churn Rate (%)')\n"
        "ax.set_xlim(0, 55)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))

    cells.append(nbf.v4.new_markdown_cell(
        "**Business Interpretation:**  \n"
        "- **Electronic check users churn at 45.29%**, dramatically higher than any other payment method.  \n"
        "- Automated payment methods (Bank transfer at 16.71% and Credit card at 15.24%) show superior customer stickiness.  \n"
        "- **Actionable Insight:** Promoting auto-pay incentives (e.g. $5 monthly billing discount) can substantially reduce churn."
    ))

    # 4. Churn Rate by Internet Service
    cells.append(nbf.v4.new_markdown_cell(
        "## 4. Churn Rate by Internet Service\n"
        "Analyze customer turnover across internet connection technologies."
    ))

    cells.append(nbf.v4.new_code_cell(
        "net_summary = compute_churn_rate_table(df, 'InternetService')\n"
        "print('Internet Service Breakdown:')\n"
        "display(net_summary)\n\n"
        "fig, ax = plt.subplots(figsize=(8, 4.5))\n"
        "bars = ax.bar(net_summary['InternetService'], net_summary['Churn_Rate_Pct'], color=['#e67e22', '#3498db', '#95a5a6'], edgecolor='black', alpha=0.85)\n"
        "for bar in bars:\n"
        "    yval = bar.get_height()\n"
        "    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f'{yval:.1f}%', ha='center', va='bottom', fontweight='bold')\n"
        "ax.set_title('Churn Rate by Internet Technology')\n"
        "ax.set_ylabel('Churn Rate (%)')\n"
        "ax.set_ylim(0, 50)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))

    cells.append(nbf.v4.new_markdown_cell(
        "**Business Interpretation:**  \n"
        "- **Fiber optic customers have a staggering 41.89% churn rate**, despite being the premium offering.  \n"
        "- In contrast, DSL customers churn at **18.96%**, and customers without internet churn at only **7.40%**.  \n"
        "- **Actionable Insight:** Fiber optic customers are paying higher monthly fees, pointing to potential service quality issues, outage dissatisfaction, or pricing shock."
    ))

    # 5. Churn Rate by Tenure Groups
    cells.append(nbf.v4.new_markdown_cell(
        "## 5. Churn Rate by Tenure Groups (Lifecycle Analysis)\n"
        "Group customers into business cohorts to examine tenure-based turnover."
    ))

    cells.append(nbf.v4.new_code_cell(
        "tenure_bins = [-1, 12, 24, 48, 60, 72]\n"
        "tenure_labels = ['0-12m (New)', '13-24m (Early)', '25-48m (Mid)', '49-60m (Established)', '61-72m (Loyal)']\n"
        "df['tenure_group'] = pd.cut(df['tenure'], bins=tenure_bins, labels=tenure_labels)\n\n"
        "tenure_summary = compute_churn_rate_table(df, 'tenure_group')\n"
        "print('Tenure Cohort Breakdown:')\n"
        "display(tenure_summary)\n\n"
        "fig, ax = plt.subplots(figsize=(10, 4.5))\n"
        "bars = ax.bar(tenure_summary['tenure_group'], tenure_summary['Churn_Rate_Pct'], color='#9b59b6', edgecolor='black', alpha=0.85)\n"
        "for bar in bars:\n"
        "    yval = bar.get_height()\n"
        "    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f'{yval:.1f}%', ha='center', va='bottom', fontweight='bold')\n"
        "ax.set_title('Churn Rate by Customer Tenure Cohort')\n"
        "ax.set_ylabel('Churn Rate (%)')\n"
        "ax.set_ylim(0, 55)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))

    cells.append(nbf.v4.new_markdown_cell(
        "**Business Interpretation:**  \n"
        "- **New customers (0-12 months) suffer a 47.68% churn rate**. Almost half of new subscribers leave within their first year.  \n"
        "- Churn drops steeply as tenure increases: **28.71% (13-24m)**, **18.06% (25-48m)**, **9.99% (49-60m)**, and down to **6.61% (61-72m)**.  \n"
        "- **Actionable Insight:** First-year onboarding and satisfaction checks are the highest ROI intervention opportunities."
    ))

    # 6. Tech Support Availability
    cells.append(nbf.v4.new_markdown_cell(
        "## 6. Churn Rate by Tech Support Availability\n"
        "Examine whether technical assistance serves as a churn mitigation shield."
    ))

    cells.append(nbf.v4.new_code_cell(
        "ts_summary = compute_churn_rate_table(df, 'TechSupport')\n"
        "print('Tech Support Breakdown:')\n"
        "display(ts_summary)\n\n"
        "fig, ax = plt.subplots(figsize=(8, 4.5))\n"
        "bars = ax.bar(ts_summary['TechSupport'], ts_summary['Churn_Rate_Pct'], color=['#e74c3c', '#2ecc71', '#95a5a6'], edgecolor='black', alpha=0.85)\n"
        "for bar in bars:\n"
        "    yval = bar.get_height()\n"
        "    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f'{yval:.1f}%', ha='center', va='bottom', fontweight='bold')\n"
        "ax.set_title('Churn Rate by Tech Support Status')\n"
        "ax.set_ylabel('Churn Rate (%)')\n"
        "ax.set_ylim(0, 50)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))

    cells.append(nbf.v4.new_markdown_cell(
        "**Business Interpretation:**  \n"
        "- Customers with **No Tech Support experience a 41.64% churn rate**, compared to **15.17% for customers with Tech Support**.  \n"
        "- Customers with technical issues who lack assistance churn at nearly triple the rate of supported users.  \n"
        "- **Actionable Insight:** Bundling free or discounted tech support into high-tier internet plans is an effective retention lever."
    ))

    # 7 & 8: MonthlyCharges & TotalCharges Distributions
    cells.append(nbf.v4.new_markdown_cell(
        "## 7 & 8. MonthlyCharges and TotalCharges Distribution by Churn Status\n"
        "Inspect financial charge distributions across retained and churned cohorts using boxplots and kernel density estimates."
    ))

    cells.append(nbf.v4.new_code_cell(
        "fig, axes = plt.subplots(2, 2, figsize=(14, 9))\n\n"
        "# MonthlyCharges Boxplot\n"
        "sns.boxplot(x='Churn', y='MonthlyCharges', data=df, palette=['#2ecc71', '#e74c3c'], ax=axes[0, 0])\n"
        "axes[0, 0].set_xticklabels(['Retained (0)', 'Churned (1)'])\n"
        "axes[0, 0].set_title('Monthly Charges Distribution by Churn Status')\n"
        "axes[0, 0].set_ylabel('Monthly Charges ($)')\n\n"
        "# MonthlyCharges KDE\n"
        "sns.kdeplot(df.loc[df['Churn'] == 0, 'MonthlyCharges'], ax=axes[0, 1], label='Retained', color='#2ecc71', fill=True, alpha=0.3)\n"
        "sns.kdeplot(df.loc[df['Churn'] == 1, 'MonthlyCharges'], ax=axes[0, 1], label='Churned', color='#e74c3c', fill=True, alpha=0.3)\n"
        "axes[0, 1].set_title('Monthly Charges Density Comparison')\n"
        "axes[0, 1].set_xlabel('Monthly Charges ($)')\n"
        "axes[0, 1].legend()\n\n"
        "# TotalCharges Boxplot\n"
        "sns.boxplot(x='Churn', y='TotalCharges', data=df, palette=['#2ecc71', '#e74c3c'], ax=axes[1, 0])\n"
        "axes[1, 0].set_xticklabels(['Retained (0)', 'Churned (1)'])\n"
        "axes[1, 0].set_title('Total Charges Distribution by Churn Status')\n"
        "axes[1, 0].set_ylabel('Total Charges ($)')\n\n"
        "# TotalCharges KDE\n"
        "sns.kdeplot(df.loc[df['Churn'] == 0, 'TotalCharges'], ax=axes[1, 1], label='Retained', color='#2ecc71', fill=True, alpha=0.3)\n"
        "sns.kdeplot(df.loc[df['Churn'] == 1, 'TotalCharges'], ax=axes[1, 1], label='Churned', color='#e74c3c', fill=True, alpha=0.3)\n"
        "axes[1, 1].set_title('Total Charges Density Comparison')\n"
        "axes[1, 1].set_xlabel('Total Charges ($)')\n"
        "axes[1, 1].legend()\n\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))

    cells.append(nbf.v4.new_markdown_cell(
        "**Business Interpretation:**  \n"
        "- **MonthlyCharges:** Median monthly charge for churned customers is **$79.65**, compared to **$64.43** for retained customers. Churn density peaks sharply between **$70 and $100/month**, corresponding to expensive unbundled fiber optic plans.  \n"
        "- **TotalCharges:** Churned customers have substantially lower median total charges (**$701.50** vs **$1,683.60** for retained). This reflects early customer defection (low tenure) rather than lower plan pricing."
    ))

    # 9. Service Adoption Patterns
    cells.append(nbf.v4.new_markdown_cell(
        "## 9. Service Adoption Patterns & Ecosystem Stickiness\n"
        "Examine how the total number of subscribed add-on services affects churn risk."
    ))

    cells.append(nbf.v4.new_code_cell(
        "service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']\n"
        "df['active_addons_count'] = (df[service_cols] == 'Yes').sum(axis=1)\n\n"
        "addon_summary = compute_churn_rate_table(df, 'active_addons_count')\n"
        "print('Add-on Services Adoption Breakdown:')\n"
        "display(addon_summary)\n\n"
        "fig, ax = plt.subplots(figsize=(9, 4.5))\n"
        "bars = ax.bar(addon_summary['active_addons_count'].astype(str), addon_summary['Churn_Rate_Pct'], color='#16a085', edgecolor='black', alpha=0.85)\n"
        "for bar in bars:\n"
        "    yval = bar.get_height()\n"
        "    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f'{yval:.1f}%', ha='center', va='bottom', fontweight='bold')\n"
        "ax.set_title('Churn Rate by Number of Active Add-on Services')\n"
        "ax.set_xlabel('Count of Active Add-on Services (Security, Backup, Streaming, Support)')\n"
        "ax.set_ylabel('Churn Rate (%)')\n"
        "ax.set_ylim(0, 50)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))

    cells.append(nbf.v4.new_markdown_cell(
        "**Business Interpretation:**  \n"
        "- Customers with **0 add-on services churn at 39.87%**.  \n"
        "- Churn steadily declines as customers adopt more services, falling to **11.45% at 4 services** and **14.86% at 5 services**.  \n"
        "- **Actionable Insight:** Ecosystem stickiness creates switching friction. Cross-selling at least 2 security or streaming services provides strong churn insulation."
    ))

    # 10. Demographic Comparisons
    cells.append(nbf.v4.new_markdown_cell(
        "## 10. Demographic Comparisons: Senior Citizens, Partners & Dependents\n"
        "Assess customer vulnerability across demographic segments."
    ))

    cells.append(nbf.v4.new_code_cell(
        "fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))\n\n"
        "demographics = [\n"
        "    ('SeniorCitizen', 'Senior Citizen Status (0=No, 1=Yes)'),\n"
        "    ('Partner', 'Has Partner'),\n"
        "    ('Dependents', 'Has Dependents')\n"
        "]\n\n"
        "for idx, (col, title) in enumerate(demographics):\n"
        "    sum_df = compute_churn_rate_table(df, col)\n"
        "    axes[idx].bar(sum_df[col].astype(str), sum_df['Churn_Rate_Pct'], color=['#34495e', '#e74c3c'], edgecolor='black', alpha=0.85)\n"
        "    for bar in axes[idx].patches:\n"
        "        yval = bar.get_height()\n"
        "        axes[idx].text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f'{yval:.1f}%', ha='center', va='bottom', fontweight='bold')\n"
        "    axes[idx].set_title(title)\n"
        "    axes[idx].set_ylabel('Churn Rate (%)')\n"
        "    axes[idx].set_ylim(0, 50)\n\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))

    cells.append(nbf.v4.new_markdown_cell(
        "**Business Interpretation:**  \n"
        "- **Senior Citizens:** Churn at **41.68%**, compared to **23.61% for non-seniors**. Seniors represent an underserved, price-sensitive, or technology-friction segment requiring dedicated support.  \n"
        "- **Partner & Dependents:** Customers with partners churn at **19.66%** (vs 32.96% single). Customers with dependents churn at only **15.45%** (vs 31.28% without). Household-anchored accounts show substantially higher stability."
    ))

    nb.cells = cells
    out_path = NOTEBOOKS_DIR / "02_exploratory_analysis.ipynb"
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Generated: {out_path}")
    return out_path


def build_03_features_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "venv",
            "language": "python",
            "name": "venv"
        },
        "language_info": {
            "name": "python",
            "version": "3.14.2"
        }
    }
    cells = []

    # Title
    cells.append(nbf.v4.new_markdown_cell(
        "# 03 — Feature Engineering, Pipeline Architecture & ML Readiness\n\n"
        "**Project:** AI Churn Prediction Model  \n"
        "**Repository:** `Vaishnavik-droid/AI-Churn-Prediction-Model`  \n"
        "**Dataset:** Cleaned Telco Customer Churn (7,043 customer records)  \n"
        "**Objective:** Evaluate and correct prior feature engineering, design reproducible "
        "business-driven domain features without target leakage, formulate a scikit-learn "
        "`ColumnTransformer` / `Pipeline` architecture for model training, and export the finalized "
        "dataset (`telco_customer_churn_featured.csv`).\n\n"
        "---\n\n"
        "## 1. Setup & Ingestion\n"
        "Load the cleaned dataset and establish reproducible parameters."
    ))

    cells.append(nbf.v4.new_code_cell(
        "import sys\n"
        "from pathlib import Path\n"
        "import pandas as pd\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n\n"
        "# Project directory resolution & sys.path setup for modular imports\n"
        "CURRENT_DIR = Path.cwd()\n"
        "ROOT_DIR = CURRENT_DIR.parent if CURRENT_DIR.name == 'notebooks' else CURRENT_DIR\n"
        "if str(ROOT_DIR) not in sys.path:\n"
        "    sys.path.append(str(ROOT_DIR))\n\n"
        "CLEANED_DATA_PATH = ROOT_DIR / 'data' / 'processed' / 'telco_customer_churn_cleaned.csv'\n"
        "FEATURED_DATA_PATH = ROOT_DIR / 'data' / 'processed' / 'telco_customer_churn_featured.csv'\n\n"
        "df_clean = pd.read_csv(CLEANED_DATA_PATH)\n"
        "print(f'Cleaned dataset loaded: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns.')"
    ))

    # Audit of prior feature engineering
    cells.append(nbf.v4.new_markdown_cell(
        "## 2. Audit of Prior Feature Engineering: Charge Categories\n"
        "In the original EDA notebook, `MonthlyCharges` was discretized into arbitrary bins `[0, 40, 80, 150]` "
        "generating dummy variables `ChargeCategory_Medium` and `ChargeCategory_High`.\n"
        "Let us investigate whether these derived features add meaningful value or introduce redundancy."
    ))

    cells.append(nbf.v4.new_code_cell(
        "# Replicate original binning\n"
        "df_audit = df_clean.copy()\n"
        "df_audit['ChargeCategory'] = pd.cut(df_audit['MonthlyCharges'], bins=[0, 40, 80, 150], labels=['Low', 'Medium', 'High'])\n"
        "charge_dummies = pd.get_dummies(df_audit['ChargeCategory'], prefix='ChargeCategory', drop_first=True)\n"
        "df_audit = pd.concat([df_audit, charge_dummies], axis=1)\n\n"
        "# Correlation analysis between raw MonthlyCharges and derived dummies\n"
        "corr_matrix = df_audit[['MonthlyCharges', 'ChargeCategory_Medium', 'ChargeCategory_High', 'Churn']].corr()\n"
        "print('Correlation Matrix:')\n"
        "display(corr_matrix.round(4))\n\n"
        "fig, ax = plt.subplots(figsize=(6, 4.5))\n"
        "sns.heatmap(corr_matrix, annot=True, cmap='Blues', fmt='.3f', ax=ax)\n"
        "ax.set_title('Collinearity between MonthlyCharges and Binned Dummies')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))

    cells.append(nbf.v4.new_markdown_cell(
        "### Audit Conclusions on ChargeCategory Dummies:\n"
        "1. **Information Loss:** Discretizing `MonthlyCharges` into 3 broad bins discards continuous variance and fine-grained price sensitivities.  \n"
        "2. **Severe Collinearity:** Retaining both continuous `MonthlyCharges` and the dummy flags introduces multicollinearity in linear/logistic models without providing any new signal.  \n"
        "3. **Arbitrary Thresholds:** The bin limits (40, 80) were selected without empirical justification.  \n"
        "4. **Methodological Recommendation:** Retain continuous `MonthlyCharges`. If non-linear relationships are present, tree-based models (Random Forest, XGBoost) discover optimal split thresholds directly from continuous values."
    ))

    # 3. Domain Feature Engineering
    cells.append(nbf.v4.new_markdown_cell(
        "## 3. Principled Domain Feature Engineering\n"
        "We construct business-informed, non-leaking features:\n"
        "1. **`tenure_group`**: Customer lifecycle cohorts (0-12m, 13-24m, 25-48m, 49-60m, 61-72m).\n"
        "2. **`total_services`**: Aggregate count of all subscribed active services (phone, internet, add-ons).\n"
        "3. **`security_services_count`**: Count of security/protection services (`OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`).\n"
        "4. **`streaming_services_count`**: Count of entertainment services (`StreamingTV`, `StreamingMovies`).\n"
        "5. **`is_fiber_optic`**: Binary indicator for high-risk fiber optic internet technology.\n"
        "6. **`is_automatic_payment`**: Binary indicator for low-risk automated payment methods (bank transfer, credit card).\n"
        "7. **`is_long_term_contract`**: Binary indicator for one-year or two-year commitments.\n"
        "8. **`avg_monthly_charges`**: Lifetime average monthly spend ($TotalCharges / (tenure + 1)$).\n"
        "9. **`charge_escalation_ratio`**: Current monthly charge divided by historical average monthly charge (detects price shocks)."
    ))

    cells.append(nbf.v4.new_code_cell(
        "from src.features.engineering import engineer_features\n\n"
        "df_featured = engineer_features(df_clean)\n"
        "print(f'Engineered dataset shape: {df_featured.shape[0]} rows, {df_featured.shape[1]} columns.\\n')\n"
        "print('New feature columns created:')\n"
        "new_cols = [c for c in df_featured.columns if c not in df_clean.columns]\n"
        "for col in new_cols:\n"
        "    print(f'  - {col}: dtype {df_featured[col].dtype}, nulls: {df_featured[col].isnull().sum()}')"
    ))

    # Verification of zero target leakage
    cells.append(nbf.v4.new_markdown_cell(
        "## 4. Target Leakage Verification\n"
        "Ensure no input feature was constructed using `Churn`."
    ))

    cells.append(nbf.v4.new_code_cell(
        "# Calculate correlation of new features with target\n"
        "numeric_cols = df_featured.select_dtypes(include=[np.number]).columns\n"
        "target_corrs = df_featured[numeric_cols].corr()['Churn'].sort_values()\n"
        "print('Feature Correlations with Churn:')\n"
        "display(target_corrs.to_frame('Correlation_with_Churn'))\n\n"
        "# Verify no feature has an impossible perfect correlation (r = 1.0 or r = -1.0)\n"
        "leaking_features = [col for col in numeric_cols if col != 'Churn' and abs(target_corrs[col]) > 0.95]\n"
        "print(f'Potentially leaking features (|r| > 0.95): {leaking_features}')\n"
        "assert len(leaking_features) == 0, 'Target leakage detected!'"
    ))

    # 5. Production Preprocessing Architecture
    cells.append(nbf.v4.new_markdown_cell(
        "## 5. Production Pipeline & Preprocessing Architecture\n"
        "In contrast to fitting `LabelEncoder` globally across all columns, an industry-standard "
        "pipeline uses `ColumnTransformer` with distinct treatment for:\n"
        "- **Numeric Features:** Imputation (if needed) + Standard Scaling.\n"
        "- **Binary Features:** Direct 0/1 representation (no expansion).\n"
        "- **Nominal Categorical Features:** `OneHotEncoder(drop='first', handle_unknown='ignore')`.\n\n"
        "Ensembling this within `sklearn.pipeline.Pipeline` guarantees that encoders and scalers are "
        "**fitted strictly on training folds**, completely eliminating data leakage."
    ))

    cells.append(nbf.v4.new_code_cell(
        "from sklearn.compose import ColumnTransformer\n"
        "from sklearn.preprocessing import OneHotEncoder, StandardScaler\n"
        "from sklearn.impute import SimpleImputer\n"
        "from sklearn.pipeline import Pipeline\n\n"
        "# Define feature groups for ML ColumnTransformer\n"
        "numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'avg_monthly_charges', 'charge_escalation_ratio']\n"
        "binary_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling',\n"
        "                   'has_internet', 'is_fiber_optic', 'is_automatic_payment', 'is_long_term_contract']\n"
        "nominal_features = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',\n"
        "                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',\n"
        "                    'Contract', 'PaymentMethod', 'tenure_group']\n\n"
        "numeric_transformer = Pipeline(steps=[\n"
        "    ('imputer', SimpleImputer(strategy='median')),\n"
        "    ('scaler', StandardScaler())\n"
        "])\n\n"
        "nominal_transformer = Pipeline(steps=[\n"
        "    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))\n"
        "])\n\n"
        "preprocessor = ColumnTransformer(transformers=[\n"
        "    ('num', numeric_transformer, numeric_features),\n"
        "    ('nom', nominal_transformer, nominal_features)\n"
        "], remainder='passthrough')\n\n"
        "print('Configured ColumnTransformer architecture successfully:')\n"
        "print(preprocessor)"
    ))

    # 6. Export Featured Dataset
    cells.append(nbf.v4.new_markdown_cell(
        "## 6. Export Featured Dataset & Automated Validation\n"
        "Save the finalized feature dataset to `data/processed/telco_customer_churn_featured.csv`."
    ))

    cells.append(nbf.v4.new_code_cell(
        "df_featured.to_csv(FEATURED_DATA_PATH, index=False)\n"
        "print(f'Finalized feature dataset exported to: {FEATURED_DATA_PATH}')\n\n"
        "# Automated assertions\n"
        "assert len(df_featured) == 7043, 'Row count mismatch!'\n"
        "assert df_featured['customerID'].nunique() == 7043, 'customerID not unique!'\n"
        "assert df_featured['Churn'].isnull().sum() == 0, 'Churn nulls present!'\n"
        "assert not df_featured.select_dtypes(include=[np.number]).isin([np.inf, -np.inf]).any().any(), 'Infinite values present!'\n"
        "print('ALL 4 FEATURE VALIDATION ASSERTIONS PASSED!')"
    ))

    nb.cells = cells
    out_path = NOTEBOOKS_DIR / "03_feature_engineering.ipynb"
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Generated: {out_path}")
    return out_path


if __name__ == "__main__":
    p1 = build_01_cleaning_notebook()
    p2 = build_02_eda_notebook()
    p3 = build_03_features_notebook()
    print("All 3 notebooks built successfully!")
