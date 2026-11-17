# Telco Customer Churn: Data Preparation Audit & Remediation Report

**Repository:** `Vaishnavik-droid/AI-Churn-Prediction-Model`  
**Branch:** `dev/changes`  
**Dataset:** Telco Customer Churn (7,043 customer records)  
**Author:** Antigravity Data Science Team  
**Date:** November 2026  
**Status:** Audit Completed & Remediation Verified  

---

## Executive Summary

An exhaustive technical and methodological audit was performed on the data preparation, exploratory data analysis (EDA), and feature engineering workflows in the repository. Several critical defects and methodological flaws were confirmed, including:
1. **Silent feature omission of `TotalCharges`**, stripping one of the strongest predictive features in the dataset.
2. **Unassigned `.fillna()` bug** and improper handling of whitespace strings.
3. **Improper categorical encoding** using global `LabelEncoder` across all columns, artificially imposing ordinal hierarchies on nominal categories and introducing severe data leakage.
4. **False duplicate alarm (27 rows)**, which arose because `TotalCharges` was omitted; in reality, all 7,043 customer records possess strictly unique primary keys (`customerID`) and represent distinct subscribers.
5. **Arbitrary, redundant feature binning** (`ChargeCategory_Medium`, `ChargeCategory_High`) creating multicollinearity without predictive gain.
6. **File extension misnomers** where CSV-formatted datasets were saved with `.xls` extensions.
7. **Brittle, machine-dependent path references** leading to execution failures.

All issues have been resolved. The repository now features modular, tested data pipelines (`src/data/cleaning.py`, `src/features/engineering.py`), automated validation unit tests (`tests/test_data_validation.py`), verified clean datasets (`telco_customer_churn_cleaned.csv`, `telco_customer_churn_featured.csv`), and three clean, fully executed, reproducible Jupyter notebooks adhering to industry standards.

---

## 1. Audit Issue Log & Detailed Remediation

### ISSUE-01: Incorrect File Extensions on Processed Datasets (Confirmed)
- **Issue ID:** `ISSUE-01`
- **Location:** `data/processed/Telco-Customer-Churn_cleaned.xls`, `data/processed/add_feature_monthly charges category.xls`
- **Original Implementation:** Files were written with `.xls` extensions despite containing standard comma-separated text (RFC 4180 CSV).
- **Why Problematic:**
  - Standard spreadsheet tools (Microsoft Excel) raise security/corruption alerts when opening CSV text disguised as binary `.xls`.
  - Calling `pd.read_excel()` throws a binary format parser error.
  - Calling `pd.read_csv()` on a `.xls` file violates convention and confuses automated ingestion scripts.
  - The notebook code (`df.to_csv("...csv")`) wrote `.csv`, but git-committed files were `.xls`, causing `FileNotFoundError` during notebook execution.
- **Potential Impact on ML Pipeline:** Breaks automated data ingestion, CI/CD test runners, and multi-developer collaboration.
- **Implemented Correction:**
  - Renamed datasets via git tracking to `.csv`: `data/processed/Telco-Customer-Churn_cleaned.csv` and `data/processed/add_feature_monthly charges category.csv`.
  - Standardized new canonical datasets: `data/processed/telco_customer_churn_cleaned.csv` and `data/processed/telco_customer_churn_featured.csv`.
  - Updated all notebook and script path references to `.csv`.
- **Validation Performed:** Verified file MIME type and executed `pd.read_csv()` successfully with zero parsing errors.
- **Final Status:** **RESOLVED**

---

### ISSUE-02: TotalCharges Preprocessing & Silent Feature Omission (Confirmed)
- **Issue ID:** `ISSUE-02`
- **Location:** `notebooks/Telco_data_Jupyter lab.ipynb` (cells 14, 16) and `data/processed/Telco-Customer-Churn_cleaned.xls`
- **Original Implementation:**
  ```python
  df["TotalCharges"] = df["TotalCharges"].replace("", np.nan)
  df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
  df["TotalCharges"].fillna(df["TotalCharges"].median())  # BUG: No assignment back to df!
  ```
  Following this step, `TotalCharges` was completely omitted from the committed processed dataset (only 19 columns instead of 20).
- **Why Problematic:**
  1. The raw dataset contains whitespace `" "` (a single space), which `replace("", np.nan)` fails to match.
  2. `df["TotalCharges"].fillna(...)` was executed without `inplace=True` or assignment back to `df["TotalCharges"] = ...`. The imputed Series was displayed in cell output and immediately discarded.
  3. `TotalCharges` was missing entirely from `data/processed/Telco-Customer-Churn_cleaned.xls`, silently stripping the cumulative customer billing history.
  4. Imputing with the global median prior to train/test split introduces **data leakage**, as test set distribution information contaminates the training set.
- **Deep Investigation & Root Cause:**
  - Querying records where `TotalCharges` was whitespace revealed exactly **11 records** out of 7,043.
  - Crucially, all 11 records have **`tenure == 0`**.
  - These are brand-new customers who enrolled in the current billing cycle and have not yet completed their first monthly billing cycle.
  - Therefore, their cumulative billed charges are mathematically and logically **$0.00**.
- **Potential Impact on ML Pipeline:**
  - Loss of `TotalCharges` removes a primary predictor of customer lifetime value and churn likelihood.
  - Silent drop causes feature mismatch between training pipelines and production inference APIs.
- **Implemented Correction:**
  - Applied `pd.to_numeric(df["TotalCharges"], errors="coerce")`.
  - Set `df.loc[df["tenure"] == 0, "TotalCharges"] = 0.0`.
  - Documented that any further missing-value imputation in the ML phase should be encapsulated inside a `SimpleImputer` within an `sklearn.pipeline.Pipeline` fitted only on training folds.
  - Retained `TotalCharges` (`float64`) in both cleaned and featured datasets.
- **Validation Performed:** Verified `df['TotalCharges'].dtype == np.float64`, `df['TotalCharges'].isnull().sum() == 0`, and the 11 `tenure == 0` customers have `TotalCharges == 0.0`.
- **Final Status:** **RESOLVED**

---

### ISSUE-03: Improper Categorical Encoding with Global LabelEncoder (Confirmed)
- **Issue ID:** `ISSUE-03`
- **Location:** `notebooks/Telco_data_Jupyter lab.ipynb` (cell 25)
- **Original Implementation:**
  ```python
  from sklearn.preprocessing import LabelEncoder
  le = LabelEncoder()
  for column in df.select_dtypes(include="object").columns:
      df[column] = le.fit_transform(df[column].astype(str))
  ```
- **Why Problematic:**
  1. `LabelEncoder` is strictly designed for encoding 1D target vectors (`y`), NOT input feature matrices (`X`).
  2. Imposes an artificial ordinal integer scale (0, 1, 2, ...) on nominal categories with no natural hierarchy (e.g., `PaymentMethod`, `InternetService`, `Contract`). Linear models, logistic regression, neural networks, and distance-based algorithms interpret these as quantitative distances ($2 > 1 > 0$), distorting model weights.
  3. Tree-based models can be forced into unnatural sequential splits.
  4. Fitting across the entire dataset before splitting creates **data leakage**.
  5. Cannot handle unseen categories at inference time without re-fitting.
- **Potential Impact on ML Pipeline:**
  - Severely degrades linear/regularized models.
  - Yields invalid coefficients and deceptive feature importances.
  - Causes runtime crashes on production data with novel categorical levels.
- **Implemented Correction:**
  - Preserved natural categorical values in `01_data_cleaning.ipynb` and `02_exploratory_analysis.ipynb`.
  - Categorized features into explicit handling tiers for ML modeling:
    - **Binary Variables** (`Partner`, `Dependents`, `PhoneService`, `PaperlessBilling`, `Churn`): Cleanly mapped to integer binary `0`/`1`.
    - **Nominal Multi-class Variables** (`InternetService`, `Contract`, `PaymentMethod`, `MultipleLines`, etc.): Targeted for `OneHotEncoder(drop='first', handle_unknown='ignore')`.
  - Designed an industry-standard `ColumnTransformer` architecture ready for the upcoming ML experimentation phase.
- **Validation Performed:** Verified that categorical columns maintain clean representations and that the `ColumnTransformer` pipeline successfully transforms feature matrices without leakage.
- **Final Status:** **RESOLVED**

---

### ISSUE-04: False Duplicate Alarm & Customer Record Integrity (Confirmed)
- **Issue ID:** `ISSUE-04`
- **Location:** `notebooks/Telco_data_Jupyter lab.ipynb` (cell 17) and `data/processed/Telco-Customer-Churn_cleaned.xls`
- **Original Implementation:**
  ```python
  df.drop("customerID", axis=1, inplace=True)
  df.duplicated().sum()  # Reported 27 duplicates
  ```
- **Why Problematic:**
  - Dropping `customerID` before checking duplicates masked whether the records were duplicate database entries or distinct customers.
  - The omission of `TotalCharges` (Issue B) artificially inflated the duplicate feature count from 22 to 27 (an extra 5 false collisions).
  - Deleting these rows would discard 22 to 27 real customer subscriptions.
- **Deep Investigation & Mathematical Proof:**
  - `df_raw['customerID'].nunique() == 7043` out of 7043 rows. **Every customer ID is 100% unique.**
  - When `TotalCharges` is properly included, exactly 22 duplicate feature rows (42 customers total) share identical plan parameters: Month-to-month, single line, basic phone, tenure = 1 month, monthly charge ~$20.
  - Crucially, within these identical plan profiles, **some customers churned while others remained subscribed**.
  - In a telecommunications provider with thousands of subscribers, multiple customers purchasing standard basic service plans is expected and normal.
- **Potential Impact on ML Pipeline:** Deleting these records reduces legitimate training data, biases churn class balance, and strips real customer behavioral variance.
- **Implemented Correction:**
  - All 7,043 customer records are **strictly retained**.
  - `customerID` is retained as a primary key/identifier during cleaning and validation, and excluded only from model input matrices.
- **Validation Performed:** Verified `len(df_clean) == 7043` and `df_clean['customerID'].nunique() == 7043`.
- **Final Status:** **RESOLVED**

---

### ISSUE-05: Redundant & Arbitrary Feature Engineering (Confirmed)
- **Issue ID:** `ISSUE-05`
- **Location:** `notebooks/Telco_exploratory data analysis.ipynb` (cells 14–17)
- **Original Implementation:**
  ```python
  df["ChargeCategory"] = pd.cut(df["MonthlyCharges"], bins=[0,40,80,150], labels=["Low","Medium","High"])
  df = pd.get_dummies(df, columns=["ChargeCategory"], drop_first=True)
  ```
  Resulting dataset retained continuous `MonthlyCharges` alongside `ChargeCategory_Medium` and `ChargeCategory_High`.
- **Why Problematic:**
  1. Coarse discretization of continuous `MonthlyCharges` destroys granular pricing variation.
  2. Retaining both continuous `MonthlyCharges` and the dummy indicators introduces **strong multicollinearity** (Pearson correlation up to $r = 0.81$).
  3. Bin cutoffs (40, 80) were selected arbitrarily without statistical or business rationale.
  4. No domain-grounded feature engineering (tenure cohorts, service bundling, payment automation) was performed.
- **Potential Impact on ML Pipeline:** Multicollinearity inflates standard errors in logistic regression, introduces redundant splits in tree models, and impairs interpretability.
- **Implemented Correction:**
  - Retained continuous `MonthlyCharges` and `TotalCharges`.
  - Replaced arbitrary bins with 9 domain-grounded, non-leaking features:
    1. `tenure_group`: Lifecycle cohorts (`0-12m`, `13-24m`, `25-48m`, `49-60m`, `61-72m`).
    2. `total_services`: Aggregate count of active subscriber services.
    3. `security_services_count`: Count of security/protection services (`OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`).
    4. `streaming_services_count`: Count of entertainment add-ons (`StreamingTV`, `StreamingMovies`).
    5. `has_internet`: Binary indicator for internet presence.
    6. `is_fiber_optic`: Binary indicator for premium high-churn fiber optic technology.
    7. `is_automatic_payment`: Binary indicator for automated payment mechanisms (bank transfer, credit card).
    8. `is_long_term_contract`: Binary indicator for 1- or 2-year commitments.
    9. `avg_monthly_charges` & `charge_escalation_ratio`: Historical billing dynamics detecting price escalation shocks.
  - Verified zero target leakage: `Churn` was never used in any feature definition.
- **Validation Performed:** Computed correlation matrices confirming no extreme collinearity with original numeric features and no correlation $|r| > 0.95$ with `Churn`.
- **Final Status:** **RESOLVED**

---

### ISSUE-06: Brittle, Machine-Specific File Paths (Confirmed)
- **Issue ID:** `ISSUE-06`
- **Location:** `notebooks/Telco_data_Jupyter lab.ipynb`, `notebooks/Telco_exploratory data analysis.ipynb`
- **Original Implementation:** Hardcoded paths such as `pd.read_csv("Telco-Customer-Churn.csv")` which fail if the notebook kernel is launched from the repository root rather than `notebooks/`.
- **Why Problematic:** Execution crashes with `FileNotFoundError` across different IDEs and command-line runners.
- **Implemented Correction:** Implemented dynamic path resolution using `pathlib.Path`:
  ```python
  CURRENT_DIR = Path.cwd()
  ROOT_DIR = CURRENT_DIR.parent if CURRENT_DIR.name == "notebooks" else CURRENT_DIR
  DATA_RAW = ROOT_DIR / "data" / "raw" / "Telco-Customer-Churn.csv"
  ```
- **Validation Performed:** Validated execution both from the root workspace and from within the `notebooks/` directory.
- **Final Status:** **RESOLVED**

---

### ISSUE-07: Unnormalized EDA & Misleading Visualizations (Confirmed)
- **Issue ID:** `ISSUE-07`
- **Location:** `notebooks/Telco_exploratory data analysis.ipynb`
- **Original Implementation:** Relied solely on raw customer counts (`sns.countplot(..., hue="Churn")`).
- **Why Problematic:** Subgroup sizes differ drastically (e.g., Month-to-month contracts vs Two-year contracts). Raw counts fail to show segment churn propensity.
- **Implemented Correction:** In `notebooks/02_exploratory_analysis.ipynb`, implemented normalized Churn Rate calculations:
  $$\text{Churn Rate (\%)} = \frac{\text{Churned Customers}}{\text{Total Customers in Segment}} \times 100$$
  Visualized all 10 required business dimensions with descriptive titles, percentage labels, and actionable interpretations.
- **Validation Performed:** Executed notebook; verified all 10 visual analyses render cleanly with exact mathematical consistency.
- **Final Status:** **RESOLVED**

---

## 2. Dataset Validation Summary

Following remediation, both the cleaned and engineered datasets were rigorously validated against baseline invariants:

| Metric / Check | Baseline Reference | Validated Cleaned Dataset | Validated Featured Dataset | Status |
| :--- | :--- | :--- | :--- | :---: |
| **Total Customer Records** | 7,043 | 7,043 | 7,043 | **PASS** |
| **Total Columns** | 21 (raw) | 21 | 31 | **PASS** |
| **Unique Customer IDs** | 7,043 | 7,043 (100% unique) | 7,043 (100% unique) | **PASS** |
| **TotalCharges Null Count** | 0 allowed | 0 (11 tenure=0 filled with 0.0) | 0 | **PASS** |
| **TotalCharges Data Type** | Numeric (`float64`) | `float64` | `float64` | **PASS** |
| **Target Variable (Churn)** | Binary `{0, 1}` | `{0, 1}` | `{0, 1}` | **PASS** |
| **Non-Churned Customers (0)**| 5,174 | 5,174 (73.46%) | 5,174 (73.46%) | **PASS** |
| **Churned Customers (1)** | 1,869 | 1,869 (26.54%) | 1,869 (26.54%) | **PASS** |
| **Baseline Churn Rate** | 26.54% | 26.54% | 26.54% | **PASS** |
| **Infinite Values Check** | Zero infinite values | None detected | None detected | **PASS** |
| **Target Leakage Check** | No leakage features | None | Max $|r| = 0.41$ (Contract) | **PASS** |
| **Primary Key Separation** | `customerID` not in features| Retained for validation only | Retained for validation only | **PASS** |

### File Locations:
- Cleaned Dataset: `data/processed/telco_customer_churn_cleaned.csv`
- Featured Dataset: `data/processed/telco_customer_churn_featured.csv`

---

## 3. Notebook Structure & Execution Results

The notebooks have been structured into three dedicated, single-responsibility documents:

### Notebook 1: `notebooks/01_data_cleaning.ipynb`
- **Responsibilities:** Ingests raw data via `pathlib`, inspects shapes and types, identifies whitespace in `TotalCharges`, resolves tenure=0 records, audits duplicate feature rows vs unique customer IDs, encodes binary target `Churn`, exports cleaned CSV, and executes 8 automated validation assertions.
- **Execution Status:** **EXECUTED CLEANLY (Exit Code 0)** — All assertions passed.

### Notebook 2: `notebooks/02_exploratory_analysis.ipynb`
- **Responsibilities:** Ingests cleaned data and conducts all 10 required analyses using normalized Churn Rates (%):
  1. Overall churn distribution (26.54% baseline).
  2. Contract type vs churn (Month-to-month: 42.71% vs Two-year: 2.83%).
  3. Payment method vs churn (Electronic check: 45.29% vs Auto-pay: ~16%).
  4. Internet technology vs churn (Fiber optic: 41.89% vs DSL: 18.96%).
  5. Tenure lifecycle cohorts (0-12m: 47.68% vs 61-72m: 6.61%).
  6. Tech support availability (No support: 41.64% vs With support: 15.17%).
  7. Monthly charges distribution (Churners concentrated between $70–$100).
  8. Total charges distribution (Churners leave early, skewing total charges low).
  9. Service adoption patterns (Churn drops from 39.87% at 0 add-ons to 11.45% at 4 add-ons).
  10. Demographic comparisons (Seniors churn at 41.68% vs non-seniors at 23.61%; partners/dependents provide churn protection).
- **Execution Status:** **EXECUTED CLEANLY (Exit Code 0)** — All figures and tables rendered.

### Notebook 3: `notebooks/03_feature_engineering.ipynb`
- **Responsibilities:** Evaluates prior charge binning, demonstrates collinearity and information loss, constructs 9 domain-grounded derived features, verifies zero target leakage, configures a production-ready `ColumnTransformer` preprocessing pipeline, exports featured dataset, and validates data invariants.
- **Execution Status:** **EXECUTED CLEANLY (Exit Code 0)** — All assertions passed.

---

## 4. Repository Structure Overview

```
AI-Churn-Prediction-Model/
│
├── data/
│   ├── raw/
│   │   └── Telco-Customer-Churn.csv             # Immutable raw data (7,043 rows)
│   └── processed/
│       ├── telco_customer_churn_cleaned.csv     # Verified cleaned dataset (21 cols)
│       ├── telco_customer_churn_featured.csv    # Finalized feature dataset (31 cols)
│       ├── Telco-Customer-Churn_cleaned.csv     # Renamed from .xls (legacy compatibility)
│       └── add_feature_monthly charges category.csv # Renamed from .xls (legacy compatibility)
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb                   # Standardized cleaning & audit notebook
│   ├── 02_exploratory_analysis.ipynb            # Standardized normalized EDA notebook
│   ├── 03_feature_engineering.ipynb             # Standardized feature pipeline notebook
│   ├── Telco_data_Jupyter lab.ipynb             # Original notebook preserved
│   └── Telco_exploratory data analysis.ipynb    # Original notebook preserved
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── cleaning.py                          # Modular cleaning and loading utilities
│   └── features/
│       ├── __init__.py
│       └── engineering.py                       # Modular feature engineering utilities
│
├── tests/
│   └── test_data_validation.py                  # Automated unit test suite (6 tests)
│
├── scripts/
│   ├── generate_notebooks.py                    # Notebook generator script
│   └── execute_notebooks.py                     # Notebook execution runner
│
├── reports/
│   ├── figures/
│   └── data_preparation_audit.md               # This comprehensive audit report
│
├── .gitignore                                   # Standard Python/venv gitignore
└── README.md
```

---

## 5. Summary of Files Changed

| Action | File Path | Description |
| :--- | :--- | :--- |
| **Added** | `reports/data_preparation_audit.md` | Master audit and remediation documentation |
| **Added** | `notebooks/01_data_cleaning.ipynb` | Standardized data cleaning notebook |
| **Added** | `notebooks/02_exploratory_analysis.ipynb` | Standardized EDA notebook with 10 business analyses |
| **Added** | `notebooks/03_feature_engineering.ipynb` | Standardized feature engineering & pipeline notebook |
| **Added** | `src/data/__init__.py` & `src/data/cleaning.py` | Reusable data loading, cleaning, and validation functions |
| **Added** | `src/features/__init__.py` & `src/features/engineering.py` | Reusable feature engineering functions |
| **Added** | `tests/test_data_validation.py` | Unit test suite (6 automated tests) |
| **Added** | `scripts/generate_notebooks.py` & `scripts/execute_notebooks.py` | Reproducible notebook generation and execution scripts |
| **Added** | `data/processed/telco_customer_churn_cleaned.csv` | Cleaned dataset (7,043 rows, 21 columns) |
| **Added** | `data/processed/telco_customer_churn_featured.csv` | Featured dataset (7,043 rows, 31 columns) |
| **Renamed**| `data/processed/Telco-Customer-Churn_cleaned.xls` $\rightarrow$ `.csv` | Corrected file extension from `.xls` to `.csv` |
| **Renamed**| `data/processed/add_feature_monthly charges category.xls` $\rightarrow$ `.csv`| Corrected file extension from `.xls` to `.csv` |
| **Modified**| `.gitignore` | Added `venv/`, Python caches, and temporary files |

---

## 6. Outstanding Issues

- **None.** All identified issues (A through E and supplementary reproducibility findings) have been investigated, mathematically proven, corrected, tested, and documented.

---

## 7. Machine Learning Readiness Assessment

The finalized dataset (`telco_customer_churn_featured.csv`) and modular pipeline are **100% READY for the Machine Learning phase**.

### Readiness Justification:
1. **Data Integrity Guaranteed:** Exactly 7,043 unique customer records, zero unhandled missing values, zero infinite values, and mathematically verified target distribution (26.54% churn rate).
2. **Zero Data Leakage:** Categorical encoders and numeric scalers are not fitted globally. Clean `ColumnTransformer` architecture is defined so preprocessing fits strictly on training folds during cross-validation.
3. **No Target Leakage:** Derived features (`tenure_group`, `total_services`, `security_services_count`, `is_fiber_optic`, `is_automatic_payment`, `charge_escalation_ratio`) depend strictly on customer subscription parameters and billing dynamics without reference to `Churn`.
4. **Reproducibility:** Tested from end to end using Python 3.14 in the virtual environment. All unit tests pass (`OK in 0.27s`).
