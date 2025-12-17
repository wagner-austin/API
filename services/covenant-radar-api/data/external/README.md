# External Datasets for Bankruptcy & Credit Risk Prediction

This folder contains 38 datasets for training and evaluating bankruptcy/credit default prediction models.

## Dataset Summary

| Dataset | Samples | Features | Target | Source |
|---------|---------|----------|--------|--------|
| **AMEX Default** | 5.5M+ | 190 | Default | Kaggle Competition |
| **Lending Club** | 2.2M+ | 150+ | Loan Status | Kaggle |
| **SBA Loans** | 900K+ | 27 | Default | Kaggle |
| **Home Credit** | 307K | 122 | Default | Kaggle Competition |
| **Risk Analytics** | 300K+ | 100+ | Default | Kaggle |
| **Credit Card Fraud** | 284K | 31 | Fraud | Kaggle |
| **Prosper Loans** | 113K | 81 | Loan Status | Kaggle |
| **US Bankruptcy** | 78K | 18 | Bankruptcy | Kaggle/GitHub |
| **Vehicle Loan** | 233K | 41 | Default | Kaggle |
| **FICO Give Me Credit** | 150K | 11 | Default | Kaggle Competition |
| **Credit Score** | 100K | 28 | Score Class | Kaggle |
| **Taiwan Bankruptcy** | 6.8K | 95 | Bankruptcy | UCI/Kaggle |
| **Polish Bankruptcy** | 13K | 64 | Bankruptcy | UCI |
| **Chinese HAT** | 13.5K | Graph | Bankruptcy | GitHub |
| **Chinese SMEsD** | 4K | Graph | Bankruptcy | GitHub |
| **German Credit** | 1K | 20 | Credit Risk | UCI |

---

## Detailed Dataset Descriptions

### Large Datasets (100K+ samples)

#### `kaggle_amex_default/` - American Express Default Prediction
- **Size**: 48GB (train: 16GB, test: 32GB)
- **Samples**: 5.5M+ credit card statements
- **Features**: 190 anonymized features
- **Target**: `target` (1 = default)
- **Source**: https://www.kaggle.com/c/amex-default-prediction
- **License**: Competition rules apply

#### `kaggle_lending_club/` - Lending Club Loan Data
- **Size**: 3.9GB
- **Samples**: 2.2M+ loans (2007-2018)
- **Features**: 150+ (loan amount, interest rate, grade, employment, etc.)
- **Target**: `loan_status` (Fully Paid, Charged Off, etc.)
- **Source**: https://www.kaggle.com/datasets/wordsforthewise/lending-club
- **Files**: `accepted_2007_to_2018Q4.csv`, `rejected_2007_to_2018Q4.csv`

#### `kaggle_sba_loans/` - SBA National Loans
- **Size**: 172MB
- **Samples**: 900K+ small business loans
- **Features**: 27 (loan amount, term, industry, state, etc.)
- **Target**: `MIS_Status` (P I F = Paid, CHGOFF = Default)
- **Source**: https://www.kaggle.com/datasets/mirbektoktogaraev/should-this-loan-be-approved-or-denied
- **License**: CC-BY-SA-4.0

#### `kaggle_home_credit/` - Home Credit Default Risk
- **Size**: 688MB (multiple files)
- **Samples**: 307K applications
- **Features**: 122 in main file + bureau/payment history
- **Target**: `TARGET` (1 = default)
- **Source**: https://www.kaggle.com/c/home-credit-default-risk
- **Files**: `application_train.csv`, `bureau.csv`, `credit_card_balance.csv`, etc.

#### `kaggle_risk_analytics/` - Loan Defaulter Risk Analytics
- **Size**: 545MB
- **Samples**: 300K+ applications
- **Features**: 100+ (demographics, income, loan details)
- **Target**: `TARGET` (1 = default)
- **Source**: https://www.kaggle.com/datasets/gauravduttakiit/loan-defaulter

#### `kaggle_credit_card_fraud/` - Credit Card Fraud Detection
- **Size**: 144MB
- **Samples**: 284,807 transactions
- **Features**: 31 (V1-V28 PCA components + Time, Amount)
- **Target**: `Class` (1 = fraud)
- **Source**: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- **License**: DbCL-1.0
- **Note**: Highly imbalanced (0.17% fraud)

#### `kaggle_prosper_loans/` - Prosper P2P Lending
- **Size**: 83MB
- **Samples**: 113K loans
- **Features**: 81 (credit grade, income, debt ratio, etc.)
- **Target**: `LoanStatus`
- **Source**: https://www.kaggle.com/datasets/nurudeenabdulsalaam/prosper-loan-dataset

#### `kaggle_vehicle_loan/` - Vehicle Loan Default
- **Size**: 61MB
- **Samples**: 233K loans
- **Features**: 41 (demographics, bureau data, loan details)
- **Target**: `loan_default` (1 = default)
- **Source**: https://www.kaggle.com/datasets/avikpaul4u/vehicle-loan-default-prediction
- **Files**: `train.csv`, `test.csv`, `Data Dictionary.xlsx`

#### `kaggle_fico/` - Give Me Some Credit (FICO)
- **Size**: 14MB
- **Samples**: 150K borrowers
- **Features**: 11 (utilization, age, income, delinquencies)
- **Target**: `SeriousDlqin2yrs` (1 = 90+ days delinquent)
- **Source**: https://www.kaggle.com/c/GiveMeSomeCredit
- **Files**: `cs-training.csv`, `cs-test.csv`, `Data Dictionary.xls`

#### `kaggle_credit_score/` - Credit Score Classification
- **Size**: 45MB
- **Samples**: 100K customers
- **Features**: 28 (payment history, credit mix, utilization)
- **Target**: `Credit_Score` (Good, Standard, Poor)
- **Source**: https://www.kaggle.com/datasets/parisrohan/credit-score-classification

---

### Medium Datasets (10K-100K samples)

#### `github_american_bankruptcy/` - US Stock Market Bankruptcy
- **Size**: 12MB
- **Samples**: 78,682 firm-year observations
- **Features**: 18 financial ratios
- **Target**: `status_label` (1 = bankruptcy)
- **Period**: 1999-2018 (NYSE & NASDAQ)
- **Source**: https://github.com/sowide/bankruptcy_dataset
- **License**: See LICENSE.md
- **Splits**: Train (1999-2011), Val (2012-2014), Test (2015-2018)

#### `kaggle_us_bankruptcy/` - US Companies Bankruptcy (Kaggle version)
- **Size**: 11MB
- **Samples**: 78K firm-years
- **Source**: https://www.kaggle.com/datasets/utkarshx27/american-companies-bankruptcy-prediction-dataset

#### `kaggle_loan_default/` - Loan Default Dataset
- **Size**: 28MB
- **Samples**: 67K loans
- **Features**: 34 (loan details, borrower info)
- **Target**: `Status` (1 = default)
- **Source**: https://www.kaggle.com/datasets/yasserh/loan-default-dataset

#### `kaggle_loan_status/` - Credit Loan Status
- **Size**: 20MB
- **Samples**: 50K+ loans
- **Source**: https://www.kaggle.com/datasets/zaurbegiev/my-dataset

#### `kaggle_money_lender/` - Loan Based on Customer Behavior
- **Size**: 22MB
- **Features**: Demographics, income, employment
- **Target**: `Risk_Flag`
- **Source**: https://www.kaggle.com/datasets/subhamjain/loan-prediction-based-on-customer-behavior

#### `github_chinese_hat/` - Chinese Company Networks (Graph Data)
- **Size**: 30MB
- **Samples**: 13,489 companies
- **Features**: Board member + shareholder network
- **Target**: Bankruptcy labels in `label_ci_l.npy`
- **Format**: NumPy arrays (.npy) and pickle (.pkl)
- **Source**: https://github.com/hetergraphforbankruptcypredict/HAT
- **Note**: Designed for Graph Neural Networks (GNN)

#### `polish_data/` - Polish Companies Bankruptcy
- **Size**: 21MB
- **Samples**: 13,288 companies (5 year horizons)
- **Features**: 64 financial ratios
- **Target**: `class` (1 = bankruptcy)
- **Format**: ARFF (1year.arff through 5year.arff)
- **Source**: UCI ML Repository
- **Period**: 2000-2013

#### `kaggle_company_bankruptcy/` - Taiwan Company Bankruptcy
- **Size**: 11MB
- **Samples**: 6,819 companies
- **Features**: 95 financial ratios
- **Target**: `Bankrupt?` (1 = bankruptcy)
- **Source**: https://www.kaggle.com/datasets/fedesoriano/company-bankruptcy-prediction

#### `taiwan_data/` - Taiwan Bankruptcy (Original)
- **Size**: 11MB
- **Samples**: 6,819 companies
- **Source**: Taiwan Economic Journal (1999-2009)

#### `kaggle_taiwan_bankruptcy/` - Taiwan Bankruptcy (Kaggle copy)
- **Size**: 11MB
- **Same as above**

---

### Small Datasets (<10K samples)

#### `kaggle_financial_distress/` - Financial Distress Prediction
- **Size**: 2.2MB
- **Samples**: 3,672 companies
- **Features**: 83 anonymized (x1-x83)
- **Target**: `Financial Distress` (continuous, >0.5 = distress)
- **Source**: https://www.kaggle.com/datasets/shebrahimi/financial-distress

#### `github_chinese_smesd/` - Chinese SME Bankruptcy (Graph Data)
- **Size**: 2.1MB
- **Samples**: 3,976 SMEs
- **Features**: Business info + lawsuit events (2000-2021)
- **Format**: Pickle files (.pkl)
- **Source**: https://github.com/shaopengw/comrisk
- **Note**: Includes lawsuit information (rare feature)
- **Splits**: Train (2014-2018), Val (2019), Test (2020-2021)

#### `kaggle_credit_risk/` - Credit Risk Dataset
- **Size**: 1.8MB
- **Samples**: 32,581 borrowers
- **Features**: 11 (age, income, employment, loan details)
- **Target**: `loan_status` (0 = non-default, 1 = default)
- **Source**: https://www.kaggle.com/datasets/laotse/credit-risk-dataset
- **License**: CC0-1.0

#### `kaggle_credit_default/` - UCI Credit Card Default
- **Size**: 2.8MB
- **Samples**: 30,000 Taiwan credit card users
- **Features**: 24 (demographics, payment history, bill amounts)
- **Target**: `default.payment.next.month` (1 = default)
- **Source**: https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset

#### `kaggle_give_me_credit/` - Give Me Some Credit (Competition)
- **Size**: 5.2MB
- **Same as kaggle_fico

#### `kaggle_credit_customers/` - Credit Card Customers Churn
- **Size**: 1.5MB
- **Samples**: 10,127 customers
- **Target**: `Attrition_Flag` (churn prediction)
- **Source**: https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers

#### `kaggle_bank_churners/` - Bank Churners
- **Size**: 1.5MB
- **Same as credit_customers

#### `german_credit/` - German Credit (UCI)
- **Size**: 88KB
- **Samples**: 1,000
- **Features**: 20 (categorical + numerical)
- **Target**: Last column (1 = good, 2 = bad)
- **Source**: UCI Statlog German Credit

#### `kaggle_german_credit/` - German Credit (Kaggle)
- **Size**: 52KB
- **Same as above, CSV format

#### `kaggle_south_german/` - South German Credit
- **Size**: 52KB
- **Same as german_credit

#### `kaggle_heloc/` - Home Equity Line of Credit
- **Size**: 664KB
- **Samples**: 10,459 applicants
- **Features**: 23 credit bureau variables
- **Target**: `RiskPerformance` (Good/Bad)
- **Source**: FICO Explainable ML Challenge

#### `kaggle_bank_loan/` - Bank Personal Loan
- **Size**: 344KB
- **Samples**: 5,000 customers
- **Target**: `Personal Loan` (accepted/declined)
- **Format**: Excel (.xlsx)
- **Source**: https://www.kaggle.com/datasets/itsmesunil/bank-loan-modelling

#### `kaggle_personal_loan/` - Personal Loan Modeling
- **Size**: 208KB
- **Same as bank_loan

#### `kaggle_loan_eligibility/` - Loan Eligibility
- **Size**: 68KB
- **Features**: Income, credit history, loan amount
- **Target**: `Loan_Status` (Y/N)

#### `kaggle_loan_prediction/` - Loan Prediction Problem
- **Size**: 68KB
- **Similar to loan_eligibility

#### `kaggle_loan_approval/` - Loan Approval Prediction
- **Size**: 376KB
- **Target**: `loan_status` (Approved/Rejected)
- **License**: MIT

#### `kaggle_bank_marketing/` - Bank Marketing
- **Size**: 5.6MB
- **Samples**: 41,188 contacts
- **Target**: `y` (subscribed to term deposit)
- **Note**: Not credit/bankruptcy, but related financial ML

#### `us_data/` - US Bankruptcy (Original)
- **Size**: 11MB
- **Original US bankruptcy dataset

---

## Data Formats

| Format | Extensions | How to Load |
|--------|------------|-------------|
| CSV | `.csv` | `pandas.read_csv()` |
| Excel | `.xls`, `.xlsx` | `pandas.read_excel()` |
| ARFF | `.arff` | `scipy.io.arff.loadarff()` |
| Pickle | `.pkl` | `pickle.load()` or `pandas.read_pickle()` |
| NumPy | `.npy` | `numpy.load()` |
| Gzip CSV | `.csv.gz` | `pandas.read_csv(compression='gzip')` |

---

## Usage Notes

1. **Class Imbalance**: Most bankruptcy datasets are highly imbalanced (1-5% positive class). Consider using:
   - SMOTE / ADASYN oversampling
   - Class weights
   - Focal loss
   - Stratified sampling

2. **Time-based Splits**: For realistic evaluation, use temporal splits (train on past, test on future) rather than random splits.

3. **Feature Engineering**: Financial ratios often benefit from:
   - Log transformations
   - Winsorization (outlier clipping)
   - Sector/industry normalization

4. **Graph Datasets**: Chinese HAT and SMEsD require GNN frameworks (PyG, DGL) for full utilization.

---

## Citations

If using these datasets in research, please cite the original sources. Key papers:

- **American Bankruptcy**: Pellegrino et al. (2024) "A Multi-Head LSTM Architecture for Bankruptcy Prediction"
- **Chinese HAT**: PAKDD 2021 "Heterogeneous Graph Attention Network for SME Bankruptcy Prediction"
- **Chinese SMEsD**: Wei et al. (2024) "Combining Intra-Risk and Contagion Risk for Enterprise Bankruptcy Prediction"
- **Polish Companies**: Emerging Markets Information Service (EMIS)
- **Taiwan Bankruptcy**: Taiwan Economic Journal

---

## Total Collection

- **38 datasets**
- **55GB total size**
- **~10M+ samples combined**
- **Bankruptcy, credit default, loan status, fraud detection**

Last updated: December 2024
