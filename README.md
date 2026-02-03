# Credit Scoring Model using Machine Learning

## Project Overview
This project implements a **Credit Scoring Model** to predict whether a loan applicant is likely to **default (1)** or **repay successfully (0)**.  
The model uses applicant financial, demographic, and credit behavior data and applies supervised machine learning techniques to estimate default risk.

This project follows a complete **end-to-end machine learning pipeline**, including:
- Data loading and preprocessing  
- Exploratory Data Analysis (EDA)  
- Feature engineering  
- Model training and evaluation  
- Threshold tuning  
- Model saving for deployment  

---

## Dataset Description
The dataset used in this project is a **realistic simulated credit dataset** designed to resemble real-world banking and loan data.

**Target Variable**
- `default`  
  - `1` → Applicant defaulted  
  - `0` → Applicant did not default  

### Features
| Feature | Description |
|------|------------|
| age | Age of applicant |
| income | Annual income |
| employment_years | Years of employment |
| credit_amount | Loan amount |
| duration_months | Loan duration |
| num_existing_loans | Number of active loans |
| num_dependents | Number of dependents |
| credit_utilization | Percentage of credit used |
| late_payments_last_year | Missed payments in last year |
| savings_balance | Savings balance |
| housing_type | own / rent / mortgage |
| loan_purpose | education / car / personal / business |

Categorical features are encoded using **One-Hot Encoding**, and numerical features are scaled after imputation.

---

## Machine Learning Pipeline
1. **Data Loading**
   - CSV-based dataset loading using pandas

2. **Exploratory Data Analysis**
   - Class imbalance analysis
   - Feature distribution visualization
   - Summary statistics

3. **Preprocessing**
   - Missing value imputation
   - Feature scaling (StandardScaler)
   - Categorical encoding (OneHotEncoder)
   - Implemented using `ColumnTransformer`

4. **Model Training**
   - Logistic Regression (baseline)
   - Random Forest Classifier (final model)

5. **Evaluation Metrics**
   - Accuracy
   - Precision
   - Recall (Sensitivity)
   - F1-score
   - ROC-AUC
   - Confusion Matrix

6. **Threshold Optimization**
   - Adjusted decision threshold to improve recall for default detection

7. **Model Persistence**
   - Trained pipeline saved using `joblib`

---

## Model Performance (Example)
- **ROC-AUC:** ~0.85 (varies per run)
- **Recall (Default Class):** Prioritized to reduce false negatives
- **Precision-Recall Tradeoff:** Controlled via threshold tuning

> Note: In credit risk modeling, recall is often prioritized to minimize approving high-risk applicants.

---

## Project Structure

Credit_Scoring_Project/
│
├── CreditScoring_Starter_Notebook.ipynb
├── sample_credit_dataset.csv
├── models/
│ └── credit_scoring_rf_pipeline.joblib
├── README.md
└── requirements.txt

---

## Required Python Libraries (Install Before Running)
Install the following libraries **before executing the notebook**:

bash:
>>>pip install numpy pandas matplotlib scikit-learn joblib

