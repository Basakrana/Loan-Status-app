## 🏦 Loan Approval Prediction App

This project predicts whether a loan application will be approved or rejected based on applicant information such as age, income, employment details, credit history, and loan attributes. The model uses machine learning classification algorithms to assist financial institutions in making faster and more consistent loan decisions.

## 📌 Project Overview

Loan approvals are often influenced by multiple factors like an applicant’s financial stability, credit history, and employment background. Manually evaluating these applications can be slow and prone to human bias. This project applies data preprocessing, feature engineering, and machine learning classification techniques to automate the decision-making process. The end result is a Streamlit-based app where users can input details and instantly receive a prediction on loan approval status.

## 📂 Dataset Information

The dataset includes customer-level financial and behavioral attributes. Key columns:

person_age → Applicant’s age

person_gender → Gender of the applicant

person_education → Education level (Graduate, Master, etc.)

person_income → Annual income

person_emp_exp → Employment experience (years)

person_home_ownership → Home ownership status (Rent, Own, Mortgage)

loan_amnt → Loan amount requested

loan_intent → Purpose of loan (Education, Medical, Personal, etc.)

loan_int_rate → Loan interest rate (%)

loan_percent_income → Loan amount as a % of income

cb_person_cred_hist_length → Credit history length (years)

credit_score → Credit score of the applicant

previous_loan_defaults_on_file → Whether applicant had previous defaults (Yes/No)

loan_status → Target variable (Approved = 1, Not Approved = 0)

## ⚙️ Features of the Project

Data Cleaning & Preprocessing: Handling missing values, encoding categorical features, scaling.

Exploratory Data Analysis (EDA): Understanding patterns in loan approval.

Classification Models Used:

Logistic Regression

Random Forest Classifier

XGBoost Classifier

Evaluation Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC.

Deployment: Streamlit app for interactive predictions.

## 🚀 Tech Stack

Python

Libraries: Pandas, NumPy, Scikit-learn, XGBoost, Imbalanced-learn

Visualization: Matplotlib, Seaborn

Deployment: Streamlit

## 📊 Model Performance

XGBoost Classifier gave the best results with high accuracy and balanced recall.

Random Forest also performed strongly, reducing overfitting.

Logistic Regression served as a baseline model.

## 📌 Insights

Applicants with higher income and strong credit history are more likely to get loan approvals.

Previous loan defaults drastically reduce approval chances.

Higher loan-to-income ratio negatively impacts approval probability.

Education level and stable employment history improve chances of approval.

## 📧 Contact

👤 Rana Basak

Data Analyst / Machine Learning Enthusiast

LinkedIn
 | GitHub
