# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python [conda env:base] *
#     language: python
#     name: conda-base-py
# ---

# %%
from sklearn.naive_bayes import GaussianNB

nb_model = GaussianNB()
nb_model.fit(x_train_scaled, y_train)

y_pred = nb_model.predict(x_test_scaled)

# %%

# %%
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# Load trained objects
# -----------------------------
model = joblib.load("naive_bayes_model.pkl")
scaler = joblib.load("scaler.pkl")
ohe = joblib.load("onehot_encoder.pkl")
le = joblib.load("label_encoder.pkl")

st.set_page_config(
    page_title="CreditWise Loan System",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 CreditWise Loan System")
st.write("Predict whether a loan application will be Approved or Rejected.")

st.divider()

col1, col2 = st.columns(2)

with col1:
    applicant_income = st.number_input("Applicant Income", min_value=0.0)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0.0)
    age = st.number_input("Age", min_value=18, max_value=80)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900)
    existing_loans = st.number_input("Existing Loans", min_value=0)
    dti_ratio = st.number_input("DTI Ratio", min_value=0.0)
    savings = st.number_input("Savings", min_value=0.0)

with col2:
    collateral_value = st.number_input("Collateral Value", min_value=0.0)
    loan_amount = st.number_input("Loan Amount", min_value=0.0)
    loan_term = st.number_input("Loan Term (Months)", min_value=1)

    dependents = st.selectbox("Dependents", [0,1,2,3,4])

    education = st.selectbox(
        "Education Level",
        ["Graduate","Postgraduate","Undergraduate"]
    )

    employment = st.selectbox(
        "Employment Status",
        ["Business","Salaried","Self-Employed"]
    )

    marital = st.selectbox(
        "Marital Status",
        ["Married","Single"]
    )

    loan_purpose = st.selectbox(
        "Loan Purpose",
        ["Business","Education","Home","Personal"]
    )

    property_area = st.selectbox(
        "Property Area",
        ["Rural","Semi-Urban","Urban"]
    )

    gender = st.selectbox(
        "Gender",
        ["Female","Male"]
    )

    employer_category = st.selectbox(
        "Employer Category",
        ["Govt","Private","Self"]
    )

if st.button("Predict Loan Status", use_container_width=True):

    education = le.transform([education])[0]

    numeric = pd.DataFrame({
        "Applicant_Income":[applicant_income],
        "Coapplicant_Income":[coapplicant_income],
        "Age":[age],
        "Dependents":[dependents],
        "Credit_Score":[credit_score],
        "Existing_Loans":[existing_loans],
        "DTI_Ratio":[dti_ratio],
        "Savings":[savings],
        "Collateral_Value":[collateral_value],
        "Loan_Amount":[loan_amount],
        "Loan_Term":[loan_term],
        "Education_Level":[education]
    })

    categorical = pd.DataFrame({
        "Employment_Status":[employment],
        "Marital_Status":[marital],
        "Loan_Purpose":[loan_purpose],
        "Property_Area":[property_area],
        "Gender":[gender],
        "Employer_Category":[employer_category]
    })

    encoded = ohe.transform(categorical)

    encoded_df = pd.DataFrame(
        encoded,
        columns=ohe.get_feature_names_out(categorical.columns)
    )

    df = pd.concat(
        [numeric.reset_index(drop=True),
         encoded_df.reset_index(drop=True)],
        axis=1
    )

    # Feature Engineering
    df["DTI_Ratio_sq"] = df["DTI_Ratio"]**2
    df["Credit_Score_sq"] = df["Credit_Score"]**2
    df["applicant_income_log"] = np.log1p(df["Applicant_Income"])

    # Drop original columns exactly like training
    df = df.drop(
        columns=[
            "Credit_Score",
            "DTI_Ratio",
            "Applicant_Income"
        ]
    )

    scaled = scaler.transform(df)

    prediction = model.predict(scaled)[0]
    probability = model.predict_proba(scaled)[0]

    st.divider()

    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    st.metric(
        "Approval Probability",
        f"{probability[1]*100:.2f}%"
    )

    st.metric(
        "Rejection Probability",
        f"{probability[0]*100:.2f}%"
    )
