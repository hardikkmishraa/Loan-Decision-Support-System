import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ----------------------------------------------------------------------
# Load saved artifacts (must sit in the same folder as this app.py)
# ----------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("naive_bayes_model.pkl")
    scaler = joblib.load("scaler.pkl")
    ohe = joblib.load("onehot_encoder.pkl")
    le_education = joblib.load("education_label_encoder.pkl")
    le_target = joblib.load("target_label_encoder.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, scaler, ohe, le_education, le_target, feature_columns


model, scaler, ohe, le_education, le_target, feature_columns = load_artifacts()

OHE_COLS = ["Employment_Status", "Marital_Status", "Loan_Purpose",
            "Property_Area", "Gender", "Employer_Category"]

st.set_page_config(page_title="Loan Approval Predictor", page_icon="💰")
st.title("💰 Loan Approval Predictor")
st.write("Fill in the applicant details below to get a prediction.")

# ----------------------------------------------------------------------
# Input form
# ----------------------------------------------------------------------
with st.form("loan_form"):
    col1, col2 = st.columns(2)

    with col1:
        applicant_income = st.number_input("Applicant Income", min_value=0.0, value=5000.0)
        coapplicant_income = st.number_input("Coapplicant Income", min_value=0.0, value=0.0)
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        dependents = st.number_input("Dependents", min_value=0, max_value=10, value=0)
        credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
        existing_loans = st.number_input("Existing Loans", min_value=0, max_value=10, value=0)
        dti_ratio = st.number_input("DTI Ratio", min_value=0.0, max_value=1.0, value=0.3, step=0.01)

    with col2:
        savings = st.number_input("Savings", min_value=0.0, value=10000.0)
        collateral_value = st.number_input("Collateral Value", min_value=0.0, value=20000.0)
        loan_amount = st.number_input("Loan Amount", min_value=0.0, value=15000.0)
        loan_term = st.number_input("Loan Term (months)", min_value=0.0, value=60.0)
        employment_status = st.selectbox("Employment Status", ohe.categories_[0])
        marital_status = st.selectbox("Marital Status", ohe.categories_[1])
        loan_purpose = st.selectbox("Loan Purpose", ohe.categories_[2])

    col3, col4 = st.columns(2)
    with col3:
        property_area = st.selectbox("Property Area", ohe.categories_[3])
        gender = st.selectbox("Gender", ohe.categories_[4])
    with col4:
        employer_category = st.selectbox("Employer Category", ohe.categories_[5])
        education_level = st.selectbox("Education Level", le_education.classes_)

    submitted = st.form_submit_button("Predict")

# ----------------------------------------------------------------------
# Preprocessing + prediction (mirrors the notebook's training pipeline)
# ----------------------------------------------------------------------
if submitted:
    raw = pd.DataFrame([{
        "Applicant_Income": applicant_income,
        "Coapplicant_Income": coapplicant_income,
        "Employment_Status": employment_status,
        "Age": age,
        "Marital_Status": marital_status,
        "Dependents": dependents,
        "Credit_Score": credit_score,
        "Existing_Loans": existing_loans,
        "DTI_Ratio": dti_ratio,
        "Savings": savings,
        "Collateral_Value": collateral_value,
        "Loan_Amount": loan_amount,
        "Loan_Term": loan_term,
        "Loan_Purpose": loan_purpose,
        "Property_Area": property_area,
        "Gender": gender,
        "Employer_Category": employer_category,
        "Education_Level": education_level,
    }])

    # 1. Label-encode Education_Level with the SAME encoder used in training
    raw["Education_Level"] = le_education.transform(raw["Education_Level"])

    # 2. One-hot encode the categorical columns with the saved encoder
    encoded = ohe.transform(raw[OHE_COLS])
    encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(OHE_COLS), index=raw.index)
    data = pd.concat([raw.drop(columns=OHE_COLS), encoded_df], axis=1)

    # 3. Engineered features (same as training)
    data["DTI_Ratio_sq"] = data["DTI_Ratio"] ** 2
    data["Credit_Score_sq"] = data["Credit_Score"] ** 2
    data["applicant_income_log"] = np.log1p(data["Applicant_Income"])
    data = data.drop(columns=["DTI_Ratio", "Credit_Score", "Applicant_Income"])

    # 4. Reindex to the exact column order used at training time
    data = data.reindex(columns=feature_columns, fill_value=0)

    # 5. Scale
    data_scaled = scaler.transform(data)

    # 6. Predict
    pred = model.predict(data_scaled)[0]
    proba = model.predict_proba(data_scaled)[0]
    label = le_target.inverse_transform([pred])[0]

    st.divider()
    if label == "Yes":
        st.success(f"✅ Loan Approved  (confidence: {proba.max()*100:.1f}%)")
    else:
        st.error(f"❌ Loan Not Approved  (confidence: {proba.max()*100:.1f}%)")

    with st.expander("See prediction probabilities"):
        st.write({cls: f"{p*100:.1f}%" for cls, p in zip(le_target.classes_, proba)})