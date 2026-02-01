import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# Load models
logistic_model = joblib.load("logistic_churn_model.pkl")
rf_model = joblib.load("rf_churn_model.pkl")

st.set_page_config(page_title="Bank Customer Churn Dashboard", layout="wide")

st.title("🏦 Bank Customer Churn Risk Dashboard")

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("Customer Details")

age = st.sidebar.slider("Age", 18, 90, 40)
credit_score = st.sidebar.slider("Credit Score", 300, 900, 650)
geography = st.sidebar.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
tenure = st.sidebar.slider("Tenure (years)", 0, 10, 3)
balance = st.sidebar.number_input("Account Balance", 0.0, 300000.0, 50000.0)
num_products = st.sidebar.slider("Number of Products", 1, 4, 2)
is_active = st.sidebar.selectbox("Is Active Member", [0, 1])
salary = st.sidebar.number_input("Estimated Salary", 10000.0, 200000.0, 60000.0)

# -----------------------------
# Create input dataframe
# -----------------------------
input_df = pd.DataFrame({
    "CreditScore": [credit_score],
    "Geography": [geography],
    "Gender": [gender],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_products],
    "IsActiveMember": [is_active],
    "EstimatedSalary": [salary]
})

# Feature Engineering
input_df["BalanceSalaryRatio"] = input_df["Balance"] / (input_df["EstimatedSalary"] + 1)
input_df["ProductDensity"] = input_df["NumOfProducts"] / (input_df["Tenure"] + 1)
input_df["EngagementScore"] = input_df["IsActiveMember"] * input_df["NumOfProducts"]
input_df["AgeTenureRatio"] = input_df["Age"] / (input_df["Tenure"] + 1)
input_df["ZeroBalanceFlag"] = (input_df["Balance"] == 0).astype(int)

# -----------------------------
# Prediction
# -----------------------------
log_prob = logistic_model.predict_proba(input_df)[0][1]
rf_prob = rf_model.predict_proba(input_df)[0][1]

# Risk Category
def risk_label(p):
    if p < 0.30:
        return "🟢 Low Risk"
    elif p < 0.60:
        return "🟡 Medium Risk"
    else:
        return "🔴 High Risk"

# -----------------------------
# Display Results
# -----------------------------
st.subheader("📊 Churn Risk Prediction")

col1, col2, col3 = st.columns(3)

col1.metric("Logistic Churn Probability", f"{log_prob:.2%}")
col2.metric("Random Forest Probability", f"{rf_prob:.2%}")
col3.metric("Risk Category", risk_label(log_prob))

# -----------------------------
# What-if Analysis
# -----------------------------
st.subheader("🔁 What-if Simulator")

new_products = st.slider("Increase Number of Products", 1, 4, num_products)
new_active = st.selectbox("Make Customer Active?", [0, 1], is_active)

sim_df = input_df.copy()
sim_df["NumOfProducts"] = new_products
sim_df["IsActiveMember"] = new_active
sim_df["EngagementScore"] = sim_df["NumOfProducts"] * sim_df["IsActiveMember"]

sim_prob = logistic_model.predict_proba(sim_df)[0][1]

st.write(f"📉 New churn probability after changes: **{sim_prob:.2%}**")
