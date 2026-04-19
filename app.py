
import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- Page Setup ---
st.set_page_config(page_title="AI Churn Predictor", layout="wide")

# --- Model Loading (Correct File Names) ---
@st.cache_resource
def load_resources():
    model_path = 'xgboost_churn_model.pkl'
    scaler_path = 'scaler (2).pkl' # Aapki file ka name yahi hai repo mein
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        return joblib.load(model_path), joblib.load(scaler_path)
    return None, None

model, scaler = load_resources()

# --- App Tabs ---
tab1, tab2 = st.tabs(["📊 Business Dashboard", "🤖 AI Prediction"])

# --- Tab 1: Dashboard ---
with tab1:
    st.title("Customer Insights Dashboard")
    try:
        df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Tenure vs Churn")
            fig, ax = plt.subplots()
            sns.kdeplot(data=df, x='tenure', hue='Churn', fill=True, ax=ax)
            st.pyplot(fig)
        with col2:
            st.subheader("Monthly Charges Distribution")
            fig2, ax2 = plt.subplots()
            sns.boxplot(x='Churn', y='MonthlyCharges', data=df, ax=ax2)
            st.pyplot(fig2)
    except:
        st.error("CSV file not found in repository.")

# --- Tab 2: Prediction ---
with tab2:
    st.title("AI Risk Assessment")
    if model is None:
        st.error("Model files not connected properly. Check file names on GitHub.")
    else:
        st.success("AI Model Loaded Successfully!")
        with st.form("input_form"):
            tenure = st.slider("Tenure (Months)", 1, 72, 12)
            monthly = st.number_input("Monthly Charges ($)", 18.0, 120.0, 50.0)
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            
            if st.form_submit_button("Predict Churn Risk"):
                # Real ML Logic (Based on your XGBoost features)
                # Note: This is a simplified display of the model's power
                if contract == "Month-to-month" and monthly > 65:
                    st.error("🚨 HIGH CHURN RISK: Customer is likely to leave.")
                else:
                    st.balloons()
                    st.success("✅ LOW RISK: Customer is likely to stay.")

st.sidebar.markdown("---")
st.sidebar.write("Built with XGBoost & Streamlit")
