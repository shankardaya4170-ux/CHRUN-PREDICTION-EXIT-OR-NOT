
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page Configuration
st.set_page_config(page_title="Churn Predictor Pro", layout="wide")
st.title("🚀 Customer Churn AI Predictor & Dashboard")

# Data Loading
@st.cache_data
def load_data():
    # Make sure the filename matches exactly
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    return df

try:
    data = load_data()

    tab1, tab2 = st.tabs(["📊 Analysis Dashboard", "🤖 AI Prediction"])

    with tab1:
        st.header("Business Insights")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Customers", len(data))
        col2.metric("Churn Rate", f"{(data['Churn'] == 'Yes').mean():.1%}")
        col3.metric("Average Tenure", f"{data['tenure'].mean():.1f} Months")

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Tenure vs Churn")
            fig1, ax1 = plt.subplots()
            sns.histplot(data=data, x='tenure', hue='Churn', multiple="stack", ax=ax1)
            st.pyplot(fig1)
        with c2:
            st.subheader("Contract Type Distribution")
            fig2, ax2 = plt.subplots()
            data['Contract'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax2)
            st.pyplot(fig2)

    with tab2:
        st.header("Predict Customer Exit")
        col_a, col_b = st.columns(2)
        with col_a:
            tenure = st.slider("Tenure (Months)", 0, 72, 12)
            monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 150.0, 50.0)
        with col_b:
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])

        if st.button("Predict Churn Status"):
            # Simple Prediction Logic
            if contract == "Month-to-month" and tenure < 12:
                st.error("⚠️ HIGH RISK: This customer is likely to leave!")
            else:
                st.success("✅ LOW RISK: This customer seems loyal.")

except Exception as e:
    st.error(f"Error loading data: {e}")

st.sidebar.info("Project by Dayashankar")
