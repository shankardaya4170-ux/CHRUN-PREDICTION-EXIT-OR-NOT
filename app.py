import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Model aur Scaler Load karein
model = joblib.load('supply_chain_rf_model.pkl')
# Agar scaler save kiya hai toh load karein, varna skip karein
try:
    scaler = joblib.load('scaler.pkl')
except:
    scaler = None

st.set_page_config(page_title="Supply Chain Analytics", layout="wide")

st.title("🚚 Real-Time Supply Chain Delivery Predictor")
st.markdown("---")

# Layout: 2 Columns
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📋 Order Details")
    # Top features ke basis par inputs
    sales = st.number_input("Total Sales ($)", value=200.0)
    profit = st.number_input("Profit Per Order ($)", value=50.0)
    days_scheduled = st.slider("Scheduled Shipping Days", 0, 7, 4)
    region = st.selectbox("Order Region (ID)", options=list(range(20))) # Base on your encoding

with col2:
    st.header("📊 Prediction & Visualization")
    if st.button('Predict Delivery Risk'):
        # Input data format (matching your X columns)
        # Note: Aapko yahan apne exact column names use karne honge jo X_train mein the
        input_data = pd.DataFrame([[sales, profit, days_scheduled, region]], 
                                 columns=['Sales', 'Order Profit Per Order', 'Days for shipment (scheduled)', 'Order Region'])
        
        # Ye part dhyan dein: Agar columns mismatch ho toh model error dega. 
        # Isliye hum dummy columns fill kar rahe hain jo missing hain:
        full_input = pd.DataFrame(0, index=[0], columns=model.feature_names_in_)
        for col in input_data.columns:
            if col in full_input.columns:
                full_input[col] = input_data[col].values

        prediction = model.predict(full_input)[0]
        prob = model.predict_proba(full_input)[0]

        if prediction == 1:
            st.error(f"🔴 HIGH RISK: Model predicts LATE DELIVERY")
        else:
            st.success(f"🟢 LOW RISK: Model predicts ON-TIME DELIVERY")
            
        # Visualization
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.barplot(x=["On-Time", "Late"], y=[prob[0], prob[1]], palette=['#2ecc71', '#e74c3c'], ax=ax)
        st.pyplot(fig)
