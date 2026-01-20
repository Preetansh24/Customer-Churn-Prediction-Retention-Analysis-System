"""
Customer Churn Prediction - Streamlit Web Application
A visually appealing UI for predicting customer churn
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium styling
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    
    /* Headers */
    h1 {
        background: linear-gradient(90deg, #00d4ff, #7b2cbf);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
    }
    
    h2, h3 {
        color: #e0e0e0 !important;
    }
    
    /* Cards/containers */
    .metric-card {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 25px;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 8px 32px 0 rgba(0,0,0,0.3);
        margin-bottom: 20px;
    }
    
    /* Risk badges */
    .risk-high {
        background: linear-gradient(135deg, #ff4757, #c44569);
        color: white;
        padding: 10px 25px;
        border-radius: 25px;
        font-weight: bold;
        font-size: 18px;
        display: inline-block;
        box-shadow: 0 4px 15px rgba(255,71,87,0.4);
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #ffa502, #ff7f50);
        color: white;
        padding: 10px 25px;
        border-radius: 25px;
        font-weight: bold;
        font-size: 18px;
        display: inline-block;
        box-shadow: 0 4px 15px rgba(255,165,2,0.4);
    }
    
    .risk-low {
        background: linear-gradient(135deg, #2ed573, #1e90ff);
        color: white;
        padding: 10px 25px;
        border-radius: 25px;
        font-weight: bold;
        font-size: 18px;
        display: inline-block;
        box-shadow: 0 4px 15px rgba(46,213,115,0.4);
    }
    
    /* Probability gauge */
    .gauge-container {
        text-align: center;
        padding: 30px;
    }
    
    .probability-value {
        font-size: 72px;
        font-weight: 800;
        background: linear-gradient(135deg, #00d4ff, #7b2cbf);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .probability-label {
        font-size: 18px;
        color: #a0a0a0;
        margin-top: 10px;
    }
    
    /* Factor bars */
    .factor-bar {
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
        height: 25px;
        margin: 10px 0;
        overflow: hidden;
    }
    
    .factor-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 0.5s ease;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #00d4ff, #7b2cbf) !important;
        color: white !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 15px 40px !important;
        font-weight: bold !important;
        font-size: 16px !important;
        box-shadow: 0 4px 15px rgba(0,212,255,0.3) !important;
        transition: transform 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(0,212,255,0.4) !important;
    }
    
    /* Select boxes and inputs */
    .stSelectbox > div > div {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .stNumberInput > div > div > input {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        color: white;
    }
    
    /* Text colors */
    p, label {
        color: #e0e0e0 !important;
    }
    
    /* Dividers */
    hr {
        border-color: rgba(255,255,255,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    model_path = 'model/churn_model.pkl'
    feature_path = 'model/feature_names.pkl'
    
    if not os.path.exists(model_path):
        return None, None
    
    model = joblib.load(model_path)
    features = joblib.load(feature_path)
    return model, features

model, feature_names = load_model()

# Title
st.markdown("<h1 style='text-align: center; font-size: 48px;'>📊 Customer Churn Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px; color: #a0a0a0;'>AI-Powered Customer Retention Analysis</p>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

if model is None:
    st.error("⚠️ Model not found! Please run `python model_trainer.py` first to train the model.")
    st.stop()

# Sidebar - Customer Input Form
with st.sidebar:
    st.markdown("### 👤 Customer Information")
    st.markdown("---")
    
    # Demographics
    st.markdown("**Demographics**")
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    
    st.markdown("---")
    st.markdown("**Account Info**")
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check",
        "Mailed check", 
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ])
    
    st.markdown("---")
    st.markdown("**Services**")
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    if phone_service == "Yes":
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No"])
    else:
        multiple_lines = "No phone service"
    
    internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    
    if internet_service != "No":
        online_security = st.selectbox("Online Security", ["Yes", "No"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No"])
        device_protection = st.selectbox("Device Protection", ["Yes", "No"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No"])
    else:
        online_security = online_backup = device_protection = "No internet service"
        tech_support = streaming_tv = streaming_movies = "No internet service"
    
    st.markdown("---")
    st.markdown("**Charges**")
    monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 50.0, step=5.0)
    total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, monthly_charges * tenure, step=50.0)

# Main content
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    predict_btn = st.button("🔮 Predict Churn", use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

if predict_btn:
    # Prepare input data
    input_data = pd.DataFrame({
        'tenure': [tenure],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges],
        'gender': [gender],
        'SeniorCitizen': [senior_citizen],
        'Partner': [partner],
        'Dependents': [dependents],
        'PhoneService': [phone_service],
        'MultipleLines': [multiple_lines],
        'InternetService': [internet_service],
        'OnlineSecurity': [online_security],
        'OnlineBackup': [online_backup],
        'DeviceProtection': [device_protection],
        'TechSupport': [tech_support],
        'StreamingTV': [streaming_tv],
        'StreamingMovies': [streaming_movies],
        'Contract': [contract],
        'PaperlessBilling': [paperless_billing],
        'PaymentMethod': [payment_method]
    })
    
    # Make prediction
    probability = model.predict_proba(input_data)[0][1]
    prediction = model.predict(input_data)[0]
    
    # Determine risk category
    if probability >= 0.6:
        risk_category = "HIGH"
        risk_class = "risk-high"
    elif probability >= 0.3:
        risk_category = "MEDIUM"
        risk_class = "risk-medium"
    else:
        risk_category = "LOW"
        risk_class = "risk-low"
    
    # Get feature importances
    try:
        importances = model.named_steps['classifier'].feature_importances_
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False).head(5)
    except:
        importance_df = None
    
    # Display results
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="gauge-container">
                <div class="probability-value">{:.0%}</div>
                <div class="probability-label">Churn Probability</div>
            </div>
        </div>
        """.format(probability), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card" style="text-align: center;">
            <h3 style="margin-bottom: 20px;">Risk Category</h3>
            <div class="{}">⚠️ {} RISK</div>
            <p style="margin-top: 20px; font-size: 14px;">
                {} customer retention intervention
            </p>
        </div>
        """.format(
            risk_class, 
            risk_category,
            "Requires immediate" if risk_category == "HIGH" else 
            "Consider" if risk_category == "MEDIUM" else "No immediate"
        ), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card" style="text-align: center;">
            <h3 style="margin-bottom: 20px;">Prediction</h3>
            <div style="font-size: 48px; margin: 20px 0;">
                {}
            </div>
            <p style="font-size: 16px; color: #a0a0a0;">
                {}
            </p>
        </div>
        """.format(
            "🚨" if prediction == 1 else "✅",
            "Will Likely Churn" if prediction == 1 else "Likely to Stay"
        ), unsafe_allow_html=True)
    
    # Key Factors
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🔍 Key Factors Influencing Prediction")
    
    if importance_df is not None:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        
        colors = ['#00d4ff', '#7b2cbf', '#ff4757', '#ffa502', '#2ed573']
        
        for idx, row in importance_df.iterrows():
            # Clean up feature name
            feature_name = row['Feature'].replace('_', ' ').title()
            importance_pct = row['Importance'] * 100 / importance_df['Importance'].max()
            
            col1, col2 = st.columns([3, 2])
            with col1:
                st.markdown(f"**{feature_name}**")
            with col2:
                st.progress(importance_pct / 100)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Recommendations
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 💡 Retention Recommendations")
    
    recommendations = []
    if contract == "Month-to-month":
        recommendations.append("🎯 Offer discounted annual contract upgrade")
    if payment_method == "Electronic check":
        recommendations.append("💳 Incentivize switch to automatic payment methods")
    if tenure < 12:
        recommendations.append("🌟 Implement new customer engagement program")
    if online_security == "No" and internet_service != "No":
        recommendations.append("🔒 Bundle Online Security service at discount")
    if tech_support == "No" and internet_service != "No":
        recommendations.append("🛠️ Offer complimentary Tech Support trial")
    
    if not recommendations:
        recommendations.append("✅ Customer profile looks healthy - continue standard engagement")
    
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    for rec in recommendations:
        st.markdown(f"- {rec}")
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<p style='text-align: center; color: #666; font-size: 12px;'>
    Powered by Machine Learning | Customer Churn Prediction System
</p>
""", unsafe_allow_html=True)
