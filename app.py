import streamlit as st
import pandas as pd
import joblib

# Set page configuration
st.set_page_config(page_title="Medical Insurance Cost Predictor", page_icon="🩺", layout="centered")

st.title("🩺 Medical Insurance Cost Estimator")
st.write("Enter patient details below to predict estimated annual medical charges.")

# Load saved preprocessor and model pipeline
@st.cache_resource
def load_assets():
    preprocessor = joblib.load('models/preprocessor.joblib')
    model = joblib.load('models/xgb_model.joblib')
    return preprocessor, model

preprocessor, model = load_assets()

# Feature Engineering function matching your Colab notebook
def create_engineered_features(df):
    df = df.copy()
    df['is_obese'] = (df['bmi'] >= 30).astype(int)
    smoker_binary = (df['smoker'] == 'yes').astype(int)
    df['obese_smoker'] = df['is_obese'] * smoker_binary
    df['age_smoker'] = df['age'] * smoker_binary
    df['is_parent'] = (df['children'] > 0).astype(int)
    return df

# Input Form
with st.form("patient_form"):
    st.subheader("Patient Attributes")
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Age", min_value=18, max_value=100, value=30)
        sex = st.selectbox("Sex", options=["female", "male"])
        children = st.number_input("Children", min_value=0, max_value=10, value=0)
        
    with col2:
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
        smoker = st.selectbox("Smoker Status", options=["no", "yes"])
        region = st.selectbox("Region", options=["southwest", "southeast", "northwest", "northeast"])
        
    submit = st.form_submit_button("Predict Charges 🚀")

if submit:
    # 1. Input DataFrame
    raw_input = pd.DataFrame([{
        'age': age,
        'sex': sex,
        'bmi': bmi,
        'children': children,
        'smoker': smoker,
        'region': region
    }])
    
    # 2. Engineer Features
    eng_input = create_engineered_features(raw_input)
    
    # 3. Transform and Predict
    transformed_input = preprocessor.transform(eng_input)
    prediction = model.predict(transformed_input)[0]
    
    st.markdown("---")
    st.subheader("Estimated Annual Charge")
    st.success(f"**${prediction:,.2f}**")