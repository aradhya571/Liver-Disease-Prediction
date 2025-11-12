# streamlit_app.py

import streamlit as st
import joblib
import pandas as pd

# Load the pre-trained model and scaler
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
poly = joblib.load("poly.pkl")

# App title and description
st.title("Liver Disease Prediction App")
st.write("This app predicts the likelihood of liver disease based on user inputs.")

# Collecting input features from the user
age = st.slider("Age", min_value=20, max_value=80, value=50, step=1)
gender = st.selectbox("Gender", ["Female", "Male"])
bmi = st.slider("BMI", min_value=15.0, max_value=40.0, value=27.7)
alcohol_consumption = st.slider("Alcohol Consumption Level", min_value=0.0, max_value=20.0, value=9.8)
smoking = st.selectbox("Smoking", ["No", "Yes"])
genetic_risk = st.selectbox("Genetic Risk", ["Low", "Medium", "High"])
physical_activity = st.slider("Physical Activity Level", min_value=0.0, max_value=10.0, value=5.0)
diabetes = st.selectbox("Diabetes", ["No", "Yes"])
hypertension = st.selectbox("Hypertension", ["No", "Yes"])
liver_function_test = st.slider("Liver Function Test Result", min_value=20.0, max_value=100.0, value=59.9)

# Convert categorical features to numerical values
gender = 1 if gender == "Male" else 0
smoking = 1 if smoking == "Yes" else 0
diabetes = 1 if diabetes == "Yes" else 0
hypertension = 1 if hypertension == "Yes" else 0
genetic_risk_mapping = {"Low": 0, "Medium": 1, "High": 2}
genetic_risk = genetic_risk_mapping[genetic_risk]

# Create a DataFrame for the input data
input_data = pd.DataFrame({
    'Age': [age],
    'Gender': [gender],
    'BMI': [bmi],
    'AlcoholConsumption': [alcohol_consumption],
    'Smoking': [smoking],
    'GeneticRisk': [genetic_risk],
    'PhysicalActivity': [physical_activity],
    'Diabetes': [diabetes],
    'Hypertension': [hypertension],
    'LiverFunctionTest': [liver_function_test]
})

# Apply scaling and polynomial transformation
input_data_scaled = scaler.transform(input_data)
input_data_poly = poly.transform(input_data_scaled)

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data_poly)
    diagnosis = "Liver Disease Detected" if prediction[0] == 1 else "No Liver Disease Detected"
    st.write(f"Prediction: **{diagnosis}**")
