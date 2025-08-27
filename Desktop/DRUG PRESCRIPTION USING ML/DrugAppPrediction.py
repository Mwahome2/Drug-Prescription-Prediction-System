# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 20:41:47 2025

@author: STUDENT
"""

# Import all necessary libraries at the top
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- 1. Load Machine Learning Model and Encoders ---
# This section now references files directly, avoiding the 'script_dir' error.
loaded_model = None
label_encoders = {}  # Dictionary to hold all label encoders

try:
    # Use direct file names assuming they are in the same folder as the script
    # This is the key change to fix the NameError.
    model_path = 'drug_prediction_model.joblib'
    sex_encoder_path = 'sex_encoder.joblib'
    bp_encoder_path = 'bp_encoder.joblib'
    cholesterol_encoder_path = 'cholesterol_encoder.joblib'
    drug_encoder_path = 'drug_label_encoder.joblib'

    # Load the machine learning model
    loaded_model = joblib.load(open(model_path, 'rb'))

    # Load all individual encoders
    label_encoders['Sex'] = joblib.load(open(sex_encoder_path, 'rb'))
    label_encoders['BP'] = joblib.load(open(bp_encoder_path, 'rb'))
    label_encoders['Cholesterol'] = joblib.load(open(cholesterol_encoder_path, 'rb'))
    label_encoders['Drug'] = joblib.load(open(drug_encoder_path, 'rb'))

except Exception as e:
    # A clear error message is displayed to the user if anything fails to load
    st.error(f"Error loading model or encoders: {e} 😞. Please check that all files are in the same directory.")
    loaded_model = None


# --- 2. Create the Prediction Function ---
def predict_drug(age, sex, bp, cholesterol, na_to_k):
    """
    Predicts the drug based on input features using the loaded model and encoders.
    """
    if loaded_model is None:
        # Prevents the app from crashing if the model isn't loaded
        return "Model not loaded. Please contact the administrator."

    try:
        # Encode categorical inputs using the loaded encoders
        sex_encoded = label_encoders['Sex'].transform([sex])[0]
        bp_encoded = label_encoders['BP'].transform([bp])[0]
        cholesterol_encoded = label_encoders['Cholesterol'].transform([cholesterol])[0]

        # Create a DataFrame for prediction
        input_df = pd.DataFrame(
            [[age, sex_encoded, bp_encoded, cholesterol_encoded, na_to_k]],
            columns=['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']
        )
        
        # Make the prediction
        prediction_encoded = loaded_model.predict(input_df)[0]
        
        # Decode the prediction back to the original drug name
        predicted_drug = label_encoders['Drug'].inverse_transform([prediction_encoded])[0]

        return predicted_drug
    except Exception as e:
        # Catches any errors during the prediction process
        st.error(f"Error during prediction: {e}")
        return "Prediction failed."


# --- 3. Streamlit App Interface ---
# The app's layout and styling are defined here.

# Add custom CSS for improved aesthetics.
st.markdown(
    """
    <style>
    .reportview-container {
        background: linear-gradient(to right, #00c6ff, #0072ff);
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stApp {
        background-color: #f0f2f6;
    }
    .st-bw {
        border-radius: 10px;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
        transition: 0.3s;
    }
    .st-bw:hover {
        box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
    }
    .st-bv {
        padding: 20px;
        border-radius: 10px;
        background-color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("💊 AI-Assisted Drug Prescription Prediction Web App")
st.markdown("A Machine Learning powered system to predict the most suitable drug.")

# Create the input form for patient information
st.header("Patient Information")
with st.container():
    st.markdown('<div class="st-bv">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=35)
        # Use an index to handle initial state for selectbox
        sex_options = ['F', 'M']
        sex = st.selectbox("Sex", options=sex_options)

        bp_options = ['HIGH', 'NORMAL', 'LOW']
        bp = st.selectbox("Blood Pressure", options=bp_options)

    with col2:
        cholesterol_options = ['HIGH', 'NORMAL']
        cholesterol = st.selectbox("Cholesterol", options=cholesterol_options)
        na_to_k = st.number_input("Na_to_K Ratio", min_value=0.0, value=10.0)

    st.markdown('</div>', unsafe_allow_html=True)


# The prediction button
if st.button("Predict Drug"):
    if loaded_model:
        result = predict_drug(age, sex, bp, cholesterol, na_to_k)
        if result:
            st.success(f"The recommended drug is: **{result}**")
    else:
        st.warning("Please check the console for errors. The model could not be loaded.")

# Add a footer
st.markdown("---")
st.markdown("Created by [Your Name] for a Capstone Project")
