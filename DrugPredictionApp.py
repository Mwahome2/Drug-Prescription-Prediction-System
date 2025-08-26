# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 15:28:33 2025

@author: STUDENT
"""


import numpy as np
import pickle
import streamlit as st
import pandas as pd


# Loading the trained model
loaded_model = pickle.load(open("C:/Users/STUDENT/Desktop/DRUG PRESCRIPTION USING ML/RandomForest_model.pkl",'rb'))
def diagnosis(input_data_features): # Renamed argument for clarity
    # input_data_features should be a tuple or list of features (e.g., age, gender, symptom1, lab_result)
    # The example line below is for demonstration of expected input structure,
    # but in a real function call, 'input_data_features' would be passed as an argument.
    # For instance, if you're calling diagnosis((40, 0, 0, 0, 0.8)), then input_data_features would be that tuple.

    # Changing the input data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data_features)

    # Reshape the array as we are predicting for a single instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Make the prediction
    # 'loaded_model' and 'le' (label encoder) must be accessible in this scope.
    # Typically, they are loaded globally in a Streamlit app or passed as arguments.
    prediction = loaded_model.predict(input_data_reshaped)

    # Print the predicted drug (the output is the encoded drug)
    print('Predicted drug (encoded): {}'.format(prediction[0]))

    # To get the actual drug name, you would need the inverse mapping from the label encoder
    # used for the 'Drug' column during your model training.
    # Ensure 'le' is your loaded LabelEncoder instance.
    predicted_drug_name = le.inverse_transform(prediction)[0]
    print('Predicted drug: {}'.format(predicted_drug_name))

    return predicted_drug_name # Return the actual drug name for use in your app
def main():
    st.title("Drug Prescription Prediction System")
    
    # ... (code to load model and encoders)
    
    st.write("Please enter the patient's information:")
    
    # Get all required inputs from the user
    sex = st.selectbox("Sex", ["F", "M"])
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    blood_pressure = st.selectbox("Blood Pressure", ["HIGH", "NORMAL", "LOW"])
    cholesterol_level = st.selectbox("Cholesterol level", ["HIGH", "NORMAL"])
    na_to_k = st.number_input("Sodium to Potassium", min_value=0.0, format="%.2f", value=25.35)
    
    # THIS IS THE CRUCIAL PART: Create the dictionary
    input_data = {
        'Sex': sex,
        'Age': age,
        'Blood Pressure': blood_pressure,
        'Cholesterol level': cholesterol_level,
        'Sodium to Potassium': na_to_k,
    }
    
    if st.button("Predict Drug"):
        diagnosis_result = diagnosis(input_data, loaded_model, sex_encoder, bp_encoder, cholesterol_encoder)
        if diagnosis_result:
            st.success(f"Predicted Drug Type: **{diagnosis_result}**")
            
# This part must be at the very beginning of the line
if __name__ == '__main__':
    main()
           
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    