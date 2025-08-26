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
    #giving a title
    st.title("Drug Prescription App")
    
    #getting the input data from the user
    Age= st.text_input("Age")
    Sex= st.text_input("Sex")
    BP= st.text_input("Blood Pressure")
    Cholesterol= st.text_input("Cholesterol level")
    Na_to_K= st.text_input("Sodium to Potassium")
    
    input_data= [
        Age,
        Sex,
        BP,
        Cholesterol,
        Na_to_K
        ]
    
    
    if st.button("Drug Prescription Result"):
        diagnosis_result = diagnosis(input_data)
        st.success(diagnosis_result)

if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    