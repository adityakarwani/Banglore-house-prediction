import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load transformed data
st.title("Bangalore House Price Prediction")
st.write("Loading transformed data...")
df = pickle.load(open('df.pkl', 'rb'))

# Display the first few rows of the dataframe
st.write(df.head())

# Load the pre-trained model
model = pickle.load(open('Ridge_model.pkl', 'rb'))

# User input
st.write("Enter the details to predict the house price:")

location = st.selectbox("Location", df['location'].unique())
total_sqft = st.number_input("Total Square Feet", min_value=300)
bath = st.number_input("Number of Bathrooms", min_value=1)
bhk = st.number_input("Number of Bedrooms", min_value=1)

# Prediction
if st.button("Predict"):
    input_data = pd.DataFrame([[location, total_sqft, bath, bhk]], columns=['location', 'total_sqft', 'bath', 'bhk'])
    prediction = model.predict(input_data)
    st.write(f"Predicted Price: {prediction[0]:.2f}lakhs")

