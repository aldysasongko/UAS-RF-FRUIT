import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

st.title('Fish Species Prediction App')
st.write("Masukkan panjang, berat, dan rasio panjang terhadap berat untuk memprediksi spesies ikan.")

# Input Form
diameter = st.slider('Panjang (cm)', min_value=0.0,max_value = 10.0, format="%.2f")
weight = st.slider('Berat (kg)', min_value=0.0,max_value = 150.0, format="%.2f")
red = st.slider('Merah', min_value=0.0,max_value = 255.0, format="%.2f")
green = st.slider('Hijau', min_value=0.0,max_value = 255.0, format="%.2f")
blue = st.slider('Biru', min_value=0.0,max_value = 255.0, format="%.2f")

if st.button('Prediksi'):
    if diameter > 0 and weight > 0 and red > 0 and green > 0 and blue > 0:
        # Load Saved Model and Scaler
        loaded_model = joblib.load('random_forest_model.pkl')
        loaded_scaler = joblib.load('scaler.pkl')
        loaded_label_encoder = joblib.load('label_encoder.pkl')
        
        # Predict Input dari User
        input_features = [[diameter, weight, red, green, blue]]
        input_scaled = loaded_scaler.transform(input_features)
        prediction = loaded_model.predict(input_scaled)

        # Hasil Prediksi
        predicted_species = loaded_label_encoder.inverse_transform(prediction)
        st.success(f"Species yang diprediksi: {predicted_species[0]}")
    else:
        st.error('Harap masukkan nilai yang valid untuk semua input.')
