import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

st.title('Fish Species Prediction App')
st.write("Masukkan panjang, berat, dan rasio panjang terhadap berat untuk memprediksi spesies ikan.")

# Input Form
length = st.slider('Panjang (cm)', min_value=0.0, format="%.2f")
weight = st.slider('Berat (kg)', min_value=0.0, format="%.2f")
w_l_ratio = st.slider('Rasio Panjang terhadap Berat', min_value=0.0, format="%.2f")

if st.button('Prediksi'):
    if length > 0 and weight > 0 and w_l_ratio > 0:
        # Load Saved Model and Scaler
        loaded_model = joblib.load('random_forest_model.pkl')
        loaded_scaler = joblib.load('scaler.pkl')
        loaded_label_encoder = joblib.load('label_encoder.pkl')
        
        # Predict Input dari User
        input_features = [[length, weight, w_l_ratio]]
        input_scaled = loaded_scaler.transform(input_features)
        prediction = loaded_model.predict(input_scaled)

        # Hasil Prediksi
        predicted_species = loaded_label_encoder.inverse_transform(prediction)
        st.success(f"Species yang diprediksi: {predicted_species[0]}")
    else:
        st.error('Harap masukkan nilai yang valid untuk semua input.')