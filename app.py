import streamlit as st
import pandas as pd
import joblib
import requests
import io

# URL directa al archivo .pkl en GitHub (usa el enlace RAW)
MODEL_URL = "https://github.com/jesusalvarado2023/prueba_borrar/raw/refs/heads/main/decision_tree_model.pkl" 

def load_model():
    response = requests.get(MODEL_URL)
    if response.status_code != 200:
        raise ValueError("No se pudo descargar el modelo desde GitHub.")
    model = joblib.load(io.BytesIO(response.content))
    return model

# Cargar modelo
model = load_model()

# Título
st.title("Clasificador de Enfermedad Renal Crónica (CKD)")

# Campos requeridos por el modelo
columnas = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc']
input_data = {}

st.subheader("Ingrese los datos del paciente:")

# Entradas de usuario
for col in columnas:
    input_data[col] = st.number_input(f"{col}", format="%.2f")

# Botón para predecir
if st.button("Predecir clase"):
    # Crear DataFrame
    df_input = pd.DataFrame([input_data])
    
    # Realizar predicción
    prediction = model.predict(df_input)[0]
    
    # Mostrar resultado
    st.success(f"Predicción: **{prediction}**")
