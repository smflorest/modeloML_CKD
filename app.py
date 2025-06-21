import streamlit as st
import pandas as pd
import joblib
import os

# Imagen en la parte superior izquierda con texto al lado
st.markdown("""
    <div style="display: flex; align-items: center;">
        <img src="https://cancuncancerinstitute.com/wp-content/uploads/2023/02/Portada-datos-sobre-el-cancer-de-rinon.png" 
             alt="Riñón" width="160" style="margin-right: 40px;">
        <h1>Enfermedad Renal Crónica (CKD)</h1>
    </div>
""", unsafe_allow_html=True)

st.set_page_config(page_title="Predicción CKD", layout="centered")
st.title("🧠 Modelo de Predicción de Enfermedad Renal Crónica (CKD)")
# Párrafo descriptivo
st.markdown("""
Esta aplicación permite predecir si un paciente tiene **Enfermedad Renal Crónica (CKD)** 
o no (**notckd**) usando un modelo de inteligencia artificial utilizando machine learning supervisado por clasificacion: Árbol de Decisión con una exactitud del 0.97, y basado en el dataset de UCI Irvine Machine learning repository (https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease). 
""")

st.markdown("""
Ingrese los datos clínicos del paciente, seleccione el modelo deseado, y haga clic en *Predecir*para predecir si presenta enfermedad renal crónica (CKD).
""")

st.write("NOTA: Esta aplicacion es con fines de entrenamiento y no con fines de uso clinico.")

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
    
# Footer con nombres alineados a la izquierda
st.markdown("""
    <div style='position: fixed; bottom: 10px; left: 20px; text-align: right; color: gray; font-size: 14px;'>
        <p><b>Integrantes:</b><br>
        Silvia Flores Toledo<br>
        Sarina Ramos Zuniga<br>
        Jose C Jara Aguirre<br>
        Vladimir Villoslada Terrones<br>
        Jose Luis Vargas</p>
    </div>
""", unsafe_allow_html=True)
