import streamlit as st
import pandas as pd
import joblib

# Título
st.title("Clasificador de Enfermedad Renal Crónica (CKD)")

# Selector de modelo
model_choice = st.selectbox("Seleccione el modelo a utilizar:", 
                            ["decision_tree_model.pkl", "best_decision_tree_model.pkl"])

# Cargar el modelo
@st.cache_resource
def load_model(path):
    return joblib.load(path)

model = load_model(model_choice)

# Columnas de entrada esperadas
columns = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr',
           'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc',
           'htn', 'dm', 'cad', 'appet', 'pe', 'ane']

# Crear diccionario para inputs
input_data = {}

# Tipos de columnas categóricas
categorical_columns = {
    'rbc': ['normal', 'abnormal'],
    'pc': ['normal', 'abnormal'],
    'pcc': ['present', 'notpresent'],
    'ba': ['present', 'notpresent'],
    'htn': ['yes', 'no'],
    'dm': ['yes', 'no'],
    'cad': ['yes', 'no'],
    'appet': ['good', 'poor'],
    'pe': ['yes', 'no'],
    'ane': ['yes', 'no']
}

# Formulario de entrada
st.subheader("Ingrese los datos del paciente:")

with st.form("input_form"):
    for col in columns:
        if col in categorical_columns:
            input_data[col] = st.selectbox(f"{col}:", categorical_columns[col])
        else:
            input_data[col] = st.text_input(f"{col}:", key=col)

    submitted = st.form_submit_button("Predecir")

# Predicción
if submitted:
    try:
        # Convertir a DataFrame
        df = pd.DataFrame([input_data])

        # Convertir numéricos a float
        for col in df.columns:
            if col not in categorical_columns:
                df[col] = pd.to_numeric(df[col], errors='raise')

        # Realizar predicción
        prediction = model.predict(df)[0]
        pred_label = "ckd" if prediction == 0 else "notckd"

        st.success(f"Predicción: **{pred_label.upper()}**")

    except Exception as e:
        st.error(f"Error en los datos: {e}")
