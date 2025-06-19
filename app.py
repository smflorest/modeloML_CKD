import streamlit as st
import pandas as pd
import joblib
import os

# Imagen en la parte superior izquierda con texto al lado
st.markdown("""
    <div style="display: flex; align-items: center;">
        <img src="https://cancuncancerinstitute.com/wp-content/uploads/2023/02/Portada-datos-sobre-el-cancer-de-rinon.png" 
             alt="Riñón" width="80" style="margin-right: 20px;">
        <h1>Clasificador de Enfermedad Renal Crónica (CKD)</h1>
    </div>
""", unsafe_allow_html=True)

st.set_page_config(page_title="Predicción CKD", layout="centered")
st.title("🧠 Predicción de Enfermedad Renal Crónica (CKD)")
# Párrafo descriptivo
st.markdown("""
Esta aplicación permite predecir si un paciente tiene **Enfermedad Renal Crónica (CKD)** 
o no (**notckd**) usando un modelo de Árbol de Decisión previamente entrenado utilizando machine learning supervisado por clasificacion, con una exactitud del 0.99, basado en el UCI Irvine Machine learning repository (https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease). 
""")
st.markdown("""
Ingrese los datos clínicos del paciente, seleccione el modelo deseado, y haga clic en *Predecir*para predecir si presenta enfermedad renal crónica (CKD).
""")
st.write("NOTA: Esta aplicacion es con fines de entrenamiento y no para fines de uso clinico.")

# Selección de modelo
model_option = st.selectbox("📦 Selecciona el modelo a usar:",
                            ("decision_tree_model.pkl", "best_decision_tree_model.pkl"))
model_path = os.path.join(os.getcwd(), model_option)
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    st.error(f"El modelo '{model_option}' no se encuentra en el directorio.")
    st.stop()

# Definir columnas, unidades y tipo de dato
column_info = {
    'age': ("Edad", "años", "numérico"),
    'bp': ("Presión arterial", "mm Hg", "entero"),
    'sg': ("Gravedad específica (sg)", "g/mL", "numérico"),
    'al': ("Proteínas en orina (albumina)", "categoría: 0–5", "entero"),
    'su': ("Azúcar en orina", "categoría: 0–5", "entero"),
    'rbc': ("Glóbulos rojos", "normal / abnormal", "categórico"),
    'pc': ("Células epiteliales", "normal / abnormal", "categórico"),
    'pcc': ("Cilindros celulares", "present / notpresent", "categórico"),
    'ba': ("Bacterias", "present / notpresent", "categórico"),
    'bgr': ("Nivel de glucosa en sangre (bgr)", "mg/dL", "numérico"),
    'bu': ("Urea en sangre", "mg/dL", "numérico"),
    'sc': ("Creatinina sérica", "mg/dL", "numérico"),
    'sod': ("Sodio sérico", "mEq/L", "numérico"),
    'pot': ("Potasio sérico", "mEq/L", "numérico"),
    'hemo': ("Hemoglobina", "g/dL", "numérico"),
    'pcv': ("Volumen corpuscular", "%", "numérico"),
    'wbcc': ("Recuento de leucocitos", "células/µL", "numérico"),
    'rbcc': ("Recuento de eritrocitos", "millones/µL", "numérico"),
    'htn': ("Hipertensión", "yes / no", "categórico"),
    'dm': ("Diabetes", "yes / no", "categórico"),
    'cad': ("Enfermedad cardíaca", "yes / no", "categórico"),
    'appet': ("Apetito", "good / poor", "categórico"),
    'pe': ("Presencia de edema", "yes / no", "categórico"),
    'ane': ("Anemia", "yes / no", "categórico")
}

# Mapas de LabelEncoder
label_maps = {
    'rbc': {'normal': 1, 'abnormal': 0},
    'pc': {'normal': 1, 'abnormal': 0},
    'pcc': {'notpresent': 0, 'present': 1},
    'ba': {'notpresent': 0, 'present': 1},
    'htn': {'no': 0, 'yes': 1},
    'dm': {'no': 0, 'yes': 1},
    'cad': {'no': 0, 'yes': 1},
    'appet': {'good': 1, 'poor': 0},
    'pe': {'no': 0, 'yes': 1},
    'ane': {'no': 0, 'yes': 1}
}

# Formulario de ingreso
st.subheader("📝 Ingreso de datos clínicos")

input_data = {}

for col, (label, unit, tipo) in column_info.items():
    display_label = f"**{label}** ({unit}, {tipo})"
    if col in label_maps:
        input_data[col] = st.selectbox(display_label, list(label_maps[col].keys()))
    elif tipo == "entero":
        input_data[col] = st.number_input(display_label, step=1, format="%d")
    else:
        input_data[col] = st.number_input(display_label, step=0.1)

# Botón de predicción
if st.button("🔍 Predecir"):
    try:
        # Convertir datos categóricos a numéricos
        processed_data = {}
        for col in column_info:
            if col in label_maps:
                processed_data[col] = label_maps[col][input_data[col]]
            else:
                processed_data[col] = input_data[col]

        input_df = pd.DataFrame([processed_data])
        prediction = model.predict(input_df)[0]
        resultado = "✅ NO tiene CKD (notckd)" if prediction == 1 else "⚠️ Tiene CKD (ckd)"
        st.success(f"Resultado de la predicción: {resultado}")

        with st.expander("📋 Ver datos ingresados"):
            display_df = pd.DataFrame([input_data])
            st.dataframe(display_df)

    except Exception as e:
        st.error(f"Error al hacer la predicción: {e}")
