import streamlit as st
import pandas as pd
import joblib
import os

# Imagen y título
st.set_page_config(page_title="Predicción CKD", layout="centered")

st.markdown("""
    <div style="display: flex; align-items: center;">
        <img src="https://cancuncancerinstitute.com/wp-content/uploads/2023/02/Portada-datos-sobre-el-cancer-de-rinon.png" 
             alt="Riñón" width="160" style="margin-right: 40px;">
        <h1>Enfermedad Renal Crónica (CKD)</h1>
    </div>
""", unsafe_allow_html=True)

st.title("🧠 Modelo de Predicción de Enfermedad Renal Crónica (CKD)")

# Descripción
st.markdown("""
Esta aplicación permite predecir si un paciente tiene **Enfermedad Renal Crónica (CKD)** 
o no (**notckd**) usando un modelo de clasificación: Árbol de Decisión (exactitud 0.97) 
basado en el dataset de UCI Irvine.
""")

st.write("NOTA: Esta aplicación es con fines educativos, no clínicos.")

# Selección de modelo
model_option = st.selectbox("📦 Selecciona el modelo a usar:",
                            ("decision_tree_model.pkl", "best_decision_tree_model.pkl"))
model_path = os.path.join(os.getcwd(), model_option)
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    st.error(f"El modelo '{model_option}' no se encuentra en el directorio.")
    st.stop()

# Definir columnas
column_info = {
    'age': ("Edad", "años", "numérico"),
    'bp': ("Presión arterial", "mm Hg", "entero"),
    'sg': ("Gravedad específica (sg)", "g/mL", "numérico"),
    'al': ("Proteínas en orina (albumina)", "categoría: 0–5", "entero"),
    'su': ("Azúcar en orina", "categoría: 0–5", "entero"),
    'rbc': ("Glóbulos rojos", ["normal", "abnormal"]),
    'pc': ("Células epiteliales", ["normal", "abnormal"]),
    'pcc': ("Cilindros celulares", ["notpresent", "present"]),
    'ba': ("Bacterias", ["notpresent", "present"]),
    'bgr': ("Nivel de glucosa en sangre (bgr)", "mg/dL", "numérico"),
    'bu': ("Urea en sangre", "mg/dL", "numérico"),
    'sc': ("Creatinina sérica", "mg/dL", "numérico"),
    'sod': ("Sodio sérico", "mEq/L", "numérico"),
    'pot': ("Potasio sérico", "mEq/L", "numérico"),
    'hemo': ("Hemoglobina", "g/dL", "numérico"),
    'pcv': ("Volumen corpuscular", "%", "numérico"),
    'wbcc': ("Recuento de leucocitos", "células/µL", "numérico"),
    'rbcc': ("Recuento de eritrocitos", "millones/µL", "numérico"),
    'htn': ("Hipertensión", ["yes", "no"]),
    'dm': ("Diabetes", ["yes", "no"]),
    'cad': ("Enfermedad cardíaca", ["yes", "no"]),
    'appet': ("Apetito", ["good", "poor"]),
    'pe': ("Presencia de edema", ["yes", "no"]),
    'ane': ("Anemia", ["yes", "no"])
}

# Formulario de entrada
st.subheader("📝 Ingreso de datos clínicos")
input_data = {}

for col, info in column_info.items():
    label = info[0]
    if isinstance(info[1], list):  # Categórico
        input_data[col] = st.selectbox(f"**{label}**", options=info[1])
    else:  # Numérico
        tipo = info[2] if len(info) > 2 else "numérico"
        step = 1 if tipo == "entero" else 0.1
        input_data[col] = st.number_input(f"**{label}** ({info[1]}, {tipo})", step=step)

# Botón de predicción
if st.button("🔍 Predecir"):
    try:
        # Crear DataFrame y aplicar get_dummies
        input_df = pd.DataFrame([input_data])
        input_df = pd.get_dummies(input_df)

        # Asegurar columnas del modelo
        for col in model.feature_names_in_:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[model.feature_names_in_]  # Reordenar columnas

        # Predicción
        prediction = model.predict(input_df)[0]
        resultado = "✅ NO tiene CKD (notckd)" if prediction == 1 else "⚠️ Tiene CKD (ckd)"
        st.success(f"Resultado de la predicción: {resultado}")

        with st.expander("📋 Ver datos ingresados"):
            st.dataframe(pd.DataFrame([input_data]))

    except Exception as e:
        st.error(f"Error al hacer la predicción: {e}")

# Footer
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
