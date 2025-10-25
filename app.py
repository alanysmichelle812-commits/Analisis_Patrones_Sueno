import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os
import numpy as np

# --- 0. CONFIGURACIÓN DE LA RUTA Y VARIABLES ---
# Usamos la ruta RELATIVA, que funciona en Streamlit Cloud
DATA_FILE = "Sleep_health_and_lifestyle_dataset.csv"

# Definiciones para codificación (deben coincidir con el preprocesamiento)
OCCUPATION_MAP = {
    'Scientist': 0, 'Sales Representative': 1, 'Software Engineer': 2, 'Doctor': 3,
    'Engineer': 4, 'Accountant': 5, 'Nurse': 6, 'Teacher': 7, 'Broker': 8, 'Lawyer': 9,
    'Manager': 10
}
BMI_MAP = {'Overweight': 0, 'Normal': 1, 'Normal Weight': 2, 'Obese': 3}


# =================================================================
# 1. CARGA Y ENTRENAMIENTO DEL MODELO (CORREGIDO el ValueError)
# =================================================================
@st.cache_data(show_spinner="Entrenando modelo y preparando datos...")
def get_trained_model(data_file_name):
    """Carga los datos y entrena el modelo Random Forest."""

    if not os.path.exists(data_file_name):
        return None, f"Error: El archivo de datos '{data_file_name}' no se encontró en el servidor."

    df = pd.read_csv(data_file_name)

    # Preprocesamiento rápido
    df['ESTRES_BINARIO'] = df['Stress Level'].apply(lambda x: 1 if x >= 7 else 0)
    df.drop(['Person ID', 'Stress Level', 'Quality of Sleep', 'Blood Pressure'], axis=1, inplace=True)

    # Codificación de Ocupación
    df['Occupation_Encoded'] = df['Occupation'].map(OCCUPATION_MAP)
    moda_codificada_occ = df['Occupation_Encoded'].mode()[0]
    df['Occupation_Encoded'] = df['Occupation_Encoded'].fillna(moda_codificada_occ).astype(int)
    
    # Codificación de BMI
    df['BMI_Encoded'] = df['BMI Category'].map(BMI_MAP)
    moda_codificada_bmi = df['BMI_Encoded'].mode()[0]
    df['BMI_Encoded'] = df['BMI_Encoded'].fillna(moda_codificada_bmi).astype(int)

    # Codificación de Género y Sleep Disorder
    df['Gender_Encoded'] = df['Gender'].apply(lambda x: 1 if x == 'Male' else 0)
    df['SleepDisorder_Encoded'] = df['Sleep Disorder'].fillna('None').apply(lambda x: 0 if x == 'None' else (1 if x == 'Insomnia' else 2))

    df.drop(['Gender', 'BMI Category', 'Occupation', 'Sleep Disorder'], axis=1, inplace=True)

    # Preparar para el modelo
    X = df.drop('ESTRES_BINARIO', axis=1)
    y = df['ESTRES_BINARIO']

    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    return model, X.columns.tolist()

# Intenta cargar/entrenar el modelo
model, feature_names = get_trained_model(DATA_FILE)

# =================================================================
# 2. INTERFAZ Y LÓGICA DE STREAMLIT
# =================================================================
st.set_page_config(page_title="Predicción de Estrés", layout="centered")
st.title("🧠 Predictor de Nivel de Estrés por Hábitos de Sueño")
st.markdown("---")


if model is None:
    st.error(feature_names) # Muestra el mensaje de error si el archivo no se encontró
else:
    st.subheader("Ingresa tus datos para predecir tu nivel de estrés")

    # --- Formulario de Inputs ---
    with st.form("input_form"):
        # Secciones de input
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Edad", min_value=18, max_value=80, value=35)
            gender_input = st.selectbox("Género", ["Male", "Female"])
            occupation_input = st.selectbox("Ocupación", list(OCCUPATION_MAP.keys()))
            sleep_duration = st.slider("Duración del Sueño (horas)", min_value=4.0, max_value=10.0, value=7.5, step=0.1)
            sleep_disorder = st.selectbox("Trastorno del Sueño (Usado para modelo)", ["None", "Insomnia", "Sleep Apnea"])


        with col2:
            bmi_input = st.selectbox("Categoría BMI", list(BMI_MAP.keys()))
            physical_activity = st.slider("Nivel de Actividad Física (min/día)", min_value=0, max_value=150, value=60)
            heart_rate = st.slider("Frecuencia Cardíaca (BPM)", min_value=50, max_value=90, value=70)
            # LÍNEA CORREGIDA: Se asegura el cierre del paréntesis.
            daily_steps = st.slider("Pasos Diarios", min_value=1000, max_value=10000, value=5000)

        st.markdown("---")
        submitted = st.form_submit_button("Predecir Nivel de Estrés")

    # --- Lógica de Predicción ---
    if submitted:
        # Codificación de los inputs de usuario
        gender_encoded = 1 if gender_input == 'Male' else 0
        bmi_encoded = BMI_MAP.get(bmi_input, 1)
        occupation_encoded = OCCUPATION_MAP.get(occupation_input, 0)
        
        if sleep_disorder == "None":
            sleep_disorder_encoded = 0
        elif sleep_disorder == "Insomnia":
            sleep_disorder_encoded = 1
        else: # Sleep Apnea
            sleep_disorder_encoded = 2


        # Crear el DataFrame de entrada
        input_data = pd.DataFrame([[
            age, 
            sleep_duration, 
            physical_activity, 
            heart_rate, 
            daily_steps, 
            gender_encoded, 
            bmi_encoded, 
            occupation_encoded, 
            sleep_disorder_encoded
        ]], columns=feature_names)

        # Predicción (con indentación corregida)
        try:
            prediction = model.predict(input_data)[0]
            prediction_proba = model.predict_proba(input_data)[0]

            # Presentación del Resultado
            st.markdown("### Resultado de la Predicción")
            if prediction == 1:
                st.error(f"⚠️ **ALERTA: Alto Nivel de Estrés**")
                st.write(f"Probabilidad de Estrés Alto: **{prediction_proba[1]:.2f}**")
                st.info("Considera ajustar tus hábitos de sueño y actividad física para reducir el riesgo.")
            else:
                st.success(f"✅ **Bajo Nivel de Estrés**")
                st.write(f"Probabilidad de Estrés Bajo: **{prediction_proba[0]:.2f}**")
                st.info("¡Tus hábitos de sueño sugieren un nivel de estrés bajo. Sigue así!")
        
        except Exception as e:
            st.error(f"Ocurrió un error durante la predicción. Por favor, verifica tus inputs. Error: {e}")

