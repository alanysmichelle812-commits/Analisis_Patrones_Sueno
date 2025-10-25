import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os
import numpy as np

# --- 0. CONFIGURACIÓN DE LA RUTA Y VARIABLES ---
# Usamos la ruta RELATIVA, que funciona en Streamlit Cloud
DATA_FILE = "Sleep_health_and_lifestyle_dataset.csv"

# Definiciones para codificación (necesarias para la predicción)
OCCUPATION_MAP = {
    'Scientist': 0, 'Sales Representative': 1, 'Software Engineer': 2, 'Doctor': 3,
    'Engineer': 4, 'Accountant': 5, 'Nurse': 6, 'Teacher': 7, 'Broker': 8, 'Lawyer': 9,
    'Manager': 10
}
BMI_MAP = {'Overweight': 0, 'Normal': 1, 'Normal Weight': 2, 'Obese': 3}


# =================================================================
# 1. CARGA Y ENTRENAMIENTO DEL MODELO
# =================================================================
@st.cache_data(show_spinner="Entrenando modelo y preparando datos...")
def get_trained_model(data_file_name):
    """Carga los datos y entrena el modelo Random Forest."""

    if not os.path.exists(data_file_name):
        return None, f"Error: El archivo de datos '{data_file_name}' no se encontró en el servidor."

    df = pd.read_csv(data_file_name)

    # Preprocesamiento rápido (igual al usado en el script 01_preparacion_datos.py)
    df['ESTRES_BINARIO'] = df['Stress Level'].apply(lambda x: 1 if x >= 7 else 0)
    df.drop(['Person ID', 'Stress Level', 'Quality of Sleep', 'Blood Pressure'], axis=1, inplace=True)

    # Codificación de variables categóricas (usando los mapas predefinidos)
    df['Gender_Encoded'] = df['Gender'].apply(lambda x: 1 if x == 'Male' else 0)
    df['BMI_Encoded'] = df['BMI Category'].map(BMI_MAP).fillna(df['BMI Category'].mode()[0]).astype(int)
    df['Occupation_Encoded'] = df['Occupation'].map(OCCUPATION_MAP).fillna(df['Occupation'].mode()[0]).astype(int)
    
    # Manejo de Sleep Disorder
    df['SleepDisorder_Encoded'] = df['Sleep Disorder'].fillna('None').apply(lambda x: 0 if x == 'None' else (1 if x == 'Insomnia' else 2))

    df.drop(['Gender', 'BMI Category', 'Occupation', 'Sleep Disorder'], axis=1, inplace=True)

    # Preparar para el modelo (usando las columnas finales)
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

        with col2:
            bmi_input = st.selectbox("Categoría BMI", list(BMI_MAP.keys()))
            physical_activity = st.slider("Nivel de Actividad Física (min/día)", min_value=0, max_value=150, value=60)
            heart_rate = st.slider("Frecuencia Cardíaca (BPM)", min_value=50, max_value=90, value=70)
            daily_steps = st.slider("Pasos Diarios", min_value=1000, max_value=10000, value=5000)

        st.markdown("---")
        submitted = st.form_submit_button("Predecir Nivel de Estrés")

    # --- Lógica de Predicción ---
    if submitted:
        # Codificación de los inputs
        gender_encoded = 1 if gender_input == 'Male' else 0
        bmi_encoded = BMI_MAP.get(bmi_input, 1) # Usa 1 (Normal) si no se encuentra
        occupation_encoded = OCCUPATION_MAP.get(occupation_input, 0) # Usa 0 si no se encuentra (Scientist)

        # Crear el DataFrame de entrada (Asegurarse que las columnas coincidan con las de 'feature_names')
        input_data

