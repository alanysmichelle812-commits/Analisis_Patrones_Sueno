import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

# --- Configuración de la ruta absoluta del archivo de datos ---
# ¡Asegúrate de que esta ruta sea correcta para tu sistema!
DATA_FILE = "c:/Users/UserEliteBook/Documents/Proyecto_Sueno_IA/Sleep_health_and_lifestyle_dataset.csv"

# =================================================================
# 1. CARGA Y ENTRENAMIENTO DEL MODELO
# =================================================================
@st.cache_data(show_spinner="Entrenando modelo...")
def get_trained_model():
    if not os.path.exists(DATA_FILE):
        return None, f"Error: El archivo '{DATA_FILE}' no se encontró."
    
    df = pd.read_csv(DATA_FILE)

    # Preprocesamiento Rápido
    df['ESTRES_BINARIO'] = df['Stress Level'].apply(lambda x: 1 if x >= 7 else 0)
    df.drop(['Person ID', 'Stress Level', 'Quality of Sleep', 'Blood Pressure'], axis=1, inplace=True)
    
    le = LabelEncoder()
    df['Gender_Encoded'] = df['Gender'].apply(lambda x: 1 if x == 'Male' else 0)
    df['BMI_Encoded'] = le.fit_transform(df['BMI Category'])
    df['Occupation_Encoded'] = le.fit_transform(df['Occupation'])
    df['SleepDisorder_Encoded'] = df['Sleep Disorder'].fillna('None').apply(lambda x: 0 if x == 'None' else (1 if x == 'Insomnia' else 2))
    
    df.drop(['Gender', 'BMI Category', 'Occupation', 'Sleep Disorder'], axis=1, inplace=True)
    
    X = df.drop('ESTRES_BINARIO', axis=1)
    y = df['ESTRES_BINARIO']
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y) # Entrenamos con todos los datos disponibles
    
    return model, X.columns.tolist()

# Intenta cargar/entrenar el modelo
model, feature_names = get_trained_model()

# =================================================================
# 2. INTERFAZ DE STREAMLIT
# =================================================================
st.set_page_config(page_title="Predicción de Estrés", layout="centered")
st.title("🧠 Predictor de Nivel de Estrés por Hábitos de Sueño")

if model is None:
    st.error(feature_names) # Muestra el error de archivo si el modelo es None
else:
    st.subheader("Modelo Cargado con Éxito. ¡Ingresa tus datos!")
    
    # --- Formulario de Inputs ---
    with st.form("input_form"):
        # ... [Dejar aquí el resto del código del formulario que ya tenías]
        # (Para ahorrar espacio, asumimos que copiaste el resto del formulario)
        
        # Inputs simplificados para prueba:
        age = st.slider("Edad", min_value=18, max_value=80, value=35)
        sleep_duration = st.number_input("Duración del Sueño (horas)", min_value=4.0, max_value=10.0, value=7.5, step=0.1)
        
        submitted = st.form_submit_button("Predecir Nivel de Estrés")

    # --- Lógica de Predicción ---
    if submitted:
        # Aquí iría el resto de la lógica de codificación y predicción...

        st.success("¡El código de predicción se ejecutaría aquí!")
