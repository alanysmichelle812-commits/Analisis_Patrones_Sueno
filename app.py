import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

# --- 0. CONFIGURACIÓN DE LA RUTA Y VARIABLES GLOBALES ---
# Usamos la ruta RELATIVA, que funciona en Streamlit Cloud
DATA_FILE = "Sleep_health_and_lifestyle_dataset.csv"

# =================================================================
# 1. CARGA Y ENTRENAMIENTO DEL MODELO
# =================================================================
# La función de cache maneja la carga y entrenamiento solo una vez.
@st.cache_data(show_spinner="Entrenando modelo...")
def get_trained_model(data_file_name):
    """Carga los datos y entrena el modelo Random Forest."""
    
    # Comprobación de existencia del archivo en el servidor
    if not os.path.exists(data_file_name):
        return None, f"Error: El archivo de datos '{data_file_name}' no se encontró en el servidor de Streamlit Cloud. Verifica que fue subido a GitHub."
    
    df = pd.read_csv(data_file_name)

    # Preprocesamiento Rápido
    df['ESTRES_BINARIO'] = df['Stress Level'].apply(lambda x: 1 if x >= 7 else 0)
    df.drop(['Person ID', 'Stress Level', 'Quality of Sleep', 'Blood Pressure'], axis=1, inplace=True)
    
    # Codificación (Label Encoding)
    le = LabelEncoder()
    df['Gender_Encoded'] = df['Gender'].apply(lambda x: 1 if x == 'Male' else 0)
    df['BMI_Encoded'] = le.fit_transform(df['BMI Category'])
    df['Occupation_Encoded'] = le.fit_transform(df['Occupation'])
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
# 2. INTERFAZ DE STREAMLIT
# =================================================================
st.set_page_config(page_title="Predicción de Estrés", layout="centered")
st.title("🧠 Predictor de Nivel de Estrés por Hábitos de Sueño")

if model is None:
    # Muestra el error de archivo si la función get_trained_model() falló
    st.error(feature_names) 
else:
    st.subheader("Modelo Cargado con Éxito. ¡Ingresa tus datos!")
    
    # --- Formulario de Inputs ---
    with st.form("input_form"):
        # Inputs simplificados (debes agregar todos tus sliders y selects aquí)
        age = st.slider("Edad", min_value=18, max_value=80, value=35)
        sleep_duration = st.number_input("Duración del Sueño (horas)", min_value=4.0, max_value=10.0, value=7.5, step=0.1)
        
        submitted = st.form_submit_button("Predecir Nivel de Estrés")

    # --- Lógica de Predicción ---
    if submitted:
        # Aquí iría el resto de la lógica de codificación y predicción...
        st.success("¡El código de predicción se ejecutaría aquí!")

