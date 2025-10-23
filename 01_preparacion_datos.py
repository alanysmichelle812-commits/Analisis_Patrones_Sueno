import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.stats import iqr

sns.set_style("whitegrid")
pd.set_option('display.max_columns', None)

# -------------------------------------------------------------
# 1. CARGA, EXPLORACIÓN INICIAL Y TRANSFORMACIONES
# -------------------------------------------------------------
try:
    # Asegúrate de que este CSV esté en la misma carpeta
    df = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv") 
    print("✅ Dataset cargado correctamente.")
except FileNotFoundError:
    print("❌ ERROR: Archivo CSV no encontrado.")
    exit()

# Análisis inicial y Tratamiento de valores vacíos
print("\n--- 1. Exploración de Datos ---")
print(df.info())
print("\nValores nulos (se espera 0):")
print(df.isnull().sum())

# Transformaciones iniciales y eliminación de columna
df = df.drop('Person ID', axis=1) 
print("\n'Person ID' eliminado.")


# -------------------------------------------------------------
# 2. ANÁLISIS UNIVARIANTE Y DEDUCCIONES
# -------------------------------------------------------------

print("\n--- 2. Análisis Univariante: Distribución ---")
# Comentar/descomentar plt.show() para visualizar gráficos
num_cols = ['Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level', 'Heart Rate', 'Daily Steps']
for col in num_cols:
    # plt.figure(figsize=(6, 4)); sns.histplot(df[col], kde=True); plt.title(f'Distribución: {col}'); plt.show()
    pass


# -------------------------------------------------------------
# 3. FILTRADO DE VARIABLES (OUTLIERS POR IQR)
# -------------------------------------------------------------
print("\n--- 3. Filtrado de Outliers (IQR en Sleep Duration) ---")
Q1 = df['Sleep Duration'].quantile(0.25)
Q3 = df['Sleep Duration'].quantile(0.75)
IQR_val = Q3 - Q1
limite_inferior = Q1 - 1.5 * IQR_val
limite_superior = Q3 + 1.5 * IQR_val

df = df[(df['Sleep Duration'] >= limite_inferior) & (df['Sleep Duration'] <= limite_superior)].copy()
print(f"Registros después de filtrar outliers: {len(df)}")


# -------------------------------------------------------------
# 4. CREACIÓN Y TRATAMIENTO DE LA VARIABLE OBJETIVO
# -------------------------------------------------------------
print("\n--- 4. Creación de la Variable Objetivo 'ESTRES_BINARIO' ---")

# Creación de la variable objetivo binaria: 0 (3-6), 1 (7+)
df['ESTRES_BINARIO'] = np.where(df['Stress Level'] >= 7, 1, 0)

# Eliminar la variable numérica "nivel de estrés" (evitar data leakage)
df = df.drop('Stress Level', axis=1)
print(f"Variable 'Stress Level' numérica eliminada. Proporción objetivo:\n{df['ESTRES_BINARIO'].value_counts(normalize=True).round(3)}")


# -------------------------------------------------------------
# 5. ANÁLISIS BIVARIANTE Y MATRIZ DE CORRELACIÓN
# -------------------------------------------------------------

# Eliminación de variables por correlación/redundancia
df = df.drop(['Quality of Sleep', 'Blood Pressure'], axis=1) 
print("\n'Quality of Sleep' y 'Blood Pressure' eliminadas por correlación/complejidad.")


# -------------------------------------------------------------
# 6. CODIFICACIÓN Y DIVISIÓN ESTRATIFICADA (TRAIN/TEST)
# -------------------------------------------------------------

# Codificación de variables categóricas restantes para el modelo
le = LabelEncoder()
df['Gender_Encoded'] = le.fit_transform(df['Gender'])
df['BMI_Encoded'] = le.fit_transform(df['BMI Category'])
df['Occupation_Encoded'] = le.fit_transform(df['Occupation'])
df['SleepDisorder_Encoded'] = le.fit_transform(df['Sleep Disorder'].fillna('None')) # Rellenar nulos de Sleep Disorder

# Seleccionar variables finales (Codificadas y Numéricas)
features_to_model = [col for col in df.columns if 'Encoded' in col or df[col].dtype != 'object']
X_final = df[features_to_model].drop('ESTRES_BINARIO', axis=1)
y_final = df['ESTRES_BINARIO']

# División estratificada 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X_final, 
    y_final, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_final
)

# Comprobación de la estratificación
print("\n--- 6. Comprobación de la Estratificación ---")
print("Proporción en Train (debe ser mantenida):\n", y_train.value_counts(normalize=True).round(3))
print("Proporción en Test (debe ser mantenida):\n", y_test.value_counts(normalize=True).round(3))

# Guardar el dataset
train_set = pd.concat([X_train, y_train], axis=1)
test_set = pd.concat([X_test, y_test], axis=1)

train_set.to_csv("train_final.csv", index=False)
test_set.to_csv("test_final.csv", index=False)
print("\n✅ Archivos 'train_final.csv' y 'test_final.csv' guardados para modelización.")
