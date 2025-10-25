import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression # Modelo Base
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import joblib # Para guardar el modelo

# =================================================================
# 1. CARGA DE DATOS
# =================================================================
try:
    # Usamos los archivos generados en la fase 1
    train_df = pd.read_csv("train_final.csv")
    test_df = pd.read_csv("test_final.csv")
    print("✅ Datos de entrenamiento y prueba cargados.")
except FileNotFoundError:
    print("❌ ERROR: Asegúrate de ejecutar 01_preparacion_datos.py primero.")
    exit()

X_train = train_df.iloc[:, :-1]
y_train = train_df['ESTRES_BINARIO']
X_test = test_df.iloc[:, :-1]
y_test = test_df['ESTRES_BINARIO']

# =================================================================
# 2. FUNCIÓN DE CÁLCULO DE MÉTRICAS AVANZADAS
# =================================================================
def calcular_metricas(y_true, y_pred, model_name):
    """Calcula todas las métricas requeridas: Precision, Recall, F1, Sensitivity, Specificity."""
    cm = confusion_matrix(y_true, y_pred)
    
    # Valores de la Matriz de Confusión: (0,0)=TN, (0,1)=FP, (1,0)=FN, (1,1)=TP
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    TP = cm[1, 1]
    
    sensibilidad = TP / (TP + FN) if (TP + FN) > 0 else 0 # Sensitivity (Recall)
    especificidad = TN / (TN + FP) if (TN + FP) > 0 else 0 # Specificity
    
    return {
        'Modelo': model_name,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall (Sensibilidad)': sensibilidad,
        'Specificity': especificidad,
        'F1-Score': f1_score(y_true, y_pred),
        'Matriz Confusión': cm
    }

# =================================================================
# 3. ENTRENAMIENTO Y COMPARACIÓN DE MODELOS
# =================================================================
modelos = {
    'Logistic Regression (BASE)': LogisticRegression(random_state=42, max_iter=1000),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Support Vector Machine': SVC(random_state=42, probability=True),
    'Random Forest (Fase 1)': RandomForestClassifier(random_state=42)
}

resultados = []
mejor_modelo = None
mejor_metrica = -1
mejor_modelo_nombre = ""

print("\n--- Comparación de Modelos de Clasificación ---")

for nombre, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    metricas = calcular_metricas(y_test, y_pred, nombre)
    resultados.append(metricas)
    print(f"✅ Modelo {nombre} entrenado. Recall: {metricas['Recall (Sensibilidad)']:.4f}")

    # Elegimos el mejor modelo basado en Recall (Sensibilidad) para la UI
    if metricas['Recall (Sensibilidad)'] > mejor_metrica:
        mejor_metrica = metricas['Recall (Sensibilidad)']
        mejor_modelo = modelo
        mejor_modelo_nombre = nombre

# =================================================================
# 4. TABULAR RESULTADOS Y GUARDAR MEJOR MODELO
# =================================================================
resultados_df = pd.DataFrame([
    {k: v for k, v in res.items() if k != 'Matriz Confusión'} for res in resultados
]).set_index('Modelo')

print("\n--- Tabla de Métricas de Modelos (en Test Set) ---")
print("Métrica de Elección: Recall (Sensibilidad), para evitar Falsos Negativos.")
print(resultados_df)
print(f"\n🏆 Mejor Modelo Elegido (por Recall): {mejor_modelo_nombre} con Recall: {mejor_metrica:.4f}")

joblib.dump(mejor_modelo, 'mejor_modelo.pkl')
print("\n✅ Mejor modelo guardado como 'mejor_modelo.pkl' para la interfaz web.")