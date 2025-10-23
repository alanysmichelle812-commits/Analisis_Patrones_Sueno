import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1️⃣ Cargar los datos ya preparados y divididos
try:
    # Los archivos deben estar en la misma carpeta que este script
    train_df = pd.read_csv("train_final.csv")
    test_df = pd.read_csv("test_final.csv")
    print("✅ Archivos de entrenamiento y prueba cargados.")
except FileNotFoundError:
    print("❌ ERROR CRÍTICO: No se encontraron 'train_final.csv' o 'test_final.csv'.")
    print("Ejecuta primero el script 01_preparacion_datos.py.")
    exit()

# Definir X (features) y y (target)
X_train = train_df.iloc[:, :-1]
y_train = train_df['ESTRES_BINARIO']

X_test = test_df.iloc[:, :-1]
y_test = test_df['ESTRES_BINARIO']

# 2️⃣ Entrenar modelo
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
print("\n✅ Modelo Random Forest entrenado con el dataset estratificado.")

# 3️⃣ Evaluar modelo
y_pred = model.predict(X_test)
print("\n--- Resultados del Modelo de Clasificación de Estrés ---")
print("Precisión:", round(accuracy_score(y_test, y_pred)*100, 2), "%")
print("\nMatriz de confusión:\n", confusion_matrix(y_test, y_pred))
print("\nReporte de clasificación:\n", classification_report(y_test, y_pred))

# 4️⃣ Visualización (Se abrirá una ventana que debes cerrar para continuar)
plt.figure(figsize=(8, 5))
plt.scatter(X_test['Sleep Duration'], y_test, alpha=0.6)
plt.title("Duración del sueño vs Estrés (0=Moderado, 1=Estresado)")
plt.xlabel("Horas de sueño")
plt.ylabel("ESTRES_BINARIO")
plt.grid(True)
plt.show()
