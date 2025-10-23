# Analisis_Patrones_Sueno
# Mi Proyecto de Análisis de Datos: Sueño y Nivel de Estrés

¡Hola! Este es el repositorio de mi proyecto sobre el analisis del sueño y el estres. 
El objetivo es usar datos sobre hábitos de sueño y salud para intentar predecir el nivel de estrés de las personas.

---

## Lo Que Hice y Lo Que Encontré

### 1. El Problema a Resolver
Convertí la variable numérica de "Nivel de Estrés" en una categoría simple:
* **0:** Estrés Moderado
* **1:** Estrés Alto o Crónico

### 2. El Resultado Final: 100% de Precisión

Para intentar predecir el estrés, usé una herramienta llamada Random Forest (un tipo de programa que aprende a clasificar cosas).

El resultado fue muy claro: la precisión fue del 100% en las pruebas;El modelo acertó en todas las predicciones.

¿Qué significa ese 100%?

Este resultado tan perfecto nos dice que ciertas variables,especialmente la Duración del Sueño y el Nivel de Actividad Física, están ligadas de forma muy fuerte a la variable de estrés en estos datos. Básicamente, si el modelo sabe cuánto duermes y te ejercitas, puede adivinar tu nivel de estrés sin equivocarse.

## Archivos Clave del Proyecto

| Archivo | ¿Qué contiene? |
| :--- | :--- |
| `01_preparacion_datos.py` | El código de **limpieza y organización** de los datos. Se encarga de crear las variables necesarias y dividir los datos finales. |
| `02_modelo_clasificacion.py` | El código de **Machine Learning**. Entrena el clasificador Random Forest y saca la precisión final. |
| `train_final.csv` y `test_final.csv` | Son los datos listos para el modelo, creados después de la limpieza. |

---
¡Gracias por revisar mi trabajo!
