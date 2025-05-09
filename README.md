# Airline Passenger Satisfaction - Machine Learning Pipeline

Este proyecto construye un pipeline completo de Machine Learning para predecir si un pasajero está satisfecho con su experiencia aérea, utilizando datos reales de encuestas de satisfacción.

## Objetivo

Clasificar a los pasajeros en:
- **Satisfechos**
- **No satisfechos o neutrales**

A partir de factores como clase del vuelo, tipo de viaje, servicio abordo, retrasos, etc.

## Estructura del Proyecto
airline_satisfaction_ml/

- `clean_data.py` → Limpieza y codificación de los datos
- `explore_data.py` → Análisis exploratorio y visualización
- `train.py` → Entrenamiento de modelos (Random Forest, Regresión Logística, XGBoost)
- `validate.py` → Evaluación final sobre el conjunto de test
- `requirements.txt` → Lista de dependencias
- `artifacts/` → Datos preprocesados serializados (.pkl)
- `models/` → Modelo final guardado (`mejor_modelo.pkl`)
- `outputs/` → Gráficas generadas por el análisis y evaluación
- `.github/workflows/ml_ci.yml` → CI/CD automático con GitHub Actions


## Ejecución local

1. Clona el repositorio:

2. Instala dependencias

3. Ejecuta en orden
python clean_data.py
python train.py
python validate.py

## CI/CD Automático
Este repositorio tiene configurado GitHub Actions. Cada vez que haces un push a main, se ejecutan automáticamente:

Limpieza de datos

Entrenamiento de modelos

Validación final

Puedes ver los resultados en la pestaña Actions del repo.

## Resultados (Modelo final: XGBoost)

| Métrica   | Valor   |
|-----------|---------|
| Accuracy  | 95.7%   |
| F1-score  | 95.7%   |

El modelo se comporta de forma equilibrada para ambas clases y generaliza bien sobre el conjunto de test.

## Fuente
Dataset: Kaggle – Airline Passenger Satisfaction> https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction/

## Notas
Los archivos .pkl y gráficos están incluidos para evaluación.

El pipeline está optimizado para ejecución en local con recursos moderados.
