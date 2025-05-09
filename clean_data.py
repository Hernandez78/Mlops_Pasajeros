import pandas as pd
import numpy as np
import os
import joblib
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ============================
# Configuración de rutas
# ============================
base_dir = os.getcwd()
data_dir = os.path.join(base_dir, "data")
artifacts_dir = os.path.join(base_dir, "artifacts")
os.makedirs(artifacts_dir, exist_ok=True)

# ============================
# Configurar MLflow
# ============================
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Preprocesamiento - Clean Data")

with mlflow.start_run(run_name="Limpieza y división de datos"):
    # ============================
    # Cargar los datos
    # ============================
    df_train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    df_test = pd.read_csv(os.path.join(data_dir, "test.csv"))

    # ============================
    # Limpieza básica
    # ============================
    df_train.drop(['Unnamed: 0', 'id'], axis=1, inplace=True)
    df_test.drop(['Unnamed: 0', 'id'], axis=1, inplace=True)

    median_delay = df_train['Arrival Delay in Minutes'].median()
    df_train = df_train.copy()  # ✅ Garantizar que estamos modificando una versión real y no una vista
    df_test = df_test.copy()

    df_train['Arrival Delay in Minutes'] = df_train['Arrival Delay in Minutes'].fillna(median_delay)
    df_test['Arrival Delay in Minutes'] = df_test['Arrival Delay in Minutes'].fillna(median_delay)
    
    df_train['satisfaction'] = df_train['satisfaction'].map({'satisfied': 1, 'neutral or dissatisfied': 0})
    df_test['satisfaction'] = df_test['satisfaction'].map({'satisfied': 1, 'neutral or dissatisfied': 0})

    # ============================
    # Codificación de variables categóricas
    # ============================
    cat_cols = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
    label_encoders = {}

    for col in cat_cols:
        le = LabelEncoder()
        df_train[col] = le.fit_transform(df_train[col])
        df_test[col] = le.transform(df_test[col])
        label_encoders[col] = le

    # ============================
    # Separar X e y
    # ============================
    X = df_train.drop("satisfaction", axis=1)
    y = df_train["satisfaction"]
    X_test = df_test.drop("satisfaction", axis=1)
    y_test = df_test["satisfaction"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # ============================
    # Guardar los datasets
    # ============================
    path_train = os.path.join(artifacts_dir, "train.pkl")
    path_val = os.path.join(artifacts_dir, "val.pkl")
    path_test = os.path.join(artifacts_dir, "test.pkl")

    joblib.dump((X_train, y_train), path_train)
    joblib.dump((X_val, y_val), path_val)
    joblib.dump((X_test, y_test), path_test)

    print("✅ Datos limpios y guardados en artifacts/*.pkl")

    # ============================
    # Loggear artefactos en MLflow
    # ============================
    mlflow.log_artifact(path_train)
    mlflow.log_artifact(path_val)
    mlflow.log_artifact(path_test)