import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import mlflow

# ================================
# Configurar rutas y carpetas
# ================================
outputs_dir = os.path.join(os.getcwd(), "outputs")
os.makedirs(outputs_dir, exist_ok=True)

# ================================
# Configurar MLflow para EDA
# ================================
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("EDA - Exploración de Datos")

with mlflow.start_run(run_name="EDA básica"):
    # ================================
    # Cargar datasets
    # ================================
    df_train = pd.read_csv(os.path.join("data", "train.csv"))
    df_test = pd.read_csv(os.path.join("data", "test.csv"))

    # Estructura
    print("Train shape:", df_train.shape)
    print("Test shape:", df_test.shape)
    print("\\nColumnas:", df_train.columns.tolist())

    # Valores nulos
    print("\\nValores nulos por columna (train):\\n", df_train.isnull().sum())

    # ================================
    # Distribución de la variable objetivo
    # ================================
    plt.figure()
    sns.countplot(data=df_train, x='satisfaction')
    plt.title("Distribución de satisfacción")
    
    path_dist = os.path.join(outputs_dir, "distribucion_satisfaccion.png")
    plt.savefig(path_dist, dpi=300, bbox_inches="tight")
    plt.close()

    # Intentar registrar el artefacto
    try:
        mlflow.log_artifact(path_dist)
    except Exception as e:
        print(f"Error al registrar el artefacto de distribución: {e}")

    # ================================
    # Variables categóricas vs satisfacción
    # ================================
    categorical_columns = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
    
    for col in categorical_columns:
        plt.figure(figsize=(6, 4))
        sns.countplot(data=df_train, x=col, hue='satisfaction')
        plt.title(f'{col} vs Satisfaction')
        plt.xticks(rotation=15)
        plt.tight_layout()

        path_cat = os.path.join(outputs_dir, f"{col.replace(' ', '_')}_vs_satisfaction.png")
        plt.savefig(path_cat, dpi=300, bbox_inches="tight")
        plt.close()

        # Intentar registrar el artefacto
        try:
            mlflow.log_artifact(path_cat)
        except Exception as e:
            print(f"Error al registrar el artefacto de {col} vs satisfacción: {e}")

    # ================================
    # Correlación entre variables numéricas
    # ================================
    numerical = df_train.select_dtypes(include='number')

    plt.figure(figsize=(12, 10))
    sns.heatmap(numerical.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Matriz de correlación")
    plt.tight_layout()

    path_corr = os.path.join(outputs_dir, "matriz_correlacion.png")
    plt.savefig(path_corr, dpi=300, bbox_inches="tight")
    plt.close()

    # Intentar registrar el artefacto
    try:
        mlflow.log_artifact(path_corr)
    except Exception as e:
        print(f"Error al registrar el artefacto de matriz de correlación: {e}")

    # ================================
    # Valores únicos en columnas clave
    # ================================
    print("\\nValores únicos en 'satisfaction':", df_train['satisfaction'].unique())
    for col in categorical_columns:
        print(f"Columna '{col}' - valores únicos:", df_train[col].unique())