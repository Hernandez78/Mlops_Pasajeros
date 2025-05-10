import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from mlflow.exceptions import MlflowException

# ===============================
# ✅ Configurar MLflow con rutas relativas
# ===============================
def setup_mlflow(experiment_name):
    mlflow.set_tracking_uri("file://mlruns")
    os.makedirs("mlruns", exist_ok=True)
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"✅ Experimento '{experiment_name}' creado.")
    except MlflowException:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        print(f"ℹ️ Experimento '{experiment_name}' ya existía. Usando ID: {experiment_id}")
    mlflow.set_experiment(experiment_name)
    return experiment_id

# ===============================
# Cargar datos preprocesados
# ===============================
def load_data():
    try:
        X_train, y_train = joblib.load("artifacts/train.pkl")
        X_val, y_val = joblib.load("artifacts/val.pkl")
        return X_train, y_train, X_val, y_val
    except FileNotFoundError as e:
        print(f"Error al cargar los datos: {e}")
        raise

# ===============================
# Modelos a comparar
# ===============================
def train_and_evaluate_models(X_train, y_train, X_val, y_val, experiment_id):
    modelos = {
        "RandomForest": RandomForestClassifier(n_estimators=50, max_depth=6, class_weight='balanced', random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=500, class_weight='balanced', random_state=42),
        "XGBoost": XGBClassifier(n_estimators=50, max_depth=4, use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    
    resultados = []

    # ===============================
    # Entrenamiento y evaluación
    # ===============================
    with mlflow.start_run(experiment_id=experiment_id):
        for nombre, modelo in modelos.items():
            print(f"\n🚀 Entrenando modelo: {nombre}")
            modelo.fit(X_train, y_train)
            y_pred = modelo.predict(X_val)
            acc = accuracy_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred, average='weighted')
            print(f"✅ {nombre} - Accuracy: {acc:.4f} | F1-score: {f1:.4f}")
            print(classification_report(y_val, y_pred, target_names=["No Satisfecho", "Satisfecho"]))
            # Log en MLflow
            mlflow.log_param(f"model_{nombre}", modelo.__class__.__name__)
            mlflow.log_metric(f"{nombre}_accuracy", acc)
            mlflow.log_metric(f"{nombre}_f1", f1)
            resultados.append({
                "nombre": nombre,
                "modelo": modelo,
                "accuracy": acc,
                "f1": f1
            })

    return resultados

# ===============================
# Guardar el mejor modelo
# ===============================
def save_best_model(resultados):
    mejor_modelo = max(resultados, key=lambda x: x['f1'])
    print(f"\n🏆 Mejor modelo: {mejor_modelo['nombre']} (F1: {mejor_modelo['f1']:.4f})")
    os.makedirs("models", exist_ok=True)
    modelo_path = os.path.abspath("models/mejor_modelo.pkl")
    joblib.dump(mejor_modelo["modelo"], modelo_path)
    print(f"💾 Modelo guardado en {modelo_path}")

    # ===============================
    # Log del modelo final
    # ===============================
    mlflow.sklearn.log_model(
        sk_model=mejor_modelo["modelo"],
        artifact_path="modelo_final"
    )
    mlflow.log_artifact(modelo_path, artifact_path="modelo_final")

if __name__ == "__main__":
    experiment_name = "airline_satisfaction"
    experiment_id = setup_mlflow(experiment_name)
    X_train, y_train, X_val, y_val = load_data()
    resultados = train_and_evaluate_models(X_train, y_train, X_val, y_val, experiment_id)
    save_best_model(resultados)