# log_model.py
import mlflow
import mlflow.sklearn
import pickle

# Muat model yang Anda punya
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with mlflow.start_run():
    mlflow.log_param("model_file", "model.pkl")

    mlflow.sklearn.log_model(model, "model")

    print("Model successfully logged to MLflow.")
