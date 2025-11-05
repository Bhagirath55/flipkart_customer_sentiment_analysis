import os
import sys
import json
import torch
import mlflow
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

from src.logger import setup_logger
from src.exception import CustomException
from src.utils import read_yaml, save_json_report


class ModelEvaluator:
    """
    Stage 05: Model Evaluation
    - Automatically detects whether the best model is RandomForest or DistilBERT.
    - Recomputes metrics on test data.
    - Saves confusion matrix and evaluation report.
    - Logs everything to MLflow.
    """

    def __init__(self, config_path: str, params_path: str):
        try:
            self.config = read_yaml(config_path)
            self.params = read_yaml(params_path)
            self.logger = setup_logger("stage_05_model_evaluation.log")

            self.transformed_data_path = self.config["data_transformation"]["transformed_data_path"]
            self.artifact_dir = self.config["artifact_dir"]
            self.model_dir = self.config["model_trainer"]["model_dir"]

            # best model now stored as a folder for both cases
            self.best_model_path = os.path.join(self.artifact_dir, "best_model.pkl")
            self.best_model_folder = os.path.join(self.artifact_dir, "best_model")

            self.eval_report_path = os.path.join(self.artifact_dir, "best_model_evaluation_report.json")
            self.conf_matrix_path = os.path.join(self.artifact_dir, "confusion_matrix.png")

            os.makedirs(self.artifact_dir, exist_ok=True)
            self.device = "cuda" if torch.cuda.is_available() and self.params["train"]["use_gpu"] else "cpu"

        except Exception as e:
            raise CustomException(e, sys)

    def load_data(self):
        if not os.path.exists(self.transformed_data_path):
            raise FileNotFoundError(f"Transformed dataset not found at {self.transformed_data_path}")
        df = pd.read_csv(self.transformed_data_path)
        df = df[self.params["features"]["selected"] + [self.params["features"]["target"]]]
        self.logger.info(f"Loaded dataset for evaluation with shape: {df.shape}")
        return df

    def plot_confusion_matrix(self, y_true, y_pred, model_name):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"{model_name} - Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(self.conf_matrix_path)
        plt.close()
        self.logger.info(f"Confusion matrix saved at {self.conf_matrix_path}")

    def evaluate_random_forest(self, df):
        self.logger.info("Evaluating RandomForest model...")
        vectorizer_path = os.path.join(self.model_dir, "tfidf_vectorizer.pkl")

        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(f"TF-IDF vectorizer not found at {vectorizer_path}")
        vectorizer = joblib.load(vectorizer_path)

        meta_cols = ["product_price", "Rate", "review_length", "avg_word_length"]
        X_tfidf = vectorizer.transform(df["clean_text"])
        X_meta = sparse.csr_matrix(df[meta_cols].fillna(0).values)
        X_combined = sparse.hstack([X_tfidf, X_meta])
        y_true = df["sentiment_encoded"]

        model = joblib.load(self.best_model_path)
        y_pred = model.predict(X_combined)

        acc = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, output_dict=True)
        self.plot_confusion_matrix(y_true, y_pred, "RandomForest")

        return {"model": "RandomForest", "accuracy": acc, "report": report}

    def evaluate_distilbert(self, df):
        self.logger.info("Evaluating DistilBERT model...")
        tokenizer = DistilBertTokenizerFast.from_pretrained(self.best_model_folder)
        model = DistilBertForSequenceClassification.from_pretrained(self.best_model_folder).to(self.device)
        model.eval()

        X = list(df["clean_text"])
        y_true = df["sentiment_encoded"].tolist()

        preds = []
        with torch.no_grad():
            for i in range(0, len(X), 16):
                batch_texts = X[i:i + 16]
                enc = tokenizer(batch_texts, truncation=True, padding=True,
                                max_length=self.params["distilbert"]["tokenizer_max_length"],
                                return_tensors="pt").to(self.device)
                outputs = model(**enc)
                batch_preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                preds.extend(batch_preds)

        acc = accuracy_score(y_true, preds)
        report = classification_report(y_true, preds, output_dict=True)
        self.plot_confusion_matrix(y_true, preds, "DistilBERT")

        return {"model": "DistilBERT", "accuracy": acc, "report": report}

    def run(self):
        try:
            self.logger.info("========== Stage 05: Model Evaluation Started ==========")

            df = self.load_data()
            _, df_test = np.split(df.sample(frac=1, random_state=self.params["train_test_split"]["random_state"]),
                                  [int(0.8 * len(df))])

            tracking_uri = self.config["mlflow"].get("tracking_uri", "")
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            else:
                local_uri = os.path.join(os.getcwd(), "mlruns")
                mlflow.set_tracking_uri(f"file:{local_uri}")

            experiment_name = self.config["mlflow"]["experiment_name"]
            mlflow.set_experiment(experiment_name)

            with mlflow.start_run(run_name="model_evaluation_stage"):
                # Updated logic: check folder structure instead of just file existence
                if os.path.exists(os.path.join(self.best_model_folder, "config.json")):
                    result = self.evaluate_distilbert(df_test)
                elif os.path.exists(self.best_model_path):
                    result = self.evaluate_random_forest(df_test)
                else:
                    raise FileNotFoundError("No best model found in artifacts folder.")

                save_json_report(result, self.eval_report_path)
                mlflow.log_metric("eval_accuracy", result["accuracy"])
                mlflow.log_artifact(self.eval_report_path, artifact_path="evaluation")
                mlflow.log_artifact(self.conf_matrix_path, artifact_path="evaluation")

                self.logger.info(f"Evaluation completed. Accuracy: {result['accuracy']:.4f}")
                self.logger.info("========== Stage 05: Model Evaluation Completed ==========")

        except Exception as e:
            self.logger.error("Model Evaluation failed.")
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        evaluator = ModelEvaluator(config_path="config/config.yaml", params_path="config/params.yaml")
        evaluator.run()
    except Exception as e:
        raise CustomException(e, sys)
