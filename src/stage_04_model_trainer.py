#stage_04_model_trainer.py
import os
import sys
import torch
import mlflow
import joblib
import shutil
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
# from transformers import (
#     DistilBertTokenizerFast,
#     DistilBertForSequenceClassification,
#     Trainer,
#     TrainingArguments
# )

from src.logger import setup_logger
from src.exception import CustomException
from src.utils import read_yaml, save_json_report


class ModelTrainer:
    """
    Stage 04: Model Training
    Trains RandomForest (TF-IDF + meta-features) [DistilBERT disabled for CI/CD].
    Logs metrics, saves all models, and stores the best inside artifacts/.
    """

    def __init__(self, config_path: str, params_path: str):
        try:
            self.config = read_yaml(config_path)
            self.params = read_yaml(params_path)
            self.logger = setup_logger("stage_04_model_trainer.log")

            self.transformed_data_path = self.config["data_transformation"]["transformed_data_path"]
            self.model_dir = self.config["model_trainer"]["model_dir"]
            self.model_report_path = self.config["model_trainer"]["model_report"]
            self.best_model_artifact = os.path.join(self.config["artifact_dir"], "best_model.pkl")

            os.makedirs(self.model_dir, exist_ok=True)
            os.makedirs(os.path.dirname(self.model_report_path), exist_ok=True)

            use_gpu = self.params["train"]["use_gpu"]
            self.device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
            self.fp16 = self.params["train"]["fp16"]
            self.logger.info(f"Device selected: {self.device}  |  fp16 enabled: {self.fp16}")

        except Exception as e:
            raise CustomException(e, sys)

    def load_data(self):
        if not os.path.exists(self.transformed_data_path):
            raise FileNotFoundError(f"Transformed dataset not found at {self.transformed_data_path}")

        df = pd.read_csv(self.transformed_data_path)
        self.logger.info(f"Loaded transformed dataset: {df.shape}")

        selected_cols = self.params["features"]["selected"] + [self.params["features"]["target"]]
        df = df[selected_cols].copy()
        df.dropna(subset=["clean_text", self.params["features"]["target"]], inplace=True)
        df["clean_text"] = df["clean_text"].fillna("").astype(str)
        return df

    # -------------------- RANDOM FOREST --------------------
    def train_random_forest(self, df_train, df_test):
        try:
            self.logger.info("Starting RandomForest training (TF-IDF + meta features)")

            df_train["clean_text"] = df_train["clean_text"].fillna("").astype(str)
            df_test["clean_text"] = df_test["clean_text"].fillna("").astype(str)

            vectorizer = TfidfVectorizer(max_features=self.params["random_forest"]["tfidf_max_features"])
            X_train_tfidf = vectorizer.fit_transform(df_train["clean_text"])
            X_test_tfidf = vectorizer.transform(df_test["clean_text"])

            meta_cols = ["product_price", "Rate", "review_length", "avg_word_length"]
            X_train_meta = df_train[meta_cols].fillna(0).values
            X_test_meta = df_test[meta_cols].fillna(0).values

            X_train_combined = sparse.hstack([X_train_tfidf, sparse.csr_matrix(X_train_meta)])
            X_test_combined = sparse.hstack([X_test_tfidf, sparse.csr_matrix(X_test_meta)])

            y_train, y_test = df_train["sentiment_encoded"], df_test["sentiment_encoded"]

            clf = RandomForestClassifier(
                n_estimators=self.params["random_forest"]["n_estimators"],
                max_depth=self.params["random_forest"]["max_depth"],
                random_state=42,
                n_jobs=-1
            )
            clf.fit(X_train_combined, y_train)
            preds = clf.predict(X_test_combined)

            acc = accuracy_score(y_test, preds)
            report = classification_report(y_test, preds, output_dict=True)

            rf_model_path = os.path.join("models", "random_forest_model.pkl")
            tfidf_path = os.path.join("models", "tfidf_vectorizer.pkl")
            os.makedirs(os.path.dirname(rf_model_path), exist_ok=True)
            joblib.dump(clf, rf_model_path)
            joblib.dump(vectorizer, tfidf_path)

            mlflow.log_metric("rf_accuracy", acc)
            mlflow.log_artifact(rf_model_path, artifact_path="models")
            self.logger.info(f"RandomForest accuracy: {acc:.4f}")

            return {"name": "RandomForest", "accuracy": acc, "report": report, "path": rf_model_path}

        except Exception as e:
            raise CustomException(e, sys)

    # -------------------- DISTILBERT (DISABLED) --------------------
    # def train_distilbert(self, df_train, df_test):
    #     try:
    #         self.logger.info("Starting DistilBERT fine-tuning...")
    #
    #         model_name = self.params["distilbert"]["pretrained_model_name"]
    #         tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
    #         max_len = self.params["distilbert"]["tokenizer_max_length"]
    #
    #         X_train, X_test = df_train["clean_text"].tolist(), df_test["clean_text"].tolist()
    #         y_train, y_test = df_train["sentiment_encoded"].tolist(), df_test["sentiment_encoded"].tolist()
    #
    #         train_enc = tokenizer(X_train, truncation=True, padding=True, max_length=max_len)
    #         test_enc = tokenizer(X_test, truncation=True, padding=True, max_length=max_len)
    #
    #         class SentimentDataset(torch.utils.data.Dataset):
    #             def __init__(self, encodings, labels):
    #                 self.encodings = encodings
    #                 self.labels = labels
    #             def __getitem__(self, idx):
    #                 item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
    #                 item["labels"] = torch.tensor(self.labels[idx])
    #                 return item
    #             def __len__(self):
    #                 return len(self.labels)
    #
    #         train_ds = SentimentDataset(train_enc, y_train)
    #         test_ds = SentimentDataset(test_enc, y_test)
    #
    #         model = DistilBertForSequenceClassification.from_pretrained(
    #             model_name,
    #             num_labels=len(set(y_train))
    #         ).to(self.device)
    #
    #         lr = float(self.params["distilbert"]["learning_rate"])
    #         wd = float(self.params["distilbert"]["weight_decay"])
    #         epochs = int(self.params["distilbert"]["epochs"])
    #         batch_size = int(self.params["distilbert"]["batch_size"])
    #
    #         args = TrainingArguments(
    #             output_dir="./results",
    #             num_train_epochs=epochs,
    #             per_device_train_batch_size=batch_size,
    #             per_device_eval_batch_size=batch_size,
    #             learning_rate=lr,
    #             save_total_limit=1,
    #             save_strategy="epoch",
    #             weight_decay=wd,
    #             fp16=self.fp16 and self.device == "cuda",
    #             logging_dir="./logs",
    #             logging_steps=50,
    #             report_to=[]
    #         )
    #
    #         trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=test_ds)
    #         trainer.train()
    #
    #         preds_output = trainer.predict(test_ds)
    #         preds = np.argmax(preds_output.predictions, axis=1)
    #         acc = accuracy_score(y_test, preds)
    #         report = classification_report(y_test, preds, output_dict=True)
    #
    #         bert_path = os.path.join("models", "distilbert_model")
    #         os.makedirs(bert_path, exist_ok=True)
    #         model.save_pretrained(bert_path)
    #         tokenizer.save_pretrained(bert_path)
    #
    #         mlflow.log_metric("bert_accuracy", acc)
    #         mlflow.log_artifact(bert_path, artifact_path="models")
    #         self.logger.info(f"DistilBERT accuracy: {acc:.4f}")
    #
    #         return {"name": "DistilBERT", "accuracy": acc, "report": report, "path": bert_path}
    #
    #     except Exception as e:
    #         raise CustomException(e, sys)

    # -------------------- PIPELINE --------------------
    def run(self):
        try:
            self.logger.info("========== Stage 04: Model Training Started ==========")

            tracking_uri = self.config["mlflow"].get("tracking_uri", "")
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            else:
                local_uri = os.path.join(os.getcwd(), "mlruns")
                self.logger.info(f"Using local MLflow at: {local_uri}")
                mlflow.set_tracking_uri(f"file:{local_uri}")
            mlflow.set_experiment(self.config["mlflow"]["experiment_name"])

            with mlflow.start_run(run_name="model_trainer_stage"):
                df = self.load_data()
                train_df, test_df = train_test_split(
                    df,
                    test_size=self.params["train_test_split"]["test_size"],
                    random_state=self.params["train_test_split"]["random_state"],
                    stratify=df[self.params["features"]["target"]]
                )

                mlflow.log_param("train_rows", len(train_df))
                mlflow.log_param("test_rows", len(test_df))

                rf_result = self.train_random_forest(train_df, test_df)

                # DistilBERT training disabled for CI/CD to reduce runtime
                bert_result = {"name": "DistilBERT", "accuracy": 0, "report": {}, "path": None}

                best = rf_result if rf_result["accuracy"] >= bert_result["accuracy"] else bert_result
                self.logger.info(f"Best model: {best['name']} ({best['accuracy']:.4f})")

                report = {
                    "RandomForest": rf_result,
                    "DistilBERT": bert_result,
                    "best_model": best["name"],
                    "best_accuracy": best["accuracy"],
                    "timestamp": datetime.utcnow().isoformat()
                }
                save_json_report(report, self.model_report_path)
                mlflow.log_artifact(self.model_report_path, artifact_path="reports")

                if best["name"] == "RandomForest":
                    shutil.copy(best["path"], self.best_model_artifact)
                # else:
                #     dest = os.path.join(self.config["artifact_dir"], "best_model")
                #     shutil.copytree(best["path"], dest, dirs_exist_ok=True)

                mlflow.log_param("best_model", best["name"])
                mlflow.log_metric("best_accuracy", best["accuracy"])

                self.logger.info("========== Stage 04: Model Training Completed ==========")

        except Exception as e:
            self.logger.error("Model Training failed.")
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        trainer = ModelTrainer(config_path="config/config.yaml", params_path="config/params.yaml")
        trainer.run()
    except Exception as e:
        raise CustomException(e, sys)
