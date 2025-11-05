import os
import sys
import torch
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

from src.logger import setup_logger
from src.exception import CustomException
from src.utils import read_yaml


class ModelInference:
    """
    Stage 06: Model Inference
    - Loads the best model from artifacts/
    - Accepts new input text(s) via terminal
    - Predicts sentiment (Positive / Negative / Neutral)
    - Logs each input + prediction into artifacts/logs/utils.log
    """

    def __init__(self, config_path: str, params_path: str):
        try:
            self.config = read_yaml(config_path)
            self.params = read_yaml(params_path)

            # Ensure logs directory exists inside artifacts
            self.artifact_dir = self.config["artifact_dir"]
            self.logs_dir = os.path.join(self.artifact_dir, "logs")
            os.makedirs(self.logs_dir, exist_ok=True)

            self.logger = setup_logger(os.path.join(self.logs_dir, "utils.log"))

            self.model_dir = self.config["model_trainer"]["model_dir"]
            self.best_model_path = os.path.join(self.artifact_dir, "best_model.pkl")
            self.best_model_folder = os.path.join(self.artifact_dir, "best_model")

            self.device = "cuda" if torch.cuda.is_available() and self.params["train"]["use_gpu"] else "cpu"

            self.model_type = self.detect_model_type()
            self.logger.info(f"Detected model type: {self.model_type}")

            if self.model_type == "RandomForest":
                self.load_random_forest_model()
            elif self.model_type == "DistilBERT":
                self.load_distilbert_model()
            else:
                raise ValueError("No valid best model found in artifacts.")

        except Exception as e:
            raise CustomException(e, sys)

    def detect_model_type(self):
        """Detect best model type based on saved structure."""
        if os.path.exists(os.path.join(self.best_model_folder, "config.json")):
            return "DistilBERT"
        elif os.path.exists(self.best_model_path):
            return "RandomForest"
        return None

    def load_random_forest_model(self):
        """Load RandomForest + TF-IDF vectorizer."""
        try:
            vectorizer_path = os.path.join(self.model_dir, "tfidf_vectorizer.pkl")
            if not os.path.exists(vectorizer_path):
                raise FileNotFoundError("TF-IDF vectorizer not found.")

            self.vectorizer = joblib.load(vectorizer_path)
            self.model = joblib.load(self.best_model_path)
            self.meta_cols = ["product_price", "Rate", "review_length", "avg_word_length"]

            self.logger.info("RandomForest model and TF-IDF vectorizer loaded successfully.")
        except Exception as e:
            raise CustomException(e, sys)

    def load_distilbert_model(self):
        """Load DistilBERT model + tokenizer."""
        try:
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(self.best_model_folder)
            self.model = DistilBertForSequenceClassification.from_pretrained(self.best_model_folder).to(self.device)
            self.model.eval()
            self.logger.info("DistilBERT model and tokenizer loaded successfully.")
        except Exception as e:
            raise CustomException(e, sys)

    def predict_random_forest(self, text: str, meta_data=None):
        """Make prediction using RandomForest."""
        if meta_data is None:
            meta_data = {
                "product_price": 0,
                "Rate": 0,
                "review_length": len(text),
                "avg_word_length": np.mean([len(w) for w in text.split()]) if len(text.split()) > 0 else 0,
            }

        X_tfidf = self.vectorizer.transform([text])
        X_meta = sparse.csr_matrix([[meta_data.get(col, 0) for col in self.meta_cols]])
        X_combined = sparse.hstack([X_tfidf, X_meta])

        pred = self.model.predict(X_combined)[0]
        return int(pred)

    def predict_distilbert(self, text: str):
        """Make prediction using DistilBERT."""
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=self.params["distilbert"]["tokenizer_max_length"],
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            pred = torch.argmax(outputs.logits, dim=1).cpu().item()
        return int(pred)

    def predict(self, text: str):
        """Predict sentiment for a single input and log it."""
        if not text.strip():
            raise ValueError("Empty input text.")

        if self.model_type == "RandomForest":
            pred = self.predict_random_forest(text)
        elif self.model_type == "DistilBERT":
            pred = self.predict_distilbert(text)
        else:
            raise ValueError("Unknown model type.")

        sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        sentiment_label = sentiment_map.get(pred, "Unknown")

        # Log both input and prediction
        self.logger.info(f"Input Text: {text}")
        self.logger.info(f"Predicted Sentiment: {sentiment_label}\n")

        return sentiment_label


if __name__ == "__main__":
    try:
        inference = ModelInference(config_path="config/config.yaml", params_path="config/params.yaml")

        print("\n========== Sentiment Prediction ==========")
        print("Type 'exit' to quit.\n")

        while True:
            user_input = input("Enter review text: ").strip()
            if user_input.lower() == "exit":
                print("Exiting inference session.")
                break

            try:
                sentiment = inference.predict(user_input)
                print(f"Predicted Sentiment: {sentiment}\n")
            except Exception as e:
                print(f"Error: {e}\n")

    except Exception as e:
        raise CustomException(e, sys)
