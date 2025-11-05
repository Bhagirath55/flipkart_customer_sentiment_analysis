import os
import sys
import re
import json
import unicodedata
import pandas as pd
import numpy as np
import mlflow
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import spacy
import wordninja

from src.logger import setup_logger
from src.exception import CustomException
from src.utils import read_yaml, save_csv

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

HTML_RE = re.compile(r"<.*?>")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
CONTRACTIONS = {"can't": "cannot", "won't": "will not", "n't": " not", "i'm": "i am", "it's": "it is"}


def expand_contractions(text):
    for c, e in CONTRACTIONS.items():
        text = text.replace(c, e)
    return text


def clean_text_basic(text):
    if pd.isna(text):
        return ""
    t = str(text).lower()
    t = re.sub(r"([.,!?])([A-Za-z])", r"\1 \2", t)
    t = re.sub(r"([a-zA-Z])(\d)", r"\1 \2", t)
    t = re.sub(r"(\d)([a-zA-Z])", r"\1 \2", t)
    t = HTML_RE.sub(" ", t)
    t = URL_RE.sub(" ", t)
    t = unicodedata.normalize("NFKD", t)
    t = expand_contractions(t)
    t = re.sub(r"[^a-z\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def segment_text(text):
    if not text:
        return ""
    return " ".join(wordninja.split(text))


def batch_lemmatize(texts, batch_size=1000):
    out = []
    for doc in nlp.pipe(texts, batch_size=batch_size):
        lemmas = [t.lemma_ for t in doc if t.is_alpha]
        out.append(" ".join(lemmas))
    return out


class DataTransformation:
    """Stage 03 — Transform and preprocess dataset."""

    def __init__(self, config_path: str, params_path: str):
        try:
            self.config = read_yaml(config_path)
            self.params = read_yaml(params_path)
            self.logger = setup_logger("stage_03_data_transformation.log")
            self.ingested_data_path = self.config["data_ingestion"]["processed_data_path"]
            self.transformed_data_path = self.config["data_transformation"]["transformed_data_path"]
            self.report_file = "artifacts/reports/data_transformation_report.json"
        except Exception as e:
            raise CustomException(e, sys)

    def run(self):
        try:
            self.logger.info("========== Stage 03: Data Transformation Started ==========")
            tracking_uri = self.config["mlflow"].get("tracking_uri", "")
            mlflow.set_tracking_uri(f"file:{os.getcwd()}/mlruns" if not tracking_uri else tracking_uri)
            mlflow.set_experiment(self.config["mlflow"].get("experiment_name", "flipkart_sentiment_experiment"))

            with mlflow.start_run(run_name="data_transformation_stage"):
                df = pd.read_csv(self.ingested_data_path)
                self.logger.info(f"Loaded data: {df.shape}")

                df["product_price"] = pd.to_numeric(df["product_price"], errors="coerce")
                df["Rate"] = pd.to_numeric(df["Rate"], errors="coerce")
                df["product_price"].fillna(df["product_price"].median(), inplace=True)
                df["Rate"].fillna(df["Rate"].median(), inplace=True)

                df["Summary"] = df["Summary"].fillna("")
                df["Review"] = df["Review"].fillna("")
                df["raw_text"] = df["Summary"].astype(str) + " " + df["Review"].astype(str)
                df["clean_text"] = df["raw_text"].map(clean_text_basic).map(segment_text)
                df["clean_text"] = batch_lemmatize(df["clean_text"].tolist())

                df["review_length"] = df["clean_text"].apply(lambda x: len(str(x).split()))
                df["avg_word_length"] = df["clean_text"].apply(
                    lambda x: np.mean([len(w) for w in x.split()]) if x else 0.0
                )

                df["Sentiment"] = df["Sentiment"].astype(str).str.lower().str.strip()
                label_encoder = LabelEncoder()
                df["sentiment_encoded"] = label_encoder.fit_transform(df["Sentiment"])

                os.makedirs(os.path.dirname(self.transformed_data_path), exist_ok=True)
                save_csv(df, self.transformed_data_path)

                report = {
                    "stage": "data_transformation",
                    "input_file": self.ingested_data_path,
                    "output_file": self.transformed_data_path,
                    "rows": int(df.shape[0]),
                    "cols": int(df.shape[1]),
                    "features_added": [
                        "raw_text", "clean_text", "review_length", "avg_word_length", "sentiment_encoded"
                    ],
                    "timestamp": datetime.utcnow().isoformat()
                }

                os.makedirs(os.path.dirname(self.report_file), exist_ok=True)
                with open(self.report_file, "w", encoding="utf-8") as f:
                    json.dump(report, f, indent=4)

                mlflow.log_artifact(self.transformed_data_path, artifact_path="data_transformation")
                mlflow.log_artifact(self.report_file, artifact_path="reports")

                self.logger.info(f"Transformation report saved at: {self.report_file}")
                self.logger.info("========== Stage 03: Data Transformation Completed ==========")
        except Exception as e:
            self.logger.error("Transformation failed.")
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        DataTransformation("config/config.yaml", "config/params.yaml").run()
    except Exception as e:
        raise CustomException(e, sys)
