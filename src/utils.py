# src/utils.py
import os
import sys
import yaml
import pickle
import pandas as pd
import numpy as np
from src.logger import setup_logger
from src.exception import CustomException
from datetime import datetime
import json

logger = setup_logger("utils.log")

# ---------- FILE I/O ----------

def read_yaml(path_to_yaml: str) -> dict:
    """
    Reads YAML configuration file safely.
    """
    try:
        with open(path_to_yaml, "r") as f:
            content = yaml.safe_load(f)
        logger.info(f"YAML file loaded: {path_to_yaml}")
        return content
    except Exception as e:
        logger.error(f"Failed to read YAML file: {path_to_yaml}")
        raise CustomException(e, sys)


def save_object(file_path: str, obj) -> None:
    """
    Saves Python object (pickle format).
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logger.info(f"Object saved successfully: {file_path}")
    except Exception as e:
        logger.error(f"Failed to save object: {file_path}")
        raise CustomException(e, sys)


def load_object(file_path: str):
    """
    Loads pickled Python object.
    """
    try:
        with open(file_path, "rb") as file_obj:
            obj = pickle.load(file_obj)
        logger.info(f"Object loaded: {file_path}")
        return obj
    except Exception as e:
        logger.error(f"Failed to load object: {file_path}")
        raise CustomException(e, sys)


def read_csv(file_path: str) -> pd.DataFrame:
    """
    Reads CSV into pandas DataFrame with logging.
    """
    try:
        df = pd.read_csv(file_path)
        logger.info(f"CSV file loaded successfully: {file_path}, shape={df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error reading CSV: {file_path}")
        raise CustomException(e, sys)


def save_csv(df: pd.DataFrame, file_path: str) -> None:
    """
    Saves DataFrame to CSV with directory creation.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.info(f"CSV saved successfully: {file_path}, shape={df.shape}")
    except Exception as e:
        logger.error(f"Error saving CSV: {file_path}")
        raise CustomException(e, sys)
    
def save_json_report(report: dict, report_path: str, logger=None):
    """
    Save a report dictionary as a JSON file in artifacts/reports/.
    """
    try:
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        report["timestamp"] = datetime.utcnow().isoformat()

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)

        if logger:
            logger.info(f"Report saved at {report_path}")

    except Exception as e:
        if logger:
            logger.error(f"Failed to save report at {report_path}: {str(e)}")
        raise


# ---------- METRICS ----------

def evaluate_classification_metrics(y_true, y_pred) -> dict:
    """
    Computes basic classification metrics: accuracy, precision, recall, F1.
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    try:
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1_score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }
        logger.info(f"Evaluation metrics computed: {metrics}")
        return metrics
    except Exception as e:
        logger.error("Error while computing evaluation metrics")
        raise CustomException(e, sys)
