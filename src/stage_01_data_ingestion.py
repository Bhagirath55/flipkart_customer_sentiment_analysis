import os
import sys
import pandas as pd
from datetime import datetime
import json
import mlflow

from src.logger import setup_logger
from src.exception import CustomException
from src.utils import read_yaml, save_csv


class DataIngestion:
    """
    Stage 1: Data Ingestion
    - Loads raw dataset
    - Performs basic validation
    - Saves a cleaned, deduplicated version for further processing
    - Logs everything to MLflow (local or remote)
    """

    def __init__(self, config_path: str, params_path: str):
        try:
            self.config = read_yaml(config_path)
            self.params = read_yaml(params_path)
            self.logger = setup_logger("stage_01_data_ingestion.log")

            self.raw_data_path = self.config["data_ingestion"]["raw_data_path"]
            self.processed_data_path = self.config["data_ingestion"]["processed_data_path"]
            self.report_file = self.config["data_ingestion"]["report_file"]

        except Exception as e:
            raise CustomException(e, sys)

    def run(self):
        try:
            self.logger.info("========== Stage 01: Data Ingestion Started ==========")

            # ---------------- MLflow Setup ---------------- #
            tracking_uri = self.config.get("mlflow", {}).get("tracking_uri", "")
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
                self.logger.info(f"Using remote MLflow tracking URI: {tracking_uri}")
            else:
                local_uri = os.path.join(os.getcwd(), "mlruns")
                mlflow.set_tracking_uri(f"file:{local_uri}")
                self.logger.info(f"Using local MLflow tracking at: {local_uri}")

            experiment_name = self.config.get("mlflow", {}).get("experiment_name", "flipkart_sentiment_experiment")
            mlflow.set_experiment(experiment_name)

            # ---------------- MLflow Run ---------------- #
            with mlflow.start_run(run_name="data_ingestion_stage") as run:
                mlflow.log_param("stage", "data_ingestion")
                mlflow.log_param("raw_data_path", self.raw_data_path)
                mlflow.log_param("processed_data_path", self.processed_data_path)

                if not os.path.exists(self.raw_data_path):
                    msg = f"Raw dataset not found at path: {self.raw_data_path}"
                    self.logger.error(msg)
                    raise FileNotFoundError(msg)

                df = pd.read_csv(self.raw_data_path)
                self.logger.info(f"Loaded raw dataset successfully with shape: {df.shape}")
                mlflow.log_metric("rows_raw", df.shape[0])
                mlflow.log_metric("cols_raw", df.shape[1])

                missing = df.isnull().sum().sum()
                duplicates = df.duplicated().sum()
                self.logger.info(f"Missing values: {missing}, Duplicates: {duplicates}")

                mlflow.log_metric("missing_values", int(missing))
                mlflow.log_metric("duplicate_rows", int(duplicates))

                if duplicates > 0:
                    df = df.drop_duplicates().reset_index(drop=True)
                    self.logger.info("Duplicate rows removed.")

                os.makedirs(os.path.dirname(self.processed_data_path), exist_ok=True)
                save_csv(df, self.processed_data_path)

                mlflow.log_artifact(self.processed_data_path, artifact_path="data_ingestion")
                mlflow.log_param("ingestion_time", datetime.utcnow().isoformat())

                # ---------- Report Generation ---------- #
                report = {
                    "stage": "data_ingestion",
                    "raw_data_path": self.raw_data_path,
                    "processed_data_path": self.processed_data_path,
                    "rows": int(df.shape[0]),
                    "columns": int(df.shape[1]),
                    "missing_values": int(missing),
                    "duplicate_rows": int(duplicates),
                    "timestamp": datetime.utcnow().isoformat()
                }
                os.makedirs(os.path.dirname(self.report_file), exist_ok=True)
                with open(self.report_file, "w", encoding="utf-8") as f:
                    json.dump(report, f, indent=4)
                mlflow.log_artifact(self.report_file, artifact_path="reports")

                self.logger.info(f"Report saved at: {self.report_file}")
                self.logger.info("========== Stage 01: Data Ingestion Completed ==========")

        except Exception as e:
            self.logger.error("Data ingestion failed.")
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        pipeline = DataIngestion("config/config.yaml", "config/params.yaml")
        pipeline.run()
    except Exception as e:
        raise CustomException(e, sys)
