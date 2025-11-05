#stage_02_data_validation.py
import os
import sys
import json
import pandas as pd
import mlflow
from datetime import datetime

from src.logger import setup_logger
from src.exception import CustomException
from src.utils import read_yaml

class DataValidation:
    """Stage 02 — Validates dataset integrity before preprocessing."""

    def __init__(self, config_path: str, params_path: str):
        try:
            self.config = read_yaml(config_path)
            self.params = read_yaml(params_path)
            self.logger = setup_logger("stage_02_data_validation.log")

            self.ingested_data_path = self.config["data_ingestion"]["processed_data_path"]
            self.report_dir = self.config["data_validation"]["report_dir"]
            self.report_file = os.path.join(self.report_dir, "data_validation_report.json")
            self.expected_columns = self.config["data_validation"]["expected_columns"]
        except Exception as e:
            raise CustomException(e, sys)

    def validate_schema(self, df):
        self.logger.info("Validating schema...")
        actual = df.columns.tolist()
        missing = [c for c in self.expected_columns if c not in actual]
        extra = [c for c in actual if c not in self.expected_columns]
        return {"schema_ok": len(missing) == 0, "missing_columns": missing, "extra_columns": extra}

    def validate_data_quality(self, df):
        total_missing = int(df.isnull().sum().sum())
        duplicates = int(df.duplicated().sum())
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        non_numeric_cols = df.select_dtypes(exclude=["int64", "float64"]).columns.tolist()
        return {
            "total_missing": total_missing,
            "duplicate_rows": duplicates,
            "numeric_columns": numeric_cols,
            "non_numeric_columns": non_numeric_cols,
        }

    def run(self):
        try:
            self.logger.info("========== Stage 02: Data Validation Started ==========")

            tracking_uri = self.config["mlflow"].get("tracking_uri", "")
            mlflow.set_tracking_uri(f"file:{os.getcwd()}/mlruns" if not tracking_uri else tracking_uri)
            mlflow.set_experiment(self.config["mlflow"].get("experiment_name", "flipkart_sentiment_experiment"))

            with mlflow.start_run(run_name="data_validation_stage"):
                df = pd.read_csv(self.ingested_data_path)
                self.logger.info(f"Loaded dataset: {df.shape}")

                schema_report = self.validate_schema(df)
                quality_report = self.validate_data_quality(df)

                report = {
                    "schema_validation": schema_report,
                    "data_quality": quality_report,
                    "timestamp": datetime.utcnow().isoformat()
                }

                os.makedirs(self.report_dir, exist_ok=True)
                with open(self.report_file, "w", encoding="utf-8") as f:
                    json.dump(report, f, indent=4)

                mlflow.log_artifact(self.report_file, artifact_path="data_validation")

                self.logger.info(f"Validation report saved at: {self.report_file}")
                self.logger.info("========== Stage 02: Data Validation Completed ==========")
        except Exception as e:
            self.logger.error("Validation failed.")
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        DataValidation("config/config.yaml", "config/params.yaml").run()
    except Exception as e:
        raise CustomException(e, sys)
