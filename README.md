Project Description

Flipkart Customer Sentiment Analysis is an end-to-end MLOps project designed to predict customer sentiment from product reviews. The pipeline automates the complete workflow from raw data ingestion to model deployment, ensuring reproducibility, scalability, and monitoring using MLflow and DagsHub.

What is Done

Data Ingestion: Loads raw Flipkart reviews, checks for missing or duplicate entries, and saves cleaned datasets.

Data Validation: Validates dataset schema and data quality (missing values, duplicate rows, numeric and non-numeric columns).

Data Transformation: Preprocesses text by cleaning, lemmatizing, and creating additional features like review length and average word length.

Model Training: Trains two models:

RandomForest with TF-IDF and meta-features

DistilBERT transformer for fine-grained sentiment analysis
Logs metrics and artifacts using MLflow.

Model Serving: Deploys a FastAPI service for real-time sentiment predictions.

How It Works

Configuration-driven: All paths, hyperparameters, and experiment settings are managed via config.yaml and params.yaml.

MLflow Tracking: Captures parameters, metrics, artifacts, and experiment runs for monitoring and reproducibility.

DagsHub Integration: Centralizes experiment tracking, metrics visualization, and model versioning.

CI/CD Pipeline: Automated via GitHub Actions, running all stages from ingestion to model training and pushing artifacts back to the repository.

Prediction API: Users can send review text to the FastAPI endpoint and receive predicted sentiment (Positive, Neutral, Negative).