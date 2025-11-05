import os
import sys
import torch
import joblib
import numpy as np
from scipy import sparse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

from src.logger import setup_logger
from src.exception import CustomException
from src.utils import read_yaml

# ---------- FastAPI setup ----------
app = FastAPI(title="Flipkart Sentiment Analysis API", version="1.0")

# ---------- Input Schema ----------
class ReviewInput(BaseModel):
    text: str

# ---------- Load model and config ----------
try:
    config = read_yaml("config/config.yaml")
    params = read_yaml("config/params.yaml")
    logger = setup_logger("serve_api.log")

    artifact_dir = config["artifact_dir"]
    model_dir = config["model_trainer"]["model_dir"]

    best_model_path = os.path.join(artifact_dir, "best_model.pkl")
    best_model_folder = os.path.join(artifact_dir, "best_model")

    device = "cuda" if torch.cuda.is_available() and params["train"]["use_gpu"] else "cpu"

    # Detect model type
    if os.path.exists(os.path.join(best_model_folder, "config.json")):
        model_type = "DistilBERT"
    elif os.path.exists(best_model_path):
        model_type = "RandomForest"
    else:
        raise FileNotFoundError("No best model found in artifacts.")

    logger.info(f"Serving model type: {model_type}")

    if model_type == "RandomForest":
        vectorizer_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")
        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError("TF-IDF vectorizer missing.")
        vectorizer = joblib.load(vectorizer_path)
        model = joblib.load(best_model_path)
        meta_cols = ["product_price", "Rate", "review_length", "avg_word_length"]
    else:
        tokenizer = DistilBertTokenizerFast.from_pretrained(best_model_folder)
        model = DistilBertForSequenceClassification.from_pretrained(best_model_folder).to(device)
        model.eval()

except Exception as e:
    raise CustomException(e, sys)


# ---------- Helper Functions ----------
def predict_rf(text: str):
    meta_data = {
        "product_price": 0,
        "Rate": 0,
        "review_length": len(text),
        "avg_word_length": np.mean([len(w) for w in text.split()]) if text.split() else 0
    }
    X_tfidf = vectorizer.transform([text])
    X_meta = sparse.csr_matrix([[meta_data.get(col, 0) for col in meta_cols]])
    X_combined = sparse.hstack([X_tfidf, X_meta])
    pred = model.predict(X_combined)[0]
    return int(pred)


def predict_bert(text: str):
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=params["distilbert"]["tokenizer_max_length"],
        return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).cpu().item()
    return int(pred)


def get_sentiment_label(pred):
    sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return sentiment_map.get(pred, "Unknown")


# ---------- API Endpoint ----------
@app.post("/predict")
async def predict_sentiment(review: ReviewInput):
    text = review.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    logger.info(f"Received text for prediction: {text}")

    try:
        if model_type == "RandomForest":
            pred = predict_rf(text)
        else:
            pred = predict_bert(text)

        sentiment = get_sentiment_label(pred)
        logger.info(f"Predicted Sentiment: {sentiment}")
        return {"text": text, "predicted_sentiment": sentiment}

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------- Root endpoint ----------
@app.get("/")
async def root():
    return {"message": "Flipkart Sentiment Analysis API is running."}


# ---------- Run with: ----------
# uvicorn serve:app --host 0.0.0.0 --port 8000
