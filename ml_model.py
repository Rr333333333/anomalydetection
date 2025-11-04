# ml_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib
import os

MODEL_PATH = "model.pkl"
METRICS_PATH = "metrics.pkl"

def train_model(dataset_path, target_column):
    """
    Trains a simple Logistic Regression model using the uploaded dataset.
    Saves the model and evaluation metrics.
    """

    # Load dataset
    df = pd.read_csv(dataset_path)

    # Basic check
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset")

    # Split features and labels
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Create and train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)

    # Save model and metrics
    joblib.dump(model, MODEL_PATH)
    joblib.dump({'accuracy': accuracy, 'precision': precision, 'recall': recall}, METRICS_PATH)

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4)
    }

def predict(new_data_path):
    """
    Uses the saved model to make predictions on new data.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found. Please train the model first.")

    model = joblib.load(MODEL_PATH)
    new_data = pd.read_csv(new_data_path)
    predictions = model.predict(new_data)
    return predictions.tolist()

def get_metrics():
    """
    Returns the stored accuracy, precision, and recall.
    """
    if os.path.exists(METRICS_PATH):
        metrics = joblib.load(METRICS_PATH)
        return metrics
    else:
        return {"accuracy": None, "precision": None, "recall": None}
