
from pathlib import Path

import joblib
import pandas as pd

# Locate project folders
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_DIR / "models"


def load_regression_model():
    """Load the trained regression model."""
    path = MODELS_DIR / "best_regression_model.joblib"

    if not path.exists():
        raise FileNotFoundError(f"Regression model not found: {path}")

    return joblib.load(path)


def load_classification_model():
    """Load the trained classification model."""
    path = MODELS_DIR / "best_classification_model.joblib"

    if not path.exists():
        raise FileNotFoundError(f"Classification model not found: {path}")

    return joblib.load(path)


def predict_demand(input_data):
    """
    Predict charging demand.

    input_data can be a dictionary or a pandas DataFrame.
    Its columns must match the model's training input columns.
    """
    model = load_regression_model()

    if isinstance(input_data, dict):
        input_data = pd.DataFrame([input_data])

    if not isinstance(input_data, pd.DataFrame):
        raise TypeError("Input must be a dictionary or pandas DataFrame.")

    return model.predict(input_data)


def predict_high_demand(input_data):
    """
    Classify demand using the trained classification model.
    Returns the model's predicted class.
    """
    model = load_classification_model()

    if isinstance(input_data, dict):
        input_data = pd.DataFrame([input_data])

    if not isinstance(input_data, pd.DataFrame):
        raise TypeError("Input must be a dictionary or pandas DataFrame.")

    return model.predict(input_data)
