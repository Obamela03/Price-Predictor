# core/ml/predict.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import joblib
import numpy as np


ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "xgboost_model.pkl"
SCALER_PATH = ARTIFACTS_DIR / "scaler.pkl"


@dataclass(frozen=True)
class PredictionResult:
    prediction: float
    model_loaded: bool


class HousePricePredictor:
    """
    Loads model artifacts once and exposes a predict() method.

    Expected input order must match training:
    Example (California Housing):
    [MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]
    """

    def __init__(self, model_path: Path = MODEL_PATH, scaler_path: Path = SCALER_PATH):
        self.model_path = model_path
        self.scaler_path = scaler_path

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self.model = joblib.load(self.model_path)

        # scaler is optional (you might not use one)
        self.scaler = joblib.load(self.scaler_path) if self.scaler_path.exists() else None

    def predict(self, features: list[float]) -> PredictionResult:
        x = np.array(features, dtype=float).reshape(1, -1)

        if self.scaler is not None:
            x = self.scaler.transform(x)

        pred = float(self.model.predict(x)[0])
        return PredictionResult(prediction=pred, model_loaded=True)


# Singleton-like instance (loaded once when Django imports the module)
_predictor: HousePricePredictor | None = None


def get_predictor() -> HousePricePredictor:
    global _predictor
    if _predictor is None:
        _predictor = HousePricePredictor()
    return _predictor


def predict_price(features: list[float]) -> float:
    """
    Convenience function for views.
    """
    predictor = get_predictor()
    return predictor.predict(features).prediction
