# core/ml/predict.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import joblib
import numpy as np


ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "xgboost_model.pkl"
SCALER_PATH = ARTIFACTS_DIR / "scaler.pkl"
FEATURES_PATH = ARTIFACTS_DIR / "feature_names.json"
GEO_CLUSTERER_PATH = ARTIFACTS_DIR / "geo_kmeans.pkl"


@dataclass(frozen=True)
class PredictionResult:
    prediction: float
    model_loaded: bool


class HousePricePredictor:
    """
    Loads model artifacts once and exposes predict().

    This version supports GeoCluster feature engineering (K-Means).
    The final feature order is loaded from feature_names.json.
    """

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        scaler_path: Path = SCALER_PATH,
        features_path: Path = FEATURES_PATH,
        geo_clusterer_path: Path = GEO_CLUSTERER_PATH,
    ):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.features_path = features_path
        self.geo_clusterer_path = geo_clusterer_path

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        if not self.features_path.exists():
            raise FileNotFoundError(f"Feature names not found: {self.features_path}")

        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path) if self.scaler_path.exists() else None

        # Geo clustering artifact is optional depending on your final features
        self.geo_clusterer = joblib.load(self.geo_clusterer_path) if self.geo_clusterer_path.exists() else None

        self.feature_names: list[str] = json.loads(self.features_path.read_text())

    def _ensure_feature_vector(
        self,
        raw_features: dict[str, float],
        *,
        latitude: float | None = None,
        longitude: float | None = None,
    ) -> np.ndarray:
        """
        Builds the feature vector in the exact order used during training.
        If 'GeoCluster' is required and not provided, it can be computed from latitude/longitude.
        """
        features = dict(raw_features)

        if "GeoCluster" in self.feature_names:
            if "GeoCluster" not in features:
                if self.geo_clusterer is None:
                    raise ValueError("GeoCluster is required but geo_clusterer artifact is missing.")
                if latitude is None or longitude is None:
                    raise ValueError("GeoCluster is required. Provide latitude and longitude to compute it.")
                geo_x = np.array([[float(latitude), float(longitude)]], dtype=float)
                features["GeoCluster"] = int(self.geo_clusterer.predict(geo_x)[0])

        # Build ordered vector
        try:
            vec = [float(features[name]) for name in self.feature_names]
        except KeyError as e:
            missing = [n for n in self.feature_names if n not in features]
            raise ValueError(f"Missing required features: {missing}") from e

        return np.array(vec, dtype=float).reshape(1, -1)

    def predict(self, raw_features: dict[str, float], *, latitude: float | None = None, longitude: float | None = None) -> PredictionResult:
        x = self._ensure_feature_vector(raw_features, latitude=latitude, longitude=longitude)

        if self.scaler is not None:
            x = self.scaler.transform(x)

        pred = float(self.model.predict(x)[0])
        return PredictionResult(prediction=pred, model_loaded=True)


_predictor: HousePricePredictor | None = None


def get_predictor() -> HousePricePredictor:
    global _predictor
    if _predictor is None:
        _predictor = HousePricePredictor()
    return _predictor


def predict_price(raw_features: dict[str, float], *, latitude: float | None = None, longitude: float | None = None) -> float:
    predictor = get_predictor()
    return predictor.predict(raw_features, latitude=latitude, longitude=longitude).prediction
