# core/ml/train.py
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

try:
    from xgboost import XGBRegressor
except ImportError as e:
    raise ImportError(
        "xgboost is not installed. Run: pip install xgboost"
    ) from e


ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "xgboost_model.pkl"
SCALER_PATH = ARTIFACTS_DIR / "scaler.pkl"
FEATURES_PATH = ARTIFACTS_DIR / "feature_names.json"


def drop_highly_correlated_features(X, threshold: float = 0.80):
    """
    Drops features that are highly correlated (upper triangle method).
    Returns: X_selected, dropped_features(list)
    """
    corr = X.corr(numeric_only=True).abs()
    upper_triangle = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    to_drop = [
        col for col in upper_triangle.columns
        if any(upper_triangle[col] > threshold)
    ]
    return X.drop(columns=to_drop), to_drop


def main():
    # 1) Load dataset
    data = fetch_california_housing(as_frame=True)
    df = data.frame.copy()
    df.rename(columns={"MedHouseVal": "target"}, inplace=True)

    X = df.drop(columns=["target"])
    y = df["target"]  # target is in $100,000 units

    # 2) Feature selection (drop high correlations)
    X_selected, dropped = drop_highly_correlated_features(X, threshold=0.80)

    # 3) Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42
    )

    # 4) Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5) Train XGBoost
    model = XGBRegressor(
        n_estimators=800,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        objective="reg:squarederror",
    )
    model.fit(X_train_scaled, y_train)

    # 6) Evaluate
    preds = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print("Dropped due to correlation:", dropped)
    print("Features used:", list(X_selected.columns))
    print(f"MAE (in $100k units): {mae:.4f}  | approx ${mae*100_000:,.0f}")
    print(f"R²: {r2:.4f}")

    # 7) Save artifacts
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    FEATURES_PATH.write_text(json.dumps(list(X_selected.columns), indent=2))

    print("\nSaved:")
    print(f"- {MODEL_PATH}")
    print(f"- {SCALER_PATH}")
    print(f"- {FEATURES_PATH} (important for input order)")


if __name__ == "__main__":
    main()
