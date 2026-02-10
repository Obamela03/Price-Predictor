# core/ml/train.py
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler as GeoScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

try:
    from xgboost import XGBRegressor
except ImportError as e:
    raise ImportError("xgboost is not installed. Run: pip install xgboost") from e


ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "xgboost_model.pkl"
SCALER_PATH = ARTIFACTS_DIR / "scaler.pkl"
FEATURES_PATH = ARTIFACTS_DIR / "feature_names.json"
GEO_CLUSTERER_PATH = ARTIFACTS_DIR / "geo_kmeans.pkl"
GEO_K_REPORT_PATH = ARTIFACTS_DIR / "geo_k_report.json"


def drop_highly_correlated_features(X: pd.DataFrame, threshold: float = 0.80):
    """
    Drops features that are highly correlated (upper triangle method).
    Returns: X_selected, dropped_features(list)
    """
    corr = X.corr(numeric_only=True).abs()
    upper_triangle = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper_triangle.columns if any(upper_triangle[col] > threshold)]
    return X.drop(columns=to_drop), to_drop


def choose_k_for_geo_clusters(lat_lon: pd.DataFrame, k_min: int = 3, k_max: int = 10) -> tuple[int, list[dict]]:
    """
    Picks a reasonable k using silhouette (higher better) and Davies–Bouldin (lower better).
    Returns: (best_k, report_rows)
    """
    Xs = GeoScaler().fit_transform(lat_lon)

    report = []
    best_k = None
    best_score = -np.inf

    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(Xs)

        sil = silhouette_score(Xs, labels)
        db = davies_bouldin_score(Xs, labels)

        # Simple combined score: prioritize silhouette; lightly penalize DB.
        combined = sil - 0.05 * db

        report.append({
            "k": k,
            "silhouette": float(sil),
            "davies_bouldin": float(db),
            "combined_score": float(combined),
            "inertia": float(km.inertia_),
        })

        if combined > best_score:
            best_score = combined
            best_k = k

    # Practical guardrail: avoid too many clusters for UX
    # (If best_k is very high, clamp to 7 unless you explicitly want more.)
    if best_k is None:
        best_k = 6
    best_k = min(best_k, 7)

    return best_k, report


def main():
    # 1) Load dataset
    data = fetch_california_housing(as_frame=True)
    df = data.frame.copy()
    df.rename(columns={"MedHouseVal": "target"}, inplace=True)

    X_full = df.drop(columns=["target"])
    y = df["target"]  # in $100,000 units

    # 2) Train geo clusterer on lat/long (feature engineering step)
    lat_lon = X_full[["Latitude", "Longitude"]].copy()

    best_k, k_report = choose_k_for_geo_clusters(lat_lon, k_min=3, k_max=10)

    geo_clusterer = Pipeline(steps=[
        ("scaler", GeoScaler()),
        ("kmeans", KMeans(n_clusters=best_k, n_init=10, random_state=42)),
    ])
    geo_labels = geo_clusterer.fit_predict(lat_lon)

    # Add GeoCluster feature
    X_geo = X_full.copy()
    X_geo["GeoCluster"] = geo_labels.astype(int)

    # Optional: drop raw lat/long so users don't need to input coordinates
    # If you want to keep them for experimentation, comment these two lines out.
    X_geo = X_geo.drop(columns=["Latitude", "Longitude"])

    # 3) Feature selection (drop high correlations) AFTER geo engineering
    X_selected, dropped = drop_highly_correlated_features(X_geo, threshold=0.80)

    # 4) Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42
    )

    # 5) Scale numeric features (optional for XGBoost; kept for consistent pipeline)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 6) Train XGBoost
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

    # 7) Evaluate
    preds = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print("Geo clusters (k):", best_k)
    print("Dropped due to correlation:", dropped)
    print("Features used:", list(X_selected.columns))
    print(f"MAE (in $100k units): {mae:.4f}  | approx ${mae*100_000:,.0f}")
    print(f"R²: {r2:.4f}")

    # 8) Save artifacts
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(geo_clusterer, GEO_CLUSTERER_PATH)

    FEATURES_PATH.write_text(json.dumps(list(X_selected.columns), indent=2))
    GEO_K_REPORT_PATH.write_text(json.dumps({
        "chosen_k": best_k,
        "k_report": k_report
    }, indent=2))

    print("\nSaved:")
    print(f"- {MODEL_PATH}")
    print(f"- {SCALER_PATH}")
    print(f"- {GEO_CLUSTERER_PATH} (for GeoCluster inference)")
    print(f"- {FEATURES_PATH} (important for input order)")
    print(f"- {GEO_K_REPORT_PATH} (k selection report)")


if __name__ == "__main__":
    main()
