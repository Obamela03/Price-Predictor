# core/views.py
from __future__ import annotations

import json
from pathlib import Path

from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
from django.views.decorators.http import require_http_methods
from django.contrib.auth import login
from django.contrib.auth.forms import UserCreationForm

from .ml.predict import predict_price

ARTIFACTS_DIR = Path(__file__).resolve().parent / "ml" / "artifacts"
FEATURES_PATH = ARTIFACTS_DIR / "feature_names.json"
GEO_K_REPORT_PATH = ARTIFACTS_DIR / "geo_k_report.json"
GEO_LABELS_PATH = ARTIFACTS_DIR / "geo_cluster_labels.json"


def load_feature_names() -> list[str]:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Missing {FEATURES_PATH}. Train first: python core/ml/train.py"
        )

    feature_names = json.loads(FEATURES_PATH.read_text(encoding="utf-8"))
    if not isinstance(feature_names, list) or not all(isinstance(x, str) for x in feature_names):
        raise ValueError(f"{FEATURES_PATH} must be a JSON list of strings.")
    return feature_names


@login_required
@require_http_methods(["GET", "POST"])
def predict_view(request):
    try:
        fields = load_feature_names()
    except Exception as e:
        return render(request, "core/predict.html", {"fields": [], "error": str(e)})

    context = {
        "fields": fields,
        "submitted": {},
    }

    geo_k_range, geo_cluster_labels = load_geo_dropdown()
    context["geo_k_range"] = geo_k_range
    context["geo_cluster_labels"] = geo_cluster_labels

    if request.method == "POST":
        submitted: dict[str, str] = {}
        raw_features: dict[str, float] = {}

        try:
            # If your final training uses GeoCluster (and dropped Latitude/Longitude),
            # the form must provide GeoCluster (as a number or a dropdown).
            for name in fields:
                raw = request.POST.get(name, "")
                submitted[name] = raw
                raw = (raw or "").strip()

                if raw == "":
                    raise ValueError(f"Missing value for {name}")

                # GeoCluster must be an integer label, but we can parse it as float then cast
                if name == "GeoCluster":
                    raw_features[name] = float(int(float(raw)))
                else:
                    raw_features[name] = float(raw)

            # Predict (returns target in $100,000 units)
            y_pred = predict_price(raw_features)

            context["prediction"] = round(y_pred * 100_000, 2)
            context["raw_prediction"] = float(y_pred)
            context["submitted"] = submitted

        except Exception as e:
            context["error"] = f"Prediction failed: {e}"
            context["submitted"] = submitted

    return render(request, "core/predict.html", context)


@require_http_methods(["GET", "POST"])
def register_view(request):
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)  # auto-login after successful registration
            return redirect("predict")
    else:
        form = UserCreationForm()

    return render(request, "registration/register.html", {"form": form})

def load_geo_dropdown():
    """
    Returns (geo_k_range, geo_cluster_labels)
    geo_cluster_labels is a list of {"value": int, "label": str} if available.
    """
    geo_k = 6  # fallback
    if GEO_K_REPORT_PATH.exists():
        report = json.loads(GEO_K_REPORT_PATH.read_text(encoding="utf-8"))
        geo_k = int(report.get("chosen_k", geo_k))

    geo_k_range = list(range(geo_k))

    geo_cluster_labels = None
    if GEO_LABELS_PATH.exists():
        labels = json.loads(GEO_LABELS_PATH.read_text(encoding="utf-8"))
        # Expecting dict: {"0": "South Coast", "1": "Central Valley", ...}
        if isinstance(labels, dict):
            geo_cluster_labels = [{"value": int(k), "label": str(v)} for k, v in labels.items()]
            geo_cluster_labels = sorted(geo_cluster_labels, key=lambda x: x["value"])

    return geo_k_range, geo_cluster_labels
