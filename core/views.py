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

    context = {"fields": fields, "submitted": {}}

    if request.method == "POST":
        submitted = {}
        try:
            features = []
            for name in fields:
                raw = request.POST.get(name, "")
                submitted[name] = raw
                raw = (raw or "").strip()
                if raw == "":
                    raise ValueError(f"Missing value for {name}")
                features.append(float(raw))

            y_pred = predict_price(features)

            # California housing target is in 100k units
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