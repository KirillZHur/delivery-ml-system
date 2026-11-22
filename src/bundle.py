"""Integrity-checked loading and scoring of a trained delivery model."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from src.features import MODEL_FEATURES, model_matrix
from src.models import predict_delivery


def load_bundle(directory: Path) -> tuple[object, dict]:
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["feature_order"] != list(MODEL_FEATURES):
        raise ValueError("Feature contract mismatch in model manifest")
    if manifest.get("target") != "delivery_hours":
        raise ValueError("Unsupported target")
    model_file = directory / "model.joblib"
    digest = hashlib.sha256(model_file.read_bytes()).hexdigest()
    if digest != manifest["model_sha256"]:
        raise ValueError("Trained model checksum mismatch")
    # Only load locally produced, trusted joblib artifacts. Never load user uploads.
    return joblib.load(model_file), manifest


def predict_frame(model: object, manifest: dict, frame) -> np.ndarray:
    x = model_matrix(frame)
    return predict_delivery(model, x, tuple(manifest["prediction_bounds_hours"]))
