"""Cross-check model and validation-derived interval before serving."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from src.bundle import load_bundle
from src.features import MODEL_FEATURES


def load_serving_bundle(artifact_dir: Path) -> tuple[object, dict, dict]:
    model, manifest = load_bundle(artifact_dir)
    serving = json.loads((artifact_dir / "serving.json").read_text(encoding="utf-8"))
    if serving.get("schema_version") != 1:
        raise ValueError("Unsupported service artifact version")
    if serving.get("manifest_sha256") != hashlib.sha256(
        (artifact_dir / "manifest.json").read_bytes()
    ).hexdigest():
        raise ValueError("Service calibration does not match current model manifest")
    for key in ("model_version", "model_sha256", "data_sha256", "feature_order"):
        if serving.get(key) != manifest.get(key):
            raise ValueError(f"Service artifact mismatch: {key}")
    if serving["feature_order"] != list(MODEL_FEATURES):
        raise ValueError("Invalid serving feature order")
    coverage = float(serving["interval_coverage"])
    width = float(serving["interval_half_width_hours"])
    if not (0 < coverage < 1 and math.isfinite(coverage)):
        raise ValueError("Invalid interval coverage")
    if not (math.isfinite(width) and width >= 0):
        raise ValueError("Invalid interval half-width")
    if int(serving["calibration_orders"]) < 1 or serving["calibration_source"] not in ("mature_validation_only", "mature_replay_calibration"):
        raise ValueError("Invalid calibration provenance")
    return model, manifest, serving
