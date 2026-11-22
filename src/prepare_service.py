"""Package the *validation-derived* prediction interval for online serving.

The service artifact is model-version-bound and contains NO test or replay
labels. A retrained model invalidates the previous service artifact and must
have its interval rebuilt before it can become ready.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from data.generate import ROOT, load_config
from src.bundle import load_bundle, predict_frame
from src.evaluate import interval_half_width
from src.train import load_model_data


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_service_artifact(
    *, features_path: Path, data_config_path: Path,
    evaluation_config_path: Path, artifact_dir: Path,
) -> dict:
    """Store residual quantile fitted exclusively on mature validation rows."""
    model, manifest = load_bundle(artifact_dir)
    if _sha256(features_path) != manifest["data_sha256"]:
        raise ValueError("Data snapshot no longer matches the trained model")
    data_config = load_config(data_config_path)
    config = json.loads(evaluation_config_path.read_text(encoding="utf-8"))
    if int(data_config["seed"]) != int(config["seed"]):
        raise ValueError("Calibration seed and data seed differ")
    _, valid = load_model_data(features_path, data_config)
    pred = predict_frame(model, manifest, valid)
    truth = valid.delivery_hours.to_numpy(dtype=float)
    coverage = float(config["prediction_interval_coverage"])
    half_width = interval_half_width(np.abs(truth - pred), coverage)
    result = {
        "schema_version": 1,
        "model_version": manifest["model_version"],
        "model_sha256": manifest["model_sha256"],
        "manifest_sha256": _sha256(artifact_dir / "manifest.json"),
        "data_sha256": manifest["data_sha256"],
        "feature_order": manifest["feature_order"],
        "interval_coverage": coverage,
        "interval_half_width_hours": half_width,
        "calibration_orders": int(len(valid)),
        "calibration_source": "mature_validation_only",
        "limitations": "Validation was also used for model selection; formal split-conformal coverage is not claimed.",
    }
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "serving.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare validation-only service calibration")
    parser.add_argument("--features", type=Path, default=ROOT / "data/generated/features.csv")
    parser.add_argument("--data-config", type=Path, default=ROOT / "configs/data.json")
    parser.add_argument("--eval-config", type=Path, default=ROOT / "configs/evaluation.json")
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    args = parser.parse_args()
    result = prepare_service_artifact(
        features_path=args.features,
        data_config_path=args.data_config,
        evaluation_config_path=args.eval_config,
        artifact_dir=args.artifacts,
    )
    print(json.dumps({k: result[k] for k in (
        "model_version", "interval_coverage", "interval_half_width_hours", "calibration_orders"
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
