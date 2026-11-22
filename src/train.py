"""Train three candidate delivery regressors with a strictly time-based split.

One command from the repository root:
    python -m src.train

Validation is used for model selection; the test and replay splits are never
loaded into the training/selection matrices.  The chosen model is packaged
with its fitted preprocessing as one local joblib artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from data.generate import ROOT, load_config
from src.features import MODEL_FEATURES, model_matrix
from src.models import model_candidates, predict_delivery
from src.splits import closed_split


def load_model_data(features_path: Path, data_config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load ONLY mature train and validation rows using known time boundaries."""
    frame = pd.read_csv(features_path)
    required = {
        "order_id", "ordered_at", "period", "delivery_hours", "label_available_at",
        *MODEL_FEATURES,
    }
    if not required.issubset(frame.columns):
        raise ValueError(f"Required columns missing: {sorted(required - set(frame.columns))}")
    if frame["order_id"].duplicated().any() or frame["order_id"].isna().any():
        raise ValueError("Order IDs must be unique and nonnull")
    periods = {item["name"]: item for item in data_config["periods"]}
    train_p = periods["train"]
    valid_p = periods["valid"]
    if pd.Timestamp(train_p["end"]) > pd.Timestamp(valid_p["start"]):
        raise ValueError("Train and validation periods overlap")
    result = []
    for name, definition in (("train", train_p), ("valid", valid_p)):
        subset = frame.loc[frame["period"] == name]
        matured = closed_split(
            subset,
            start=definition["start"],
            end=definition["end"],
            as_of=definition["end"],
        )
        if matured.empty:
            raise ValueError(f"No mature labels for {name} at snapshot")
        x = model_matrix(matured)
        target = matured["delivery_hours"].to_numpy(dtype=float)
        if not np.isfinite(target).all() or ((target < 1) | (target > 168)).any():
            raise ValueError("Delivery target outside 1..168 hours")
        if not np.isfinite(x).all():
            raise ValueError("Invalid features")
        result.append(matured)
    train, valid = result
    if set(train["order_id"]) & set(valid["order_id"]):
        raise ValueError("Train and validation orders intersect")
    return train, valid


def regression_metrics(y: np.ndarray, prediction: np.ndarray) -> dict:
    y = np.asarray(y, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if y.ndim != 1 or prediction.shape != y.shape or not len(y):
        raise ValueError("Expected nonempty aligned one-dimensional arrays")
    if not np.isfinite(y).all() or not np.isfinite(prediction).all():
        raise ValueError("Inputs must be finite")
    diff = y - prediction  # Positive values mean the model underestimated delivery time.
    return {
        "n": int(len(y)),
        "mae_hours": float(np.mean(np.abs(diff))),
        "rmse_hours": float(np.sqrt(np.mean(np.square(diff)))),
        "mean_signed_error_hours": float(np.mean(diff)),
        "underestimation_share": float(np.mean(diff > 0)),
        "mean_underestimation_hours": float(np.mean(np.maximum(diff, 0))),
        "p95_underestimation_hours": float(np.quantile(np.maximum(diff, 0), 0.95)),
    }


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train_and_select(
    *,
    features_path: Path,
    data_config_path: Path,
    training_config_path: Path,
    artifact_dir: Path,
    metrics_path: Path,
) -> dict:
    """Fit on mature train, compare on mature valid, save selected model and audit."""
    data_config = load_config(data_config_path)
    training_config = json.loads(training_config_path.read_text(encoding="utf-8"))
    if int(training_config["seed"]) != int(data_config["seed"]):
        raise ValueError("Data/training seeds differ")
    train, valid = load_model_data(features_path, data_config)
    x_train = model_matrix(train)
    x_valid = model_matrix(valid)
    y_train = train["delivery_hours"].to_numpy(dtype=float)
    y_valid = valid["delivery_hours"].to_numpy(dtype=float)
    bounds = tuple(training_config["prediction_range_hours"])

    candidates = model_candidates(training_config)
    measurements = {}
    for name, model in candidates.items():
        model.fit(x_train, y_train)
        train_pred = predict_delivery(model, x_train, bounds)
        valid_pred = predict_delivery(model, x_valid, bounds)
        measurements[name] = {
            "train": regression_metrics(y_train, train_pred),
            "valid": regression_metrics(y_valid, valid_pred),
        }

    metric_name = training_config["selection_metric"]
    if metric_name != "mae_hours":
        raise ValueError("Only validation MAE is allowed for automatic selection")
    # Dict insertion order breaks exact ties in favor of simpler models.
    winner = min(candidates, key=lambda name: measurements[name]["valid"][metric_name])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / "model.joblib"
    joblib.dump(candidates[winner], model_path, compress=3)
    manifest = {
        "model_version": f"delivery-2026.1-{winner}",
        "selected_model": winner,
        "feature_order": list(MODEL_FEATURES),
        "target": "delivery_hours",
        "prediction_bounds_hours": list(bounds),
        "selection_metric": metric_name,
        "model_sha256": _digest(model_path),
        "training_seed": int(training_config["seed"]),
        "train_as_of": next(p["end"] for p in data_config["periods"] if p["name"] == "train"),
        "valid_as_of": next(p["end"] for p in data_config["periods"] if p["name"] == "valid"),
        "data_sha256": _digest(features_path),
        "config_sha256": _digest(training_config_path),
    }
    (artifact_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    result = {
        "stage": "02_training_and_model_selection",
        "selected_model": winner,
        "selection_metric": metric_name,
        "train_orders_mature": int(len(train)),
        "valid_orders_mature": int(len(valid)),
        "models": measurements,
        "manifest": manifest,
        "versions": {
            "numpy": np.__version__, "pandas": pd.__version__,
            "sklearn": sklearn.__version__, "joblib": joblib.__version__,
        },
    }
    # Keep machine-readable metrics separate from the human-facing final report.
    metrics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and compare delivery models")
    parser.add_argument("--features", type=Path, default=ROOT / "data/generated/features.csv")
    parser.add_argument("--data-config", type=Path, default=ROOT / "configs/data.json")
    parser.add_argument("--train-config", type=Path, default=ROOT / "configs/training.json")
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--metrics", type=Path, default=ROOT / "reports/training_metrics.json")
    args = parser.parse_args()
    result = train_and_select(
        features_path=args.features,
        data_config_path=args.data_config,
        training_config_path=args.train_config,
        artifact_dir=args.artifacts,
        metrics_path=args.metrics,
    )
    print(json.dumps({
        "selected_model": result["selected_model"],
        "train_mature": result["train_orders_mature"],
        "valid_mature": result["valid_orders_mature"],
        "validation_mae_hours": {
            name: round(detail["valid"]["mae_hours"], 6)
            for name, detail in result["models"].items()
        },
        "model_version": result["manifest"]["model_version"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
