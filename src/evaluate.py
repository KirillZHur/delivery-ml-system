"""Final *closed* test evaluation for the selected delivery regressor.

Re-uses the selected, integrity-checked artifact from stage 2. Fits only the
constant baseline to mature *train* labels and estimates the 90% prediction
interval half-width from mature *validation* residuals. Opens the March test
once for a final paired comparison and evaluates predeclared slices.

Machine-readable files only; README / human-facing final report come later.
From the repository root:
    python -m src.evaluate
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from data.generate import ROOT, load_config
from src.bundle import load_bundle, predict_frame
from src.features import MODEL_FEATURES, model_matrix
from src.models import MedianRegressor, predict_delivery
from src.splits import closed_split
from src.train import regression_metrics


def paired_day_bootstrap_mae(
    y: np.ndarray,
    baseline_prediction: np.ndarray,
    candidate_prediction: np.ndarray,
    days: np.ndarray,
    *,
    n_boot: int = 1999,
    seed: int = 2026,
    confidence: float = 0.95,
) -> dict:
    """Paired cluster bootstrap of MAE(baseline) - MAE(candidate).

    Each replicate draws UTC day groups with replacement. All orders and both
    forecasts in a sampled group travel together. Day sizes may differ, so
    group *sums* and group *counts* are resampled before computing the MAE.
    """
    target = np.asarray(y, dtype=float)
    a = np.asarray(baseline_prediction, dtype=float)
    b = np.asarray(candidate_prediction, dtype=float)
    days = np.asarray(days)
    if target.ndim != 1 or len(target) == 0 or a.shape != target.shape or b.shape != target.shape:
        raise ValueError("Expected nonempty, aligned one-dimensional predictions")
    if days.ndim != 1 or len(days) != len(target):
        raise ValueError("Day identifiers must match the number of orders")
    if not (np.isfinite(target).all() and np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("Target and predictions must be finite")
    if int(n_boot) != n_boot or n_boot < 1 or not 0 < confidence < 1:
        raise ValueError("Invalid bootstrap configuration")
    if any(pd.isna(days)):
        raise ValueError("Day identifiers must not be null")
    codes, uniques = pd.factorize(days, sort=True)
    if len(uniques) < 2:
        raise ValueError("At least two independent time blocks are required")

    loss_a = np.abs(target - a)
    loss_b = np.abs(target - b)
    delta = float(np.mean(loss_a - loss_b))
    counts = np.bincount(codes, minlength=len(uniques)).astype(float)
    sum_a = np.bincount(codes, weights=loss_a, minlength=len(uniques))
    sum_b = np.bincount(codes, weights=loss_b, minlength=len(uniques))
    rng = np.random.default_rng(seed)
    # Shape: [repetitions, number of original day groups]. Repeated group
    # indices repeat both models' losses and every associated order.
    sampled = rng.integers(0, len(uniques), size=(n_boot, len(uniques)))
    denom = counts[sampled].sum(axis=1)
    differences = (sum_a[sampled] - sum_b[sampled]).sum(axis=1) / denom
    alpha = (1 - confidence) / 2
    lo, hi = np.quantile(differences, [alpha, 1 - alpha])
    return {
        "metric": "baseline_mae_minus_candidate_mae_hours",
        "estimate_hours": delta,
        "ci_lower_hours": float(lo),
        "ci_upper_hours": float(hi),
        "confidence": float(confidence),
        "resampling_unit": "utc_order_day",
        "n_blocks": int(len(uniques)),
        "n_orders": int(len(target)),
        "repeats": int(n_boot),
        "seed": int(seed),
    }


def interval_half_width(abs_validation_residuals: np.ndarray, coverage: float = 0.90) -> float:
    """Finite-sample residual quantile at ceil((n+1)*coverage) / n.

    This is a held-out-residual-style quantile, but model *selection* has
    already inspected the same validation split in stage 2. Consequently, the
    formal split-conformal coverage theorem is NOT claimed for this project.
    Final empirical coverage is measured once on test instead.
    """
    scores = np.asarray(abs_validation_residuals, dtype=float)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError("Residuals must be nonempty, finite and nonnegative")
    if not 0 < coverage < 1:
        raise ValueError("Coverage must be between 0 and 1")
    rank = math.ceil((len(scores) + 1) * coverage)
    if rank > len(scores):
        raise ValueError("Too few calibration points for finite requested quantile")
    return float(np.partition(scores, rank - 1)[rank - 1])


def prediction_intervals(
    prediction: np.ndarray,
    half_width: float,
    *,
    bounds: tuple[float, float] = (1.0, 168.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Clip symmetric nonconformity intervals to the target's known domain."""
    pred = np.asarray(prediction, dtype=float)
    low, high = map(float, bounds)
    if pred.ndim != 1 or not np.isfinite(pred).all():
        raise ValueError("Expected finite 1D predictions")
    if not np.isfinite(half_width) or half_width < 0 or low >= high:
        raise ValueError("Invalid interval width or bounds")
    return np.maximum(low, pred - half_width), np.minimum(high, pred + half_width)


def interval_diagnostics(
    y: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> dict:
    target, lo, hi = (np.asarray(v, dtype=float) for v in (y, lower, upper))
    if target.ndim != 1 or not len(target) or lo.shape != target.shape or hi.shape != target.shape:
        raise ValueError("Interval inputs must be nonempty aligned arrays")
    if not all(np.isfinite(v).all() for v in (target, lo, hi)) or (lo > hi).any():
        raise ValueError("Invalid interval values")
    return {
        "n": int(len(target)),
        "empirical_coverage": float(np.mean((target >= lo) & (target <= hi))),
        "mean_width_hours": float(np.mean(hi - lo)),
        "median_width_hours": float(np.median(hi - lo)),
    }


def select_slice(frame: pd.DataFrame, definition: dict) -> np.ndarray:
    """Apply declarative, prespecified feature-only filters (never labels)."""
    conditions = definition["conditions"]
    if not isinstance(conditions, list) or not conditions:
        raise ValueError("Slice needs one or more conditions")
    included = np.ones(len(frame), dtype=bool)
    for item in conditions:
        feature, operator, value = item["feature"], item["op"], item["value"]
        if feature not in MODEL_FEATURES:
            raise ValueError(f"Slice cannot use undeclared feature: {feature}")
        col = pd.to_numeric(frame[feature], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(col).all() or not np.isfinite(value):
            raise ValueError("Slice conditions need finite numbers")
        if operator == "ge":
            included &= col >= float(value)
        elif operator == "lt":
            included &= col < float(value)
        elif operator == "eq":
            included &= col == float(value)
        else:
            raise ValueError(f"Unsupported slice operator: {operator}")
    return included


def evaluate_predictions(
    *,
    test: pd.DataFrame,
    valid: pd.DataFrame,
    train: pd.DataFrame,
    candidate: object,
    manifest: dict,
    config: dict,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Pure scoring core: no training or selection on test or replay labels."""
    if not all(len(f) for f in (test, valid, train)):
        raise ValueError("Train, validation, and test must be nonempty")
    ids = [set(f.order_id) for f in (train, valid, test)]
    if any(ids[i] & ids[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Train/valid/test order IDs must be disjoint")
    bounds = tuple(manifest["prediction_bounds_hours"])
    baseline = MedianRegressor().fit(model_matrix(train), train.delivery_hours.to_numpy(dtype=float))
    true_valid = valid.delivery_hours.to_numpy(dtype=float)
    pred_valid = predict_frame(candidate, manifest, valid)
    q = interval_half_width(np.abs(true_valid - pred_valid), config["prediction_interval_coverage"])
    val_lower, val_upper = prediction_intervals(pred_valid, q, bounds=bounds)
    val_pi = interval_diagnostics(true_valid, val_lower, val_upper)

    target = test.delivery_hours.to_numpy(dtype=float)
    baseline_pred = predict_delivery(baseline, model_matrix(test), bounds)
    candidate_pred = predict_frame(candidate, manifest, test)
    lower, upper = prediction_intervals(candidate_pred, q, bounds=bounds)
    pi = interval_diagnostics(target, lower, upper)
    blocks = pd.to_datetime(test.ordered_at, utc=True).dt.strftime("%Y-%m-%d").to_numpy()
    kwargs = dict(
        n_boot=int(config["bootstrap_repeats"]),
        seed=int(config["seed"]),
        confidence=float(config["bootstrap_confidence"]),
    )
    overall_delta = paired_day_bootstrap_mae(target, baseline_pred, candidate_pred, blocks, **kwargs)

    slice_rows: list[dict] = []
    for i, definition in enumerate(config["mandatory_slices"]):
        mask = select_slice(test, definition)
        n = int(mask.sum())
        result = {
            "slice": definition["name"],
            "mandatory": True,
            "n_orders": n,
            "n_days": int(len(np.unique(blocks[mask]))),
            "status": "evaluated" if n >= int(config["min_slice_orders"]) else "insufficient_data",
            "baseline_mae_hours": None,
            "candidate_mae_hours": None,
            "delta_mae_hours": None,
            "ci_lower_hours": None,
            "ci_upper_hours": None,
            "empirical_interval_coverage": None,
            "mean_interval_width_hours": None,
            "candidate_mean_underestimation_hours": None,
        }
        if result["status"] == "evaluated" and result["n_days"] >= 2:
            part_delta = paired_day_bootstrap_mae(
                target[mask], baseline_pred[mask], candidate_pred[mask], blocks[mask],
                **{**kwargs, "seed": int(config["seed"]) + i + 1},
            )
            part_pi = interval_diagnostics(target[mask], lower[mask], upper[mask])
            part_metrics = regression_metrics(target[mask], candidate_pred[mask])
            result.update({
                "baseline_mae_hours": regression_metrics(target[mask], baseline_pred[mask])["mae_hours"],
                "candidate_mae_hours": part_metrics["mae_hours"],
                "delta_mae_hours": part_delta["estimate_hours"],
                "ci_lower_hours": part_delta["ci_lower_hours"],
                "ci_upper_hours": part_delta["ci_upper_hours"],
                "empirical_interval_coverage": part_pi["empirical_coverage"],
                "mean_interval_width_hours": part_pi["mean_width_hours"],
                "candidate_mean_underestimation_hours": part_metrics["mean_underestimation_hours"],
            })
        elif result["n_days"] < 2:
            result["status"] = "insufficient_data"
        slice_rows.append(result)

    predictions = test.loc[:, ["order_id", "ordered_at", "label_available_at", "delivery_hours"]].copy()
    predictions["baseline_prediction_hours"] = baseline_pred
    predictions["candidate_prediction_hours"] = candidate_pred
    predictions["interval_lower_hours"] = lower
    predictions["interval_upper_hours"] = upper
    predictions["within_interval"] = (target >= lower) & (target <= upper)
    predictions["underestimated_hours"] = np.maximum(target - candidate_pred, 0.0)

    result = {
        "stage": "03_offline_evaluation",
        "evaluation_policy": {
            "model_selected_on": "mature_validation_february_2026",
            "calibration_residuals_from": "mature_validation_february_2026",
            "final_holdout": "mature_test_march_2026",
            "test_used_in_selection_or_calibration": False,
            "interval_note": "Empirical validation residual interval; validation was also used for model selection, so distribution-free split-conformal guarantee is NOT claimed.",
            "bootstrap_unit": "UTC calendar day (all orders and both predictions resampled together)",
            "confidence_level": kwargs["confidence"],
            "bootstrap_repeats": kwargs["n_boot"],
            "seed": kwargs["seed"],
        },
        "model_version": manifest["model_version"],
        "selected_model": manifest["selected_model"],
        "split_counts": {
            "train_mature": int(len(train)),
            "valid_mature": int(len(valid)),
            "test_mature": int(len(test)),
        },
        "baseline_test": regression_metrics(target, baseline_pred),
        "candidate_test": regression_metrics(target, candidate_pred),
        "paired_mae_improvement": overall_delta,
        "prediction_interval": {
            "target_coverage": float(config["prediction_interval_coverage"]),
            "calibration_residual_count": int(len(true_valid)),
            "half_width_hours": q,
            "validation_diagnostic": val_pi,
            "test": pi,
        },
        "mandatory_slices": slice_rows,
    }
    return result, pd.DataFrame(slice_rows), predictions


def evaluate_final(
    *,
    features_path: Path,
    data_config_path: Path,
    evaluation_config_path: Path,
    artifact_dir: Path,
    output_dir: Path,
) -> dict:
    data_cfg = load_config(data_config_path)
    cfg = json.loads(Path(evaluation_config_path).read_text(encoding="utf-8"))
    if cfg["seed"] != data_cfg["seed"]:
        raise ValueError("Evaluation and data seeds differ")
    if len({s["name"] for s in cfg["mandatory_slices"]}) != len(cfg["mandatory_slices"]):
        raise ValueError("Slice names must be unique")
    model, manifest = load_bundle(artifact_dir)
    data_hash = hashlib.sha256(Path(features_path).read_bytes()).hexdigest()
    if data_hash != manifest["data_sha256"]:
        raise ValueError("Feature data checksum differs from the training bundle")
    frame = pd.read_csv(features_path)
    if frame.order_id.isna().any() or frame.order_id.duplicated().any():
        raise ValueError("Invalid order identifiers")
    periods = {p["name"]: p for p in data_cfg["periods"]}
    required = {"train", "valid", "test", "replay"}
    if not required.issubset(periods):
        raise ValueError("Missing mandatory data periods")
    samples = {}
    counts = {}
    for name in ("train", "valid", "test"):
        p = periods[name]
        group = frame.loc[frame.period == name]
        samples[name] = closed_split(group, start=p["start"], end=p["end"], as_of=p["end"])
        counts[name] = {"total": int(len(group)), "mature": int(len(samples[name])),
                        "immature": int(len(group) - len(samples[name]))}
    # Replay must remain entirely unopened by this evaluation stage.
    result, slices, predictions = evaluate_predictions(
        test=samples["test"], valid=samples["valid"], train=samples["train"],
        candidate=model, manifest=manifest, config=cfg,
    )
    result["split_readiness"] = counts
    result["sha256"] = {
        "features": data_hash,
        "selected_model": manifest["model_sha256"],
        "evaluation_config": hashlib.sha256(evaluation_config_path.read_bytes()).hexdigest(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluation_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    slices.to_csv(output_dir / "evaluation_slices.csv", index=False)
    predictions.to_csv(output_dir / "evaluation_predictions.csv", index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired offline evaluation on mature delivery labels")
    parser.add_argument("--features", type=Path, default=ROOT / "data/generated/features.csv")
    parser.add_argument("--data-config", type=Path, default=ROOT / "configs/data.json")
    parser.add_argument("--eval-config", type=Path, default=ROOT / "configs/evaluation.json")
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--output", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    result = evaluate_final(
        features_path=args.features,
        data_config_path=args.data_config,
        evaluation_config_path=args.eval_config,
        artifact_dir=args.artifacts,
        output_dir=args.output,
    )
    ci = result["paired_mae_improvement"]
    pred_int = result["prediction_interval"]
    print(json.dumps({
        "stage": result["stage"],
        "selected_model": result["selected_model"],
        "test_mature_orders": result["split_counts"]["test_mature"],
        "baseline_mae_hours": result["baseline_test"]["mae_hours"],
        "candidate_mae_hours": result["candidate_test"]["mae_hours"],
        "mae_improvement_hours": ci["estimate_hours"],
        "mae_improvement_95_ci": [ci["ci_lower_hours"], ci["ci_upper_hours"]],
        "prediction_interval_90_coverage": pred_int["test"]["empirical_coverage"],
        "prediction_interval_mean_width_hours": pred_int["test"]["mean_width_hours"],
        "slices_evaluated": sum(s["status"] == "evaluated" for s in result["mandatory_slices"]),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
