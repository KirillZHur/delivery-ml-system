"""Stage 8: mature-only, gated model update with an atomic active registry.

No March test labels or held-out April gate labels enter training/calibration.
The managed FastAPI process loads the active registry on startup; promotion
requires process restart (this is a local exercise, not a live control plane).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from data.generate import ROOT, load_config
from src.bundle import load_bundle, predict_frame
from src.evaluate import interval_half_width, interval_diagnostics, prediction_intervals, select_slice
from src.features import MODEL_FEATURES, model_matrix
from src.models import model_candidates
from src.service import create_app
from src.serving_bundle import load_serving_bundle
from src.train import regression_metrics


def _digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".registry-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            out.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _append_event(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def _period_rows(frame: pd.DataFrame, period: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    timestamp = pd.to_datetime(frame["ordered_at"], utc=True)
    return frame.loc[frame.period.eq(period) & timestamp.ge(start) & timestamp.lt(end)].copy()


def load_update_splits(features_path: Path, data_config: dict, policy: dict) -> dict[str, pd.DataFrame]:
    """Use complete UTC cohorts whose every delayed label arrived at snapshot."""
    if int(policy["seed"]) != int(data_config["seed"]):
        raise ValueError("Retraining/data seed mismatch")
    frame = pd.read_csv(features_path)
    required = {"order_id", "period", "ordered_at", "label_available_at", "delivery_hours",
                "warehouse_event_at", "warehouse_ingested_at", "weather_event_at",
                "weather_ingested_at", *MODEL_FEATURES}
    if not required.issubset(frame):
        raise ValueError(f"Missing feature audit fields: {sorted(required - set(frame))}")
    if frame.order_id.isna().any() or frame.order_id.duplicated().any():
        raise ValueError("Order IDs must be unique")
    cutoff = pd.Timestamp(policy["snapshot_as_of"])
    t_start = pd.Timestamp(next(p["start"] for p in data_config["periods"] if p["name"] == policy["new_period"]))
    t_train_end = pd.Timestamp(policy["training_end"])
    t_cal_end = pd.Timestamp(policy["calibration_end"])
    t_gate_end = pd.Timestamp(policy["gate_end"])
    if (any(t.tzinfo is None for t in (cutoff, t_start, t_train_end, t_cal_end, t_gate_end))
            or not t_start < t_train_end < t_cal_end < t_gate_end <= cutoff):
        raise ValueError("Retraining times must be strictly ordered with UTC offsets")
    period_end = pd.Timestamp(next(p["end"] for p in data_config["periods"] if p["name"] == policy["new_period"]))
    if t_gate_end != period_end:
        raise ValueError("The gate must include the end of the new-period dataset")
    # Refuse training on a dataset whose as-of PIT audit is inconsistent.
    event_times = ["warehouse_event_at", "warehouse_ingested_at", "weather_event_at", "weather_ingested_at"]
    predicted = pd.to_datetime(frame.ordered_at, utc=True)
    for field in event_times:
        ts = pd.to_datetime(frame[field], utc=True, format="mixed")
        if ts.isna().any() or (ts > predicted).any():
            raise ValueError(f"Point-in-time leak detected: {field}")
    target = frame["delivery_hours"].to_numpy(dtype=float)
    if not np.isfinite(target).all() or ((target < 1) | (target > 168)).any():
        raise ValueError("Invalid delivery labels")
    if not np.isfinite(model_matrix(frame)).all():
        raise ValueError("Invalid input feature values")
    for field in ("warehouse_load", "rain_forecast"):
        if field == "warehouse_load" and not frame[field].between(0, 1).all():
            raise ValueError("Invalid warehouse load")
        if field == "rain_forecast" and not frame[field].isin((0, 1)).all():
            raise ValueError("Invalid rain values")
    new_train = _period_rows(frame, policy["new_period"], t_start, t_train_end)
    calibration = _period_rows(frame, policy["new_period"], t_train_end, t_cal_end)
    gate = _period_rows(frame, policy["new_period"], t_cal_end, t_gate_end)
    historical = frame.loc[frame.period.isin(policy["historical_training_periods"])].copy()
    train = pd.concat([historical, new_train], ignore_index=True)
    groups = {"training": train, "new_training": new_train, "calibration": calibration, "gate": gate}
    for name, part in groups.items():
        if part.empty:
            raise ValueError(f"Empty update group: {name}")
        when = pd.to_datetime(part.label_available_at, utc=True)
        if when.isna().any() or when.gt(cutoff).any():
            raise ValueError(f"Unmatured or undelivered label in {name}; no partial cohorts")
    for name, limit_key in (("new_training", "min_new_training_orders"),
                            ("calibration", "min_calibration_orders"), ("gate", "min_gate_orders")):
        if len(groups[name]) < int(policy[limit_key]):
            raise ValueError(f"Insufficient mature {name} rows")
    sets = [set(groups[n].order_id) for n in ("training", "calibration", "gate")]
    if any(sets[a] & sets[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Update groups have overlapping order IDs")
    if (train.period.eq("test").any() or calibration.period.eq("test").any()
            or gate.period.eq("test").any()):
        raise ValueError("March test labels cannot be used by retraining")
    return {k: v.reset_index(drop=True) for k, v in groups.items()}


def bootstrap_registry(registry_dir: Path, stable_dir: Path) -> dict:
    """Create immutable baseline copy and resolve the active bundle by registry."""
    registry_dir = Path(registry_dir)
    file = registry_dir / "registry.json"
    if file.exists():
        return _read_json(file)
    _, stable_manifest, _ = load_serving_bundle(stable_dir)
    stable_target = registry_dir / "bundles" / "stable"
    if stable_target.exists():
        raise ValueError("Unregistered baseline directory exists; refusing overwrite")
    stable_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(stable_dir, stable_target)
    load_serving_bundle(stable_target)
    state = {"schema": 1, "active_bundle": "bundles/stable", "active_version": stable_manifest["model_version"],
             "previous_bundle": None, "previous_version": None, "generation": 0}
    _atomic_json(file, state)
    _append_event(registry_dir / "events.jsonl", {"action": "bootstrap", "active_version": state["active_version"],
                                                      "active_bundle": state["active_bundle"], "generation": 0})
    return state


def active_bundle(registry_dir: Path) -> Path:
    registry_dir = Path(registry_dir)
    state = _read_json(registry_dir / "registry.json")
    relative = Path(state["active_bundle"])
    root = registry_dir.resolve()
    candidate = (root / relative).resolve()
    if relative.is_absolute() or not candidate.is_relative_to(root):
        raise ValueError("Registry bundle must remain inside managed directory")
    _, manifest, _ = load_serving_bundle(candidate)
    if state["active_version"] != manifest["model_version"]:
        raise ValueError("Registry version does not match active bundle")
    return candidate


def rollback_registry(registry_dir: Path) -> dict:
    """Revert a promoted version with an atomic registry pointer update."""
    registry_dir = Path(registry_dir)
    file = registry_dir / "registry.json"
    state = _read_json(file)
    previous = state.get("previous_bundle")
    if previous is None:
        raise ValueError("No prior compatible version to restore")
    root = registry_dir.resolve()
    target = (root / previous).resolve()
    if Path(previous).is_absolute() or not target.is_relative_to(root):
        raise ValueError("Unsafe previous bundle")
    _, manifest, _ = load_serving_bundle(target)
    if manifest["model_version"] != state["previous_version"]:
        raise ValueError("Previous version integrity mismatch")
    before = dict(state)
    state.update({"active_bundle": previous, "active_version": manifest["model_version"],
                  "previous_bundle": before["active_bundle"], "previous_version": before["active_version"],
                  "generation": before["generation"] + 1})
    _atomic_json(file, state)
    _append_event(registry_dir / "events.jsonl", {"action": "rollback", "from_version": before["active_version"],
                                                      "to_version": state["active_version"], "generation": state["generation"]})
    return state


def _candidate_bundle(model: object, source_manifest: dict, calib: pd.DataFrame,
                      config: dict, version: str, destination: Path, features_path: Path,
                      training_config_path: Path, retrain_config_path: Path,
                      training_rows: int) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    model_path = destination / "model.joblib"
    joblib.dump(model, model_path, compress=3)
    manifest = {**source_manifest, "model_version": version, "selected_model": "linear",
                "model_sha256": _digest(model_path), "data_sha256": _digest(features_path),
                "config_sha256": _digest(training_config_path),
                "retraining_config_sha256": _digest(retrain_config_path),
                "retrained_as_of": config["snapshot_as_of"],
                "retraining_examples": int(training_rows), "calibration_examples": len(calib)}
    _json(destination / "manifest.json", manifest)
    forecast = predict_frame(model, manifest, calib)
    half_width = interval_half_width(np.abs(calib.delivery_hours.to_numpy(dtype=float) - forecast), float(config["coverage"]))
    serving = {"schema_version": 1, "model_version": version, "model_sha256": manifest["model_sha256"],
               "manifest_sha256": _digest(destination / "manifest.json"), "data_sha256": manifest["data_sha256"],
               "feature_order": manifest["feature_order"], "interval_coverage": config["coverage"],
               "interval_half_width_hours": half_width, "calibration_orders": len(calib),
               "calibration_source": "mature_replay_calibration"}
    _json(destination / "serving.json", serving)
    load_serving_bundle(destination)
    return serving


def _service_probe(candidate_dir: Path, gate: pd.DataFrame, policy: dict, events_path: Path,
                   service_config_path: Path, audit_path: Path) -> dict:
    # Real TestClient (HTTP stack), not synthetic timings or a direct sklearn call.
    samples = gate.iloc[np.linspace(0, len(gate) - 1, int(policy["service_probe_requests"]), dtype=int)]
    model, manifest, _ = load_serving_bundle(candidate_dir)
    reference = predict_frame(model, manifest, samples)
    app = create_app(artifact_dir=candidate_dir, events_path=events_path,
                     config_path=service_config_path, log_path=audit_path)
    timings = []
    degraded = 0
    max_error = 0.0
    with TestClient(app) as client:
        ready = client.get("/health/ready")
        if ready.status_code != 200 or not ready.json()["ready"]:
            raise ValueError("Candidate service not ready")
        for idx, (_, row) in enumerate(samples.iterrows()):
            payload = {"order_id": str(row.order_id), "ordered_at": str(row.ordered_at),
                       "distance_km": float(row.distance_km)}
            started = time.perf_counter()
            response = client.post("/predict", json=payload)
            timings.append((time.perf_counter() - started) * 1000.0)
            if response.status_code != 200:
                raise ValueError(f"Candidate HTTP failure: {response.status_code}")
            body = response.json()
            degraded += bool(body["degraded"])
            if body["model_version"] != manifest["model_version"] or body["order_id"] != str(row.order_id):
                raise ValueError("Incorrect candidate HTTP contract")
            if body["source"] != "model" or body["prediction_interval_90"] is None:
                raise ValueError("Candidate failed to serve a model response")
            difference = abs(float(body["estimated_delivery_hours"]) - float(reference[idx]))
            max_error = max(max_error, difference)
            if difference > 1e-10:
                raise ValueError("Online/offline prediction mismatch")
    return {"requests": len(timings), "degraded": degraded, "p95_ms": float(np.quantile(timings, 0.95)),
            "max_online_offline_difference_hours": max_error,
            "measurement_scope": "local FastAPI TestClient request including response serialization; not a 20-rps external load test"}


def run_update(*, features_path: Path, data_config_path: Path, training_config_path: Path,
               retraining_config_path: Path, evaluation_config_path: Path,
               stable_dir: Path, registry_dir: Path, events_path: Path, service_config_path: Path,
               failure_mode: str | None = None) -> dict:
    """Rebuild candidate, calibrate and gate; never mutate active bundle on failure."""
    if failure_mode not in (None, "quality_bias"):
        raise ValueError("Unknown exercise failure mode")
    policy = _read_json(retraining_config_path)
    data_config = load_config(data_config_path)
    training_cfg = _read_json(training_config_path)
    eval_cfg = _read_json(evaluation_config_path)
    groups = load_update_splits(features_path, data_config, policy)
    previous = bootstrap_registry(registry_dir, stable_dir)
    current_path = active_bundle(registry_dir)
    incumbent, incumbent_manifest, incumbent_serving = load_serving_bundle(current_path)
    if incumbent_manifest["feature_order"] != list(MODEL_FEATURES):
        raise ValueError("Incompatible active feature contract")
    version = policy["candidate_version"] + ("-quality-failure" if failure_mode else "")
    candidate_dir = registry_dir / "candidates" / version
    if candidate_dir.exists():
        # Never overwrite an active release. The same cycle can be replayed
        # deterministically without mutating an already promoted candidate.
        if candidate_dir.resolve() == current_path.resolve():
            old_digest = _digest(candidate_dir / "model.joblib")
            temp = registry_dir / "candidates" / (version + "-rebuild")
            if temp.exists():
                shutil.rmtree(temp)
            destination = temp
        else:
            shutil.rmtree(candidate_dir)
            destination = candidate_dir
    else:
        destination = candidate_dir
    training = groups["training"]
    if training_cfg["seed"] != policy["seed"]:
        raise ValueError("Training/retraining seed mismatch")
    candidate = model_candidates(training_cfg)[policy["model_family"]]
    candidate.fit(model_matrix(training), training.delivery_hours.to_numpy(dtype=float))
    if failure_mode == "quality_bias":
        candidate.named_steps["ridge"].intercept_ += float(policy["bad_candidate_bias_hours"])
    _candidate_bundle(candidate, incumbent_manifest, groups["calibration"], policy,
                                    version, destination, features_path, training_config_path,
                                    retraining_config_path, len(training))
    if destination != candidate_dir:
        if old_digest != _digest(destination / "model.joblib"):
            shutil.rmtree(destination)
            raise ValueError("Cannot silently overwrite a promoted model with different weights")
        shutil.rmtree(destination)
        destination = candidate_dir
    candidate, candidate_manifest, candidate_serving = load_serving_bundle(destination)
    gate = groups["gate"]
    y = gate.delivery_hours.to_numpy(dtype=float)
    incumbent_pred = predict_frame(incumbent, incumbent_manifest, gate)
    candidate_pred = predict_frame(candidate, candidate_manifest, gate)
    active_metrics = regression_metrics(y, incumbent_pred)
    candidate_metrics = regression_metrics(y, candidate_pred)
    lo, hi = prediction_intervals(candidate_pred, float(candidate_serving["interval_half_width_hours"]),
                                  bounds=tuple(candidate_manifest["prediction_bounds_hours"]))
    intervals = interval_diagnostics(y, lo, hi)
    checks = {"mae": candidate_metrics["mae_hours"] <= active_metrics["mae_hours"] + float(policy["max_mae_regression_hours"]),
              "interval_coverage": intervals["empirical_coverage"] >= float(policy["min_interval_coverage"]),
              "interval_width": intervals["mean_width_hours"] <= float(policy["max_mean_interval_width_hours"]),
              "artifact_integrity": True,
              "feature_contract": candidate_manifest["feature_order"] == incumbent_manifest["feature_order"]
                                  and candidate_manifest["prediction_bounds_hours"] == incumbent_manifest["prediction_bounds_hours"]}
    slices = {}
    for definition in eval_cfg["mandatory_slices"]:
        mask = select_slice(gate, definition)
        n = int(mask.sum())
        baseline_mae = float(np.mean(np.abs(y[mask] - incumbent_pred[mask]))) if n else None
        new_mae = float(np.mean(np.abs(y[mask] - candidate_pred[mask]))) if n else None
        ok = (n >= int(policy["min_slice_orders"]) and new_mae is not None
              and new_mae <= baseline_mae + float(policy["max_slice_mae_regression_hours"]))
        slices[definition["name"]] = {"n": n, "active_mae_hours": baseline_mae,
                                      "candidate_mae_hours": new_mae, "passed": bool(ok)}
        checks["slice_" + definition["name"]] = bool(ok)
    try:
        service = _service_probe(destination, gate, policy, events_path, service_config_path,
                                 registry_dir / "probes" / (version + ".jsonl"))
    except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
        # Candidate failure must be recorded as a rejection, not result in an
        # accidental promotion because the probe did not return measurements.
        service = {"error": f"{type(exc).__name__}: {exc}", "requests": 0,
                   "degraded": None, "p95_ms": None,
                   "max_online_offline_difference_hours": None}
    checks["service_latency"] = service["p95_ms"] is not None and service["p95_ms"] <= float(policy["max_service_p95_ms"])
    checks["service_no_degraded"] = service["degraded"] == 0
    checks["online_offline_parity"] = (service["max_online_offline_difference_hours"] is not None
                                       and service["max_online_offline_difference_hours"] <= 1e-10)
    admitted = all(checks.values())
    decision = {"stage": "08_retraining", "outcome": ("already_active" if admitted and previous["active_version"] == version
                                                           else "promoted" if admitted else "rejected"),
                "candidate_version": version, "active_version_before": previous["active_version"],
                "snapshot_as_of": policy["snapshot_as_of"], "seed": int(policy["seed"]),
                "counts": {name: len(rows) for name, rows in groups.items()},
                "boundaries": {"training_end": policy["training_end"], "calibration_end": policy["calibration_end"],
                               "gate_end": policy["gate_end"]},
                "model": {"active": active_metrics, "candidate": candidate_metrics,
                          "gate_intervals": intervals, "calibration_half_width_hours": candidate_serving["interval_half_width_hours"],
                          "slices": slices}, "service": service, "checks": checks,
                "rejection_reasons": [name for name, passed in checks.items() if not passed],
                "candidate_bundle": str(destination.relative_to(registry_dir))}
    if admitted and previous["active_version"] != version:
        before = _read_json(registry_dir / "registry.json")
        new = {**before, "active_bundle": str(destination.relative_to(registry_dir)),
               "active_version": version, "previous_bundle": before["active_bundle"],
               "previous_version": before["active_version"], "generation": before["generation"] + 1}
        _atomic_json(registry_dir / "registry.json", new)
    decision["active_version_after"] = _read_json(registry_dir / "registry.json")["active_version"]
    _json(registry_dir / "decisions" / (version + ".json"), decision)
    _append_event(registry_dir / "events.jsonl", {"action": decision["outcome"],
              "from_version": decision["active_version_before"], "candidate_version": version,
              "active_version": decision["active_version_after"], "counts": decision["counts"],
              "checks": checks, "rejection_reasons": decision["rejection_reasons"]})
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description="Mature-only delivery model refresh and quality gate")
    parser.add_argument("--features", type=Path, default=ROOT / "data/generated/features.csv")
    parser.add_argument("--registry", type=Path, default=ROOT / "retraining")
    parser.add_argument("--exercise-rejection", action="store_true", help="Also gate a deliberately biased candidate")
    parser.add_argument("--rollback", action="store_true", help="Revert the active managed version")
    args = parser.parse_args()
    if args.rollback:
        print(json.dumps(rollback_registry(args.registry), indent=2, ensure_ascii=False))
        return
    params = dict(features_path=args.features,
                  data_config_path=ROOT / "configs/data.json", training_config_path=ROOT / "configs/training.json",
                  retraining_config_path=ROOT / "configs/retraining.json", evaluation_config_path=ROOT / "configs/evaluation.json",
                  stable_dir=ROOT / "artifacts", registry_dir=args.registry,
                  events_path=ROOT / "data/generated/events.csv", service_config_path=ROOT / "configs/service.json")
    records = [run_update(**params)]
    if args.exercise_rejection:
        records.append(run_update(**params, failure_mode="quality_bias"))
    print(json.dumps({"decisions": [{"version": item["candidate_version"], "result": item["outcome"],
                                      "active": item["active_version_after"],
                                      "failed_checks": item["rejection_reasons"],
                                      "gate_mae_active": item["model"]["active"]["mae_hours"],
                                      "gate_mae_candidate": item["model"]["candidate"]["mae_hours"]} for item in records]},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
