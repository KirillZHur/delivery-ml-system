"""Stage 8: mature-only refresh, service admission, failed update, rollback.

The module-scoped state runs a complete real training and HTTP gate once.
Earlier stage test suites remain part of the full regression run.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from data.generate import ROOT, load_config
from src.bundle import predict_frame
from src.retrain import (
    active_bundle, bootstrap_registry, load_update_splits, rollback_registry, run_update,
)
from src.active_service import create_managed_app
from src.serving_bundle import load_serving_bundle


FEATURES = ROOT / "data/generated/features.csv"
POLICY = ROOT / "configs/retraining.json"


def params(registry: Path, **overrides) -> dict:
    config = dict(
        features_path=FEATURES,
        data_config_path=ROOT / "configs/data.json",
        training_config_path=ROOT / "configs/training.json",
        retraining_config_path=POLICY,
        evaluation_config_path=ROOT / "configs/evaluation.json",
        stable_dir=ROOT / "artifacts",
        registry_dir=registry,
        events_path=ROOT / "data/generated/events.csv",
        service_config_path=ROOT / "configs/service.json",
    )
    return {**config, **overrides}


@pytest.fixture(scope="module")
def complete_cycle(tmp_path_factory):
    root = tmp_path_factory.mktemp("retraining-cycle")
    before_artifact = {
        name: hashlib.sha256((ROOT / "artifacts" / name).read_bytes()).hexdigest()
        for name in ("model.joblib", "manifest.json", "serving.json")
    }
    ok = run_update(**params(root))
    registry_after_ok = json.loads((root / "registry.json").read_text())
    bad = run_update(**params(root), failure_mode="quality_bias")
    return root, ok, bad, registry_after_ok, before_artifact


def test_update_time_splits_are_disjoint_and_mature():
    policy = json.loads(POLICY.read_text())
    groups = load_update_splits(FEATURES, load_config(ROOT / "configs/data.json"), policy)
    assert {key: len(value) for key, value in groups.items()} == {
        "training": 18684, "new_training": 2684, "calibration": 662, "gate": 654}
    assert groups["training"].period.isin(["train", "valid", "replay"]).all()
    assert set(groups["new_training"].period) == {"replay"}
    assert set(groups["calibration"].period) == {"replay"}
    assert set(groups["gate"].period) == {"replay"}
    a, b, c = (set(groups[k].order_id) for k in ("training", "calibration", "gate"))
    assert not (a & b or b & c or a & c)
    assert not any(part.period.eq("test").any() for part in groups.values())
    assert (pd.to_datetime(groups["gate"].label_available_at, utc=True)
            <= pd.Timestamp(policy["snapshot_as_of"])).all()
    assert pd.to_datetime(groups["new_training"].ordered_at, utc=True).max() < pd.Timestamp(policy["training_end"])
    assert pd.to_datetime(groups["calibration"].ordered_at, utc=True).max() < pd.Timestamp(policy["calibration_end"])


def test_update_rejects_early_snapshot_and_future_feature_arrival(tmp_path):
    config = load_config(ROOT / "configs/data.json")
    policy = json.loads(POLICY.read_text())
    early = dict(policy, snapshot_as_of="2026-05-01T00:00:00Z")
    with pytest.raises(ValueError, match="Unmatured"):
        load_update_splits(FEATURES, config, early)
    frame = pd.read_csv(FEATURES)
    frame.loc[frame.period.eq("replay").idxmax(), "warehouse_ingested_at"] = "2026-06-01T00:00:00Z"
    altered = tmp_path / "future.csv"
    frame.to_csv(altered, index=False)
    with pytest.raises(ValueError, match="Point-in-time leak"):
        load_update_splits(altered, config, policy)


def test_rejects_missing_mature_label_even_if_only_one_row(tmp_path):
    frame = pd.read_csv(FEATURES)
    affected = frame.loc[frame.period.eq("replay")].index[-1]
    frame.loc[affected, "label_available_at"] = "2026-05-12T00:00:00Z"
    altered = tmp_path / "immature.csv"
    frame.to_csv(altered, index=False)
    with pytest.raises(ValueError, match="Unmatured"):
        load_update_splits(altered, load_config(ROOT / "configs/data.json"), json.loads(POLICY.read_text()))


def test_successful_promotion_and_fail_closed_rejection(complete_cycle):
    registry, good, bad, snapshot, old = complete_cycle
    assert good["outcome"] == "promoted", good["rejection_reasons"]
    assert good["active_version_before"] == "delivery-2026.1-linear"
    assert good["active_version_after"] == "delivery-2026.2-linear"
    assert good["seed"] == 2026
    assert good["model"]["candidate"]["mae_hours"] <= good["model"]["active"]["mae_hours"] + 0.15
    assert all(good["checks"].values())
    assert bad["outcome"] == "rejected"
    assert bad["active_version_after"] == good["active_version_after"]
    assert "mae" in bad["rejection_reasons"]
    assert all(snapshot[k] == json.loads((registry / "registry.json").read_text())[k] for k in snapshot)
    assert all(hashlib.sha256((ROOT / "artifacts" / key).read_bytes()).hexdigest() == value
               for key, value in old.items())
    assert len((registry / "events.jsonl").read_text().splitlines()) == 3


def test_candidate_has_separate_calibration_and_heldout_gate(complete_cycle):
    registry, good, _, _, _ = complete_cycle
    candidate_path = registry / good["candidate_bundle"]
    _, manifest, interval = load_serving_bundle(candidate_path)
    assert interval["calibration_source"] == "mature_replay_calibration"
    assert interval["calibration_orders"] == good["counts"]["calibration"]
    assert manifest["retraining_examples"] == good["counts"]["training"]
    assert manifest["feature_order"] == ["distance_km", "warehouse_load", "rain_forecast"]
    assert 0.85 <= good["model"]["gate_intervals"]["empirical_coverage"] <= 1.0
    assert good["model"]["gate_intervals"]["n"] == 654
    assert all(s["passed"] for s in good["model"]["slices"].values())


def test_managed_fastapi_serves_promoted_bundle(complete_cycle, tmp_path):
    registry, good, _, _, _ = complete_cycle
    app = create_managed_app(registry, log_path=tmp_path / "audit.jsonl")
    frame = pd.read_csv(FEATURES)
    row = frame.loc[frame.period.eq("replay")].iloc[-10]
    model, manifest, _ = load_serving_bundle(active_bundle(registry))
    expected = float(predict_frame(model, manifest, pd.DataFrame([row]))[0])
    with TestClient(app) as client:
        ready = client.get("/health/ready")
        assert ready.status_code == 200
        assert ready.json()["model_version"] == good["candidate_version"]
        response = client.post("/predict", json={"order_id": str(row.order_id),
                      "ordered_at": str(row.ordered_at), "distance_km": float(row.distance_km)})
        assert response.status_code == 200
        result = response.json()
        assert result["source"] == "model" and not result["degraded"]
        assert result["model_version"] == good["candidate_version"]
        assert result["estimated_delivery_hours"] == pytest.approx(expected, abs=1e-10)


def test_promoted_candidate_is_not_overwritten_on_repeated_run(complete_cycle):
    registry, _, _, original, _ = complete_cycle
    before_hash = hashlib.sha256((active_bundle(registry) / "model.joblib").read_bytes()).hexdigest()
    again = run_update(**params(registry))
    assert again["outcome"] == "already_active"
    assert hashlib.sha256((active_bundle(registry) / "model.joblib").read_bytes()).hexdigest() == before_hash
    after = json.loads((registry / "registry.json").read_text())
    assert after == original


def test_corrupt_registry_pointer_and_model_are_rejected(complete_cycle, tmp_path):
    registry, _, _, _, _ = complete_cycle
    copy = tmp_path / "registry"
    shutil.copytree(registry, copy)
    file = copy / "registry.json"
    state = json.loads(file.read_text())
    state["active_bundle"] = "../../untrusted"
    file.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="inside managed"):
        active_bundle(copy)
    state["active_bundle"] = "candidates/delivery-2026.2-linear"
    file.write_text(json.dumps(state))
    with (copy / state["active_bundle"] / "model.joblib").open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        active_bundle(copy)


def test_rollback_restores_previous_compatible_version(complete_cycle, tmp_path):
    registry, _, _, _, _ = complete_cycle
    copy = tmp_path / "registry"
    shutil.copytree(registry, copy)
    reverted = rollback_registry(copy)
    assert reverted["active_version"] == "delivery-2026.1-linear"
    assert reverted["generation"] == 2
    _, manifest, serving = load_serving_bundle(active_bundle(copy))
    assert manifest["model_version"] == "delivery-2026.1-linear"
    assert serving["calibration_source"] == "mature_validation_only"
    assert json.loads((copy / "events.jsonl").read_text().splitlines()[-1])["action"] == "rollback"


def test_fallback_without_previous_version_is_prohibited(tmp_path):
    registry = tmp_path / "fresh"
    baseline = bootstrap_registry(registry, ROOT / "artifacts")
    assert baseline["generation"] == 0
    with pytest.raises(ValueError, match="No prior"):
        rollback_registry(registry)


def test_candidate_fails_service_budget_without_switching(tmp_path):
    policy = json.loads(POLICY.read_text())
    policy["max_service_p95_ms"] = 0.00001
    cfg = tmp_path / "strict.json"
    cfg.write_text(json.dumps(policy))
    registry = tmp_path / "registry"
    result = run_update(**params(registry, retraining_config_path=cfg))
    assert result["outcome"] == "rejected"
    assert "service_latency" in result["rejection_reasons"]
    assert json.loads((registry / "registry.json").read_text())["active_version"] == "delivery-2026.1-linear"


def test_previous_test_period_labels_cannot_affect_retraining_inputs(tmp_path):
    frame = pd.read_csv(FEATURES)
    frame.loc[frame.period.eq("test"), "delivery_hours"] = 100.0
    changed = tmp_path / "test_labels_modified.csv"
    frame.to_csv(changed, index=False)
    c = load_config(ROOT / "configs/data.json")
    policy = json.loads(POLICY.read_text())
    original = load_update_splits(FEATURES, c, policy)
    altered = load_update_splits(changed, c, policy)
    for split in ("training", "calibration", "gate"):
        pd.testing.assert_frame_equal(original[split], altered[split])


def test_unavailable_service_features_reject_candidate_instead_of_exception(tmp_path):
    registry = tmp_path / "registry"
    result = run_update(**params(registry, events_path=tmp_path / "missing-events.csv"))
    assert result["outcome"] == "rejected"
    assert not result["checks"]["service_no_degraded"]
    assert result["active_version_after"] == "delivery-2026.1-linear"
    assert (registry / "decisions" / "delivery-2026.2-linear.json").exists()
