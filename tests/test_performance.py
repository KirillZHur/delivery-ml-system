"""Stage 5: accelerated path numerical parity and honest latency aggregation."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import HistGradientBoostingRegressor

from data.generate import ROOT
from src.benchmark import percentile_metrics, sample_replay_requests
from src.bundle import load_bundle, predict_frame
from src.fast_predict import FastLinearPredictor
from src.features import model_matrix
from src.models import predict_delivery
from src.service import create_app


def test_fast_linear_matches_full_pipeline_on_replay():
    model, manifest = load_bundle(ROOT / "artifacts")
    fast = FastLinearPredictor.from_model(model)
    assert fast is not None
    data = pd.read_csv(ROOT / "data/generated/features.csv")
    sample = data[data.period == "replay"].iloc[::5]
    predictions = predict_frame(model, manifest, sample)
    matrix = model_matrix(sample)
    bounds = tuple(manifest["prediction_bounds_hours"])
    fast_output = [fast.predict_one(tuple(float(x) for x in row), bounds) for row in matrix]
    np.testing.assert_allclose(fast_output, predictions, rtol=0, atol=1e-12)


def test_fast_predictor_also_matches_exact_clipping_boundaries():
    model, manifest = load_bundle(ROOT / "artifacts")
    fast = FastLinearPredictor.from_model(model)
    bounds = tuple(manifest["prediction_bounds_hours"])
    matrix = np.array([[0, 0, 0], [1000, 1, 1], [1e6, 0.5, 1]], dtype=float)
    reference = predict_delivery(model, matrix, bounds)
    actual = [fast.predict_one(tuple(row), bounds) for row in matrix]
    np.testing.assert_allclose(actual, reference, rtol=0, atol=1e-12)
    with pytest.raises(ValueError):
        fast.predict_one((10.0, np.nan, 1.0), bounds)
    with pytest.raises(ValueError):
        fast.predict_one((10.0, 0.1, 1.0), (5, 5))


def test_fast_predictor_only_adapts_supported_frozen_pipeline():
    assert FastLinearPredictor.from_model(HistGradientBoostingRegressor()) is None
    assert FastLinearPredictor.from_model(object()) is None


def test_reference_and_fast_http_are_equivalent(tmp_path):
    payloads, predictions = sample_replay_requests(count=30)
    app_old = create_app(prediction_mode="baseline", log_path=tmp_path / "reference.jsonl")
    app_new = create_app(prediction_mode="fast", log_path=tmp_path / "optimized.jsonl")
    with TestClient(app_old) as left, TestClient(app_new) as right:
        assert app_old.state.fast_predictor is None
        assert app_new.state.fast_predictor is not None
        for payload, reference in zip(payloads, predictions):
            a = left.post("/predict", json=payload)
            b = right.post("/predict", json=payload)
            assert a.status_code == 200 == b.status_code
            aa, bb = a.json(), b.json()
            assert aa["source"] == "model" == bb["source"]
            assert not aa["degraded"] and not bb["degraded"]
            for key in ("estimated_delivery_hours",):
                assert aa[key] == pytest.approx(reference, abs=1e-12)
                assert bb[key] == pytest.approx(reference, abs=1e-12)
                assert aa[key] == pytest.approx(bb[key], abs=1e-12)
            for key in ("lower_hours", "upper_hours", "nominal_coverage"):
                assert aa["prediction_interval_90"][key] == pytest.approx(
                    bb["prediction_interval_90"][key], abs=1e-12
                )
        assert "delivery_predict_seconds_count 30.0" in left.get("/metrics").text
        assert "delivery_predict_seconds_count 30.0" in right.get("/metrics").text
    assert len((tmp_path / "reference.jsonl").read_text().splitlines()) == 30
    assert len((tmp_path / "optimized.jsonl").read_text().splitlines()) == 30


def test_invalid_prediction_mode_rejected():
    with pytest.raises(ValueError, match="prediction_mode"):
        create_app(prediction_mode="unknown")


def test_latency_percentile_method_and_bad_inputs():
    metrics = percentile_metrics([10, 20, 30, 40, 50])
    assert metrics["median_ms"] == 30
    assert metrics["p95_ms"] == pytest.approx(48)
    assert metrics["p99_ms"] == pytest.approx(49.6)
    with pytest.raises(ValueError):
        percentile_metrics([])
    with pytest.raises(ValueError):
        percentile_metrics([1, -2])
    with pytest.raises(ValueError):
        percentile_metrics([1, float("nan")])


def test_benchmark_payloads_are_repeatable_and_do_not_include_targets():
    one, targets = sample_replay_requests(count=12, seed=2026)
    two, targets2 = sample_replay_requests(count=12, seed=2026)
    assert one == two and targets == targets2
    assert len({r["order_id"] for r in one}) == 12
    assert all(set(r) == {"order_id", "ordered_at", "distance_km"} for r in one)
    assert all(isinstance(x, float) for x in targets)
    with pytest.raises(ValueError):
        sample_replay_requests(count=5000)
