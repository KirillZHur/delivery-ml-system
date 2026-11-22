"""Stage 4: HTTP contract, PIT feature parity, failure policy and observability."""

from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from data.generate import ROOT
from src.bundle import predict_frame
from src.evaluate import prediction_intervals
from src.feature_store import (
    EventFeatureStore, FeatureMissingError, FeatureStaleError,
    FeatureUnavailableError, OnlineFeatures,
)
from src.prepare_service import prepare_service_artifact
from src.service import create_app
from src.serving_bundle import load_serving_bundle


ARTIFACTS = ROOT / "artifacts"
EVENTS = ROOT / "data/generated/events.csv"
FEATURES = ROOT / "data/generated/features.csv"


def _request(row: pd.Series, *, order_id: str | None = None) -> dict:
    return {
        "order_id": order_id or str(row.order_id),
        "ordered_at": str(row.ordered_at),
        "distance_km": float(row.distance_km),
    }


@pytest.fixture(scope="module")
def real_orders() -> pd.DataFrame:
    return pd.read_csv(FEATURES).iloc[[0, 1, 10, 150, 999, 5000, 8000, 13000]].copy()


def test_service_interval_built_from_validation_only(tmp_path):
    dest = tmp_path / "bundle"
    dest.mkdir()
    for name in ("manifest.json", "model.joblib"):
        shutil.copy2(ARTIFACTS / name, dest / name)
    serving = prepare_service_artifact(
        features_path=FEATURES,
        data_config_path=ROOT / "configs/data.json",
        evaluation_config_path=ROOT / "configs/evaluation.json",
        artifact_dir=dest,
    )
    reported = json.loads((ROOT / "reports/evaluation_metrics.json").read_text())
    assert serving["calibration_orders"] == reported["split_counts"]["valid_mature"] == 2990
    assert serving["interval_half_width_hours"] == pytest.approx(
        reported["prediction_interval"]["half_width_hours"], abs=1e-12
    )
    assert "test" not in serving["calibration_source"]
    assert serving["manifest_sha256"] == hashlib.sha256((dest / "manifest.json").read_bytes()).hexdigest()
    _, _, loaded = load_serving_bundle(dest)
    assert loaded == serving


def test_service_artifact_rejects_stale_or_tampered_model(tmp_path):
    dest = tmp_path / "bundle"
    shutil.copytree(ARTIFACTS, dest)
    serving_path = dest / "serving.json"
    serving = json.loads(serving_path.read_text())
    serving["model_version"] = "another-model"
    serving_path.write_text(json.dumps(serving))
    with pytest.raises(ValueError, match="model_version"):
        load_serving_bundle(dest)
    shutil.copy2(ARTIFACTS / "serving.json", serving_path)
    with (dest / "model.joblib").open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        load_serving_bundle(dest)


def test_online_feature_store_parity_with_offline_joins(real_orders):
    store = EventFeatureStore.from_csv(
        EVENTS, max_age_hours={"warehouse_load": 8, "rain_forecast": 24}
    )

    async def check():
        for row in real_orders.itertuples():
            online = await store.fetch(pd.Timestamp(row.ordered_at).to_pydatetime())
            assert online.warehouse_load == pytest.approx(row.warehouse_load, abs=1e-13)
            assert online.rain_forecast == row.rain_forecast
            assert online.warehouse_event_id == row.warehouse_event_id
            assert online.weather_event_id == row.weather_event_id

    asyncio.run(check())


def test_late_old_events_do_not_override_new_known_events():
    t = pd.Timestamp("2026-01-01T00:00:00Z")
    events = pd.DataFrame([
        {"event_id": "w1", "feature": "warehouse_load", "event_at": t,
         "ingested_at": t + pd.Timedelta(minutes=1), "value": 0.1},
        {"event_id": "w2", "feature": "warehouse_load", "event_at": t + pd.Timedelta(hours=1),
         "ingested_at": t + pd.Timedelta(hours=1, minutes=2), "value": 0.7},
        {"event_id": "wlate", "feature": "warehouse_load", "event_at": t + pd.Timedelta(minutes=30),
         "ingested_at": t + pd.Timedelta(hours=2), "value": 0.9},
        {"event_id": "r1", "feature": "rain_forecast", "event_at": t,
         "ingested_at": t + pd.Timedelta(minutes=10), "value": 1},
        {"event_id": "rfuture", "feature": "rain_forecast", "event_at": t + pd.Timedelta(hours=4),
         "ingested_at": t + pd.Timedelta(hours=4, minutes=1), "value": 0},
    ])
    store = EventFeatureStore(events, max_age_hours={"warehouse_load": 6, "rain_forecast": 8})

    async def check():
        with pytest.raises(FeatureMissingError):
            await store.fetch((t + pd.Timedelta(seconds=30)).to_pydatetime())
        initial = await store.fetch((t + pd.Timedelta(minutes=45)).to_pydatetime())
        assert initial.warehouse_event_id == "w1"
        after = await store.fetch((t + pd.Timedelta(hours=3)).to_pydatetime())
        assert after.warehouse_event_id == "w2"  # wlate arrived later but event is older
        assert after.weather_event_id == "r1"  # rfuture is not available yet
        with pytest.raises(FeatureStaleError):
            await store.fetch((t + pd.Timedelta(hours=12)).to_pydatetime())

    asyncio.run(check())


def test_http_ready_and_exact_prediction_parity(tmp_path, real_orders):
    app = create_app(log_path=tmp_path / "audit.jsonl")
    with TestClient(app) as client:
        assert client.get("/health/live").json() == {"live": True}
        assert client.get("/health/ready").status_code == 200
        assert client.get("/health/ready").json()["ready"] is True
        assert client.get("/openapi.json").status_code == 200
        assert client.get("/docs").status_code == 200
        model, manifest, serving = load_serving_bundle(ARTIFACTS)
        expected = predict_frame(model, manifest, real_orders)
        lower, upper = prediction_intervals(
            expected, serving["interval_half_width_hours"],
            bounds=tuple(manifest["prediction_bounds_hours"]),
        )
        ids = set()
        for i, (_, row) in enumerate(real_orders.iterrows()):
            response = client.post("/predict", json=_request(row))
            assert response.status_code == 200, response.text
            body = response.json()
            assert body["source"] == "model"
            assert body["degraded"] is False
            assert body["degradation_reason"] is None
            assert body["model_version"] == manifest["model_version"]
            assert body["estimated_delivery_hours"] == pytest.approx(expected[i], abs=1e-12)
            pi = body["prediction_interval_90"]
            assert pi["nominal_coverage"] == 0.9
            assert pi["lower_hours"] == pytest.approx(lower[i], abs=1e-12)
            assert pi["upper_hours"] == pytest.approx(upper[i], abs=1e-12)
            assert body["latency_ms"] >= 0
            assert datetime.fromisoformat(body["predicted_at"]).tzinfo is not None
            ids.add(body["request_id"])
        assert len(ids) == len(real_orders)
        metrics = client.get("/metrics").text
        assert 'delivery_predictions_total{degraded="false",source="model"} 8.0' in metrics
        assert "delivery_predict_seconds_count 8.0" in metrics
    log = [json.loads(line) for line in (tmp_path / "audit.jsonl").read_text().splitlines()]
    assert len(log) == len(real_orders)
    assert {entry["request_id"] for entry in log} == ids
    assert all("distance_km" not in entry and "warehouse_load" not in entry for entry in log)


@pytest.mark.parametrize("change", [
    {"distance_km": -0.1}, {"distance_km": 1000.1},
    {"distance_km": "100"}, {"distance_km": None}, {"distance_km": True},
    {"ordered_at": "2026-01-01T00:00:00"},
    {"ordered_at": "not-a-date"},
    {"order_id": ""}, {"order_id": "order with spaces"},
    {"warehouse_load": 0.5}, {"rain_forecast": 1},
    {"category": "unknown-category"}, {"delivery_hours": 40},
])
def test_invalid_http_inputs_are_rejected_not_logged(tmp_path, real_orders, change):
    app = create_app(log_path=tmp_path / "audit.jsonl")
    payload = _request(real_orders.iloc[0])
    payload.update(change)
    with TestClient(app) as client:
        assert client.post("/predict", json=payload).status_code == 422
        assert "delivery_predictions_total{" not in client.get("/metrics").text
    assert not (tmp_path / "audit.jsonl").exists()


class SleepySource:
    def __init__(self):
        self.cancelled = threading.Event()
        self.called = 0

    async def fetch(self, ordered_at):
        self.called += 1
        try:
            await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        return OnlineFeatures(0.5, 0, "w", "r")


def test_feature_timeout_is_cancelled_and_uses_audited_rule(tmp_path, real_orders):
    source = SleepySource()
    app = create_app(
        log_path=tmp_path / "timeout.jsonl", feature_store=source,
        feature_timeout_ms=10,
    )
    with TestClient(app) as client:
        response = client.post("/predict", json=_request(real_orders.iloc[0]))
        assert response.status_code == 200
        body = response.json()
        assert body["degraded"] is True
        assert body["source"] == "rule"
        assert body["degradation_reason"] == "feature_timeout"
        assert body["model_version"] == "delivery-rule-v1"
        assert body["prediction_interval_90"] is None
        assert body["estimated_delivery_hours"] == pytest.approx(
            36 + .04 * real_orders.iloc[0].distance_km,
        )
        assert source.cancelled.wait(timeout=0.3)
        assert source.called == 1
        assert body["latency_ms"] < 250
        assert 'delivery_degradations_total{reason="feature_timeout"} 1.0' in client.get("/metrics").text
    recorded = json.loads((tmp_path / "timeout.jsonl").read_text().strip())
    assert recorded["degradation_reason"] == "feature_timeout"


class BrokenSource:
    def __init__(self, error):
        self.error = error

    async def fetch(self, ordered_at):
        raise self.error("test")


@pytest.mark.parametrize(("error", "reason"), [
    (FeatureMissingError, "feature_missing"),
    (FeatureStaleError, "feature_stale"),
    (FeatureUnavailableError, "feature_unavailable"),
])
def test_feature_failures_are_distinct(tmp_path, real_orders, error, reason):
    app = create_app(log_path=tmp_path / "audit.jsonl", feature_store=BrokenSource(error))
    with TestClient(app) as client:
        assert client.get("/health/ready").status_code == 200
        output = client.post("/predict", json=_request(real_orders.iloc[0])).json()
        assert output["degradation_reason"] == reason
        assert output["degraded"] is True
        assert output["source"] == "rule"
        assert output["prediction_interval_90"] is None
        assert f'delivery_degradations_total{{reason="{reason}"}} 1.0' in client.get("/metrics").text


def test_expired_and_missing_history_no_online_imputation(tmp_path, real_orders):
    app = create_app(log_path=tmp_path / "audit.jsonl")
    payload = _request(real_orders.iloc[0])
    with TestClient(app) as client:
        payload["ordered_at"] = "2025-01-01T00:00:00Z"
        assert client.post("/predict", json=payload).json()["degradation_reason"] == "feature_missing"
        payload["ordered_at"] = "2027-01-01T00:00:00Z"
        assert client.post("/predict", json=payload).json()["degradation_reason"] == "feature_stale"


def test_model_corruption_does_not_prevent_liveness_or_call_feature_store(tmp_path, real_orders):
    corrupt = tmp_path / "invalid-artifacts"
    shutil.copytree(ARTIFACTS, corrupt)
    (corrupt / "model.joblib").write_bytes(b"invalid model")
    source = SleepySource()
    app = create_app(artifact_dir=corrupt, feature_store=source, log_path=tmp_path / "audit.jsonl")
    with TestClient(app) as client:
        assert client.get("/health/live").status_code == 200
        ready = client.get("/health/ready")
        assert ready.status_code == 503
        assert ready.json()["model_loaded"] is False
        result = client.post("/predict", json=_request(real_orders.iloc[0])).json()
        assert result["source"] == "rule"
        assert result["degradation_reason"] == "model_unavailable"
        assert source.called == 0


def test_missing_feature_history_changes_readiness_but_rule_still_answers(tmp_path, real_orders):
    app = create_app(events_path=tmp_path / "absent.csv", log_path=tmp_path / "audit.jsonl")
    with TestClient(app) as client:
        assert client.get("/health/ready").status_code == 503
        body = client.post("/predict", json=_request(real_orders.iloc[0])).json()
        assert body["degradation_reason"] == "feature_unavailable"
        assert body["prediction_interval_90"] is None


def test_model_loader_called_once_at_startup(monkeypatch, tmp_path, real_orders):
    import src.service as module
    original = module.load_serving_bundle
    n = 0

    def counted(path):
        nonlocal n
        n += 1
        return original(path)

    monkeypatch.setattr(module, "load_serving_bundle", counted)
    app = create_app(log_path=tmp_path / "audit.jsonl")
    with TestClient(app) as client:
        for i in range(5):
            result = client.post("/predict", json=_request(real_orders.iloc[0], order_id=f"o-test-{i}"))
            assert result.status_code == 200
    assert n == 1


def test_concurrent_requests_have_unique_ids_and_noninterleaved_jsonl(tmp_path, real_orders):
    app = create_app(log_path=tmp_path / "audit.jsonl")
    with TestClient(app) as client:
        def go(i):
            body = _request(real_orders.iloc[0], order_id=f"o-concurrent-{i}")
            response = client.post("/predict", json=body)
            assert response.status_code == 200, response.text
            return response.json()

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(go, range(20)))
        assert len({r["request_id"] for r in results}) == 20
        assert all(r["source"] == "model" for r in results)
        assert 'delivery_predictions_total{degraded="false",source="model"} 20.0' in client.get("/metrics").text
        log = [json.loads(line) for line in (tmp_path / "audit.jsonl").read_text().splitlines()]
        assert len(log) == 20
        assert {v["request_id"] for v in log} == {v["request_id"] for v in results}


def test_audit_io_failure_returns_503_and_is_not_counted(tmp_path, real_orders):
    # A regular file at parent path makes directory creation fail.
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory")
    app = create_app(log_path=blocked / "logs.jsonl")
    with TestClient(app) as client:
        response = client.post("/predict", json=_request(real_orders.iloc[0]))
        assert response.status_code == 503
        assert response.json()["detail"] == "audit_unavailable"
        assert "delivery_predictions_total{" not in client.get("/metrics").text


def test_metrics_are_isolated_for_multiple_apps(tmp_path, real_orders):
    left = create_app(log_path=tmp_path / "a.jsonl")
    right = create_app(log_path=tmp_path / "b.jsonl")
    with TestClient(left) as a, TestClient(right) as b:
        a.post("/predict", json=_request(real_orders.iloc[0]))
        assert 'delivery_predictions_total{degraded="false",source="model"} 1.0' in a.get("/metrics").text
        assert "delivery_predictions_total{" not in b.get("/metrics").text


def test_missing_required_fields_are_rejected(tmp_path, real_orders):
    app = create_app(log_path=tmp_path / "audit.jsonl")
    with TestClient(app) as client:
        valid = _request(real_orders.iloc[0])
        for field in ("distance_km", "ordered_at", "order_id"):
            payload = valid.copy()
            del payload[field]
            assert client.post("/predict", json=payload).status_code == 422
    assert not (tmp_path / "audit.jsonl").exists()


def test_malformed_server_owned_features_do_not_reach_model(tmp_path, real_orders):
    class InvalidSource:
        async def fetch(self, ordered_at):
            return OnlineFeatures(warehouse_load=2.0, rain_forecast=0,
                                  warehouse_event_id="w", weather_event_id="r")

    app = create_app(log_path=tmp_path / "audit.jsonl", feature_store=InvalidSource())
    with TestClient(app) as client:
        response = client.post("/predict", json=_request(real_orders.iloc[0]))
        assert response.status_code == 200
        assert response.json()["degradation_reason"] == "feature_invalid"
        assert response.json()["source"] == "rule"
        assert response.json()["prediction_interval_90"] is None
