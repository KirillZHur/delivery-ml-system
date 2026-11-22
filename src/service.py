"""FastAPI delivery estimates with PIT features, versioned model and fallback.

Example (after python -m src.prepare_service):
    uvicorn src.service:app --host 127.0.0.1 --port 8000

The HTTP request contains ONLY order data. Warehouse and weather features are
owned by the service and recovered from observations available at order time.
All successful decisions are written to a JSONL audit log; no label is read.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol
from uuid import uuid4

import pandas as pd
from fastapi import FastAPI, HTTPException, Response, status
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Histogram, generate_latest

from data.generate import ROOT
from src.bundle import predict_frame
from src.evaluate import prediction_intervals
from src.fast_predict import FastLinearPredictor
from src.feature_store import (
    EventFeatureStore,
    FeatureMissingError,
    FeatureStaleError,
    FeatureUnavailableError,
    OnlineFeatures,
)
from src.serving_bundle import load_serving_bundle


class FeatureSource(Protocol):
    async def fetch(self, ordered_at: datetime) -> OnlineFeatures: ...


class DeliveryRequest(BaseModel):
    """Explicit request contract; clients cannot forge online features."""

    model_config = ConfigDict(extra="forbid")
    order_id: str = Field(strict=True, min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")
    ordered_at: AwareDatetime
    distance_km: float = Field(strict=True, ge=0.0, le=1000.0, allow_inf_nan=False)


class PredictionInterval(BaseModel):
    lower_hours: float
    upper_hours: float
    nominal_coverage: float


class DeliveryResponse(BaseModel):
    request_id: str
    order_id: str
    predicted_at: datetime
    model_version: str
    source: str
    degraded: bool
    degradation_reason: str | None
    estimated_delivery_hours: float
    prediction_interval_90: PredictionInterval | None
    latency_ms: float


class JsonlAudit:
    """Write complete, noninterleaved response records across concurrent requests."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def append(self, item: DeliveryResponse) -> None:
        # Never include distance, online feature values, or labels in the log.
        row = item.model_dump(mode="json")
        line = json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line)


def _fallback_hours(distance_km: float, cfg: dict) -> float:
    rule = cfg["fallback"]
    estimate = (
        float(rule["base_hours"])
        + float(rule["distance_hours_per_km"]) * distance_km
        + float(rule["warehouse_hours_per_unit"]) * float(rule["assumed_warehouse_load"])
        + float(rule["rain_delay_hours"]) * int(rule["assumed_rain"])
    )
    lo, hi = (float(v) for v in rule["prediction_bounds_hours"])
    if not (math.isfinite(estimate) and 0 < lo < hi):
        raise ValueError("Invalid service fallback settings")
    return max(lo, min(hi, estimate))


def _validate_config(cfg: dict) -> None:
    timeout_ms = cfg["feature_timeout_ms"]
    if not (isinstance(timeout_ms, (int, float)) and math.isfinite(timeout_ms) and timeout_ms > 0):
        raise ValueError("feature_timeout_ms must be positive")
    for feature in ("warehouse_load", "rain_forecast"):
        hours = cfg["feature_max_age_hours"][feature]
        if not (math.isfinite(hours) and hours > 0):
            raise ValueError("feature max age must be positive")
    if not cfg["fallback"]["version"]:
        raise ValueError("fallback version cannot be empty")
    _fallback_hours(100, cfg)


def create_app(
    *,
    artifact_dir: Path | None = None,
    events_path: Path | None = None,
    config_path: Path | None = None,
    log_path: Path | None = None,
    feature_store: FeatureSource | None = None,
    feature_timeout_ms: float | None = None,
    prediction_mode: str | None = None,
) -> FastAPI:
    """Build isolated app state and metrics; models are loaded in lifespan only.

    Optional injected feature_store supports deterministic timeout/unavailability
    tests without patching application code or contacting external services.
    """
    artifact_dir = Path(artifact_dir) if artifact_dir is not None else ROOT / "artifacts"
    events_path = Path(events_path) if events_path is not None else ROOT / "data/generated/events.csv"
    config_path = Path(config_path) if config_path is not None else ROOT / "configs/service.json"
    log_path = Path(log_path) if log_path is not None else Path(
        os.environ.get("DELIVERY_AUDIT_PATH", str(ROOT / "runtime/predictions.jsonl"))
    )
    prediction_mode = prediction_mode or os.environ.get("DELIVERY_INFERENCE_MODE", "fast")
    if prediction_mode not in ("baseline", "fast"):
        raise ValueError("prediction_mode must be baseline or fast")
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    if feature_timeout_ms is not None:
        cfg["feature_timeout_ms"] = feature_timeout_ms
    _validate_config(cfg)

    registry = CollectorRegistry(auto_describe=True)
    predictions = Counter(
        "delivery_predictions_total", "Delivery HTTP predictions by source and degraded state",
        labelnames=("source", "degraded"), registry=registry,
    )
    degradations = Counter(
        "delivery_degradations_total", "Delivery degraded responses by reason",
        labelnames=("reason",), registry=registry,
    )
    latency = Histogram(
        "delivery_predict_seconds", "Time spent on delivery prediction and audit logging",
        buckets=(0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0),
        registry=registry,
    )

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.model = None
        application.state.manifest = None
        application.state.calibration = None
        application.state.fast_predictor = None
        application.state.feature_store = None
        application.state.model_error = False
        application.state.store_error = False
        try:
            model, manifest, serving = load_serving_bundle(artifact_dir)
        except (OSError, KeyError, ValueError, TypeError, json.JSONDecodeError, ImportError, EOFError):
            application.state.model_error = True
        else:
            application.state.model = model
            application.state.manifest = manifest
            application.state.calibration = serving
            if prediction_mode == "fast":
                application.state.fast_predictor = FastLinearPredictor.from_model(model)
        if feature_store is not None:
            application.state.feature_store = feature_store
        else:
            try:
                application.state.feature_store = EventFeatureStore.from_csv(
                    events_path, max_age_hours=cfg["feature_max_age_hours"]
                )
            except (OSError, KeyError, ValueError, TypeError, pd.errors.ParserError):
                application.state.store_error = True
        yield
        application.state.model = None
        application.state.manifest = None
        application.state.calibration = None
        application.state.fast_predictor = None
        application.state.feature_store = None

    app = FastAPI(
        title="Delivery Prediction Service",
        description="Versioned point-in-time delivery estimates with auditable fallback.",
        version="0.4.0",
        lifespan=lifespan,
    )
    app.state.model = None
    app.state.manifest = None
    app.state.calibration = None
    app.state.fast_predictor = None
    app.state.inference_mode = prediction_mode
    app.state.feature_store = None
    app.state.model_error = False
    app.state.store_error = False
    app.state.audit = JsonlAudit(log_path)
    app.state.registry = registry

    @app.get("/health/live")
    async def liveness() -> dict:
        return {"live": True}

    @app.get("/health/ready")
    async def readiness(response: Response) -> dict:
        model_ready = app.state.model is not None and app.state.calibration is not None
        feature_ready = app.state.feature_store is not None
        ready = model_ready and feature_ready
        if not ready:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {
            "ready": ready,
            "model_loaded": model_ready,
            "feature_source_loaded": feature_ready,
            "model_version": app.state.manifest["model_version"] if model_ready else None,
        }

    @app.get("/metrics", include_in_schema=False)
    async def metrics() -> Response:
        return Response(content=generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    @app.post("/predict", response_model=DeliveryResponse)
    async def predict(request: DeliveryRequest) -> DeliveryResponse:
        start = time.perf_counter()
        reason: str | None = None
        output: PredictionInterval | None = None
        model = app.state.model
        manifest = app.state.manifest
        calibration = app.state.calibration
        store = app.state.feature_store
        prediction: float | None = None
        source = "rule"
        version = cfg["fallback"]["version"]

        if model is None or calibration is None or manifest is None:
            reason = "model_unavailable"
        elif store is None:
            reason = "feature_unavailable"
        else:
            try:
                values = await asyncio.wait_for(
                    store.fetch(request.ordered_at),
                    timeout=float(cfg["feature_timeout_ms"]) / 1000.0,
                )
            except TimeoutError:
                reason = "feature_timeout"
            except FeatureMissingError:
                reason = "feature_missing"
            except FeatureStaleError:
                reason = "feature_stale"
            except FeatureUnavailableError:
                reason = "feature_unavailable"
            else:
                try:
                    load = float(values.warehouse_load)
                    rain = float(values.rain_forecast)
                    if not (math.isfinite(load) and 0 <= load <= 1 and rain in (0, 1)):
                        raise ValueError("Invalid server-owned feature values")
                except (AttributeError, TypeError, ValueError):
                    reason = "feature_invalid"
                else:
                    # Reference path keeps the original DataFrame -> sklearn pipeline.
                    # Fast path uses prevalidated folded affine coefficients.
                    fast_predictor = app.state.fast_predictor
                    bounds = tuple(manifest["prediction_bounds_hours"])
                    if fast_predictor is not None:
                        prediction = fast_predictor.predict_one(
                            (request.distance_km, load, rain), bounds
                        )
                    else:
                        row = pd.DataFrame([{
                            "distance_km": request.distance_km,
                            "warehouse_load": load,
                            "rain_forecast": int(rain),
                        }])
                        prediction = float(predict_frame(model, manifest, row)[0])
                    bounds = tuple(manifest["prediction_bounds_hours"])
                    width = float(calibration["interval_half_width_hours"])
                    lower, upper = prediction_intervals(
                        [prediction], width, bounds=bounds
                    )
                    output = PredictionInterval(
                        lower_hours=float(lower[0]),
                        upper_hours=float(upper[0]),
                        nominal_coverage=float(calibration["interval_coverage"]),
                    )
                    source = "model"
                    version = manifest["model_version"]

        if prediction is None:
            prediction = _fallback_hours(request.distance_km, cfg)
            output = None  # Rule forecasts have no validated 90% interval.
        result = DeliveryResponse(
            request_id=str(uuid4()),
            order_id=request.order_id,
            predicted_at=datetime.now(timezone.utc),
            model_version=version,
            source=source,
            degraded=reason is not None,
            degradation_reason=reason,
            estimated_delivery_hours=prediction,
            prediction_interval_90=output,
            # Internal timer stops before journal write and HTTP serialization.
            latency_ms=(time.perf_counter() - start) * 1000.0,
        )
        try:
            app.state.audit.append(result)
        except OSError as exc:
            raise HTTPException(status_code=503, detail="audit_unavailable") from exc
        latency.observe(time.perf_counter() - start)
        predictions.labels(source=source, degraded=str(result.degraded).lower()).inc()
        if reason is not None:
            degradations.labels(reason=reason).inc()
        return result

    return app


app = create_app()
