"""Versioned shadow/canary rollout with deterministic user routing and rollback.

Operational observations in src.replay are SYNTHETIC; neither simulated latency
nor a healthy canary proves real-world safety. Versions are real, integrity-
checked model/preprocessing/calibration bundles, verified over FastAPI in replay.
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from src.serving_bundle import load_serving_bundle


def route_user(user_id: str, share: float) -> Literal["v1", "v2"]:
    """Sticky monotone route, independent of process hash randomization."""
    if not isinstance(user_id, str) or not user_id:
        raise ValueError("user_id must be a nonempty string")
    if isinstance(share, bool) or not isinstance(share, (float, int)) or not math.isfinite(share) or not 0 <= share <= 1:
        raise ValueError("share must be between 0 and 1")
    if share == 0:
        return "v1"
    if share == 1:
        return "v2"
    h = int.from_bytes(hashlib.sha256(user_id.encode("utf-8")).digest()[:8], "big")
    return "v2" if h < int(share * (1 << 64)) else "v1"


def nearest_rank_p95(values: np.ndarray | list[float]) -> float | None:
    """Nearest-rank percentile; technical failures and timeouts are included."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError("latencies must be one-dimensional")
    if not len(values):
        return None
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("latencies must be finite and nonnegative")
    return float(np.sort(values)[math.ceil(0.95 * len(values)) - 1])


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def create_release_bundles(source: Path, root: Path, *, replace: bool = False) -> dict:
    """Create standalone, checksum-linked v1/v2 bundles from a trusted model.

    v1 uses original sklearn pipeline; v2 uses folded linear coefficients.
    Each manifest/serving record is internally checksummed. The source stays intact.
    """
    source, root = Path(source), Path(root)
    _, original, original_serving = load_serving_bundle(source)
    if original["selected_model"] != "linear":
        raise ValueError("Fast candidate requires the selected linear model")
    if root.resolve() == source.resolve() or source.resolve() in root.resolve().parents:
        raise ValueError("Release root must not overwrite the training artifacts")
    if root.exists():
        if not replace:
            raise FileExistsError("Release bundles already exist")
        shutil.rmtree(root)
    root.mkdir(parents=True)
    result = {}
    for version, mode in (("v1", "baseline"), ("v2", "fast")):
        dest = root / version
        dest.mkdir()
        shutil.copy2(source / "model.joblib", dest / "model.joblib")
        manifest = dict(original)
        manifest["model_version"] = f"{original['model_version']}-{version}"
        manifest["model_sha256"] = _sha(dest / "model.joblib")
        _write_json(dest / "manifest.json", manifest)
        serving = dict(original_serving)
        serving["model_version"] = manifest["model_version"]
        serving["model_sha256"] = manifest["model_sha256"]
        serving["manifest_sha256"] = _sha(dest / "manifest.json")
        _write_json(dest / "serving.json", serving)
        _write_json(dest / "release.json", {
            "version": version,
            "inference_mode": mode,
            "manifest_sha256": _sha(dest / "manifest.json"),
            "serving_sha256": _sha(dest / "serving.json"),
            "model_sha256": manifest["model_sha256"],
        })
        result[version] = {"path": str(dest), "mode": mode, "model_version": manifest["model_version"]}
    validate_release_bundles(root)
    return result


def validate_release_bundles(root: Path) -> dict:
    """Fail closed if any version or its fallback contract is incompatible."""
    root = Path(root)
    loaded = {}
    for version in ("v1", "v2"):
        directory = root / version
        _, manifest, serving = load_serving_bundle(directory)
        metadata = json.loads((directory / "release.json").read_text(encoding="utf-8"))
        expected_mode = "baseline" if version == "v1" else "fast"
        if (metadata.get("version") != version or metadata.get("inference_mode") != expected_mode
                or metadata.get("manifest_sha256") != _sha(directory / "manifest.json")
                or metadata.get("serving_sha256") != _sha(directory / "serving.json")
                or metadata.get("model_sha256") != _sha(directory / "model.joblib")):
            raise ValueError(f"Integrity or inference mode mismatch for {version}")
        if not manifest["model_version"].endswith(f"-{version}"):
            raise ValueError("Release version suffix mismatch")
        loaded[version] = {"manifest": manifest, "serving": serving, "metadata": metadata}
    # This rollout changes the inference implementation, NOT the trained model,
    # feature/data snapshot, prediction limits, or calibration.
    for key in ("selected_model", "feature_order", "target", "prediction_bounds_hours", "model_sha256", "data_sha256", "config_sha256"):
        if loaded["v1"]["manifest"].get(key) != loaded["v2"]["manifest"].get(key):
            raise ValueError(f"Incompatible release manifests: {key}")
    for key in ("feature_order", "interval_half_width_hours", "interval_coverage", "calibration_source", "calibration_orders"):
        if loaded["v1"]["serving"].get(key) != loaded["v2"]["serving"].get(key):
            raise ValueError(f"Incompatible calibration: {key}")
    return loaded


@dataclass(frozen=True)
class VersionHealth:
    requests: int
    errors: int
    degraded: int
    p95_ms: float | None
    shadow: bool = False

    def __post_init__(self):
        if (type(self.requests) is not int or type(self.errors) is not int or
                type(self.degraded) is not int or not (0 <= self.errors <= self.requests)
                or not (0 <= self.degraded <= self.requests)):
            raise ValueError("Invalid version request/error/degraded counts")
        if self.requests == 0 and self.p95_ms is not None:
            raise ValueError("Empty group must have undefined p95")
        if self.requests > 0 and (self.p95_ms is None or not math.isfinite(self.p95_ms) or self.p95_ms < 0):
            raise ValueError("Nonempty group needs finite nonnegative p95")

    @property
    def error_rate(self) -> float | None:
        return self.errors / self.requests if self.requests else None

    @property
    def degraded_rate(self) -> float | None:
        return self.degraded / self.requests if self.requests else None

    def as_dict(self) -> dict:
        return {"requests": self.requests, "errors": self.errors, "degraded": self.degraded,
                "error_rate": self.error_rate, "degraded_rate": self.degraded_rate,
                "p95_ms": self.p95_ms, "shadow": self.shadow}


@dataclass(frozen=True)
class Window:
    id: int
    share: float
    stable: VersionHealth
    candidate: VersionHealth
    unique_order_ids: int
    # candidate.shadow is true in shadow stage and at 100% only stable.shadow true.


@dataclass
class RolloutController:
    """Keep operational safety evidence and traffic state separate from data generation.

    A window's share is taken from controller before simulation; immediately
    on confirmed rollback the *next* window uses 0. Old in-flight requests keep
    the actual version recorded when admitted (never rewritten by the switch).
    """
    config: dict
    status: str = "shadow"
    active_share: float = 0.0
    level: int = 0
    healthy_streak: int = 0
    shadow_streak: int = 0
    error_streak: int = 0
    last_window: int = 0
    events: list[dict] = field(default_factory=list)

    def __post_init__(self):
        plan = self.config["canary_shares"]
        if (not isinstance(plan, list) or len(plan) < 1 or plan[-1] != 1.0
                or any(type(v) not in (float, int) or not 0 < v <= 1 for v in plan)
                or any(left >= right for left, right in zip(plan, plan[1:]))):
            raise ValueError("Rollout shares must be increasing and finish at 1")
        for key in ("shadow_windows", "healthy_windows_per_step", "min_observations_per_version", "error_confirmation_windows", "error_confirmation_min_requests"):
            if type(self.config[key]) is not int or self.config[key] <= 0:
                raise ValueError(f"{key} must be positive integer")
        for key in ("max_error_rate", "max_degraded_rate", "max_p95_latency_ms"):
            if type(self.config[key]) not in (float, int) or not math.isfinite(self.config[key]) or self.config[key] <= 0:
                raise ValueError(f"{key} must be finite positive")

    def _record(self, window: Window, action: str, previous: str, reasons: list[str]) -> dict:
        row = {"window_id": window.id, "previous_status": previous, "status": self.status,
               "action": action, "share_before": window.share, "share_after": self.active_share,
               "reasons": reasons, "stable": window.stable.as_dict(),
               "candidate": window.candidate.as_dict(),
               "healthy_streak": self.healthy_streak, "error_streak": self.error_streak,
               "limits": {"min_version_requests": self.config["min_observations_per_version"],
                          "p95_ms": self.config["max_p95_latency_ms"],
                          "error_rate": self.config["max_error_rate"],
                          "degraded_rate": self.config["max_degraded_rate"]}}
        self.events.append(row)
        return row

    def observe(self, window: Window) -> dict:
        if self.status in ("rolled_back", "promoted"):
            raise ValueError("Terminal rollout cannot process new windows")
        if window.id <= self.last_window or window.id < 1:
            raise ValueError("Window IDs must be strictly increasing")
        if not math.isclose(window.share, self.active_share, abs_tol=1e-12):
            raise ValueError("Window share does not match active route")
        if window.unique_order_ids <= 0:
            raise ValueError("Window needs unique requests")
        if self.status == "shadow":
            if not window.candidate.shadow or window.stable.shadow:
                raise ValueError("Shadow stage must serve v1 and shadow v2")
            if window.stable.requests != window.unique_order_ids or window.candidate.requests != window.unique_order_ids:
                raise ValueError("Shadow comparison must observe every request in both versions")
        elif self.active_share == 1:
            if not window.stable.shadow or window.candidate.shadow:
                raise ValueError("100% canary needs stable shadow control")
            if window.candidate.requests != window.unique_order_ids or window.stable.requests != window.unique_order_ids:
                raise ValueError("Shadow control must receive full input copies")
        else:
            if window.stable.shadow or window.candidate.shadow:
                raise ValueError("Partial canary versions must both serve live traffic")
            if window.candidate.requests + window.stable.requests != window.unique_order_ids:
                raise ValueError("Requests lost or double-served")
        self.last_window = window.id
        previous = self.status
        stable, candidate = window.stable, window.candidate
        min_n = self.config["min_observations_per_version"]
        if stable.requests < min_n or candidate.requests < min_n:
            self.healthy_streak = self.shadow_streak = self.error_streak = 0
            return self._record(window, "hold", previous, ["insufficient_data"])

        stable_degraded = stable.degraded_rate > self.config["max_degraded_rate"]
        candidate_degraded = candidate.degraded_rate > self.config["max_degraded_rate"]
        if stable_degraded and candidate_degraded:
            self.healthy_streak = self.shadow_streak = self.error_streak = 0
            return self._record(window, "hold", previous, ["shared_feature_outage"])

        # A healthy baseline is a precondition for confident release decisions.
        if stable_degraded or stable.error_rate > self.config["max_error_rate"] or stable.p95_ms > self.config["max_p95_latency_ms"]:
            self.healthy_streak = self.shadow_streak = self.error_streak = 0
            return self._record(window, "hold", previous, ["stable_unhealthy"])

        reasons = []
        if candidate_degraded:
            reasons.append("candidate_degraded_rate")
        if candidate.p95_ms > self.config["max_p95_latency_ms"]:
            reasons.append("candidate_p95_latency")
        # Only *two* adequate windows with >1% errors trigger the error-rate rollback.
        if candidate.requests >= self.config["error_confirmation_min_requests"]:
            if candidate.error_rate > self.config["max_error_rate"]:
                self.error_streak += 1
            else:
                self.error_streak = 0
            if self.error_streak >= self.config["error_confirmation_windows"]:
                reasons.append("candidate_technical_errors")
        else:
            self.error_streak = 0
        if reasons:
            self.status = "rolled_back"
            self.active_share = 0.0
            self.healthy_streak = self.shadow_streak = 0
            return self._record(window, "rollback", previous, reasons)
        if candidate.error_rate > self.config["max_error_rate"]:
            self.healthy_streak = self.shadow_streak = 0
            return self._record(window, "hold", previous, ["candidate_errors_wait_confirmation"])

        if self.status == "shadow":
            self.shadow_streak += 1
            if self.shadow_streak < self.config["shadow_windows"]:
                return self._record(window, "hold", previous, ["shadow_observation"])
            self.status = "canary"
            self.active_share = self.config["canary_shares"][0]
            self.error_streak = 0
            self.healthy_streak = 0
            return self._record(window, "advance", previous, ["shadow_healthy"])

        self.healthy_streak += 1
        if self.healthy_streak < self.config["healthy_windows_per_step"]:
            return self._record(window, "hold", previous, ["await_second_healthy_window"])
        self.healthy_streak = 0
        if self.active_share == 1:
            self.status = "promoted"
            return self._record(window, "complete", previous, ["full_traffic_observed"])
        self.level += 1
        self.error_streak = 0
        self.active_share = self.config["canary_shares"][self.level]
        return self._record(window, "advance", previous, ["healthy_window_confirmation"])
