"""Exercise shadow/canary/rollback against real bundles, with SYNTHETIC operational events.

Run: python -m src.replay
All state/metrics written below reports/rollout_*. TestClient verifies v1/v2
predictions, but window timings are generated, NOT HTTP load measurements.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from data.generate import ROOT
from src.rollout import (
    RolloutController, VersionHealth, Window, create_release_bundles,
    nearest_rank_p95, route_user, validate_release_bundles,
)
from src.service import create_app


def _json_dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _jsonl_dump(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in rows), encoding="utf-8")


def verify_http_versions(release_dir: Path, features_path: Path, events_path: Path,
                         *, n: int = 12) -> dict:
    """Real FastAPI handlers, real PIT store, real model and saved calibration.

    Two variants share the *same verified model weights*; only inference mode
    differs. This is a parity/readiness check, not an operational load test.
    """
    validate_release_bundles(release_dir)
    frame = pd.read_csv(features_path).iloc[100:100 + n]
    if len(frame) != n:
        raise ValueError("Not enough sample orders for HTTP parity")
    app_v1 = create_app(artifact_dir=release_dir / "v1", events_path=events_path,
                        log_path=release_dir / "http_v1.jsonl", prediction_mode="baseline")
    app_v2 = create_app(artifact_dir=release_dir / "v2", events_path=events_path,
                        log_path=release_dir / "http_v2.jsonl", prediction_mode="fast")
    predictions = []
    with TestClient(app_v1) as old, TestClient(app_v2) as new:
        for client, version in ((old, "v1"), (new, "v2")):
            readiness = client.get("/health/ready")
            if readiness.status_code != 200 or not readiness.json()["model_version"].endswith(f"-{version}"):
                raise RuntimeError(f"Version {version} is not ready")
        for row in frame.itertuples():
            payload = {"order_id": row.order_id, "ordered_at": row.ordered_at,
                       "distance_km": float(row.distance_km)}
            before = old.post("/predict", json=payload)
            after = new.post("/predict", json=payload)
            if before.status_code != 200 or after.status_code != 200:
                raise RuntimeError("Canary parity check HTTP request failed")
            a, b = before.json(), after.json()
            if a["source"] != "model" or b["source"] != "model" or a["degraded"] or b["degraded"]:
                raise RuntimeError("Canary parity samples must use healthy models")
            if a["prediction_interval_90"]["nominal_coverage"] != b["prediction_interval_90"]["nominal_coverage"]:
                raise RuntimeError("Candidate interval coverage differs")
            predictions.append(abs(a["estimated_delivery_hours"] - b["estimated_delivery_hours"]))
            for bound in ("lower_hours", "upper_hours"):
                if abs(a["prediction_interval_90"][bound] - b["prediction_interval_90"][bound]) > 1e-12:
                    raise RuntimeError("Candidate interval numerical parity failed")
    for filename in ("http_v1.jsonl", "http_v2.jsonl"):
        (release_dir / filename).unlink(missing_ok=True)
    maximum = max(predictions)
    if maximum > 1e-12:
        raise RuntimeError(f"Numerical parity failure: {maximum}")
    return {"http_requests_per_version": n, "ready_v1": True, "ready_v2": True,
            "max_abs_prediction_error_hours": maximum,
            "model_quality_checked_on_test": False}


def verify_rollback_http(release_dir: Path, events_path: Path, features_path: Path) -> dict:
    """Request the *original* ready service after switching traffic off v2."""
    validate_release_bundles(release_dir)
    row = pd.read_csv(features_path).iloc[100]
    payload = {"order_id": str(row.order_id), "ordered_at": str(row.ordered_at),
               "distance_km": float(row.distance_km)}
    app = create_app(artifact_dir=release_dir / "v1", events_path=events_path,
                     log_path=release_dir / "rollback_http.jsonl", prediction_mode="baseline")
    with TestClient(app) as client:
        ready = client.get("/health/ready")
        response = client.post("/predict", json=payload)
    (release_dir / "rollback_http.jsonl").unlink(missing_ok=True)
    if ready.status_code != 200 or response.status_code != 200:
        raise RuntimeError("Stable serving bundle is not available after rollback")
    body = response.json()
    if not body["model_version"].endswith("-v1") or body["source"] != "model" or body["degraded"]:
        raise RuntimeError("Rollback did not restore healthy version v1")
    return {"status": "verified", "new_actual_version": "v1",
            "model_version": body["model_version"], "healthy_model_response": True}


def _rng(seed: int, scenario: str, window_id: int) -> np.random.Generator:
    scenario_code = int.from_bytes(hashlib.sha256(scenario.encode()).digest()[:4], "big")
    return np.random.default_rng(np.random.SeedSequence([seed, scenario_code, window_id]))


def simulate_window(controller: RolloutController, *, scenario: str, window_id: int,
                    n: int | None = None, event: str | None = None) -> tuple[Window, list[dict]]:
    """Produce an independent observable window, without inventing HTTP benchmarks."""
    cfg = controller.config
    n = cfg["window_requests"] if n is None else n
    if not isinstance(n, int) or n < 1:
        raise ValueError("Window size must be positive")
    if event not in (None, "shared_feature_outage", "candidate_latency", "candidate_errors", "candidate_feature_outage"):
        raise ValueError("Unsupported event injection")
    rng = _rng(cfg["seed"], scenario, window_id)
    share = controller.active_share
    shadow_v2 = controller.status == "shadow"
    shadow_v1 = share == 1.0 and controller.status == "canary"
    customers = [f"u{i % cfg['user_pool']:06d}" for i in range(n)]
    routes = np.array([route_user(user, share) for user in customers])
    is_v2 = routes == "v2"
    # Shadow versions receive a copy of every request but never decide its route.
    mask_v1 = np.ones(n, dtype=bool) if shadow_v1 else ~is_v2
    mask_v2 = np.ones(n, dtype=bool) if shadow_v2 else is_v2
    jitter_old = rng.normal(0, cfg["latency_jitter_ms"], n)
    jitter_new = rng.normal(0, cfg["latency_jitter_ms"], n)
    old_latency = np.clip(cfg["stable_latency_ms"] + jitter_old, 0, None)
    new_latency = np.clip(cfg["candidate_latency_ms"] + jitter_new, 0, None)
    if event == "candidate_latency":
        new_latency = np.clip(cfg["failure_latency_ms"] + jitter_new, 0, None)
    p_stable = cfg["healthy_error_probability"]
    p_candidate = cfg["failure_error_probability"] if event == "candidate_errors" else p_stable
    old_error = rng.random(n) < p_stable
    new_error = rng.random(n) < p_candidate
    p_degraded = cfg["healthy_degraded_probability"]
    old_degraded = rng.random(n) < (1.0 if event == "shared_feature_outage" else p_degraded)
    new_degraded = rng.random(n) < (1.0 if event in ("shared_feature_outage", "candidate_feature_outage") else p_degraded)

    def health(mask, errors, degraded, latency, *, shadow=False):
        return VersionHealth(int(np.sum(mask)), int(np.sum(errors[mask])),
                             int(np.sum(degraded[mask])), nearest_rank_p95(latency[mask]), shadow=shadow)

    stable = health(mask_v1, old_error, old_degraded, old_latency, shadow=shadow_v1)
    candidate = health(mask_v2, new_error, new_degraded, new_latency, shadow=shadow_v2)
    w = Window(window_id, share, stable, candidate, n)
    # Stable records prove that assignment is not silently rewritten on rollback;
    # sample both routings without logging private features or synthetic labels.
    selected = list(range(min(12, n)))
    if n > 12:
        selected += list(np.flatnonzero(is_v2)[:12])
    sample = []
    for i in dict.fromkeys(selected):
        actual = "v1" if shadow_v2 else routes[i]
        sample.append({"window_id": window_id, "order_id": f"rollout-{scenario}-{window_id:03d}-{i:06d}",
                       "user_id": customers[i], "assigned_version": str(actual),
                       "actual_version": str(actual),
                       "shadow_version": "v2" if shadow_v2 else ("v1" if shadow_v1 else None),
                       "request_error": bool(old_error[i] if actual == "v1" else new_error[i]),
                       "degraded": bool(old_degraded[i] if actual == "v1" else new_degraded[i]),
                       "latency_ms": float(old_latency[i] if actual == "v1" else new_latency[i])})
    return w, sample


def _event(scenario: str, controller: RolloutController, window_id: int) -> str | None:
    # Stage-dependent faults; automatic code does not inspect model metrics/test.
    if scenario == "shared_outage" and window_id == 3:
        return "shared_feature_outage"
    if controller.active_share == 0.25 and scenario == "candidate_latency":
        return "candidate_latency"
    if controller.active_share == 0.25 and scenario == "candidate_errors":
        return "candidate_errors"
    if controller.active_share == 0.25 and scenario == "candidate_degraded":
        return "candidate_feature_outage"
    return None


def run_scenario(config: dict, scenario: str, *, max_windows: int | None = None) -> dict:
    if scenario not in ("healthy", "shared_outage", "candidate_latency", "candidate_errors", "candidate_degraded"):
        raise ValueError("Unknown rollout scenario")
    controller = RolloutController(config)
    requests = []
    for id_ in range(1, 1 + (max_windows or config["default_scenario_windows"])):
        injection = _event(scenario, controller, id_)
        window, records = simulate_window(controller, scenario=scenario, window_id=id_, event=injection)
        decision = controller.observe(window)
        decision["scenario"] = scenario
        decision["injected_event"] = injection
        requests.extend(records)
        if controller.status in ("promoted", "rolled_back"):
            break
    result = {"scenario": scenario, "final_status": controller.status,
              "final_share": controller.active_share,
              "windows_seen": controller.last_window, "events": controller.events,
              "sampled_requests": requests}
    if controller.status == "rolled_back":
        # The first NEW request is correctly served by stable v1, without
        # relabeling already-admitted canary requests from the failed window.
        tail_id = controller.last_window + 1
        followup = {"window_id": tail_id, "new_request_id": f"rollout-{scenario}-{tail_id:03d}-000000",
                    "new_assigned_version": route_user("u000000", controller.active_share),
                    "stable_ready": True,
                    "inflight_admitted_before_rollback_version": "v2",
                    "inflight_completed_after_rollback_version": "v2",
                    "inflight_admitted_at_ms": tail_id * 60_000 - 100,
                    "inflight_completed_at_ms": tail_id * 60_000 + 50,
                    "rollback_switch_at_ms": tail_id * 60_000}
        result["recovery"] = followup
    return result


def small_window_check(config: dict) -> dict:
    controller = RolloutController(config)
    # First enough shadow traffic to authorize 5% of traffic.
    first, _ = simulate_window(controller, scenario="small_window", window_id=1)
    controller.observe(first)
    second, _ = simulate_window(controller, scenario="small_window", window_id=2, n=100)
    event = controller.observe(second)
    return {"window_size": 100, "stable_requests": second.stable.requests,
            "candidate_requests": second.candidate.requests,
            "status": controller.status, "action": event["action"],
            "reasons": event["reasons"], "share_after": controller.active_share}


def run_all(*, config_path: Path, source_dir: Path, release_dir: Path,
            features_path: Path, events_path: Path, out_dir: Path) -> dict:
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    RolloutController(cfg)  # validate config before writing artifacts
    create_release_bundles(source_dir, release_dir, replace=True)
    http_parity = verify_http_versions(release_dir, features_path, events_path)
    scenarios = [run_scenario(cfg, name) for name in (
        "healthy", "shared_outage", "candidate_latency", "candidate_errors", "candidate_degraded"
    )]
    sample = small_window_check(cfg)
    # Verify stable fallback is not merely a routing label; perform real HTTP.
    rollback_http = verify_rollback_http(release_dir, events_path, features_path)
    if not (scenarios[0]["final_status"] == scenarios[1]["final_status"] == "promoted"
            and all(x["final_status"] == "rolled_back" for x in scenarios[2:])
            and sample["action"] == "hold"):
        raise AssertionError("Unexpected release scenario outcomes")
    out_dir.mkdir(parents=True, exist_ok=True)
    events = [row for result in scenarios for row in result["events"]]
    _jsonl_dump(out_dir / "rollout_events.jsonl", events)
    _jsonl_dump(out_dir / "rollout_request_samples.jsonl",
                [row for result in scenarios for row in result["sampled_requests"]])
    with (out_dir / "rollout_windows.csv").open("w", encoding="utf-8", newline="") as fd:
        writer = csv.DictWriter(fd, fieldnames=[
            "scenario", "window_id", "injected_event", "share_before", "share_after",
            "action", "status", "reasons", "stable_n", "stable_errors", "stable_degraded",
            "stable_p95_ms", "candidate_n", "candidate_errors", "candidate_degraded", "candidate_p95_ms",
        ])
        writer.writeheader()
        for row in events:
            writer.writerow({"scenario": row["scenario"], "window_id": row["window_id"],
                             "injected_event": row["injected_event"], "share_before": row["share_before"],
                             "share_after": row["share_after"], "action": row["action"],
                             "status": row["status"], "reasons": ",".join(row["reasons"]),
                             "stable_n": row["stable"]["requests"],
                             "stable_errors": row["stable"]["errors"],
                             "stable_degraded": row["stable"]["degraded"],
                             "stable_p95_ms": row["stable"]["p95_ms"],
                             "candidate_n": row["candidate"]["requests"],
                             "candidate_errors": row["candidate"]["errors"],
                             "candidate_degraded": row["candidate"]["degraded"],
                             "candidate_p95_ms": row["candidate"]["p95_ms"]})
    _json_dump(out_dir / "rollout_summary.json", {
        "stage": "06_shadow_canary_and_rollback", "seed": cfg["seed"],
        "config_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
        "http_parity": http_parity,
        "rollback_http": rollback_http,
        "scenarios": [{"name": result["scenario"], "status": result["final_status"],
                       "windows": result["windows_seen"], "share": result["final_share"],
                       "recovery": result.get("recovery")} for result in scenarios],
        "small_window": sample,
        "notes": ["Operational latency, errors and degraded counts are synthetic, NOT HTTP load measurements.",
                  "Real FastAPI TestClient checks readiness and numerical parity of both versioned bundles.",
                  "Technical error rollback requires two consecutive windows with >=1000 candidate requests each.",
                  "Candidate canary traffic is sticky by user_id; stable uses shadow copies at 100% candidate traffic."],
    })
    return json.loads((out_dir / "rollout_summary.json").read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce versioned canary and automatic rollback")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/rollout.json")
    parser.add_argument("--source", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--release", type=Path, default=ROOT / "releases")
    parser.add_argument("--features", type=Path, default=ROOT / "data/generated/features.csv")
    parser.add_argument("--events", type=Path, default=ROOT / "data/generated/events.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    print(json.dumps(run_all(config_path=args.config, source_dir=args.source,
                             release_dir=args.release, features_path=args.features,
                             events_path=args.events, out_dir=args.out), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
