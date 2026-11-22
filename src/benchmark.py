"""Stage 5 reproducible localhost HTTP load test for baseline vs fast inference.

Run from repository root:
    python -m src.benchmark --out reports/performance.json --raw reports/latency_raw.csv

Both modes run in separate, single-worker Uvicorn processes, with the SAME
request sequence, scheduled arrival rate, model, event history, and threads.
HTTP timings include localhost networking, validation, PIT lookup, prediction,
audit logging, and response serialization. This is not a production SLO test.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import platform
import socket
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import sklearn

from data.generate import ROOT
from src.bundle import load_bundle, predict_frame


def percentile_metrics(values: list[float]) -> dict:
    """Summarize completed requests only; p95 uses linear interpolation."""
    if not values or not all(math.isfinite(x) and x >= 0 for x in values):
        raise ValueError("Expected nonempty, nonnegative, finite latency values")
    return {
        "count": len(values),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95, method="linear")),
        "p99_ms": float(np.percentile(values, 99, method="linear")),
        "max_ms": max(values),
    }


def sample_replay_requests(*, count: int, seed: int = 2026) -> tuple[list[dict], list[float]]:
    """Fixed sample from held-out replay, with offline expected predictions."""
    if count < 1:
        raise ValueError("count must be positive")
    data = pd.read_csv(ROOT / "data/generated/features.csv")
    # Take only replay rows. IDs, target and future timestamps are NEVER inputs.
    replay = data.loc[data["period"] == "replay"]
    if count > len(replay):
        raise ValueError(f"Requested {count} samples but only {len(replay)} replay rows exist")
    rng = np.random.default_rng(seed)
    chosen = replay.iloc[rng.permutation(len(replay))[:count]]
    model, manifest = load_bundle(ROOT / "artifacts")
    expected = predict_frame(model, manifest, chosen)
    payloads = [
        {"order_id": str(r.order_id), "ordered_at": str(r.ordered_at),
         "distance_km": float(r.distance_km)}
        for r in chosen.itertuples(index=False)
    ]
    return payloads, [float(x) for x in expected]


def _available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _rss_megabytes(pid: int) -> float | None:
    # `ps` is available on both macOS and Linux; no psutil dependency needed.
    try:
        result = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)],
            check=False, text=True, capture_output=True, timeout=3,
        )
        if result.returncode == 0 and result.stdout.strip():
            return int(result.stdout.strip().split()[0]) / 1024.0
    except (OSError, ValueError, subprocess.TimeoutExpired):
        pass
    return None


async def send_load(
    base_url: str, payloads: list[dict], expected: list[float], *,
    mode: str, phase: str, rps: float,
) -> tuple[list[dict], float]:
    """Open-loop scheduled arrival: one new request each 1/rps seconds.

    A single AsyncClient keeps connections alive. HTTP request/response, local
    network, JSON parsing, and server work are timed; pre-send scheduling
    lateness is tracked separately in schedule_lag_ms.
    """
    if rps <= 0 or len(payloads) != len(expected):
        raise ValueError("Invalid load profile")
    limits = httpx.Limits(max_connections=100, max_keepalive_connections=100)
    async with httpx.AsyncClient(base_url=base_url, timeout=5, limits=limits) as client:
        start_at = time.perf_counter()

        async def request_one(index: int) -> dict:
            scheduled = start_at + index / rps
            await asyncio.sleep(max(0.0, scheduled - time.perf_counter()))
            started = time.perf_counter()
            error: str | None = None
            status: int | None = None
            body: dict = {}
            try:
                response = await client.post("/predict", json=payloads[index])
                status = response.status_code
                body = response.json()
            except (httpx.HTTPError, ValueError) as exc:
                error = type(exc).__name__
            completed = time.perf_counter()
            successful = (
                error is None and status == 200 and body.get("source") == "model"
                and body.get("degraded") is False
            )
            difference = (
                abs(float(body["estimated_delivery_hours"]) - expected[index])
                if successful else None
            )
            return {
                "mode": mode, "phase": phase, "index": index,
                "order_id": payloads[index]["order_id"],
                "status": status, "error": error,
                "source": body.get("source"), "degraded": body.get("degraded"),
                "latency_ms": (completed - started) * 1000.0,
                "schedule_lag_ms": max(0.0, (started - scheduled) * 1000.0),
                "server_latency_ms": body.get("latency_ms"),
                "parity_abs_error": difference,
                "completed_at": completed,
            }

        records = await asyncio.gather(*(request_one(i) for i in range(len(payloads))))
    elapsed = max(row["completed_at"] for row in records) - start_at
    return records, elapsed


def _launch_server(*, mode: str, audit_path: Path) -> tuple[subprocess.Popen, str]:
    port = _available_port()
    env = os.environ.copy()
    env.update({
        "DELIVERY_INFERENCE_MODE": mode,
        "DELIVERY_AUDIT_PATH": str(audit_path),
        "PYTHONDONTWRITEBYTECODE": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    })
    command = [sys.executable, "-m", "uvicorn", "src.service:app", "--host", "127.0.0.1",
               "--port", str(port), "--workers", "1", "--log-level", "error", "--no-access-log"]
    process = subprocess.Popen(command, cwd=str(ROOT), env=env, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}"
    limit = time.monotonic() + 30
    while time.monotonic() < limit:
        if process.poll() is not None:
            break
        try:
            response = httpx.get(url + "/health/ready", timeout=1)
            if response.status_code == 200 and response.json().get("ready"):
                return process, url
        except (httpx.HTTPError, ValueError):
            pass
        time.sleep(0.15)
    process.terminate()
    try:
        process.wait(timeout=4)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=4)
    raise RuntimeError(f"Local HTTP server not ready for mode={mode}; check dependencies")


def _stop_server(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=6)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=4)


def benchmark(
    *, steady_count: int = 240, burst_count: int = 80, warmup_count: int = 20,
    steady_rps: float = 20.0, burst_rps: float = 40.0,
) -> tuple[dict, pd.DataFrame]:
    if min(steady_count, burst_count, warmup_count) < 1 or min(steady_rps, burst_rps) <= 0:
        raise ValueError("All request counts and rates must be positive")
    total_needed = steady_count + burst_count + warmup_count
    payloads, expected = sample_replay_requests(count=total_needed)
    groups = {
        "warmup": (payloads[:warmup_count], expected[:warmup_count], steady_rps),
        "steady": (
            payloads[warmup_count:warmup_count + steady_count],
            expected[warmup_count:warmup_count + steady_count], steady_rps,
        ),
        "burst": (payloads[-burst_count:], expected[-burst_count:], burst_rps),
    }
    details: dict = {}
    raw_rows: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="delivery-benchmark-") as temp:
        for mode in ("baseline", "fast"):
            print(f"[benchmark] starting {mode}", file=sys.stderr, flush=True)
            audit_path = Path(temp) / f"{mode}-audit.jsonl"
            process, url = _launch_server(mode=mode, audit_path=audit_path)
            try:
                rss_before = _rss_megabytes(process.pid)
                mode_output = {}
                for phase, (batch, target, rps) in groups.items():
                    print(f"[benchmark] {mode}/{phase}, n={len(batch)}", file=sys.stderr, flush=True)
                    records, elapsed = asyncio.run(
                        send_load(url, batch, target, mode=mode, phase=phase, rps=rps)
                    )
                    expected_success = len(batch)
                    actual_success = sum(
                        r["status"] == 200 and r["source"] == "model" and not r["degraded"]
                        and r["parity_abs_error"] is not None and r["parity_abs_error"] <= 1e-12
                        for r in records
                    )
                    if actual_success != expected_success:
                        raise RuntimeError(
                            f"{mode}/{phase}: {actual_success}/{expected_success} valid parity responses"
                        )
                    times = [float(r["latency_ms"]) for r in records]
                    internal = [float(r["server_latency_ms"]) for r in records]
                    mode_output[phase] = {
                        "scheduled_rps": rps,
                        "completed_rps": float(len(records) / elapsed),
                        "latency_ms": percentile_metrics(times),
                        "server_handler_ms": percentile_metrics(internal),
                        "errors": len(records) - actual_success,
                        "degraded": sum(bool(r["degraded"]) for r in records),
                        "max_prediction_abs_difference": max(float(r["parity_abs_error"]) for r in records),
                    }
                    # Warmup records are not used in reported latency distribution.
                    if phase != "warmup":
                        for row in records:
                            clean = {key: val for key, val in row.items() if key != "completed_at"}
                            raw_rows.append(clean)
                rss_after = _rss_megabytes(process.pid)
                # Check count of successfully journaled responses, including warmup.
                audit_lines = audit_path.read_text(encoding="utf-8").splitlines()
                if len(audit_lines) != total_needed:
                    raise RuntimeError(f"Audit count mismatch for {mode}: {len(audit_lines)}")
                mode_output["rss_mb"] = {"before": rss_before, "after": rss_after}
                mode_output["audited_responses"] = len(audit_lines)
                details[mode] = mode_output
                print(f"[benchmark] finished {mode}", file=sys.stderr, flush=True)
            finally:
                _stop_server(process)
    earlier = json.loads((ROOT / "reports/evaluation_metrics.json").read_text(encoding="utf-8"))
    if earlier["model_version"] != json.loads(
        (ROOT / "artifacts/manifest.json").read_text(encoding="utf-8")
    )["model_version"]:
        raise RuntimeError("Model and earlier evaluation report disagree")
    largest_parity_error = max(
        details[mode][phase]["max_prediction_abs_difference"]
        for mode in ("baseline", "fast")
        for phase in ("warmup", "steady", "burst")
    )
    if largest_parity_error > 1e-12:
        raise RuntimeError("Optimization changed the model prediction beyond tolerance")
    output = {
        "stage": "05_http_optimization",
        "measurement": "localhost HTTP, separate Uvicorn worker per mode; 1 worker, persistent client",
        "latency_scope": "client POST to parsed HTTP body; includes JSON, network loopback, PIT, predict, audit",
        "handler_scope": "server handler start to completed audit; excludes HTTP serialization",
        "percentile_method": "numpy percentile linear on per-request elapsed milliseconds",
        "seed": 2026,
        "dataset": "replay inputs only, no use of replay labels for optimization",
        "feature_order": ["distance_km", "warehouse_load", "rain_forecast"],
        "threads": 1,
        "warmup_requests_per_mode": warmup_count,
        "model_artifact_bytes": (ROOT / "artifacts/model.joblib").stat().st_size,
        "offline_quality_unchanged": {
            "test_mae_hours": earlier["candidate_test"]["mae_hours"],
            "test_orders": earlier["candidate_test"]["n"],
            "model_version": earlier["model_version"],
            "max_http_parity_absolute_error": largest_parity_error,
            "equivalence_tolerance": 1e-12,
            "result": "same frozen model and test MAE, equivalent fast path within tolerance",
        },
        "system": {
            "platform": platform.platform(), "machine": platform.machine(),
            "python": platform.python_version(),
            "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__,
            "httpx": httpx.__version__,
        },
        "modes": details,
        "caveat": "Local CPU/loopback synthetic load; not a production network or p95 under peak conditions",
    }
    return output, pd.DataFrame(raw_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare delivery baseline vs fast on localhost HTTP")
    parser.add_argument("--out", type=Path, default=ROOT / "reports/performance.json")
    parser.add_argument("--raw", type=Path, default=ROOT / "reports/latency_raw.csv")
    parser.add_argument("--steady-count", type=int, default=240)
    parser.add_argument("--burst-count", type=int, default=80)
    parser.add_argument("--warmup-count", type=int, default=20)
    parser.add_argument("--steady-rps", type=float, default=20.0)
    parser.add_argument("--burst-rps", type=float, default=40.0)
    args = parser.parse_args()
    result, frame = benchmark(
        steady_count=args.steady_count, burst_count=args.burst_count,
        warmup_count=args.warmup_count, steady_rps=args.steady_rps,
        burst_rps=args.burst_rps,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.raw.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    frame.to_csv(args.raw, index=False, float_format="%.12g")
    summary = {
        "report": str(args.out), "raw": str(args.raw),
        "model_bytes": result["model_artifact_bytes"],
        "steady_rps": args.steady_rps, "burst_rps": args.burst_rps,
        "modes": {
            name: {
                phase: {
                    "median_ms": round(data[phase]["latency_ms"]["median_ms"], 3),
                    "p95_ms": round(data[phase]["latency_ms"]["p95_ms"], 3),
                    "completed_rps": round(data[phase]["completed_rps"], 2),
                }
                for phase in ("steady", "burst")
            } for name, data in result["modes"].items()
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
