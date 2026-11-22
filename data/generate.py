"""Deterministic synthetic delivery data with honest feature/label availability.

Run from repository root:
    python -m data.generate --config configs/data.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import MODEL_FEATURES, point_in_time_join
from src.splits import closed_split

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path | str) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    periods = config["periods"]
    if not periods or len({item["name"] for item in periods}) != len(periods):
        raise ValueError("Period names must be nonempty and distinct")
    previous_end = None
    for period in periods:
        start = pd.Timestamp(period["start"])
        end = pd.Timestamp(period["end"])
        if start.tzinfo is None or end.tzinfo is None or start >= end:
            raise ValueError("All periods need valid timezone-aware [start, end) bounds")
        if previous_end is not None and start < previous_end:
            raise ValueError("Periods must not overlap")
        if int(period["orders"]) < 1:
            raise ValueError("Every period needs at least one order")
        previous_end = end
    return config


def _iso(ts: pd.Timestamp) -> str:
    """Canonical RFC 3339 UTC timestamp, round-trippable via pandas."""
    return ts.isoformat().replace("+00:00", "Z")


def _generate_events(config: dict, rng: np.random.Generator) -> pd.DataFrame:
    first = pd.Timestamp(config["periods"][0]["start"]) - pd.Timedelta(days=1)
    last = pd.Timestamp(config["periods"][-1]["end"])
    rows = []
    event_num = 0

    sources = (
        ("warehouse_load", config["warehouse_event_frequency_hours"]),
        ("rain_forecast", config["weather_event_frequency_hours"]),
    )
    for feature, hours in sources:
        if int(hours) < 1:
            raise ValueError("Event frequency must be positive")
        for event_at in pd.date_range(first, last, freq=f"{hours}h", inclusive="left"):
            if feature == "warehouse_load":
                a, b = config["warehouse_load_beta"]
                value = float(rng.beta(a, b))
            else:
                value = float(rng.random() < config["rain_probability"])

            if rng.random() < config["late_event_probability"]:
                lag = int(rng.integers(
                    config["event_delay_late_min_minutes"],
                    config["event_delay_late_max_minutes"] + 1,
                ))
            else:
                lag = int(rng.integers(0, config["event_delay_regular_max_minutes"] + 1))
            event_num += 1
            rows.append({
                "event_id": f"e{event_num:07d}",
                "feature": feature,
                "event_at": _iso(event_at),
                "ingested_at": _iso(event_at + pd.Timedelta(minutes=lag)),
                "value": value,
            })
    return pd.DataFrame.from_records(rows)


def _generate_orders(config: dict, rng: np.random.Generator) -> pd.DataFrame:
    records = []
    offset = 0
    for period in config["periods"]:
        start = pd.Timestamp(period["start"])
        end = pd.Timestamp(period["end"])
        n = int(period["orders"])
        seconds = int((end - start).total_seconds())
        # Sampling within a calendar period, then sorting, keeps the periods
        # disjoint without relying on a random train/test split.
        times = np.sort(rng.integers(0, seconds, size=n))
        distances = rng.uniform(0.0, config["distance_max_km"], size=n)
        for index in range(n):
            offset += 1
            records.append({
                "order_id": f"o{offset:08d}",
                "ordered_at": _iso(start + pd.Timedelta(seconds=int(times[index]))),
                "distance_km": float(distances[index]),
                "period": period["name"],
            })
    return pd.DataFrame.from_records(records)


def delivery_hours_before_noise(distance_km: np.ndarray, load: np.ndarray, rain: np.ndarray) -> np.ndarray:
    """Synthetic conditional mean in hours: 12 + 0.04d + 18load + 6rain."""
    return 12.0 + 0.04 * distance_km + 18.0 * load + 6.0 * rain


def make_dataset(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate orders, event log and full PIT audit table (without saving)."""
    rng = np.random.default_rng(int(config["seed"]))
    events = _generate_events(config, rng)
    orders = _generate_orders(config, rng)
    assembled = point_in_time_join(orders, events)
    noise = rng.normal(0.0, config["delivery_noise_std_hours"], size=len(orders))
    delivery = np.clip(
        delivery_hours_before_noise(
            assembled["distance_km"].to_numpy(),
            assembled["warehouse_load"].to_numpy(),
            assembled["rain_forecast"].to_numpy(),
        ) + noise,
        1.0,
        168.0,
    )
    orders = orders.copy()
    orders["delivery_hours"] = delivery
    ordered_times = pd.to_datetime(orders["ordered_at"], utc=True)
    orders["delivered_at"] = (
        ordered_times + pd.to_timedelta(delivery, unit="h")
    ).map(_iso)
    orders["label_available_at"] = (
        ordered_times
        + pd.Timedelta(days=int(config["label_window_days"]))
        + pd.Timedelta(hours=int(config["label_delivery_delay_hours"]))
    ).map(_iso)

    # Order records contain the future outcome, but the separately returned
    # feature matrix is assembled only from contemporaneously available events.
    assembled = point_in_time_join(orders, events)
    assert (pd.to_datetime(orders["delivered_at"], utc=True)
            <= pd.to_datetime(orders["label_available_at"], utc=True)).all()
    return orders, events, assembled


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _code_sha256() -> str:
    digest = hashlib.sha256()
    for name in ("configs/data.json", "data/generate.py", "src/features.py", "src/splits.py"):
        digest.update(name.encode("utf-8"))
        digest.update((ROOT / name).read_bytes())
    return digest.hexdigest()


def save_dataset(config: dict, out_dir: Path, report_path: Path) -> dict:
    """Produce reproducible CSVs and a machine-checkable summary."""
    orders, events, features = make_dataset(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        "orders": out_dir / "orders.csv",
        "events": out_dir / "events.csv",
        "features": out_dir / "features.csv",
    }
    for frame, key in ((orders, "orders"), (events, "events"), (features, "features")):
        frame.to_csv(paths[key], index=False, float_format="%.17g", lineterminator="\n")

    ready = {}
    for p in config["periods"]:
        subset = orders.loc[orders["period"] == p["name"]]
        n_ready = len(closed_split(subset, start=p["start"], end=p["end"], as_of=p["end"]))
        ready[p["name"]] = {
            "period_start": p["start"],
            "period_end": p["end"],
            "snapshot_as_of": p["end"],
            "orders_total": int(len(subset)),
            "labels_ready": int(n_ready),
            "labels_immature": int(len(subset) - n_ready),
        }

    future_used = sum(
        (
            pd.to_datetime(features[f"{prefix}_event_at"], utc=True)
            > pd.to_datetime(features["ordered_at"], utc=True)
        ).sum()
        + (
            pd.to_datetime(features[f"{prefix}_ingested_at"], utc=True)
            > pd.to_datetime(features["ordered_at"], utc=True)
        ).sum()
        for prefix in ("warehouse", "weather")
    )
    if future_used:
        raise AssertionError("Point-in-time feature leakage detected")

    summary = {
        "phase": "01_data_and_point_in_time",
        "seed": int(config["seed"]),
        "feature_order": list(MODEL_FEATURES),
        "orders": int(len(orders)),
        "events": int(len(events)),
        "label_window_days": int(config["label_window_days"]),
        "label_delay_hours": int(config["label_delivery_delay_hours"]),
        "point_in_time_future_events_used": int(future_used),
        "mean_delivery_hours": float(orders["delivery_hours"].mean()),
        "split_readiness": ready,
        "sha256": {key: _sha256(path) for key, path in paths.items()},
        "code_sha256": _code_sha256(),
    }
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic delivery data and PIT audit")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/data.json")
    parser.add_argument("--out", type=Path, default=ROOT / "data/generated")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/data_summary.json")
    args = parser.parse_args()
    result = save_dataset(load_config(args.config), args.out, args.report)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
