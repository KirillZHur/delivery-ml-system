"""Point-in-time feature assembly shared by training and later API serving.

Each event must be known to the service at prediction time: BOTH timestamps
must be <= ordered_at. An event timestamp alone is not sufficient.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MODEL_FEATURES = ("distance_km", "warehouse_load", "rain_forecast")
EVENT_FEATURES = ("warehouse_load", "rain_forecast")


def point_in_time_join(orders: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Add the newest *available* warehouse and weather observations per order.

    Events are processed in ingestion order. As orders advance chronologically,
    the latest event_at among events ingested by then is retained for each source.
    This avoids the common incorrect pattern of joining on event_at and then
    filtering on ingested_at (which can lose a valid older observation).

    The function does not use outcomes (delivery_hours or future timestamps).
    Missing observations are rejected explicitly, never silently interpreted as 0.
    """
    needed_orders = {"order_id", "ordered_at", "distance_km"}
    needed_events = {"event_id", "feature", "event_at", "ingested_at", "value"}
    if not needed_orders.issubset(orders.columns):
        raise ValueError(f"Missing order columns: {sorted(needed_orders - set(orders.columns))}")
    if not needed_events.issubset(events.columns):
        raise ValueError(f"Missing event columns: {sorted(needed_events - set(events.columns))}")
    if orders["order_id"].isna().any() or orders["order_id"].duplicated().any():
        raise ValueError("order_id must be present and unique")
    if events["event_id"].isna().any() or events["event_id"].duplicated().any():
        raise ValueError("event_id must be present and unique")
    if not events["feature"].isin(EVENT_FEATURES).all():
        raise ValueError("Unsupported feature name in event log")

    base = orders.copy()
    event_log = events.copy()
    base["ordered_at"] = pd.to_datetime(base["ordered_at"], utc=True, errors="raise")
    event_log["event_at"] = pd.to_datetime(event_log["event_at"], utc=True, errors="raise")
    event_log["ingested_at"] = pd.to_datetime(event_log["ingested_at"], utc=True, errors="raise")
    if base["ordered_at"].isna().any() or event_log[["event_at", "ingested_at"]].isna().any().any():
        raise ValueError("Null timestamps are not permitted")
    if (event_log["ingested_at"] < event_log["event_at"]).any():
        raise ValueError("An event cannot be ingested before its event_at")
    if not np.isfinite(pd.to_numeric(event_log["value"], errors="coerce")).all():
        raise ValueError("Event values must be finite numbers")

    ordered = base.sort_values(["ordered_at", "order_id"], kind="stable")
    logged = event_log.sort_values(["ingested_at", "event_at", "event_id"], kind="stable")
    available = logged.to_dict(orient="records")
    latest: dict[str, dict] = {}
    pointer = 0
    matches: list[dict] = []

    for row in ordered.itertuples(index=False):
        t = row.ordered_at
        while pointer < len(available) and available[pointer]["ingested_at"] <= t:
            event = available[pointer]
            # The timestamp contract above implies event_at <= ingested_at.
            # Retain the most recent real observation, not the most recent
            # *arrival*, as a delayed older event may arrive after a newer one.
            old = latest.get(event["feature"])
            if old is None or (event["event_at"], event["ingested_at"]) > (
                old["event_at"], old["ingested_at"]
            ):
                latest[event["feature"]] = event
            pointer += 1

        if any(feature not in latest for feature in EVENT_FEATURES):
            raise ValueError(f"No eligible point-in-time features for order {row.order_id}")
        load = latest["warehouse_load"]
        weather = latest["rain_forecast"]
        rain = float(weather["value"])
        if not (0.0 <= float(load["value"]) <= 1.0) or rain not in (0.0, 1.0):
            raise ValueError("Event feature values violate their domains")

        matches.append(
            {
                "order_id": row.order_id,
                "warehouse_load": float(load["value"]),
                "rain_forecast": int(rain),
                "warehouse_event_id": load["event_id"],
                "warehouse_event_at": load["event_at"],
                "warehouse_ingested_at": load["ingested_at"],
                "weather_event_id": weather["event_id"],
                "weather_event_at": weather["event_at"],
                "weather_ingested_at": weather["ingested_at"],
            }
        )

    right = pd.DataFrame.from_records(
        matches,
        columns=[
            "order_id", "warehouse_load", "rain_forecast", "warehouse_event_id",
            "warehouse_event_at", "warehouse_ingested_at", "weather_event_id",
            "weather_event_at", "weather_ingested_at",
        ],
    )
    return base.merge(right, how="left", on="order_id", validate="one_to_one", sort=False)


def model_matrix(frame: pd.DataFrame) -> np.ndarray:
    """Return only three permitted numeric inputs, in their fixed order."""
    missing = set(MODEL_FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing model features: {sorted(missing)}")
    matrix = frame.loc[:, list(MODEL_FEATURES)].to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(matrix).all():
        raise ValueError("Features must be finite")
    if (matrix[:, 0] < 0).any() or (matrix[:, 0] > 1000).any():
        raise ValueError("distance_km outside 0..1000")
    if (matrix[:, 1] < 0).any() or (matrix[:, 1] > 1).any():
        raise ValueError("warehouse_load outside 0..1")
    if not np.isin(matrix[:, 2], (0.0, 1.0)).all():
        raise ValueError("rain_forecast must be binary")
    return matrix
