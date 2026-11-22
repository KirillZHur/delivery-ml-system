"""Point-in-time lookup of server-owned delivery features.

Events are indexed once, at service startup. A historical observation may only
be selected when both its event time and its ingestion time precede the order.
Late arrivals of older observations cannot overwrite newer observed events.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import EVENT_FEATURES


class FeatureMissingError(Exception):
    """An eligible observation has not arrived by the prediction time."""


class FeatureStaleError(Exception):
    """The newest eligible observation has exceeded the freshness contract."""


class FeatureUnavailableError(Exception):
    """The online feature source is temporarily unavailable."""


@dataclass(frozen=True)
class FeatureObservation:
    feature: str
    value: float
    event_id: str
    event_at: datetime
    ingested_at: datetime


@dataclass(frozen=True)
class OnlineFeatures:
    warehouse_load: float
    rain_forecast: int
    warehouse_event_id: str
    weather_event_id: str


class EventFeatureStore:
    """Index the same versioned event history used by offline PIT assembly.

    The *lookup* method is async so the service can apply a real deadline to
    network-based stores later. For the local CSV implementation it returns
    immediately without blocking on IO (all parsing occurs during startup).
    """

    def __init__(self, events: pd.DataFrame, *, max_age_hours: dict[str, float]):
        expected = {"event_id", "feature", "event_at", "ingested_at", "value"}
        if not expected.issubset(events.columns):
            raise ValueError(f"Missing event columns: {sorted(expected - set(events.columns))}")
        if events["event_id"].isna().any() or events["event_id"].duplicated().any():
            raise ValueError("Event identifiers must be unique and nonnull")
        if events.empty or not events["feature"].isin(EVENT_FEATURES).all():
            raise ValueError("Feature event history is empty or has unsupported features")
        if set(max_age_hours) != set(EVENT_FEATURES):
            raise ValueError("A maximum observation age is needed for every feature")
        if any(not np.isfinite(v) or v <= 0 for v in max_age_hours.values()):
            raise ValueError("Maximum observation ages must be finite and positive")

        prepared = events.copy()
        for col in ("event_at", "ingested_at"):
            prepared[col] = pd.to_datetime(prepared[col], utc=True, errors="raise")
        if prepared[["event_at", "ingested_at"]].isna().any().any():
            raise ValueError("Missing event timestamps")
        if (prepared["event_at"] > prepared["ingested_at"]).any():
            raise ValueError("An observation cannot arrive before it happens")
        prepared["value"] = pd.to_numeric(prepared["value"], errors="coerce")
        if not np.isfinite(prepared["value"]).all():
            raise ValueError("Feature values must be finite")
        load_values = prepared.loc[prepared.feature == "warehouse_load", "value"]
        rain_values = prepared.loc[prepared.feature == "rain_forecast", "value"]
        if not load_values.between(0, 1).all() or not rain_values.isin((0, 1)).all():
            raise ValueError("Feature values violate their domains")

        self._max_age_seconds = {k: float(v) * 3600.0 for k, v in max_age_hours.items()}
        self._index: dict[str, tuple[list[datetime], list[FeatureObservation]]] = {}
        for feature in EVENT_FEATURES:
            matching = prepared.loc[prepared.feature == feature]
            if matching.empty:
                raise ValueError(f"No observations for {feature}")
            ordered = matching.sort_values(
                ["ingested_at", "event_at", "event_id"], kind="stable"
            )
            ingested_times: list[datetime] = []
            best_at_arrival: list[FeatureObservation] = []
            best: FeatureObservation | None = None
            for row in ordered.itertuples(index=False):
                observation = FeatureObservation(
                    feature=feature,
                    value=float(row.value),
                    event_id=str(row.event_id),
                    event_at=row.event_at.to_pydatetime(),
                    ingested_at=row.ingested_at.to_pydatetime(),
                )
                if best is None or (observation.event_at, observation.ingested_at) > (
                    best.event_at, best.ingested_at
                ):
                    best = observation
                ingested_times.append(observation.ingested_at)
                best_at_arrival.append(best)
            self._index[feature] = ingested_times, best_at_arrival

    @classmethod
    def from_csv(cls, path: Path, *, max_age_hours: dict[str, float]):
        return cls(pd.read_csv(path), max_age_hours=max_age_hours)

    def _at(self, feature: str, ordered_at: datetime) -> FeatureObservation:
        ingestion_times, observed = self._index[feature]
        offset = bisect_right(ingestion_times, ordered_at) - 1
        if offset < 0:
            raise FeatureMissingError(f"No available {feature} observation")
        observation = observed[offset]
        if observation.event_at > ordered_at or observation.ingested_at > ordered_at:
            raise RuntimeError("Point-in-time observation contract violated")
        if (ordered_at - observation.event_at).total_seconds() > self._max_age_seconds[feature]:
            raise FeatureStaleError(f"Stale {feature} observation")
        return observation

    async def fetch(self, ordered_at: datetime) -> OnlineFeatures:
        if ordered_at.tzinfo is None or ordered_at.utcoffset() is None:
            raise ValueError("Order time must have an explicit UTC offset")
        when = ordered_at.astimezone(timezone.utc)
        load = self._at("warehouse_load", when)
        rain = self._at("rain_forecast", when)
        return OnlineFeatures(
            warehouse_load=load.value,
            rain_forecast=int(rain.value),
            warehouse_event_id=load.event_id,
            weather_event_id=rain.event_id,
        )
