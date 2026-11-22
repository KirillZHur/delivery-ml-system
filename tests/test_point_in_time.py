import numpy as np
import pandas as pd
import pytest

from data.generate import make_dataset
from src.features import model_matrix, point_in_time_join


def sample_orders():
    return pd.DataFrame([
        {"order_id": "1", "ordered_at": "2026-01-01T10:00:00Z", "distance_km": 100.0},
    ])


def sample_events():
    return pd.DataFrame([
        {"event_id": "w-old", "feature": "warehouse_load", "event_at": "2026-01-01T08:00:00Z",
         "ingested_at": "2026-01-01T08:05:00Z", "value": 0.2},
        {"event_id": "w-new", "feature": "warehouse_load", "event_at": "2026-01-01T09:00:00Z",
         "ingested_at": "2026-01-01T10:05:00Z", "value": 0.8},
        {"event_id": "rain", "feature": "rain_forecast", "event_at": "2026-01-01T07:00:00Z",
         "ingested_at": "2026-01-01T07:05:00Z", "value": 1.0},
    ])


def test_late_event_not_used_for_historical_order():
    result = point_in_time_join(sample_orders(), sample_events())
    assert result.loc[0, "warehouse_event_id"] == "w-old"
    assert result.loc[0, "warehouse_load"] == 0.2
    assert result.loc[0, "rain_forecast"] == 1


def test_out_of_order_arrivals_do_not_override_newer_event():
    events = sample_events().copy()
    events.loc[events["event_id"].eq("w-new"), "ingested_at"] = "2026-01-01T09:05:00Z"
    events.loc[events["event_id"].eq("w-old"), "ingested_at"] = "2026-01-01T09:55:00Z"
    result = point_in_time_join(sample_orders(), events)
    assert result.loc[0, "warehouse_event_id"] == "w-new"
    assert result.loc[0, "warehouse_load"] == 0.8


def test_observation_at_exact_decision_time_is_available():
    events = sample_events()
    events.loc[events["event_id"].eq("w-new"), "ingested_at"] = "2026-01-01T10:00:00Z"
    result = point_in_time_join(sample_orders(), events)
    assert result.loc[0, "warehouse_event_id"] == "w-new"


def test_missing_event_fails_explicitly():
    only_warehouse = sample_events().query("feature == 'warehouse_load'")
    with pytest.raises(ValueError, match="No eligible"):
        point_in_time_join(sample_orders(), only_warehouse)


def test_event_cannot_be_ingested_before_it_happens():
    events = sample_events()
    events.loc[events["event_id"].eq("w-new"), "ingested_at"] = "2026-01-01T06:00:00Z"
    with pytest.raises(ValueError, match="before its event_at"):
        point_in_time_join(sample_orders(), events)


def test_duplicate_ids_fail():
    doubled = pd.concat([sample_orders(), sample_orders()], ignore_index=True)
    with pytest.raises(ValueError, match="unique"):
        point_in_time_join(doubled, sample_events())


def test_feature_ranges_validated():
    for value in (float("nan"), float("inf"), -1.0):
        frame = pd.DataFrame([{"distance_km": value, "warehouse_load": 0.3, "rain_forecast": 0}])
        with pytest.raises(ValueError):
            model_matrix(frame)


def test_generated_events_never_from_future(tiny_config):
    _, _, features = make_dataset(tiny_config)
    t = pd.to_datetime(features["ordered_at"], utc=True)
    for source in ("warehouse", "weather"):
        assert (pd.to_datetime(features[f"{source}_event_at"], utc=True) <= t).all()
        assert (pd.to_datetime(features[f"{source}_ingested_at"], utc=True) <= t).all()
