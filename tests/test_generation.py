import copy

import numpy as np
import pandas as pd
import pytest

from data.generate import delivery_hours_before_noise, load_config, make_dataset, save_dataset
from src.features import model_matrix
from src.splits import closed_split


def test_manual_delivery_formula():
    distance = np.array([100.0, 100.0])
    warehouse_load = np.array([0.5, 0.5])
    rain = np.array([0, 1])
    assert delivery_hours_before_noise(distance, warehouse_load, rain).tolist() == [25.0, 31.0]


def test_tiny_generation_is_deterministic(tiny_config):
    first = make_dataset(tiny_config)
    second = make_dataset(tiny_config)
    for left, right in zip(first, second):
        pd.testing.assert_frame_equal(left, right)


def test_second_seed_changes_values(tiny_config):
    first, _, _ = make_dataset(tiny_config)
    alternate = copy.deepcopy(tiny_config)
    alternate["seed"] = 2027
    second, _, _ = make_dataset(alternate)
    assert not first["distance_km"].equals(second["distance_km"])
    assert not first["delivery_hours"].equals(second["delivery_hours"])


def test_no_overlapping_periods_or_ids(tiny_config):
    orders, events, _ = make_dataset(tiny_config)
    assert len(orders) == 50
    assert orders["order_id"].is_unique
    assert events["event_id"].is_unique
    starts = pd.to_datetime(orders["ordered_at"], utc=True)
    for period in tiny_config["periods"]:
        mask = orders["period"].eq(period["name"])
        assert (starts[mask] >= pd.Timestamp(period["start"])).all()
        assert (starts[mask] < pd.Timestamp(period["end"])).all()


def test_label_arrives_only_after_seven_days_plus_one_hour(tiny_config):
    orders, _, _ = make_dataset(tiny_config)
    start = pd.to_datetime(orders["ordered_at"], utc=True)
    delivered = pd.to_datetime(orders["delivered_at"], utc=True)
    arrived = pd.to_datetime(orders["label_available_at"], utc=True)
    assert (delivered >= start).all()
    assert (delivered <= arrived).all()
    assert ((arrived - start) == pd.Timedelta(days=7, hours=1)).all()
    assert orders["delivery_hours"].between(1.0, 168.0).all()


def test_feature_matrix_is_only_approved_inputs(tiny_config):
    _, _, features = make_dataset(tiny_config)
    matrix = model_matrix(features)
    assert matrix.shape == (50, 3)
    # The target, timestamps, IDs and period must not enter the model.
    assert np.array_equal(matrix[:, 0], features["distance_km"].to_numpy())
    assert np.array_equal(matrix[:, 1], features["warehouse_load"].to_numpy())
    assert np.array_equal(matrix[:, 2], features["rain_forecast"].to_numpy())
    hacked = features.copy()
    hacked["delivery_hours"] = -1e10
    np.testing.assert_array_equal(model_matrix(hacked), matrix)


def test_split_does_not_use_immature_labels(tiny_config):
    orders, _, _ = make_dataset(tiny_config)
    period = tiny_config["periods"][0]
    closed = closed_split(orders, start=period["start"], end=period["end"], as_of=period["end"])
    assert len(closed) == 0  # The split lasts four days, but labels need seven.


def test_generator_output_and_hashes_reproducible(tmp_path, tiny_config):
    a = tmp_path / "a"
    b = tmp_path / "b"
    ra = save_dataset(tiny_config, a, tmp_path / "r1.json")
    rb = save_dataset(tiny_config, b, tmp_path / "r2.json")
    assert ra["sha256"] == rb["sha256"]
    assert ra["code_sha256"] == rb["code_sha256"]
    assert ra["point_in_time_future_events_used"] == 0
    assert (a / "orders.csv").exists()
    assert (a / "features.csv").exists()
    assert (a / "events.csv").exists()


def test_invalid_overlap_rejected(tmp_path, tiny_config):
    cfg = copy.deepcopy(tiny_config)
    cfg["periods"][1]["start"] = "2026-01-04T00:00:00Z"
    path = tmp_path / "config.json"
    import json
    path.write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="overlap"):
        load_config(path)
