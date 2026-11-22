"""Stage 3: paired uncertainty, prediction intervals and closed holdout tests."""

import json

import numpy as np
import pandas as pd
import pytest

from data.generate import ROOT
from src.bundle import load_bundle
from src.evaluate import (
    evaluate_final,
    evaluate_predictions,
    interval_diagnostics,
    interval_half_width,
    paired_day_bootstrap_mae,
    prediction_intervals,
    select_slice,
)
from src.splits import closed_split


def test_paired_bootstrap_identical_forecasts_are_exactly_zero():
    y = np.array([1, 2, 3, 4, 5, 6], dtype=float)
    same = np.array([0, 2, 2, 4, 4, 7], dtype=float)
    days = np.array(["mon", "mon", "tue", "tue", "wed", "wed"])
    result = paired_day_bootstrap_mae(y, same, same, days, n_boot=199, seed=17)
    assert result["estimate_hours"] == 0
    assert result["ci_lower_hours"] == 0
    assert result["ci_upper_hours"] == 0
    assert result["n_blocks"] == 3


def test_paired_bootstrap_constant_row_improvement_has_degenerate_interval():
    target = np.zeros(11)
    baseline = np.full(11, 3.0)
    candidate = np.full(11, 2.0)
    groups = np.array(["day1"] * 2 + ["day2"] * 3 + ["day3"] * 6)
    result = paired_day_bootstrap_mae(target, baseline, candidate, groups, n_boot=400)
    assert result["estimate_hours"] == pytest.approx(1.0)
    assert result["ci_lower_hours"] == pytest.approx(1.0)
    assert result["ci_upper_hours"] == pytest.approx(1.0)


def test_day_resampling_preserves_order_weight_and_is_deterministic():
    y = np.zeros(7)
    baseline = np.array([9, 9, 1, 1, 1, 1, 1], dtype=float)
    candidate = np.array([1, 1, 0, 0, 0, 0, 0], dtype=float)
    days = np.array(["a", "a", "b", "b", "b", "b", "b"])
    a = paired_day_bootstrap_mae(y, baseline, candidate, days, n_boot=321, seed=2027)
    b = paired_day_bootstrap_mae(y, baseline, candidate, days, n_boot=321, seed=2027)
    assert a == b
    assert a["estimate_hours"] == pytest.approx((8 * 2 + 1 * 5) / 7)
    assert a["ci_lower_hours"] <= a["estimate_hours"] <= a["ci_upper_hours"]


@pytest.mark.parametrize("invalid", [0.0, -1.0, 1.0, float("nan")])
def test_bad_interval_coverage_is_rejected(invalid):
    with pytest.raises(ValueError):
        interval_half_width(np.arange(10) / 10, coverage=invalid)


def test_bootstrap_rejects_bad_inputs():
    with pytest.raises(ValueError, match="two independent"):
        paired_day_bootstrap_mae([0, 1], [0, 1], [1, 0], ["a", "a"])
    with pytest.raises(ValueError, match="aligned"):
        paired_day_bootstrap_mae([0, 1], [0], [0, 1], ["a", "b"])
    with pytest.raises(ValueError, match="finite"):
        paired_day_bootstrap_mae([0, np.nan], [0, 1], [0, 1], ["a", "b"])


def test_empirical_interval_rank_and_boundary_clipping():
    # n=9, rank=ceil(10*0.8)=8: eighth smallest residual, not an interpolated one.
    q = interval_half_width([9, 8, 4, 1, 7, 3, 5, 2, 6], coverage=0.8)
    assert q == 8.0
    lower, upper = prediction_intervals(np.array([2, 167]), q)
    np.testing.assert_array_equal(lower, [1, 159])
    np.testing.assert_array_equal(upper, [10, 168])
    summary = interval_diagnostics(np.array([1, 170]), lower, upper)
    assert summary["empirical_coverage"] == 0.5
    assert summary["mean_width_hours"] == pytest.approx((9 + 9) / 2)


def test_slice_cannot_use_labels_or_future_values():
    frame = pd.DataFrame({
        "distance_km": [100, 700, 850], "warehouse_load": [.1, .8, .6],
        "rain_forecast": [0, 1, 1], "delivery_hours": [1, 2, 3],
    })
    spec = {"name": "long_and_rain", "conditions": [
        {"feature": "distance_km", "op": "ge", "value": 700},
        {"feature": "rain_forecast", "op": "eq", "value": 1},
    ]}
    np.testing.assert_array_equal(select_slice(frame, spec), [False, True, True])
    with pytest.raises(ValueError, match="undeclared"):
        select_slice(frame, {"conditions": [{"feature": "delivery_hours", "op": "ge", "value": 2}]})


def test_full_final_evaluation_writes_reproducible_machine_results(tmp_path):
    kwargs = dict(
        features_path=ROOT / "data/generated/features.csv",
        data_config_path=ROOT / "configs/data.json",
        evaluation_config_path=ROOT / "configs/evaluation.json",
        artifact_dir=ROOT / "artifacts",
    )
    first = evaluate_final(**kwargs, output_dir=tmp_path / "run1")
    second = evaluate_final(**kwargs, output_dir=tmp_path / "run2")
    assert first == second
    assert first["selected_model"] == "linear"
    assert first["split_counts"] == {"train_mature": 9259, "valid_mature": 2990, "test_mature": 3092}
    assert first["split_readiness"]["test"] == {"total": 4000, "mature": 3092, "immature": 908}
    assert first["baseline_test"]["mae_hours"] > first["candidate_test"]["mae_hours"]
    assert first["paired_mae_improvement"]["ci_lower_hours"] > 0
    assert all(slice_["status"] == "evaluated" for slice_ in first["mandatory_slices"])
    assert first["prediction_interval"]["test"]["empirical_coverage"] >= 0.88
    table = pd.read_csv(tmp_path / "run1/evaluation_predictions.csv")
    assert len(table) == 3092
    assert table["order_id"].nunique() == 3092
    assert (table["interval_lower_hours"] <= table["interval_upper_hours"]).all()
    assert np.mean(table["within_interval"]) == pytest.approx(
        first["prediction_interval"]["test"]["empirical_coverage"]
    )
    assert len(pd.read_csv(tmp_path / "run1/evaluation_slices.csv")) == 4
    assert json.loads((tmp_path / "run1/evaluation_metrics.json").read_text()) == first
    # No human-facing reports or README are mutated by this step.


def test_model_and_data_integrity_checked_before_holdout(tmp_path):
    raw = ROOT / "data/generated/features.csv"
    modified = tmp_path / "features.csv"
    content = raw.read_bytes()
    modified.write_bytes(content + b"\n")
    with pytest.raises(ValueError, match="checksum"):
        evaluate_final(
            features_path=modified,
            data_config_path=ROOT / "configs/data.json",
            evaluation_config_path=ROOT / "configs/evaluation.json",
            artifact_dir=ROOT / "artifacts",
            output_dir=tmp_path / "reports",
        )


def test_test_labels_do_not_change_calibration_or_baseline_training():
    cfg = json.loads((ROOT / "configs/data.json").read_text())
    config = json.loads((ROOT / "configs/evaluation.json").read_text())
    periods = {p["name"]: p for p in cfg["periods"]}
    frame = pd.read_csv(ROOT / "data/generated/features.csv")
    samples = {}
    for name in ("train", "valid", "test"):
        p = periods[name]
        samples[name] = closed_split(
            frame.loc[frame.period == name], start=p["start"], end=p["end"], as_of=p["end"]
        )
    model, manifest = load_bundle(ROOT / "artifacts")
    original, _, _ = evaluate_predictions(
        train=samples["train"], valid=samples["valid"], test=samples["test"],
        candidate=model, manifest=manifest, config=config,
    )
    changed = samples["test"].copy()
    changed["delivery_hours"] = np.clip(changed.delivery_hours + 2.0, 1, 168)
    new, _, _ = evaluate_predictions(
        train=samples["train"], valid=samples["valid"], test=changed,
        candidate=model, manifest=manifest, config=config,
    )
    assert new["prediction_interval"]["half_width_hours"] == original["prediction_interval"]["half_width_hours"]
    assert new["model_version"] == original["model_version"]
    assert new["baseline_test"]["mae_hours"] != original["baseline_test"]["mae_hours"]
    assert new["candidate_test"]["mae_hours"] != original["candidate_test"]["mae_hours"]


def test_small_mandatory_slice_is_unknown_not_falsely_accepted():
    cfg = json.loads((ROOT / "configs/evaluation.json").read_text())
    cfg["mandatory_slices"] = [{"name": "empty", "conditions": [
        {"feature": "distance_km", "op": "ge", "value": 999.9}
    ]}]
    periods = json.loads((ROOT / "configs/data.json").read_text())["periods"]
    p = {p["name"]: p for p in periods}
    frame = pd.read_csv(ROOT / "data/generated/features.csv")
    samples = {name: closed_split(
        frame.loc[frame.period == name], start=p[name]["start"],
        end=p[name]["end"], as_of=p[name]["end"]
    ) for name in ("train", "valid", "test")}
    model, manifest = load_bundle(ROOT / "artifacts")
    result, _, _ = evaluate_predictions(
        train=samples["train"], valid=samples["valid"], test=samples["test"],
        candidate=model, manifest=manifest, config=cfg,
    )
    entry = result["mandatory_slices"][0]
    assert entry["status"] == "insufficient_data"
    assert entry["candidate_mae_hours"] is None
    assert entry["empirical_interval_coverage"] is None
