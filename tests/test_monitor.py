"""Stage 7 contracts: statistical alarms vs label maturity and source failures."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from data.generate import ROOT
from src.monitor import (
    bonferroni, categorical_comparison, inject_monitoring_incident,
    mature_cohort_quality, monitor_daily, operational_monitor, psi, run_monitor,
)


def cfg():
    return json.loads((ROOT / "configs/monitor.json").read_text())


def test_psi_identical_arrays_zeros_even_for_repeated_values():
    values = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
    assert psi(values, values) == pytest.approx(0, abs=1e-15)
    assert psi(np.ones(10), np.ones(10)) == pytest.approx(0, abs=1e-15)


def test_psi_shift_increases_and_keeps_out_of_range_values():
    ref = np.arange(1000, dtype=float)
    mild = ref + 100
    strong = ref + 600
    assert 0 < psi(ref, mild) < psi(ref, strong)
    assert psi(ref, np.arange(1000, dtype=float) + 5000) > 0


@pytest.mark.parametrize("ref, cur, bins", [([1, np.nan], [1], 10),
                                           ([], [2], 10), ([1], [2], 1)])
def test_invalid_psi_rejected(ref, cur, bins):
    with pytest.raises(ValueError):
        psi(ref, cur, bins)


def test_categorical_new_category_is_in_union():
    a = np.array(["card"] * 100 + ["wallet"] * 100)
    b = np.array(["card"] * 50 + ["wallet"] * 50 + ["new"] * 100)
    result = categorical_comparison(a, b)
    assert result["test_status"] == "ok"
    assert result["tv"] == pytest.approx(0.5)
    assert result["p_raw"] < 0.01


def test_categorical_rare_expected_count_does_not_falsify_significance():
    ref = np.array(["card"] * 99 + ["rare"])
    cur = np.array(["card"] * 99 + ["rare"])
    result = categorical_comparison(ref, cur)
    assert result["test_status"] == "rare_or_single_category"
    assert result["p_raw"] is None
    assert result["tv"] == 0


def test_bonferroni_corrects_only_present_finite_tests():
    assert bonferroni([0.001, 0.5, 0.9]) == pytest.approx([0.003, 1, 1])
    with pytest.raises(ValueError):
        bonferroni([0.01, float("nan")])


def test_delayed_cohorts_exact_window_boundary_and_early_label_not_ready():
    pred = pd.DataFrame({"order_id": ["a", "b", "c"],
                         "ordered_at": ["2026-04-01T00:00:00Z", "2026-04-01T01:00:00Z", "2026-04-02T00:00:00Z"],
                         "prediction_hours": [10., 20., 30.]})
    labels = pd.DataFrame({"order_id": ["a", "b", "c"],
                           "delivery_hours": [9., 17., 27.],
                           "available_at": ["2026-04-02T00:00:00Z", "2026-04-03T00:00:00Z", "2026-04-03T00:00:00Z"]})
    cohorts, result = mature_cohort_quality(pred, labels, snapshot_as_of="2026-04-08T00:00:00Z")
    assert cohorts.iloc[0].status == "immature"  # second order still open
    assert cohorts.iloc[1].status == "immature"
    assert result["mae_denominator"] == 0
    exact, exact_result = mature_cohort_quality(
        pred.iloc[[0]], labels.iloc[[0]], snapshot_as_of="2026-04-08T00:00:00Z")
    assert exact.iloc[0].status == "ready"
    assert exact_result["mae_denominator"] == 1
    assert exact_result["ready_mae_hours"] == 1


def test_closed_cohort_missing_one_label_becomes_incomplete():
    pred = pd.DataFrame({"order_id": ["a", "b"], "ordered_at": ["2026-04-01T00:00:00Z"] * 2,
                         "prediction_hours": [1., 2.]})
    labels = pd.DataFrame({"order_id": ["a", "b"], "delivery_hours": [1., 4.],
                           "available_at": ["2026-04-08T00:00:00Z", "2026-04-10T00:00:00Z"]})
    cohorts, result = mature_cohort_quality(pred, labels, snapshot_as_of="2026-04-09T00:00:00Z")
    assert cohorts.iloc[0].status == "incomplete"
    assert cohorts.iloc[0].label_coverage == 0.5
    assert cohorts.iloc[0].mae_denominator == 0
    assert result["ready_mae_hours"] is None


def test_ready_cohorts_are_weighted_by_orders():
    pred = pd.DataFrame({"order_id": ["a", "b", "c", "d"],
                         "ordered_at": ["2026-04-01T00:00:00Z"] * 3 + ["2026-04-02T00:00:00Z"],
                         "prediction_hours": [0.] * 4})
    labels = pd.DataFrame({"order_id": ["a", "b", "c", "d"],
                           "delivery_hours": [1., 1., 1., 9.],
                           "available_at": ["2026-04-12T00:00:00Z"] * 4})
    _, result = mature_cohort_quality(pred, labels, snapshot_as_of="2026-04-15T00:00:00Z")
    assert result["mae_denominator"] == 4
    assert result["ready_mae_hours"] == pytest.approx(3.0)  # NOT average(1,9)=5


def test_missing_or_repeated_cohort_ids_rejected():
    pred = pd.DataFrame({"order_id": ["a", "a"], "ordered_at": ["2026-04-01T00:00:00Z"] * 2,
                         "prediction_hours": [0., 0.]})
    labels = pd.DataFrame({"order_id": ["a"], "delivery_hours": [1.],
                           "available_at": ["2026-04-10T00:00:00Z"]})
    with pytest.raises(ValueError, match="unique"):
        mature_cohort_quality(pred, labels, snapshot_as_of="2026-05-01T00:00:00Z")


def test_source_order_duplicates_are_not_independent_pvalue_observations():
    data = pd.read_csv(ROOT / "data/generated/features.csv")
    reference = data.loc[data.period == "train"]
    replay = data.loc[data.period == "replay"]
    daily, alerts = monitor_daily(reference, replay, cfg(), "clean")
    assert not alerts, "Unchanged source event stream should not produce confirmed alerts"
    warehouse = daily[daily.feature == "warehouse_load"]
    assert (warehouse.effective_n < warehouse.n).all()
    assert (warehouse.reference_effective_n < len(reference)).all()
    assert (warehouse.loc[warehouse.day < "2026-04-07", "test_status"] == "not_run").all()


def test_immediate_missing_feature_and_confirmed_shift_on_real_replay():
    frame = pd.read_csv(ROOT / "data/generated/features.csv")
    ref = frame.loc[frame.period == "train"]
    replay = frame.loc[frame.period == "replay"]
    current = inject_monitoring_incident(replay, cfg())
    daily, alerts = monitor_daily(ref, current, cfg(), "incident")
    assert [(a["type"], a["feature"], a["day"]) for a in alerts] == [
        ("data_quality", "warehouse_load", "2026-04-10"),
        ("drift", "distance_km", "2026-04-19"),
    ]
    assert daily[(daily.day == "2026-04-10") & (daily.feature == "warehouse_load")].iloc[0].streak == 0
    assert daily[(daily.day == "2026-04-17") & (daily.feature == "distance_km")].iloc[0].streak == 1
    assert daily[(daily.day == "2026-04-19") & (daily.feature == "distance_km")].iloc[0].streak == 3


def test_missing_calendar_day_resets_series_and_requires_reconfirmation():
    frame = pd.read_csv(ROOT / "data/generated/features.csv")
    reference = frame[frame.period == "train"]
    replay = inject_monitoring_incident(frame[frame.period == "replay"], cfg())
    skipped = replay.loc[~pd.to_datetime(replay.ordered_at, utc=True).dt.day.isin([18])]
    daily, alarms = monitor_daily(reference, skipped, cfg(), "gap")
    # After 17 -> 19, one calendar day is missing, so 19 cannot confirm the shift.
    row = daily[(daily.day == "2026-04-19") & (daily.feature == "distance_km")].iloc[0]
    assert row.streak == 1
    assert row.calendar_gap
    assert not any(a["feature"] == "distance_km" and a["day"] == "2026-04-19" for a in alarms)
    assert any(a["feature"] == "distance_km" and a["day"] == "2026-04-21" for a in alarms)


def test_frozen_feature_triggers_immediate_quality_alarm():
    frame = pd.read_csv(ROOT / "data/generated/features.csv")
    reference = frame[frame.period == "train"]
    replay = frame[frame.period == "replay"].copy()
    first_day = pd.to_datetime(replay.ordered_at, utc=True).dt.day == 4
    replay.loc[first_day, "warehouse_load"] = 0.0
    _, alarms = monitor_daily(reference, replay, cfg(), "frozen")
    assert any(a["type"] == "data_quality" and a["feature"] == "warehouse_load"
               and a["day"] == "2026-04-04" for a in alarms)


def test_operational_monitor_distinguishes_shared_outage_from_candidate_regression():
    windows = pd.read_csv(ROOT / "reports/rollout_windows.csv")
    alarms = operational_monitor(windows)
    assert any(a["scenario"] == "shared_outage" and
               a["type"] == "shared_feature_dependency" for a in alarms)
    assert any(a["scenario"] == "candidate_latency" and a["type"] == "latency_p95" and
               a["version"] == "v2" for a in alarms)
    assert any(a["scenario"] == "candidate_errors" and a["type"] == "error_rate" for a in alarms)
    assert any(a["scenario"] == "candidate_degraded" and a["type"] == "degraded_rate" for a in alarms)


def test_operational_minimum_sample_does_not_claim_healthy_or_bad():
    frame = pd.DataFrame([{"scenario": "small", "window_id": 1,
                           "stable_n": 95, "candidate_n": 5,
                           "stable_errors": 0, "candidate_errors": 5,
                           "stable_degraded": 0, "candidate_degraded": 5,
                           "stable_p95_ms": 50., "candidate_p95_ms": 500.}])
    assert operational_monitor(frame) == []


def test_complete_monitor_reproducible_and_mature_only(tmp_path):
    first = run_monitor(output_dir=tmp_path / "first")
    second = run_monitor(output_dir=tmp_path / "second")
    assert first == second
    for name in ("monitor_daily.csv", "monitor_alerts.jsonl", "monitor_cohorts.csv",
                 "monitor_summary.json", "monitor_predictions.csv", "monitor_labels.csv"):
        assert (tmp_path / "first" / name).read_bytes() == (tmp_path / "second" / name).read_bytes()
    quality = first["mature_quality"]
    assert quality["mature_ready_orders"] == quality["mae_denominator"]
    assert 0 < quality["mae_denominator"] < first["replay_orders"]
    assert quality["incomplete_cohorts"] >= 1
    assert quality["immature_cohorts"] >= 1
    assert first["clean_alerts"] == []
    assert len(first["injected_alerts"]) == 2
    assert first["replay_orders"] == 4000
    daily = pd.read_csv(tmp_path / "first" / "monitor_daily.csv")
    assert len(daily) == 30 * 3 * 2
    assert set(daily.scenario) == {"clean_replay", "injected_incident"}
    rows = pd.read_csv(tmp_path / "first" / "monitor_cohorts.csv")
    assert rows.mae_denominator.sum() == quality["mae_denominator"]
    assert rows.loc[rows.status != "ready", "mae_hours"].isna().all()
    assert not (tmp_path / "first" / "report.md").exists()
