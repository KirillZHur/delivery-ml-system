"""Release contract, reproducible observations and safe rollback on real artifacts."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from data.generate import ROOT
from src.replay import run_all, run_scenario, simulate_window, small_window_check, verify_http_versions
from src.rollout import (
    RolloutController, VersionHealth, Window, create_release_bundles,
    nearest_rank_p95, route_user, validate_release_bundles,
)


@pytest.fixture(scope="module")
def rollout_config():
    return json.loads((ROOT / "configs/rollout.json").read_text(encoding="utf-8"))


@pytest.fixture
def versions(tmp_path):
    release = tmp_path / "versions"
    create_release_bundles(ROOT / "artifacts", release)
    return release


def good(requests=1200, p95=40, errors=2, degraded=1, *, shadow=False):
    return VersionHealth(requests, errors, degraded, p95, shadow)


def test_routing_is_repeatable_sticky_monotone_and_validates_input():
    assert route_user("u17", 0) == "v1"
    assert route_user("u17", 1) == "v2"
    counts = [sum(route_user(f"u{i}", share) == "v2" for i in range(20_000)) for share in (0.05, 0.25, 1)]
    assert abs(counts[0] / 20000 - 0.05) < 0.01
    assert abs(counts[1] / 20000 - 0.25) < 0.01
    assert counts[0] < counts[1] < counts[2] == 20000
    for i in range(200):
        if route_user(f"u{i}", 0.05) == "v2":
            assert route_user(f"u{i}", 0.25) == "v2"
    for user, share in (("", .5), ("u", -1), ("u", 1.1), ("u", float("nan")), ("u", True)):
        with pytest.raises(ValueError):
            route_user(user, share)


def test_exact_p95_nearest_rank_includes_failures():
    assert nearest_rank_p95([]) is None
    assert nearest_rank_p95(list(range(1, 101))) == 95
    assert nearest_rank_p95([40] * 94 + [150] * 6) == 150
    with pytest.raises(ValueError):
        nearest_rank_p95([1, float("nan")])
    with pytest.raises(ValueError):
        nearest_rank_p95([-1])


def test_release_artifacts_are_consistent_and_source_unmodified(versions):
    loaded = validate_release_bundles(versions)
    assert loaded["v1"]["metadata"]["inference_mode"] == "baseline"
    assert loaded["v2"]["metadata"]["inference_mode"] == "fast"
    assert loaded["v1"]["manifest"]["model_sha256"] == loaded["v2"]["manifest"]["model_sha256"]
    assert (ROOT / "artifacts/model.joblib").read_bytes() == (versions / "v1/model.joblib").read_bytes()
    with pytest.raises(FileExistsError):
        create_release_bundles(ROOT / "artifacts", versions)


def test_corrupted_old_or_new_bundle_prevents_release(versions):
    with (versions / "v1/model.joblib").open("ab") as fd:
        fd.write(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        validate_release_bundles(versions)


def test_incompatible_feature_contract_prevents_switch(versions):
    path = versions / "v2/manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["feature_order"] = list(reversed(manifest["feature_order"]))
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="feature contract|Feature contract|mismatch|checksum"):
        validate_release_bundles(versions)


def test_real_fastapi_versions_are_ready_and_predictions_have_parity(versions):
    result = verify_http_versions(versions, ROOT / "data/generated/features.csv",
                                   ROOT / "data/generated/events.csv", n=5)
    assert result["ready_v1"] and result["ready_v2"]
    assert result["max_abs_prediction_error_hours"] < 1e-12
    assert not (versions / "http_v1.jsonl").exists()


def test_shadow_does_not_serve_candidate_to_customers(rollout_config):
    controller = RolloutController(rollout_config)
    w, records = simulate_window(controller, scenario="shadow", window_id=1)
    assert w.share == 0
    assert w.stable.requests == w.candidate.requests == w.unique_order_ids
    assert w.candidate.shadow and not w.stable.shadow
    assert {x["actual_version"] for x in records} == {"v1"}
    assert {x["shadow_version"] for x in records} == {"v2"}
    event = controller.observe(w)
    assert event["action"] == "advance"
    assert controller.active_share == .05


def test_full_traffic_still_has_shadow_stable_control(rollout_config):
    controller = RolloutController(rollout_config)
    for i in range(1, 6):
        w, _ = simulate_window(controller, scenario="healthy", window_id=i)
        controller.observe(w)
    assert controller.active_share == 1
    w, rows = simulate_window(controller, scenario="healthy", window_id=6)
    assert w.candidate.requests == w.stable.requests == 6000
    assert w.stable.shadow and not w.candidate.shadow
    assert {r["actual_version"] for r in rows} == {"v2"}
    assert {r["shadow_version"] for r in rows} == {"v1"}


def test_healthy_goes_all_levels_with_2_stable_windows(rollout_config):
    result = run_scenario(rollout_config, "healthy")
    assert result["final_status"] == "promoted"
    assert result["final_share"] == 1
    assert result["windows_seen"] == 7
    assert [e["share_after"] for e in result["events"] if e["action"] == "advance"] == [.05, .25, 1.]
    assert result["events"][-1]["action"] == "complete"


def test_common_feature_outage_holds_candidate_and_then_recovers(rollout_config):
    result = run_scenario(rollout_config, "shared_outage")
    assert result["final_status"] == "promoted"
    assert result["windows_seen"] == 9
    outage = next(event for event in result["events"] if event["injected_event"] == "shared_feature_outage")
    assert outage["action"] == "hold"
    assert outage["reasons"] == ["shared_feature_outage"]
    assert outage["share_before"] == outage["share_after"] == 0.05
    assert outage["stable"]["degraded_rate"] == outage["candidate"]["degraded_rate"] == 1


def test_candidate_latency_causes_immediate_rollback_and_new_requests_use_v1(rollout_config):
    result = run_scenario(rollout_config, "candidate_latency")
    assert result["final_status"] == "rolled_back"
    assert result["final_share"] == 0
    last = result["events"][-1]
    assert last["reasons"] == ["candidate_p95_latency"]
    assert last["candidate"]["p95_ms"] > 100
    assert result["recovery"]["new_assigned_version"] == "v1"
    assert result["recovery"]["inflight_completed_after_rollback_version"] == "v2"
    assert result["recovery"]["inflight_admitted_at_ms"] < result["recovery"]["rollback_switch_at_ms"]
    assert result["recovery"]["inflight_completed_at_ms"] > result["recovery"]["rollback_switch_at_ms"]
    candidate_samples = [x for x in result["sampled_requests"] if x["window_id"] == last["window_id"] and x["actual_version"] == "v2"]
    assert candidate_samples
    assert all(x["actual_version"] == "v2" for x in candidate_samples)


def test_two_consecutive_error_windows_before_rollback(rollout_config):
    result = run_scenario(rollout_config, "candidate_errors")
    assert result["final_status"] == "rolled_back"
    first = result["events"][-2]
    last = result["events"][-1]
    assert first["reasons"] == ["candidate_errors_wait_confirmation"]
    assert first["action"] == "hold" and first["error_streak"] == 1
    assert last["reasons"] == ["candidate_technical_errors"]
    assert last["action"] == "rollback" and last["error_streak"] == 2
    assert first["candidate"]["requests"] >= 1000
    assert last["candidate"]["requests"] >= 1000


def test_canary_only_degraded_causes_rollback(rollout_config):
    result = run_scenario(rollout_config, "candidate_degraded")
    assert result["final_share"] == 0
    assert result["events"][-1]["reasons"] == ["candidate_degraded_rate"]


def test_small_window_never_promotes(rollout_config):
    result = small_window_check(rollout_config)
    assert result["action"] == "hold" and result["share_after"] == .05
    assert result["reasons"] == ["insufficient_data"]
    assert result["candidate_requests"] < 200


def test_invalid_counts_shares_and_time_order_are_rejected(rollout_config):
    with pytest.raises(ValueError):
        VersionHealth(10, 11, 0, 40)
    with pytest.raises(ValueError):
        VersionHealth(0, 0, 0, 0)
    controller = RolloutController(rollout_config)
    w, _ = simulate_window(controller, scenario="hello", window_id=1)
    controller.observe(w)
    with pytest.raises(ValueError, match="strictly increasing"):
        controller.observe(w)
    w = Window(2, 0.0, good(), good(), 1200)
    with pytest.raises(ValueError, match="share"):
        controller.observe(w)


def test_operational_equality_does_not_trigger_rollback(rollout_config):
    controller = RolloutController(rollout_config)
    controller.status, controller.active_share = "canary", 0.25
    for day in range(1, 3):
        w = Window(day, .25, good(900, 100, 9, 0), good(300, 100, 3, 0), 1200)
        event = controller.observe(w)
        assert event["action"] != "rollback"
    assert controller.status == "canary"


def test_rollout_configuration_and_error_streak_reset(rollout_config):
    cfg = copy.deepcopy(rollout_config)
    cfg["canary_shares"] = [.25, .05, 1]
    with pytest.raises(ValueError):
        RolloutController(cfg)
    controller = RolloutController(rollout_config, status="canary", active_share=.25)
    for day, errors in enumerate((20, 0, 20, 20), start=1):
        w = Window(day, .25, good(4500, 40, 4, 41), good(1500, 40, errors, 36), 6000)
        decision = controller.observe(w)
    assert decision["action"] == "rollback"
    assert [row["error_streak"] for row in controller.events] == [1, 0, 1, 2]


def test_after_terminal_controller_cannot_inadvertently_promote(rollout_config):
    controller = RolloutController(rollout_config, status="rolled_back")
    w = Window(1, 0, good(), good(shadow=True), 1200)
    with pytest.raises(ValueError, match="Terminal"):
        controller.observe(w)


def test_all_scenario_reports_reproducible_and_have_no_hidden_labels(tmp_path):
    kwargs = {"config_path": ROOT / "configs/rollout.json", "source_dir": ROOT / "artifacts",
              "release_dir": tmp_path / "release", "features_path": ROOT / "data/generated/features.csv",
              "events_path": ROOT / "data/generated/events.csv", "out_dir": tmp_path / "reports"}
    a = run_all(**kwargs)
    originals = {name: (kwargs["out_dir"] / name).read_bytes() for name in (
        "rollout_summary.json", "rollout_windows.csv", "rollout_events.jsonl", "rollout_request_samples.jsonl"
    )}
    b = run_all(**kwargs)
    assert a == b
    assert a["rollback_http"]["status"] == "verified"
    assert a["rollback_http"]["healthy_model_response"] is True
    for name, original in originals.items():
        current = (kwargs["out_dir"] / name).read_bytes()
        assert current == original
    assert all(key not in str(originals["rollout_request_samples.jsonl"]) for key in (
        "warehouse_load", "rain_forecast", "delivery_hours"))
