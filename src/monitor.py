"""Stage 7: delivery drift, immediate contract/operational alarms, mature-cohort MAE.

All injected incidents are SYNTHETIC; distribution diagnostics must not be
interpreted as evidence that delivery MAE degraded. Source predictions are
scored from the untouched replay population only after labels mature.

Execute from root: python -m src.monitor
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp

from data.generate import ROOT
from src.bundle import load_bundle, predict_frame
from src.features import MODEL_FEATURES


NUMERIC = ("distance_km", "warehouse_load")
CATEGORICAL = ("rain_forecast",)


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Quantile PSI with pooled duplicate reference edges and 0.5 smoothing.

    The original reference quantiles, NOT current quantiles, define bins;
    both tails extend to infinity. The result is exactly zero for identical
    samples (including constant and duplicate-heavy arrays).
    """
    a, b = (np.asarray(x, dtype=float) for x in (reference, current))
    if (a.ndim != 1 or b.ndim != 1 or not len(a) or not len(b)
            or not np.isfinite(a).all() or not np.isfinite(b).all()
            or not isinstance(bins, int) or bins < 2):
        raise ValueError("PSI needs finite nonempty 1D samples and bins >= 2")
    edges = np.unique(np.quantile(a, np.linspace(0.0, 1.0, bins + 1)[1:-1]))
    boundaries = np.concatenate(([-np.inf], edges, [np.inf]))
    ra, _ = np.histogram(a, bins=boundaries)
    rb, _ = np.histogram(b, bins=boundaries)
    pa = (ra + 0.5) / (len(a) + 0.5 * len(ra))
    pb = (rb + 0.5) / (len(b) + 0.5 * len(rb))
    return float(np.sum((pb - pa) * np.log(pb / pa)))


def categorical_comparison(reference: np.ndarray, current: np.ndarray) -> dict:
    """Chi-square and total variation on the union of observed categories.

    Small expected cells are reported as an explicit invalid test, not
    silently accepted. The caller still records the total-variation distance.
    """
    a, b = np.asarray(reference), np.asarray(current)
    if a.ndim != 1 or b.ndim != 1 or not len(a) or not len(b):
        raise ValueError("Categorical samples must be nonempty 1D arrays")
    if pd.isna(a).any() or pd.isna(b).any():
        raise ValueError("Null categories are not valid")
    categories = np.union1d(np.unique(a), np.unique(b))
    counts = np.array([[np.count_nonzero(a == val) for val in categories],
                       [np.count_nonzero(b == val) for val in categories]], dtype=float)
    pa = counts[0] / len(a)
    pb = counts[1] / len(b)
    total_variation = float(0.5 * np.abs(pa - pb).sum())
    expected = counts.sum(axis=1, keepdims=True) * counts.sum(axis=0, keepdims=True) / counts.sum()
    if (expected < 5).any() or len(categories) < 2:
        return {"chi2_stat": None, "p_raw": None, "tv": total_variation,
                "test_status": "rare_or_single_category"}
    statistic, pvalue, _, _ = chi2_contingency(counts, correction=False)
    return {"chi2_stat": float(statistic), "p_raw": float(pvalue),
            "tv": total_variation, "test_status": "ok"}


def bonferroni(pvalues: list[float]) -> list[float]:
    if not pvalues or not all(np.isfinite(x) and 0 <= x <= 1 for x in pvalues):
        raise ValueError("Expected one or more finite p-values in [0,1]")
    return [float(min(1.0, x * len(pvalues))) for x in pvalues]


def _day(value: object) -> str:
    return pd.Timestamp(value).tz_convert("UTC").strftime("%Y-%m-%d")


def _check_day_quality(part: pd.DataFrame, config: dict) -> dict[str, list[str]]:
    issues: dict[str, list[str]] = {}
    n = len(part)
    for feature in MODEL_FEATURES:
        reasons: list[str] = []
        if feature not in part:
            reasons.append("missing_column")
        else:
            values = pd.to_numeric(part[feature], errors="coerce").to_numpy(dtype=float)
            if not np.isfinite(values).all():
                reasons.append("missing_or_nonfinite")
            else:
                if feature == "distance_km" and ((values < 0) | (values > 1000)).any():
                    reasons.append("outside_domain")
                elif feature == "warehouse_load" and ((values < 0) | (values > 1)).any():
                    reasons.append("outside_domain")
                elif feature == "rain_forecast" and not np.isin(values, [0.0, 1.0]).all():
                    reasons.append("outside_domain")
                if feature in config["minimum_std"] and n >= 2:
                    if float(np.std(values)) < float(config["minimum_std"][feature]):
                        reasons.append("frozen_or_low_variance")
        if reasons:
            issues[feature] = reasons
    # Existing point-in-time audit columns are checked again on the replay stream.
    timestamp_fields = (("warehouse_load", "warehouse"), ("rain_forecast", "weather"))
    for feature, prefix in timestamp_fields:
        event_at, arrived_at = f"{prefix}_event_at", f"{prefix}_ingested_at"
        if all(col in part for col in (event_at, arrived_at, "ordered_at")):
            ordered = pd.to_datetime(part.ordered_at, utc=True)
            events = pd.to_datetime(part[event_at], utc=True)
            arrivals = pd.to_datetime(part[arrived_at], utc=True)
            if ((events > ordered) | (arrivals > ordered) | (arrivals < events)).any():
                issues.setdefault(feature, []).append("point_in_time_violation")
    return issues


def monitor_daily(reference: pd.DataFrame, current: pd.DataFrame, config: dict,
                  scenario: str) -> tuple[pd.DataFrame, list[dict]]:
    """Keep per-feature streaks, suppress duplicate alarms, reset on day gaps.

    Statistical candidates require corrected p plus effect size. Contract
    failure is immediate and never waits for 3 days. Labels are not accessed.
    """
    for feature in MODEL_FEATURES:
        if feature not in reference:
            raise ValueError(f"Reference lacks {feature}")
    if "ordered_at" not in current:
        raise ValueError("Current batch needs ordered_at")
    if len(reference) < 10 or not len(current):
        raise ValueError("Reference and current batches cannot be empty")
    # Orders share source observations: compare such features by UNIQUE event,
    # not one independent observation per order. Otherwise KS/chi2 p-values
    # are spuriously tiny and create false alerts on an unchanged generator.
    ref = {}
    for feature in MODEL_FEATURES:
        prefix = "warehouse" if feature == "warehouse_load" else "weather"
        event_col = f"{prefix}_event_id"
        source = (reference.drop_duplicates(event_col) if feature != "distance_km"
                  and event_col in reference else reference)
        ref[feature] = pd.to_numeric(source[feature], errors="raise").to_numpy(dtype=float)
    if not all(np.isfinite(value).all() for value in ref.values()):
        raise ValueError("Invalid reference features")
    base = current.copy()
    base["day"] = pd.to_datetime(base.ordered_at, utc=True).dt.strftime("%Y-%m-%d")
    state = {feature: {"streak": 0, "active": None} for feature in MODEL_FEATURES}
    rows: list[dict] = []
    alerts: list[dict] = []
    previous_day = None
    # Seven calendar days provide enough unique weather/warehouse observations.
    # A broken source within the window invalidates that feature's test until
    # the broken day leaves the window. Data-quality itself is still immediate.
    quality_by_day = {d: _check_day_quality(p, config)
                      for d, p in base.groupby("day", sort=True)}
    for day, part in base.sort_values("ordered_at", kind="stable").groupby("day", sort=True):
        calendar_day = pd.Timestamp(day)
        gap = previous_day is not None and (calendar_day - previous_day).days != 1
        previous_day = calendar_day
        enough = len(part) >= int(config["min_daily_requests"])
        issues = quality_by_day[day]
        statistics: dict[str, dict] = {}
        raw_pvalues: list[tuple[str, float]] = []
        for feature in MODEL_FEATURES:
            eligible = enough and feature not in issues
            source = part
            if feature in ("warehouse_load", "rain_forecast"):
                calendar = pd.Timestamp(day)
                previous_days = [(calendar - pd.Timedelta(days=i)).strftime("%Y-%m-%d")
                                 for i in range(7)]
                eligible = eligible and all(d in quality_by_day and
                                            feature not in quality_by_day[d]
                                            for d in previous_days)
                if eligible:
                    source = base.loc[base.day.isin(previous_days)]
                    column = "warehouse_event_id" if feature == "warehouse_load" else "weather_event_id"
                    if column in source:
                        source = source.drop_duplicates(column)
                    eligible = len(source) >= (28 if feature == "warehouse_load" else 20)
            if eligible:
                sample = source[feature].to_numpy(dtype=float)
                if feature in NUMERIC:
                    statistic, probability = ks_2samp(ref[feature], sample, method="auto")
                    result = {"psi": psi(ref[feature], sample, int(config["psi_bins"])),
                              "test_stat": float(statistic), "p_raw": float(probability),
                              "tv": None, "test_status": "ok"}
                else:
                    cat = categorical_comparison(ref[feature], sample)
                    result = {"psi": None, "test_stat": cat["chi2_stat"],
                              "p_raw": cat["p_raw"], "tv": cat["tv"],
                              "test_status": cat["test_status"]}
                result["effective_n"] = len(source)
                statistics[feature] = result
                if result["p_raw"] is not None:
                    raw_pvalues.append((feature, result["p_raw"]))
        adjusted = dict(zip([f for f, _ in raw_pvalues],
                            bonferroni([p for _, p in raw_pvalues]))) if raw_pvalues else {}
        for feature in MODEL_FEATURES:
            stats = statistics.get(feature, {})
            state_for_feature = state[feature]
            prev_active = state_for_feature["active"]
            kind = None
            if feature in issues:
                kind = "data_quality"
            elif enough and stats.get("p_raw") is not None:
                if feature in NUMERIC:
                    significant_effect = stats["psi"] >= config["numeric_min_psi"]
                else:
                    significant_effect = stats["tv"] >= config["categorical_min_total_variation"]
                if adjusted[feature] < config["bonferroni_alpha"] and significant_effect:
                    kind = "drift_candidate"
            if gap:
                # A missing calendar day ends the previous series; the first
                # observed candidate after the gap starts a NEW series at 1.
                state_for_feature["streak"] = 0
                state_for_feature["active"] = None
            if not enough or kind != "drift_candidate":
                state_for_feature["streak"] = 0
            else:
                state_for_feature["streak"] += 1
            confirmed = (kind == "drift_candidate" and
                         state_for_feature["streak"] >= int(config["persistent_days"]))
            incident = "data_quality" if kind == "data_quality" else "drift" if confirmed else None
            if kind == "data_quality" and prev_active != "data_quality":
                alerts.append({"scenario": scenario, "day": day, "feature": feature,
                               "type": "data_quality", "reason": ",".join(issues[feature]),
                               "observations": len(part), "confirmation_days": 1,
                               "action": "degrade_and_investigate_feature_source"})
            elif confirmed and prev_active != "drift":
                alerts.append({"scenario": scenario, "day": day, "feature": feature,
                               "type": "drift", "reason": "corrected_p_and_effect_for_consecutive_days",
                               "observations": len(part),
                               "confirmation_days": int(config["persistent_days"]),
                               "action": "investigate_data_mix_and_wait_for_mature_labels"})
            # A failed data contract interrupts the running drift streak.
            if incident:
                state_for_feature["active"] = incident
            else:
                state_for_feature["active"] = None
            rows.append({"scenario": scenario, "day": day, "feature": feature,
                         "n": int(len(part)), "sufficient_data": bool(enough),
                         "data_quality": ",".join(issues.get(feature, [])),
                         "psi": stats.get("psi"), "test_stat": stats.get("test_stat"),
                         "p_raw": stats.get("p_raw"), "p_bonferroni": adjusted.get(feature),
                         "tv": stats.get("tv"), "test_status": stats.get("test_status", "not_run"),
                         "daily_tests": len(raw_pvalues),
                         "effective_n": stats.get("effective_n", 0),
                         "reference_effective_n": len(ref[feature]),
                         "candidate": kind == "drift_candidate",
                         "streak": int(state_for_feature["streak"]),
                         "active_incident": state_for_feature["active"],
                         "calendar_gap": bool(gap)})
    return pd.DataFrame.from_records(rows), alerts


def inject_monitoring_incident(current: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Alter *observations only*. Do NOT use altered rows for outcome MAE."""
    incident = config["incident"]
    altered = current.copy()
    days = pd.to_datetime(altered.ordered_at, utc=True).dt.day
    failed = days.isin(incident["feature_failure_days"])
    altered.loc[failed, "warehouse_load"] = np.nan
    shifted = days >= int(incident["distance_shift_from_day"])
    altered.loc[shifted, "distance_km"] = np.minimum(
        altered.loc[shifted, "distance_km"] + float(incident["additional_distance_km"]), 1000.0)
    return altered


def mature_cohort_quality(predictions: pd.DataFrame, labels: pd.DataFrame, *,
                          snapshot_as_of: str, window_days: int = 7) -> tuple[pd.DataFrame, dict]:
    """Cohort ready IFF ALL prediction windows closed and ALL labels delivered.

    Incomplete and immature groups contribute zero rows to the MAE denominator.
    The overall MAE is order-weighted, not a mean of unweighted daily MAEs.
    """
    required_predictions = {"order_id", "ordered_at", "prediction_hours"}
    required_labels = {"order_id", "delivery_hours", "available_at"}
    if not required_predictions.issubset(predictions) or not required_labels.issubset(labels):
        raise ValueError("Predictions/labels do not match the cohort contract")
    if predictions.order_id.isna().any() or predictions.order_id.duplicated().any():
        raise ValueError("Prediction IDs must be present and unique")
    if labels.order_id.isna().any() or labels.order_id.duplicated().any():
        raise ValueError("Label IDs must be present and unique")
    joined = predictions.loc[:, list(required_predictions)].merge(
        labels.loc[:, list(required_labels)], on="order_id", how="left", validate="one_to_one")
    if not len(joined):
        raise ValueError("No predictions")
    ordered = pd.to_datetime(joined.ordered_at, utc=True, errors="raise")
    available = pd.to_datetime(joined.available_at, utc=True, errors="coerce")
    snapshot = pd.Timestamp(snapshot_as_of)
    if snapshot.tzinfo is None or window_days < 1:
        raise ValueError("Maturity parameters must be positive and UTC aware")
    pred = pd.to_numeric(joined.prediction_hours, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(pred).all():
        raise ValueError("Invalid predictions")
    delivered = available.notna() & (available <= snapshot)
    closed = ordered + pd.Timedelta(days=window_days) <= snapshot
    joined["day"] = ordered.dt.strftime("%Y-%m-%d")
    joined["window_closed"] = closed
    joined["label_received"] = delivered
    rows = []
    numerator = 0.0
    denominator = 0
    for day, group in joined.groupby("day", sort=True):
        n = len(group)
        if not group.window_closed.all():
            status = "immature"
        elif not group.label_received.all():
            status = "incomplete"
        else:
            status = "ready"
        eligible = status == "ready"
        absolute_errors = (group.delivery_hours - group.prediction_hours).abs() if eligible else None
        count = n if eligible else 0
        if eligible:
            if not np.isfinite(group.delivery_hours.to_numpy(dtype=float)).all():
                raise ValueError("Delivered labels must contain valid outcomes")
            numerator += float(absolute_errors.sum())
            denominator += count
        rows.append({"day": day, "status": status, "n_orders": n,
                     "windows_closed": int(group.window_closed.sum()),
                     "labels_received": int(group.label_received.sum()),
                     "label_coverage": float(group.label_received.mean()),
                     "mae_denominator": count,
                     "mae_hours": float(absolute_errors.mean()) if eligible else None})
    cohorts = pd.DataFrame.from_records(rows)
    aggregate = {"snapshot_as_of": str(snapshot.isoformat()), "total_orders": len(joined),
                 "mature_ready_orders": denominator, "mae_numerator_hours": numerator,
                 "mae_denominator": denominator,
                 "ready_mae_hours": numerator / denominator if denominator else None,
                 "ready_cohorts": int((cohorts.status == "ready").sum()),
                 "incomplete_cohorts": int((cohorts.status == "incomplete").sum()),
                 "immature_cohorts": int((cohorts.status == "immature").sum()),
                 "delivered_label_coverage": float(joined.label_received.mean())}
    return cohorts, aggregate


def operational_monitor(windows: pd.DataFrame, *, min_observations: int = 200,
                        max_error_rate: float = 0.01, max_degraded_rate: float = 0.05,
                        max_p95_ms: float = 100.0) -> list[dict]:
    """Read existing stage-6 window counters (synthetic, not live HTTP data)."""
    required = {"scenario", "window_id", "stable_n", "candidate_n", "stable_errors",
                "candidate_errors", "stable_degraded", "candidate_degraded",
                "stable_p95_ms", "candidate_p95_ms"}
    if not required.issubset(windows):
        raise ValueError("Stage-6 rollout window schema mismatch")
    alerts = []
    previous: set[tuple[str, str, str]] = set()
    final = (windows.sort_values("window_id").groupby("scenario").tail(1)
             .set_index("scenario"))
    for row in windows.itertuples(index=False):
        present: set[tuple[str, str, str]] = set()
        measurements = {}
        for version, name in (("v1", "stable"), ("v2", "candidate")):
            n = int(getattr(row, f"{name}_n"))
            if n < min_observations:
                continue
            er = int(getattr(row, f"{name}_errors")) / n
            degraded = int(getattr(row, f"{name}_degraded")) / n
            p95 = float(getattr(row, f"{name}_p95_ms"))
            breaches = []
            if er > max_error_rate:
                breaches.append("error_rate")
            if degraded > max_degraded_rate:
                breaches.append("degraded_rate")
            if np.isfinite(p95) and p95 > max_p95_ms:
                breaches.append("latency_p95")
            measurements[version] = {"n": n, "error_rate": er,
                                     "degraded_rate": degraded, "p95_ms": p95,
                                     "breaches": breaches}
        shared = ("v1" in measurements and "v2" in measurements and
                  "degraded_rate" in measurements["v1"]["breaches"] and
                  "degraded_rate" in measurements["v2"]["breaches"])
        if shared:
            key = (row.scenario, "both", "shared_feature_dependency")
            present.add(key)
            if key not in previous:
                alerts.append({"scenario": row.scenario, "window_id": int(row.window_id),
                               "type": "shared_feature_dependency", "version": "both",
                               "observations": measurements["v1"]["n"] + measurements["v2"]["n"],
                               "action": "hold_release_and_restore_shared_features",
                               "controller_action_at_signal": row.action if hasattr(row, "action") else None,
                               "controller_final_action": final.loc[row.scenario, "action"] if "action" in final else None,
                               "controller_final_state": final.loc[row.scenario, "status"] if "status" in final else None,
                               "source": "synthetic_stage06_rollout_windows"})
        else:
            for version, data in measurements.items():
                for breach in data["breaches"]:
                    key = (row.scenario, version, breach)
                    present.add(key)
                    if key not in previous:
                        alerts.append({"scenario": row.scenario, "window_id": int(row.window_id),
                                       "type": breach, "version": version,
                                       "observations": data["n"],
                                       "action": "investigate_and_apply_rollout_controller_policy",
                                       "controller_action_at_signal": row.action if hasattr(row, "action") else None,
                                       "controller_final_action": final.loc[row.scenario, "action"] if "action" in final else None,
                                       "controller_final_state": final.loc[row.scenario, "status"] if "status" in final else None,
                                       "source": "synthetic_stage06_rollout_windows"})
        # Each rollout scenario is an independent controller timeline.
        previous = {key for key in previous if key[0] != row.scenario} | present
    return alerts


def run_monitor(*, config_path: Path = ROOT / "configs/monitor.json",
                features_path: Path = ROOT / "data/generated/features.csv",
                artifact_dir: Path = ROOT / "artifacts",
                rollout_windows_path: Path = ROOT / "reports/rollout_windows.csv",
                output_dir: Path = ROOT / "reports") -> dict:
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    frame = pd.read_csv(features_path)
    reference = frame.loc[frame.period == cfg["reference_period"]].copy()
    replay = frame.loc[frame.period == cfg["monitored_period"]].copy()
    if reference.empty or replay.empty:
        raise ValueError("Monitoring periods are empty")
    clean_daily, clean_alerts = monitor_daily(reference, replay, cfg, "clean_replay")
    injected = inject_monitoring_incident(replay, cfg)
    incident_daily, incident_alerts = monitor_daily(reference, injected, cfg, "injected_incident")
    combined = pd.concat((clean_daily, incident_daily), ignore_index=True)

    model, manifest = load_bundle(artifact_dir)
    # Source-of-truth prediction records never contain future delivery outcomes.
    predictions = replay[["order_id", "ordered_at"]].copy()
    predictions["prediction_hours"] = predict_frame(model, manifest, replay)
    predictions["model_version"] = manifest["model_version"]
    # Simulated label-delivery backlog is stored in the *label log*, not features.
    labels = replay[["order_id", "ordered_at", "delivery_hours", "label_available_at"]].copy()
    labels = labels.rename(columns={"label_available_at": "available_at"})
    selected_day = int(cfg["incident"]["late_label_day"])
    selected = labels[pd.to_datetime(labels.ordered_at, utc=True).dt.day == selected_day].head(
        int(cfg["incident"]["late_label_count"])).index
    labels.loc[selected, "available_at"] = (
        pd.to_datetime(labels.loc[selected, "available_at"], utc=True)
        + pd.Timedelta(days=int(cfg["incident"]["late_label_extra_days"]))).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    labels = labels[["order_id", "delivery_hours", "available_at"]]
    cohorts, quality = mature_cohort_quality(predictions, labels,
                  snapshot_as_of=cfg["snapshot_as_of"], window_days=7)
    rollout_frame = pd.read_csv(rollout_windows_path)
    operational = operational_monitor(rollout_frame)
    alerts = [{**alert, "source": "synthetic_monitor_feature_injection"}
              for alert in clean_alerts + incident_alerts] + operational
    output_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_dir / "monitor_daily.csv", index=False, float_format="%.17g")
    cohorts.to_csv(output_dir / "monitor_cohorts.csv", index=False, float_format="%.17g")
    predictions.to_csv(output_dir / "monitor_predictions.csv", index=False, float_format="%.17g")
    labels.to_csv(output_dir / "monitor_labels.csv", index=False, float_format="%.17g")
    (output_dir / "monitor_alerts.jsonl").write_text(
        "".join(json.dumps(a, sort_keys=True, allow_nan=False) + "\n" for a in alerts), encoding="utf-8")
    result = {"stage": "07_monitoring", "seed": cfg["seed"],
              "config_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
              "model_version": manifest["model_version"],
              "reference_orders": len(reference), "replay_orders": len(replay),
              "daily_rows": len(combined), "alerts": len(alerts),
              "clean_alerts": clean_alerts, "injected_alerts": incident_alerts,
              "operational_alerts": operational, "mature_quality": quality,
              "limitations": [
                  "Injected changes alter monitored features without regenerating ground-truth delivery times; no claim about quality under injected incidents.",
                  "Quality is scored only on untouched replay predictions and fully ready daily cohorts.",
                  "Operational alerts read stage-6 synthetic windows, not real HTTP load measurements.",
                  "The generator is synthetic and has no weekly/holiday seasonality; thresholds require field calibration.",
                  "No online ground truth is available until the seven-day horizon and label-delivery delay close."
              ]}
    (output_dir / "monitor_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce operational and mature-cohort monitoring")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/monitor.json")
    parser.add_argument("--features", type=Path, default=ROOT / "data/generated/features.csv")
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--rollout-windows", type=Path, default=ROOT / "reports/rollout_windows.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    result = run_monitor(config_path=args.config, features_path=args.features,
                         artifact_dir=args.artifacts, rollout_windows_path=args.rollout_windows,
                         output_dir=args.out)
    print(json.dumps({key: result[key] for key in ("stage", "seed", "model_version", "replay_orders",
            "daily_rows", "alerts", "injected_alerts", "mature_quality")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
