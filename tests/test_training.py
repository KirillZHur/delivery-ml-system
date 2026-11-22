"""Stage-2 checks: strict split maturity, baseline ladder and saved scoring."""

import json

import numpy as np
import pandas as pd
import pytest

from data.generate import ROOT
from src.bundle import load_bundle, predict_frame
from src.features import MODEL_FEATURES, model_matrix
from src.models import MedianRegressor, model_candidates, predict_delivery
from src.train import load_model_data, regression_metrics, train_and_select


@pytest.fixture(scope="module")
def generated_data():
    config = json.loads((ROOT / "configs/data.json").read_text())
    return ROOT / "data/generated/features.csv", config


def test_time_split_filters_all_unready_rows(generated_data):
    path, config = generated_data
    train, valid = load_model_data(path, config)
    assert len(train) == 9259
    assert len(valid) == 2990
    assert set(train.period) == {"train"}
    assert set(valid.period) == {"valid"}
    assert set(train.order_id).isdisjoint(set(valid.order_id))
    assert (pd.to_datetime(train.label_available_at, utc=True) < pd.Timestamp("2026-02-02", tz="UTC")).all()
    assert (pd.to_datetime(valid.label_available_at, utc=True) <= pd.Timestamp("2026-03-01", tz="UTC")).all()


def test_median_only_depends_on_training_labels():
    x = np.arange(12, dtype=float).reshape(4, 3)
    baseline = MedianRegressor().fit(x, np.array([1.0, 2.0, 3.0, 100.0]))
    np.testing.assert_array_equal(baseline.predict(x), [2.5] * 4)
    with pytest.raises(ValueError):
        baseline.predict(np.zeros((2, 2)))


def test_models_are_fresh_and_independent():
    config = json.loads((ROOT / "configs/training.json").read_text())
    first = model_candidates(config)
    second = model_candidates(config)
    assert list(first) == ["median", "linear", "hist_gradient_boosting"]
    assert all(first[k] is not second[k] for k in first)


def test_model_matrix_excludes_future_outcomes(generated_data):
    path, _ = generated_data
    original = pd.read_csv(path).head(7)
    x = model_matrix(original)
    modified = original.copy()
    modified["delivery_hours"] = 99999
    modified["delivered_at"] = "3000-01-01T00:00:00Z"
    np.testing.assert_array_equal(x, model_matrix(modified))
    assert x.shape == (7, len(MODEL_FEATURES))


def test_underestimation_and_mae_signs():
    out = regression_metrics(np.array([10, 20]), np.array([5, 30]))
    assert out["mae_hours"] == 7.5
    assert out["underestimation_share"] == 0.5
    assert out["mean_underestimation_hours"] == 2.5
    assert out["mean_signed_error_hours"] == -2.5


def test_clipping_shared_by_inference_and_training():
    class Dummy:
        def predict(self, x):
            return np.array([-5.0, 200.0])
    np.testing.assert_array_equal(predict_delivery(Dummy(), np.zeros((2, 3))), [1.0, 168.0])


def test_bundle_is_reproducible_and_rejects_tampering(tmp_path, generated_data):
    path, _ = generated_data
    kwargs = dict(
        features_path=path,
        data_config_path=ROOT / "configs/data.json",
        training_config_path=ROOT / "configs/training.json",
    )
    first = train_and_select(
        **kwargs,
        artifact_dir=tmp_path / "run_a",
        metrics_path=tmp_path / "a.json",
    )
    second = train_and_select(
        **kwargs,
        artifact_dir=tmp_path / "run_b",
        metrics_path=tmp_path / "b.json",
    )
    assert first["models"] == second["models"]
    assert first["selected_model"] == second["selected_model"]
    assert all(first["models"][k]["valid"]["n"] == 2990 for k in first["models"])
    best = min(first["models"], key=lambda name: first["models"][name]["valid"]["mae_hours"])
    assert first["selected_model"] == best
    model, manifest = load_bundle(tmp_path / "run_a")
    reference = pd.read_csv(path).head(12)
    a = predict_frame(model, manifest, reference)
    b = predict_delivery(model, model_matrix(reference))
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)
    with open(tmp_path / "run_a/model.joblib", "ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        load_bundle(tmp_path / "run_a")


def test_never_uses_test_or_replay_labels(tmp_path, generated_data):
    path, config = generated_data
    orig = pd.read_csv(path)
    changed = orig.copy()
    changed.loc[changed.period.isin(["test", "replay"]), "delivery_hours"] = 77777.0
    alternate_path = tmp_path / "modified.csv"
    changed.to_csv(alternate_path, index=False)
    train, valid = load_model_data(alternate_path, config)
    baseline_train, baseline_valid = load_model_data(path, config)
    np.testing.assert_allclose(train.delivery_hours, baseline_train.delivery_hours)
    np.testing.assert_allclose(valid.delivery_hours, baseline_valid.delivery_hours)


def test_feature_order_guard(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({
        "feature_order": ["rain_forecast", "distance_km", "warehouse_load"]
    }))
    with pytest.raises(ValueError, match="Feature contract"):
        load_bundle(tmp_path)
