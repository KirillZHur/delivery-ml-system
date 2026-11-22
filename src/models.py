"""Deterministic baseline ladder and a single prediction interface.

A fitted estimator is always bundled with its preprocessing.  Only features in
MODEL_FEATURES are passed to models, in the specified order.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.features import MODEL_FEATURES


class MedianRegressor(RegressorMixin, BaseEstimator):
    """Constant predictor using the target median observed during fit."""

    def fit(self, x: np.ndarray, y: np.ndarray) -> "MedianRegressor":
        if len(x) == 0 or len(y) == 0 or len(x) != len(y):
            raise ValueError("MedianRegressor needs nonempty aligned training data")
        self.median_ = float(np.median(np.asarray(y, dtype=float)))
        self.n_features_in_ = np.asarray(x).shape[1]
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not hasattr(self, "median_"):
            raise ValueError("MedianRegressor must be fitted")
        x = np.asarray(x)
        if x.ndim != 2 or x.shape[1] != self.n_features_in_:
            raise ValueError("Invalid input shape")
        return np.full(x.shape[0], self.median_, dtype=float)


def model_candidates(config: dict) -> dict:
    """Build fresh, independent estimators. No model has seen validation labels."""
    return {
        "median": MedianRegressor(),
        "linear": make_pipeline(
            StandardScaler(),
            Ridge(alpha=float(config["ridge_alpha"])),
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=float(config["hgb_learning_rate"]),
            max_iter=int(config["hgb_max_iter"]),
            max_leaf_nodes=int(config["hgb_max_leaf_nodes"]),
            min_samples_leaf=int(config["hgb_min_samples_leaf"]),
            l2_regularization=float(config["hgb_l2_regularization"]),
            random_state=int(config["seed"]),
            early_stopping=False,
        ),
    }


def predict_delivery(model: object, x: np.ndarray, bounds: tuple[float, float] = (1.0, 168.0)) -> np.ndarray:
    """Apply the same final clipping for offline evaluation and serving."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != len(MODEL_FEATURES):
        raise ValueError("Expected a matrix with three delivery features")
    prediction = np.asarray(model.predict(arr), dtype=np.float64)
    if prediction.shape != (len(arr),) or not np.isfinite(prediction).all():
        raise ValueError("Model returned invalid predictions")
    low, high = map(float, bounds)
    if low >= high:
        raise ValueError("Invalid prediction bounds")
    return np.clip(prediction, low, high)
