"""Validated single-row affine predictor for the fitted StandardScaler -> Ridge.

This optional serving optimization is algebraically equivalent to the trained
pipeline: (x - mean) / scale @ coef + intercept becomes x @ weights + bias.
No model selection, fitting, or transformation statistics are recomputed here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import MODEL_FEATURES


@dataclass(frozen=True)
class FastLinearPredictor:
    weights: tuple[float, ...]
    intercept: float

    @classmethod
    def from_model(cls, model: object) -> "FastLinearPredictor | None":
        """Return None for other model families; never approximate their outputs."""
        if not isinstance(model, Pipeline) or len(model.steps) != 2:
            return None
        scaler, ridge = (step[1] for step in model.steps)
        if not isinstance(scaler, StandardScaler) or not isinstance(ridge, Ridge):
            return None
        if not scaler.with_mean or not scaler.with_std:
            return None
        dim = len(MODEL_FEATURES)
        try:
            means = np.asarray(scaler.mean_, dtype=np.float64)
            scales = np.asarray(scaler.scale_, dtype=np.float64)
            coefficients = np.asarray(ridge.coef_, dtype=np.float64)
            base_intercept = float(ridge.intercept_)
        except (AttributeError, TypeError, ValueError):
            return None
        if any(value.shape != (dim,) for value in (means, scales, coefficients)):
            return None
        if not (
            np.isfinite(means).all()
            and np.isfinite(scales).all()
            and (scales > 0).all()
            and np.isfinite(coefficients).all()
            and math.isfinite(base_intercept)
        ):
            return None
        weights = coefficients / scales
        bias = float(base_intercept - np.dot(means, weights))
        if not (np.isfinite(weights).all() and math.isfinite(bias)):
            return None
        return cls(tuple(float(x) for x in weights), bias)

    def predict_one(self, values: tuple[float, float, float], bounds: tuple[float, float]) -> float:
        if len(values) != len(self.weights) or not all(math.isfinite(x) for x in values):
            raise ValueError("Invalid inference feature vector")
        lo, hi = bounds
        if not (math.isfinite(lo) and math.isfinite(hi) and 0 < lo < hi):
            raise ValueError("Invalid prediction bounds")
        # Exact same scaling/affine transform as the frozen training pipeline,
        # up to a tiny floating-point reassociation error (< 1e-12 in tests).
        value = self.intercept + sum(w * x for w, x in zip(self.weights, values))
        if not math.isfinite(value):
            raise ValueError("Nonfinite model prediction")
        return float(min(hi, max(lo, value)))
