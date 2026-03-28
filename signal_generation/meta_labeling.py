"""Calibrated long/short meta-labeling and probability-based position sizing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.stats import rankdata
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression


@dataclass
class MetaLabelConfig:
    """Configuration for calibration and position sizing."""

    calibration_method: str = "isotonic"  # "sigmoid" (Platt) or "isotonic"
    min_confidence: float = 0.5
    sops_k: float = 8.0
    sizing_method: str = "sops"  # "sops" or "ecdf"


class DiscreteLongShortMetaLabeler:
    """
    Two specialized calibrated meta-models: one for long, one for short.

    Primary side input controls which model probability is translated to size.
    """

    def __init__(self, config: Optional[MetaLabelConfig] = None):
        self.config = config or MetaLabelConfig()
        self.long_model: Optional[CalibratedClassifierCV] = None
        self.short_model: Optional[CalibratedClassifierCV] = None

    def fit(self, x: np.ndarray, y: np.ndarray, side: np.ndarray) -> "DiscreteLongShortMetaLabeler":
        """Fit separate calibrated models for long and short subsets."""
        x = np.asarray(x)
        y = np.asarray(y).astype(int)
        side = np.asarray(side).astype(int)

        self.long_model = self._fit_one(x[side > 0], y[side > 0])
        self.short_model = self._fit_one(x[side < 0], y[side < 0])
        return self

    def predict_proba(self, x: np.ndarray, side: np.ndarray) -> np.ndarray:
        """Predict calibrated probabilities using side-specific model."""
        if self.long_model is None or self.short_model is None:
            raise RuntimeError("Meta-labeler is not fitted")

        x = np.asarray(x)
        side = np.asarray(side).astype(int)
        proba = np.zeros(len(x), dtype=float)

        long_idx = np.where(side > 0)[0]
        short_idx = np.where(side < 0)[0]

        if long_idx.size:
            proba[long_idx] = self.long_model.predict_proba(x[long_idx])[:, 1]
        if short_idx.size:
            proba[short_idx] = self.short_model.predict_proba(x[short_idx])[:, 1]

        return proba

    def predict_position_sizes(self, x: np.ndarray, primary_side: np.ndarray) -> np.ndarray:
        """Convert calibrated probabilities into signed position sizes."""
        side = np.asarray(primary_side).astype(int)
        proba = self.predict_proba(x, side)
        magnitude = self._size_from_probability(proba)
        return np.where(side > 0, magnitude, np.where(side < 0, -magnitude, 0.0))

    def _fit_one(self, x: np.ndarray, y: np.ndarray) -> CalibratedClassifierCV:
        if len(x) < 10 or len(np.unique(y)) < 2:
            raise ValueError("Each side requires at least 10 samples and both classes")

        base = LogisticRegression(max_iter=500, random_state=42)
        model = CalibratedClassifierCV(base, cv=3, method=self.config.calibration_method)
        return model.fit(x, y)

    def _size_from_probability(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        adj = np.clip((p - self.config.min_confidence) / max(1e-9, 1.0 - self.config.min_confidence), 0.0, 1.0)

        if self.config.sizing_method == "sops":
            centered = 2.0 * adj - 1.0
            return np.clip(1.0 / (1.0 + np.exp(-self.config.sops_k * centered)), 0.0, 1.0)

        if self.config.sizing_method == "ecdf":
            ranks = rankdata(adj, method="average")
            return np.clip(ranks / len(adj), 0.0, 1.0)

        raise ValueError("sizing_method must be 'sops' or 'ecdf'")
