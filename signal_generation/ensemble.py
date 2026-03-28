"""
Machine-learning ensemble signal generation.

This module builds lag-safe feature matrices and trains a tree-based ensemble
for FX return prediction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor


@dataclass(frozen=True)
class EnsembleConfig:
    """Configuration for model ensemble composition."""

    model_weights: Mapping[str, float] = field(
        default_factory=lambda: {
            "gbr": 0.5,
            "rf": 0.3,
            "gbt_alt": 0.2,
        }
    )
    random_state: int = 42


@dataclass
class EnsembleModelBundle:
    """Container with fitted models and metadata."""

    models: Dict[str, Any]
    weights: Dict[str, float]
    feature_columns: List[str]
    target_column: str


def lag_feature_columns(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    lag: int = 1,
) -> pd.DataFrame:
    """
    Return a lagged copy of selected feature columns.

    Args:
        frame: Input DataFrame.
        feature_columns: Columns to lag.
        lag: Number of bars to lag by.

    Returns:
        New DataFrame with lagged feature columns.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({"f1": [1, 2, 3], "f2": [4, 5, 6]})
        >>> lagged = lag_feature_columns(df, ["f1", "f2"], lag=1)
        >>> float(lagged["f1"].iloc[1])
        1.0
    """
    if lag < 1:
        raise ValueError("lag must be >= 1")

    out = frame.copy()
    for col in feature_columns:
        if col not in out.columns:
            raise KeyError(f"Feature column not found: {col}")
        out[col] = out[col].shift(lag)
    return out


def _default_models(random_state: int) -> Dict[str, Any]:
    return {
        "gbr": GradientBoostingRegressor(random_state=random_state),
        "rf": RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1,
        ),
        "gbt_alt": GradientBoostingRegressor(
            learning_rate=0.03,
            n_estimators=250,
            max_depth=3,
            random_state=random_state + 7,
        ),
    }


def fit_ensemble(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
    config: Optional[EnsembleConfig] = None,
) -> EnsembleModelBundle:
    """
    Fit a tree-ensemble on lagged feature data.

    Args:
        frame: Data containing lagged features and target.
        feature_columns: Feature column names.
        target_column: Forward return target column.
        config: Optional ensemble config.

    Returns:
        EnsembleModelBundle with fitted models.

    Example:
        >>> import numpy as np, pandas as pd
        >>> rng = np.random.default_rng(42)
        >>> df = pd.DataFrame({
        ...     "f1": rng.normal(size=200),
        ...     "f2": rng.normal(size=200),
        ...     "target": rng.normal(size=200),
        ... })
        >>> bundle = fit_ensemble(df, ["f1", "f2"], "target")
        >>> sorted(bundle.models.keys())
        ['gbt_alt', 'gbr', 'rf']
    """
    cfg = config or EnsembleConfig()
    missing = [c for c in [*feature_columns, target_column] if c not in frame.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    train = frame.loc[:, [*feature_columns, target_column]].dropna().copy()
    x = train.loc[:, feature_columns].to_numpy()
    y = train.loc[:, target_column].to_numpy()

    base_models = _default_models(cfg.random_state)
    fitted: Dict[str, Any] = {}
    for name, model in base_models.items():
        fitted[name] = model.fit(x, y)

    weights = dict(cfg.model_weights)
    total_weight = sum(weights.values())
    if total_weight <= 0:
        raise ValueError("Model weights must sum to a positive value")
    norm_weights = {k: v / total_weight for k, v in weights.items()}

    return EnsembleModelBundle(
        models=fitted,
        weights=norm_weights,
        feature_columns=list(feature_columns),
        target_column=target_column,
    )


def predict_ensemble(
    frame: pd.DataFrame,
    bundle: EnsembleModelBundle,
) -> pd.Series:
    """
    Produce weighted ensemble predictions.

    Args:
        frame: Feature frame.
        bundle: Fitted model bundle.

    Returns:
        Series of ensemble predictions aligned to frame index.

    Example:
        >>> import numpy as np, pandas as pd
        >>> rng = np.random.default_rng(3)
        >>> df = pd.DataFrame({"f1": rng.normal(size=50), "f2": rng.normal(size=50), "target": rng.normal(size=50)})
        >>> b = fit_ensemble(df, ["f1", "f2"], "target")
        >>> pred = predict_ensemble(df, b)
        >>> len(pred) == len(df)
        True
    """
    x = frame.loc[:, bundle.feature_columns]
    valid_mask = ~x.isna().any(axis=1)

    prediction = pd.Series(np.nan, index=frame.index, dtype=float)
    if not valid_mask.any():
        return prediction

    x_valid = x.loc[valid_mask].to_numpy()
    weighted = np.zeros(len(x_valid), dtype=float)

    for name, model in bundle.models.items():
        weight = bundle.weights.get(name, 0.0)
        if weight == 0.0:
            continue
        weighted += weight * model.predict(x_valid)

    prediction.loc[valid_mask] = weighted
    return prediction


def generate_signal_frame(
    frame: pd.DataFrame,
    predictions: pd.Series,
    threshold: float = 0.0,
) -> pd.DataFrame:
    """
    Create directional trading signals from prediction scores.

    Args:
        frame: Input frame with at least timestamp and symbol if available.
        predictions: Predicted forward returns.
        threshold: Minimum absolute score to take a position.

    Returns:
        DataFrame with prediction and signal columns.

    Example:
        >>> import pandas as pd
        >>> out = generate_signal_frame(pd.DataFrame({"x": [1,2,3]}), pd.Series([-0.1, 0.01, 0.4]), threshold=0.05)
        >>> out["signal"].tolist()
        [-1, 0, 1]
    """
    if len(frame) != len(predictions):
        raise ValueError("frame and predictions length mismatch")

    out = frame.copy()
    out["prediction"] = predictions.values
    out["signal"] = np.where(
        out["prediction"] > threshold,
        1,
        np.where(out["prediction"] < -threshold, -1, 0),
    )
    out["confidence"] = out["prediction"].abs()
    return out


class MLEnsemble:
    """Stateful wrapper around pure fit/predict helpers."""

    def __init__(self, config: Optional[EnsembleConfig] = None):
        self.config = config or EnsembleConfig()
        self.bundle: Optional[EnsembleModelBundle] = None

    def fit(
        self,
        frame: pd.DataFrame,
        feature_columns: Sequence[str],
        target_column: str,
    ) -> "MLEnsemble":
        self.bundle = fit_ensemble(frame, feature_columns, target_column, self.config)
        return self

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        if self.bundle is None:
            raise RuntimeError("Model is not fitted")
        return predict_ensemble(frame, self.bundle)
