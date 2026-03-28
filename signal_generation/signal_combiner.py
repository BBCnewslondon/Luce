"""Signal post-processing and risk-aware combination helpers."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def apply_vix_risk_scaling(
    signal_frame: pd.DataFrame,
    vix_column: str = "vix",
    signal_column: str = "signal",
    high_vix_threshold: float = 25.0,
    extreme_vix_threshold: float = 35.0,
) -> pd.DataFrame:
    """
    Scale directional signal magnitude based on VIX regime.

    Args:
        signal_frame: Input frame containing VIX and signal columns.
        vix_column: Name of VIX column.
        signal_column: Name of signal column.
        high_vix_threshold: High-volatility threshold.
        extreme_vix_threshold: Crisis-volatility threshold.

    Returns:
        DataFrame with added scaled_signal column.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({"signal": [1, 1, -1], "vix": [15, 30, 40]})
        >>> out = apply_vix_risk_scaling(df)
        >>> out["scaled_signal"].tolist()
        [1.0, 0.5, -0.25]
    """
    out = signal_frame.copy()
    if vix_column not in out.columns or signal_column not in out.columns:
        raise KeyError("signal_frame must contain signal and VIX columns")

    scale = np.where(
        out[vix_column] >= extreme_vix_threshold,
        0.25,
        np.where(out[vix_column] >= high_vix_threshold, 0.50, 1.00),
    )
    out["scaled_signal"] = out[signal_column].astype(float) * scale
    return out


def combine_signals_mean(
    signals: Iterable[pd.Series],
    threshold: float = 0.0,
) -> pd.Series:
    """
    Combine multiple signal score series by arithmetic mean.

    Args:
        signals: Iterable of equal-length prediction score series.
        threshold: Decision threshold for sign conversion.

    Returns:
        Series with directional signal (-1, 0, 1).

    Example:
        >>> import pandas as pd
        >>> s1 = pd.Series([0.1, -0.2, 0.0])
        >>> s2 = pd.Series([0.2, -0.1, 0.0])
        >>> combine_signals_mean([s1, s2]).tolist()
        [1, -1, 0]
    """
    signal_list = list(signals)
    if not signal_list:
        raise ValueError("signals cannot be empty")

    base_index = signal_list[0].index
    matrix = pd.concat([s.reindex(base_index) for s in signal_list], axis=1)
    score = matrix.mean(axis=1)

    return pd.Series(
        np.where(score > threshold, 1, np.where(score < -threshold, -1, 0)),
        index=base_index,
    )


class SignalCombiner:
    """Compatibility wrapper exposing static combiners."""

    @staticmethod
    def apply_vix_risk_scaling(
        signal_frame: pd.DataFrame,
        vix_column: str = "vix",
        signal_column: str = "signal",
        high_vix_threshold: float = 25.0,
        extreme_vix_threshold: float = 35.0,
    ) -> pd.DataFrame:
        return apply_vix_risk_scaling(
            signal_frame=signal_frame,
            vix_column=vix_column,
            signal_column=signal_column,
            high_vix_threshold=high_vix_threshold,
            extreme_vix_threshold=extreme_vix_threshold,
        )

    @staticmethod
    def combine_signals_mean(
        signals: Iterable[pd.Series],
        threshold: float = 0.0,
    ) -> pd.Series:
        return combine_signals_mean(signals=signals, threshold=threshold)
