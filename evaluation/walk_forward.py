"""Walk-forward validation and simple cost-aware backtest engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Sequence, Tuple

import numpy as np
import pandas as pd

from evaluation.metrics import TradingCostModel, apply_costs_and_liquidity
from signal_generation.ensemble import fit_ensemble, predict_ensemble


@dataclass(frozen=True)
class WalkForwardConfig:
    """Configuration for walk-forward splitting."""

    train_size: int = 24 * 365
    test_size: int = 24 * 30
    step_size: int = 24 * 30
    purge_gap: int = 24
    embargo_gap: int = 6


class WalkForwardValidator:
    """Generate purged/embargoed train-test slices for time series."""

    def __init__(self, config: WalkForwardConfig | None = None):
        self.config = config or WalkForwardConfig()

    def split(self, frame: pd.DataFrame) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Yield train and test integer index arrays.

        Args:
            frame: Time-indexed training DataFrame.

        Yields:
            Tuple of train_idx, test_idx arrays.

        Example:
            >>> import pandas as pd
            >>> df = pd.DataFrame({"x": range(2000)})
            >>> wf = WalkForwardValidator(WalkForwardConfig(train_size=500, test_size=100, step_size=100))
            >>> first = next(wf.split(df))
            >>> len(first[0]) == 500
            True
        """
        n = len(frame)
        c = self.config

        start_test = c.train_size + c.purge_gap
        while start_test + c.test_size <= n:
            train_end = start_test - c.purge_gap
            train_start = max(0, train_end - c.train_size)
            test_start = start_test
            test_end = start_test + c.test_size

            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(test_start, test_end)

            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx

            start_test = test_end + c.embargo_gap + max(0, c.step_size - c.test_size)


def run_walk_forward_backtest(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
    cost_model: TradingCostModel,
    prediction_threshold: float = 0.0,
    wf_config: WalkForwardConfig | None = None,
) -> pd.DataFrame:
    """
    Train and evaluate the ensemble on walk-forward slices.

    Args:
        frame: Full feature frame including target column.
        feature_columns: Feature columns.
        target_column: Forward return target column.
        cost_model: TradingCostModel for netting returns.
        prediction_threshold: Neutral zone threshold for signal generation.
        wf_config: Optional split config.

    Returns:
        DataFrame with timestamp-indexed backtest predictions and net returns.

    Example:
        >>> import numpy as np, pandas as pd
        >>> n = 1200
        >>> df = pd.DataFrame({
        ...     "f1": np.random.randn(n),
        ...     "f2": np.random.randn(n),
        ...     "target": np.random.randn(n) * 0.001,
        ... })
        >>> out = run_walk_forward_backtest(df, ["f1", "f2"], "target", TradingCostModel(), wf_config=WalkForwardConfig(train_size=500, test_size=100, step_size=100))
        >>> set(["prediction", "signal", "net_return"]).issubset(out.columns)
        True
    """
    frame_clean = frame.dropna(subset=[*feature_columns, target_column]).copy()
    wf = WalkForwardValidator(wf_config)

    chunks = []
    for train_idx, test_idx in wf.split(frame_clean):
        train = frame_clean.iloc[train_idx]
        test = frame_clean.iloc[test_idx]

        bundle = fit_ensemble(train, feature_columns, target_column)
        prediction = predict_ensemble(test, bundle)

        signal = pd.Series(
            np.where(prediction > prediction_threshold, 1.0, np.where(prediction < -prediction_threshold, -1.0, 0.0)),
            index=test.index,
        )
        net = apply_costs_and_liquidity(signal, test[target_column], cost_model)

        chunk = test.copy()
        chunk["prediction"] = prediction
        chunk["signal"] = signal
        chunk["net_return"] = net
        chunks.append(chunk)

    if not chunks:
        return pd.DataFrame(columns=[*frame.columns, "prediction", "signal", "net_return"])

    return pd.concat(chunks).sort_index()
