"""Evaluation report helpers."""

from __future__ import annotations

import pandas as pd


class ReportGenerator:
    """Build concise summary tables from backtest output."""

    @staticmethod
    def summary(backtest_frame: pd.DataFrame) -> pd.Series:
        """
        Compute top-level performance summary.

        Args:
            backtest_frame: Frame expected to contain net_return.

        Returns:
            Series with cumulative return, hit rate, and Sharpe-like ratio.

        Example:
            >>> import pandas as pd
            >>> df = pd.DataFrame({"net_return": [0.01, -0.005, 0.002]})
            >>> s = ReportGenerator.summary(df)
            >>> "cumulative_return" in s.index
            True
        """
        if "net_return" not in backtest_frame.columns:
            raise KeyError("backtest_frame must contain net_return")

        r = pd.to_numeric(backtest_frame["net_return"], errors="coerce").dropna()
        if r.empty:
            return pd.Series(dtype=float)

        cumulative = float((1.0 + r.astype(float)).prod())
        sharpe_like = (r.mean() / r.std()) * (252.0**0.5) if r.std() > 0 else 0.0
        return pd.Series(
            {
                "cumulative_return": cumulative - 1.0,
                "mean_return": float(r.mean()),
                "hit_rate": float((r > 0).mean()),
                "sharpe_like": float(sharpe_like),
            }
        )
