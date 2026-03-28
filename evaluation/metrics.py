"""Evaluation metrics for predictive edge and cost-adjusted trading performance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

try:
    import alphalens as al
except ImportError:  # pragma: no cover - optional dependency in runtime
    al = None


@dataclass(frozen=True)
class TradingCostModel:
    """Cost model inputs in decimal return units per trade turnover."""

    spread_cost: float = 0.00008
    commission_cost: float = 0.000035
    slippage_cost: float = 0.00005
    max_abs_position: float = 1.0


def apply_costs_and_liquidity(
    signals: pd.Series,
    gross_forward_returns: pd.Series,
    cost_model: TradingCostModel,
) -> pd.Series:
    """
    Apply transaction costs, slippage, and liquidity clipping.

    Args:
        signals: Target position signal in [-1, 1].
        gross_forward_returns: Forward return series.
        cost_model: TradingCostModel parameters.

    Returns:
        Net strategy return series.

    Example:
        >>> import pandas as pd
        >>> s = pd.Series([1.0, 0.5, -1.0])
        >>> r = pd.Series([0.001, -0.002, 0.0015])
        >>> net = apply_costs_and_liquidity(s, r, TradingCostModel())
        >>> len(net)
        3
    """
    if len(signals) != len(gross_forward_returns):
        raise ValueError("signals and returns must have identical length")

    clipped_signal = signals.clip(-cost_model.max_abs_position, cost_model.max_abs_position)
    turnover = clipped_signal.diff().abs().fillna(clipped_signal.abs())
    unit_cost = cost_model.spread_cost + cost_model.commission_cost + cost_model.slippage_cost

    gross = clipped_signal * gross_forward_returns
    net = gross - unit_cost * turnover
    return net.astype(float)


def rolling_spearman_ic(
    factor_scores: pd.Series,
    forward_returns: pd.Series,
    window: int = 252,
) -> pd.Series:
    """
    Compute rolling Spearman Information Coefficient.

    Args:
        factor_scores: Factor or model prediction values.
        forward_returns: Realized forward returns.
        window: Rolling window length.

    Returns:
        Rolling Spearman IC series.

    Example:
        >>> import numpy as np, pandas as pd
        >>> x = pd.Series(np.arange(20, dtype=float))
        >>> y = pd.Series(np.arange(20, dtype=float))
        >>> ic = rolling_spearman_ic(x, y, window=5)
        >>> float(ic.iloc[-1])
        1.0
    """
    if len(factor_scores) != len(forward_returns):
        raise ValueError("factor_scores and forward_returns length mismatch")

    pair = pd.concat([factor_scores, forward_returns], axis=1).dropna()
    pair.columns = ["factor", "forward"]

    values = [np.nan] * len(pair)
    for i in range(window - 1, len(pair)):
        sl = pair.iloc[i - window + 1 : i + 1]
        values[i] = spearmanr(sl["factor"], sl["forward"], nan_policy="omit")[0]

    return pd.Series(values, index=pair.index, name="rolling_ic")


def alphalens_rolling_ic(
    factor_data: pd.DataFrame,
    prices_wide: pd.DataFrame,
    periods: Sequence[int] = (1, 5, 10),
) -> pd.DataFrame:
    """
    Run Alphalens factor cleaning and return IC table.

    Args:
        factor_data: Long-form DataFrame with columns timestamp, symbol, factor.
        prices_wide: Price matrix indexed by timestamp, columns by symbol.
        periods: Forward return horizons.

    Returns:
        DataFrame of mean IC values by period.

    Example:
        >>> import pandas as pd
        >>> _ = pd.DataFrame(columns=["timestamp", "symbol", "factor"])
    """
    if al is None:
        raise ImportError("alphalens is not installed. Add alphalens to requirements.")

    expected = {"timestamp", "symbol", "factor"}
    if not expected.issubset(set(factor_data.columns)):
        raise KeyError("factor_data must contain timestamp, symbol, factor")

    fi = factor_data.copy()
    fi["timestamp"] = pd.to_datetime(fi["timestamp"], utc=True)
    fi = fi.set_index(["timestamp", "symbol"]).sort_index()

    clean = al.utils.get_clean_factor_and_forward_returns(
        factor=fi["factor"],
        prices=prices_wide,
        periods=list(periods),
    )
    return al.performance.mean_information_coefficient(clean)


def masters_style_profit_factor(
    trade_path_returns: pd.DataFrame,
    sample_every: int = 4,
) -> pd.Series:
    """
    Compute Master's-style profit factor by periodic sampling in trade life.

    Args:
        trade_path_returns: Columns required: trade_id, bar_in_trade, net_return.
        sample_every: Sampling stride in bars.

    Returns:
        Series indexed by sampled bar_in_trade with profit factor values.

    Example:
        >>> import pandas as pd
        >>> d = pd.DataFrame({"trade_id": [1,1,2,2], "bar_in_trade": [1,2,1,2], "net_return": [0.1,-0.02,0.05,-0.01]})
        >>> pf = masters_style_profit_factor(d, sample_every=1)
        >>> 1 in pf.index
        True
    """
    required = {"trade_id", "bar_in_trade", "net_return"}
    if not required.issubset(trade_path_returns.columns):
        raise KeyError(f"trade_path_returns must contain {sorted(required)}")

    sampled = trade_path_returns[trade_path_returns["bar_in_trade"] % sample_every == 0].copy()
    if sampled.empty:
        return pd.Series(dtype=float, name="masters_profit_factor")

    out = {}
    for bar, grp in sampled.groupby("bar_in_trade"):
        profits = grp.loc[grp["net_return"] > 0, "net_return"]
        losses = grp.loc[grp["net_return"] < 0, "net_return"].abs()
        mean_profit = profits.mean() if not profits.empty else 0.0
        mean_loss = losses.mean() if not losses.empty else np.nan
        out[bar] = np.nan if pd.isna(mean_loss) or mean_loss == 0 else mean_profit / mean_loss

    return pd.Series(out, name="masters_profit_factor").sort_index()


def alpha_decay_profile(
    factor_scores: pd.Series,
    forward_return_by_horizon: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute alpha decay curve across multiple forward horizons.

    Args:
        factor_scores: Prediction/factor score series.
        forward_return_by_horizon: DataFrame with columns as horizon labels.

    Returns:
        DataFrame with horizon, spearman_ic, mean_signed_return.

    Example:
        >>> import pandas as pd
        >>> f = pd.Series([0.1, -0.1, 0.2])
        >>> r = pd.DataFrame({"h1": [0.01, -0.02, 0.03], "h2": [0.02, -0.01, 0.01]})
        >>> out = alpha_decay_profile(f, r)
        >>> "horizon" in out.columns
        True
    """
    rows = []
    signed_position = pd.Series(np.sign(factor_scores), index=factor_scores.index)

    for horizon in forward_return_by_horizon.columns:
        fr = forward_return_by_horizon[horizon]
        aligned = pd.concat([factor_scores, fr, signed_position], axis=1).dropna()
        aligned.columns = ["factor", "forward", "sign"]

        if aligned.empty:
            ic = np.nan
            mean_signed = np.nan
        else:
            ic = spearmanr(aligned["factor"], aligned["forward"], nan_policy="omit")[0]
            mean_signed = (aligned["sign"] * aligned["forward"]).mean()

        rows.append(
            {
                "horizon": horizon,
                "spearman_ic": float(ic) if ic is not None else np.nan,
                "mean_signed_return": float(mean_signed) if mean_signed is not None else np.nan,
            }
        )

    return pd.DataFrame(rows)
