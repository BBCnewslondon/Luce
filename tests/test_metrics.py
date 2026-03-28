import numpy as np
import pandas as pd

from evaluation.metrics import (
    TradingCostModel,
    alpha_decay_profile,
    apply_costs_and_liquidity,
    masters_style_profit_factor,
    rolling_spearman_ic,
)


def test_apply_costs_and_liquidity_penalizes_turnover():
    signal = pd.Series([1.0, 1.0, -1.0, -1.0])
    returns = pd.Series([0.001, 0.001, 0.001, 0.001])
    net = apply_costs_and_liquidity(signal, returns, TradingCostModel())
    assert len(net) == 4
    assert net.iloc[2] < net.iloc[1]


def test_rolling_spearman_ic_positive_on_monotonic_data():
    x = pd.Series(np.arange(40, dtype=float))
    y = pd.Series(np.arange(40, dtype=float))
    ic = rolling_spearman_ic(x, y, window=10)
    assert ic.dropna().iloc[-1] > 0.95


def test_masters_profit_factor_and_alpha_decay_outputs():
    trades = pd.DataFrame(
        {
            "trade_id": [1, 1, 1, 2, 2, 2],
            "bar_in_trade": [1, 2, 3, 1, 2, 3],
            "net_return": [0.01, -0.005, 0.006, 0.012, -0.004, 0.003],
        }
    )
    pf = masters_style_profit_factor(trades, sample_every=1)
    assert not pf.empty

    factor = pd.Series([0.2, -0.3, 0.1, 0.4])
    horizon_returns = pd.DataFrame({"h1": [0.01, -0.01, 0.005, 0.02], "h3": [0.008, -0.004, 0.006, 0.015]})
    decay = alpha_decay_profile(factor, horizon_returns)
    assert set(["horizon", "spearman_ic", "mean_signed_return"]).issubset(decay.columns)
