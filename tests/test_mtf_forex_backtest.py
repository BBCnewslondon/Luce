import numpy as np
import pandas as pd

from evaluation.mtf_forex_backtest import MTFBacktestConfig, build_mtf_signal_frame, make_mtf_plot


def _make_ohlcv(index, base=1.10, drift=0.00005):
    n = len(index)
    close = base + np.arange(n) * drift + np.sin(np.arange(n) / 11.0) * 0.00005
    open_ = close - 0.00002
    high = close + 0.0001
    low = close - 0.0001
    vol = np.full(n, 1000)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=index)


def test_build_mtf_signal_frame_has_required_outputs_and_shifted_h4_columns():
    idx5 = pd.date_range("2024-01-01", periods=1200, freq="5min", tz="UTC")
    idx4 = pd.date_range("2023-12-25", periods=400, freq="4h", tz="UTC")
    m5 = _make_ohlcv(idx5, base=1.08, drift=0.00003)
    h4 = _make_ohlcv(idx4, base=1.06, drift=0.0004)

    out = build_mtf_signal_frame(m5, h4, MTFBacktestConfig(spread_pips=0.5))
    assert not out.empty
    assert set(
        [
            "signal",
            "signals",
            "signal_event",
            "position",
            "strategy_return",
            "cumulative_return",
            "drawdown",
            "h4_trend_active",
            "ema_8",
        ]
    ).issubset(out.columns)
    assert out["signal"].isin([-1, 0, 1]).all()
    assert out["signals"].isin(["long_entry", "short_entry", "stop_exit", "filter_exit", "flat"]).all()
    assert out["h4_trend_active"].dtype == bool
    assert out["drawdown"].le(0).all()
    mapped_h4_ema = h4["close"].ewm(span=8, adjust=False).mean().shift(1).reindex(out.index, method="ffill")
    comparable = out[["h4_ema_8"]].join(mapped_h4_ema.rename("expected")).dropna()
    assert np.allclose(comparable["h4_ema_8"].values, comparable["expected"].values)


def test_mandatory_global_filter_exit_closes_open_positions():
    idx5 = pd.date_range("2024-01-01", periods=1500, freq="5min", tz="UTC")
    idx4 = pd.date_range("2023-12-20", periods=450, freq="4h", tz="UTC")

    m5 = _make_ohlcv(idx5, base=1.10, drift=0.00002)
    h4_up = _make_ohlcv(idx4[:320], base=1.00, drift=0.0006)
    h4_dn = _make_ohlcv(idx4[320:], base=float(h4_up["close"].iloc[-1]), drift=-0.0008)
    h4 = pd.concat([h4_up, h4_dn]).sort_index()

    out = build_mtf_signal_frame(m5, h4, MTFBacktestConfig(ema_periods=(3, 5, 8, 13), rsi_periods=(5, 3)))
    assert "filter_exit" in set(out["signal_event"])
    exits = out[out["signal_event"] == "filter_exit"]
    assert (exits["position"] == 0).all()


def test_plot_contains_entries_ema_and_active_filter_shading():
    idx5 = pd.date_range("2024-01-01", periods=1000, freq="5min", tz="UTC")
    idx4 = pd.date_range("2023-12-25", periods=350, freq="4h", tz="UTC")
    m5 = _make_ohlcv(idx5, base=1.07, drift=0.00003)
    h4 = _make_ohlcv(idx4, base=1.05, drift=0.00035)

    out = build_mtf_signal_frame(m5, h4)
    fig = make_mtf_plot(out)
    names = {trace.name for trace in fig.data}
    assert {"Close", "EMA 8", "Long Entry", "Short Entry"}.issubset(names)
    assert fig.layout.shapes is not None
