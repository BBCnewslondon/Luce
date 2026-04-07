import numpy as np
import pandas as pd
import pandas_ta as ta
import evaluation.mtf_forex_backtest as mtf_module

from evaluation.mtf_forex_backtest import (
    MAJOR_FX_TICKERS,
    MTFBacktestConfig,
    build_trade_log,
    build_mtf_signal_frame,
    make_major_pairs_subplots,
    make_mtf_plot,
)


def _make_ohlcv(index, base=1.10, drift=0.00005):
    n = len(index)
    close = base + np.arange(n) * drift + np.sin(np.arange(n) / 11.0) * 0.00005
    open_ = close - 0.00002
    high = close + 0.0001
    low = close - 0.0001
    vol = np.full(n, 1000)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=index)


def _make_linear_ohlcv(index, start=1.10, step=0.0001):
    n = len(index)
    close = start + np.arange(n) * step
    spread = max(abs(step) * 2.0, 0.0001)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + spread,
            "low": close - spread,
            "close": close,
            "volume": np.full(n, 1000),
        },
        index=index,
    )


def _make_trade_event_frame():
    idx = pd.date_range("2024-01-01", periods=7, freq="5min", tz="UTC")
    close = np.array([1.1000, 1.1010, 1.1020, 1.1015, 1.1005, 1.0995, 1.0990])
    return pd.DataFrame(
        {
            "close": close,
            "ema_8": close,
            "signal_event": ["flat", "long_entry", "flat", "filter_exit", "short_entry", "stop_exit", "flat"],
            "position": [0.0, 1.0, 1.0, 0.0, -1.0, 0.0, 0.0],
            "h4_trend_active": [False, True, True, True, True, False, False],
            "macd_line": np.array([0.01, 0.02, 0.03, 0.01, -0.01, -0.02, -0.01]),
            "macd_signal": np.array([0.005, 0.01, 0.02, 0.015, -0.005, -0.01, -0.008]),
        },
        index=idx,
    )


def test_build_mtf_signal_frame_has_required_outputs_and_synchronized_h4_columns():
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
            "stop_price",
            "take_profit_price",
            "strategy_return",
            "cumulative_return",
            "drawdown",
            "h4_trend_active",
            "h4_price_above_ema_fast",
            "h4_price_below_ema_fast",
            "ema_8",
        ]
    ).issubset(out.columns)
    assert out["signal"].isin([-1, 0, 1]).all()
    assert out["signals"].isin(["long_entry", "short_entry", "stop_exit", "take_profit_exit", "filter_exit", "flat"]).all()
    assert out["signals"].equals(out["signal_event"])
    assert out["h4_trend_active"].dtype == bool
    assert out["drawdown"].le(0).all()
    mapped_h4_ema = h4["close"].ewm(span=8, adjust=False).mean().reindex(out.index).ffill()
    comparable = out[["h4_ema_8"]].join(mapped_h4_ema.rename("expected")).dropna()
    assert np.allclose(comparable["h4_ema_8"].values, comparable["expected"].values)

    macd = ta.macd(h4["close"], fast=8, slow=21, signal=5)
    h4_macd_line = macd["MACD_8_21_5"].reindex(out.index).ffill()
    h4_macd_signal = macd["MACDs_8_21_5"].reindex(out.index).ffill()
    h4_rsi_13 = ta.rsi(h4["close"], length=13).reindex(out.index).ffill()
    h4_rsi_5 = ta.rsi(h4["close"], length=5).reindex(out.index).ffill()
    merged = out[["h4_macd_line", "h4_macd_signal", "h4_rsi_13", "h4_rsi_5"]].join(
        pd.DataFrame(
            {
                "exp_macd_line": h4_macd_line,
                "exp_macd_signal": h4_macd_signal,
                "exp_rsi_13": h4_rsi_13,
                "exp_rsi_5": h4_rsi_5,
            }
        )
    ).dropna()
    assert np.allclose(merged["h4_macd_line"], merged["exp_macd_line"])
    assert np.allclose(merged["h4_macd_signal"], merged["exp_macd_signal"])
    assert np.allclose(merged["h4_rsi_13"], merged["exp_rsi_13"])
    assert np.allclose(merged["h4_rsi_5"], merged["exp_rsi_5"])


def test_mandatory_global_filter_exit_closes_open_positions(monkeypatch):
    idx5 = pd.date_range("2024-01-01", periods=1500, freq="5min", tz="UTC")
    idx4 = pd.date_range("2023-12-20", periods=450, freq="4h", tz="UTC")

    m5 = _make_ohlcv(idx5, base=1.10, drift=0.00002)

    h4 = _make_ohlcv(idx4, base=1.00, drift=0.0002)

    def fake_bullish_filter(ind, cfg):
        # Keep bullish regime active early in the sample, then disable it to force filter exit.
        cutoff = pd.Timestamp("2024-01-03", tz="UTC")
        return pd.Series(ind.index < cutoff, index=ind.index)

    def fake_bearish_filter(ind, cfg):
        return pd.Series(False, index=ind.index)

    monkeypatch.setattr(mtf_module, "_bullish_filter", fake_bullish_filter)
    monkeypatch.setattr(mtf_module, "_bearish_filter", fake_bearish_filter)

    out = build_mtf_signal_frame(m5, h4, MTFBacktestConfig(ema_periods=(3, 5, 8, 13), rsi_periods=(5, 3)))
    assert "filter_exit" in set(out["signal_event"])
    exits = out[out["signal_event"] == "filter_exit"]
    assert (exits["position"] == 0).all()


def test_entry_requires_m5_confirmation_and_triggers_when_aligned(monkeypatch):
    idx5 = pd.date_range("2024-01-01", periods=1200, freq="5min", tz="UTC")
    idx4 = pd.date_range("2023-12-20", periods=450, freq="4h", tz="UTC")

    # Force a 5m downtrend so bullish 5m confirmation remains false.
    n = len(idx5)
    close = 1.20 - np.arange(n) * 0.00003
    m5 = pd.DataFrame(
        {
            "open": close + 0.00001,
            "high": close + 0.00008,
            "low": close - 0.00008,
            "close": close,
            "volume": np.full(n, 1200),
        },
        index=idx5,
    )

    h4 = _make_ohlcv(idx4, base=1.00, drift=0.0007)

    def fake_bullish_filter(ind, cfg):
        return pd.Series(True, index=ind.index)

    def fake_bearish_filter(ind, cfg):
        return pd.Series(False, index=ind.index)

    monkeypatch.setattr(mtf_module, "_bullish_filter", fake_bullish_filter)
    monkeypatch.setattr(mtf_module, "_bearish_filter", fake_bearish_filter)

    out = build_mtf_signal_frame(m5, h4, MTFBacktestConfig(ema_periods=(3, 5, 8, 13), rsi_periods=(5, 3)))

    assert out["h4_trend_bull"].any()
    assert not out["m5_bull_confirm"].any()
    assert "long_entry" not in set(out["signal_event"])

    # With a strong 5m uptrend, confirmation should align and permit long entries.
    m5_aligned = _make_ohlcv(idx5, base=1.10, drift=0.00008)
    out_aligned = build_mtf_signal_frame(m5_aligned, h4, MTFBacktestConfig(ema_periods=(3, 5, 8, 13), rsi_periods=(5, 3)))

    assert out_aligned["m5_bull_confirm"].any()
    assert "long_entry" in set(out_aligned["signal_event"])


def test_entry_requires_h4_close_vs_ema8_price_bias(monkeypatch):
    idx5 = pd.date_range("2024-01-01", periods=1200, freq="5min", tz="UTC")
    idx4 = pd.date_range("2023-12-20", periods=450, freq="4h", tz="UTC")

    m5_up = _make_linear_ohlcv(idx5, start=1.10, step=0.00008)
    h4_down = _make_linear_ohlcv(idx4, start=1.30, step=-0.0010)
    h4_up = _make_linear_ohlcv(idx4, start=1.00, step=0.0010)

    def fake_bullish_filter(ind, cfg):
        return pd.Series(True, index=ind.index)

    def fake_bearish_filter(ind, cfg):
        return pd.Series(False, index=ind.index)

    monkeypatch.setattr(mtf_module, "_bullish_filter", fake_bullish_filter)
    monkeypatch.setattr(mtf_module, "_bearish_filter", fake_bearish_filter)

    cfg = MTFBacktestConfig(ema_periods=(3, 5, 8, 13), rsi_periods=(5, 3), stop_lookback=3)
    out_blocked = build_mtf_signal_frame(m5_up, h4_down, cfg)

    assert not out_blocked["h4_price_above_ema_fast"].any()
    assert "long_entry" not in set(out_blocked["signal_event"])

    out_allowed = build_mtf_signal_frame(m5_up, h4_up, cfg)

    assert out_allowed["h4_price_above_ema_fast"].any()
    assert out_allowed["m5_bull_confirm"].any()
    assert "long_entry" in set(out_allowed["signal_event"])


def test_plot_contains_entries_ema_and_active_filter_shading():
    idx5 = pd.date_range("2024-01-01", periods=1000, freq="5min", tz="UTC")
    idx4 = pd.date_range("2023-12-25", periods=350, freq="4h", tz="UTC")
    m5 = _make_ohlcv(idx5, base=1.07, drift=0.00003)
    h4 = _make_ohlcv(idx4, base=1.05, drift=0.00035)

    out = build_mtf_signal_frame(m5, h4)
    fig = make_mtf_plot(out)
    names = {trace.name for trace in fig.data}
    assert {
        "Close",
        "EMA 8",
        "Long Entry",
        "Short Entry",
        "Long Exit",
        "Short Exit",
        "MACD Histogram",
        "MACD",
        "MACD Signal",
    }.issubset(names)
    assert fig.layout.shapes is not None


def test_plot_marks_long_and_short_exit_positions():
    frame = _make_trade_event_frame()
    fig = make_mtf_plot(frame)
    traces = {trace.name: trace for trace in fig.data}

    assert list(traces["Long Entry"].x) == [frame.index[1]]
    assert list(traces["Short Entry"].x) == [frame.index[4]]
    assert list(traces["Long Exit"].x) == [frame.index[3]]
    assert list(traces["Short Exit"].x) == [frame.index[5]]


def test_trade_log_records_open_close_timestamps_and_prices():
    frame = _make_trade_event_frame()
    log = build_trade_log(frame, ticker="EURUSD=X")

    assert len(log) == 2
    assert list(log["ticker"]) == ["EURUSD=X", "EURUSD=X"]

    first = log.iloc[0]
    assert first["side"] == "long"
    assert first["open_time"] == frame.index[1]
    assert first["close_time"] == frame.index[3]
    assert first["open_price"] == frame.loc[frame.index[1], "close"]
    assert first["close_price"] == frame.loc[frame.index[3], "close"]
    assert first["close_event"] == "filter_exit"

    second = log.iloc[1]
    assert second["side"] == "short"
    assert second["open_time"] == frame.index[4]
    assert second["close_time"] == frame.index[5]
    assert second["open_price"] == frame.loc[frame.index[4], "close"]
    assert second["close_price"] == frame.loc[frame.index[5], "close"]
    assert second["close_event"] == "stop_exit"


def test_trade_log_closes_open_position_at_end_of_data():
    idx = pd.date_range("2024-01-01", periods=3, freq="5min", tz="UTC")
    frame = pd.DataFrame(
        {
            "close": [1.1000, 1.1010, 1.1020],
            "signal_event": ["flat", "long_entry", "flat"],
            "position": [0.0, 1.0, 1.0],
        },
        index=idx,
    )

    log = build_trade_log(frame, ticker="EURUSD=X")

    assert len(log) == 1
    assert log.iloc[0]["open_time"] == idx[1]
    assert log.iloc[0]["close_time"] == idx[-1]
    assert log.iloc[0]["close_event"] == "end_of_data"


def test_trade_log_includes_take_profit_exit_event():
    idx = pd.date_range("2024-01-01", periods=3, freq="5min", tz="UTC")
    frame = pd.DataFrame(
        {
            "close": [1.1000, 1.1010, 1.1020],
            "signal_event": ["long_entry", "take_profit_exit", "flat"],
            "position": [1.0, 0.0, 0.0],
        },
        index=idx,
    )

    log = build_trade_log(frame, ticker="EURUSD=X")

    assert len(log) == 1
    assert log.iloc[0]["close_event"] == "take_profit_exit"


def test_stop_exit_is_emitted_when_price_hits_five_candle_stop(monkeypatch):
    idx5 = pd.date_range("2024-01-01", periods=800, freq="5min", tz="UTC")
    idx4 = pd.date_range("2023-12-20", periods=400, freq="4h", tz="UTC")

    m5 = _make_ohlcv(idx5, base=1.07, drift=0.00008)

    h4_close = 1.04 + (np.arange(len(idx4)) ** 1.2) * 0.0002
    h4 = pd.DataFrame(
        {
            "open": h4_close - 0.0001,
            "high": h4_close + 0.0002,
            "low": h4_close - 0.0002,
            "close": h4_close,
            "volume": np.full(len(idx4), 3000),
        },
        index=idx4,
    )

    def fake_bullish_filter(ind, cfg):
        return pd.Series(True, index=ind.index)

    def fake_bearish_filter(ind, cfg):
        return pd.Series(False, index=ind.index)

    monkeypatch.setattr(mtf_module, "_bullish_filter", fake_bullish_filter)
    monkeypatch.setattr(mtf_module, "_bearish_filter", fake_bearish_filter)

    cfg = MTFBacktestConfig(ema_periods=(3, 5, 8, 13), rsi_periods=(5, 3), stop_lookback=3, risk_reward_ratio=1000.0)

    baseline = build_mtf_signal_frame(
        m5,
        h4,
        cfg,
    )
    entry_idx = baseline.index[baseline["signal_event"] == "long_entry"]
    assert len(entry_idx) > 0

    m5_shock = m5.copy()
    first_entry_i = m5_shock.index.get_loc(entry_idx[0])
    shock_i = min(first_entry_i + 1, len(m5_shock) - 1)
    m5_shock.iloc[shock_i, m5_shock.columns.get_loc("low")] = m5_shock.iloc[shock_i]["low"] - 0.02

    out = build_mtf_signal_frame(
        m5_shock,
        h4,
        cfg,
    )
    assert "stop_exit" in set(out["signal_event"])


def test_take_profit_exit_is_emitted_at_two_to_one_reward(monkeypatch):
    idx5 = pd.date_range("2024-01-01", periods=1000, freq="5min", tz="UTC")
    idx4 = pd.date_range("2023-12-20", periods=400, freq="4h", tz="UTC")

    m5 = _make_ohlcv(idx5, base=1.07, drift=0.00002)
    m5["high"] = m5["close"] + 0.005

    h4 = _make_ohlcv(idx4, base=1.00, drift=0.0004)

    def fake_bullish_filter(ind, cfg):
        return pd.Series(True, index=ind.index)

    def fake_bearish_filter(ind, cfg):
        return pd.Series(False, index=ind.index)

    monkeypatch.setattr(mtf_module, "_bullish_filter", fake_bullish_filter)
    monkeypatch.setattr(mtf_module, "_bearish_filter", fake_bearish_filter)

    out = build_mtf_signal_frame(
        m5,
        h4,
        MTFBacktestConfig(ema_periods=(3, 5, 8, 13), rsi_periods=(5, 3), stop_lookback=3, risk_reward_ratio=2.0),
    )

    assert "take_profit_exit" in set(out["signal_event"])
    first_entry_idx = out.index[out["signal_event"] == "long_entry"][0]
    events_after = out.loc[first_entry_idx:, "signal_event"]
    first_exit_event = events_after[events_after.isin(["stop_exit", "take_profit_exit", "filter_exit"])].iloc[0]
    assert first_exit_event == "take_profit_exit"


def test_make_major_pairs_subplots_creates_panel_per_pair():
    idx5 = pd.date_range("2024-01-01", periods=900, freq="5min", tz="UTC")
    idx4 = pd.date_range("2023-12-25", periods=300, freq="4h", tz="UTC")

    eur = build_mtf_signal_frame(_make_ohlcv(idx5, base=1.08, drift=0.00003), _make_ohlcv(idx4, base=1.06, drift=0.00035))
    gbp = build_mtf_signal_frame(_make_ohlcv(idx5, base=1.25, drift=0.00004), _make_ohlcv(idx4, base=1.22, drift=0.00030))

    fig = make_major_pairs_subplots({"EURUSD=X": eur, "GBPUSD=X": gbp})
    titles = {ann.text for ann in fig.layout.annotations}
    names = {trace.name for trace in fig.data}

    assert {"EURUSD=X", "GBPUSD=X"}.issubset(titles)
    assert {"Close", "EMA 8"}.issubset(names)
    assert len(fig.data) >= 4


def test_major_pairs_subplots_include_trade_entry_and_exit_markers():
    frame = _make_trade_event_frame()
    fig = make_major_pairs_subplots({"EURUSD=X": frame})
    names = {trace.name for trace in fig.data}

    assert {"Long Entry", "Short Entry", "Long Exit", "Short Exit"}.issubset(names)


def test_default_transaction_costs_are_disabled():
    cfg = MTFBacktestConfig()
    assert cfg.spread_pips == 0.0
    assert cfg.commission_pips == 0.0
    assert cfg.slippage_pips == 0.0


def test_major_pairs_summary_includes_win_loss_pct_and_profit_factor(monkeypatch):
    idx = pd.date_range("2024-01-01", periods=4, freq="5min", tz="UTC")
    mocked_signals = pd.DataFrame(
        {
            "close": [1.0, 0.9, 1.0, 1.2],
            "signal_event": ["long_entry", "stop_exit", "long_entry", "take_profit_exit"],
            "position": [1.0, 0.0, 1.0, 0.0],
            "cumulative_return": [0.0, -0.1, -0.1, 0.1],
            "drawdown": [0.0, -0.1, -0.1, -0.05],
        },
        index=idx,
    )
    stub_ohlcv = pd.DataFrame(
        {
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "close": [1.0],
            "volume": [1000],
        },
        index=pd.date_range("2024-01-01", periods=1, freq="5min", tz="UTC"),
    )

    monkeypatch.setattr(
        mtf_module,
        "fetch_forex_data_oanda",
        lambda ticker, start, end, config=None: {"5m": stub_ohlcv, "4h": stub_ohlcv},
    )
    monkeypatch.setattr(
        mtf_module,
        "build_mtf_signal_frame",
        lambda data_5m, data_4h, config=None: mocked_signals.copy(),
    )
    monkeypatch.setattr(mtf_module, "make_major_pairs_subplots", lambda signals_by_ticker, title: None)

    result = mtf_module.run_mtf_major_pairs_backtest(tickers=["EURUSD=X"], start="2024-01-01", end="2024-01-02")
    summary = result["summary"]
    row = summary.iloc[0]

    assert row["closed_trades"] == 2
    assert row["winning_trades"] == 1
    assert row["losing_trades"] == 1
    assert np.isclose(row["win_pct"], 50.0)
    assert np.isclose(row["loss_pct"], 50.0)
    assert np.isclose(row["profit_factor"], 2.0)


def test_major_fx_universe_includes_gbpnzd():
    assert "GBPNZD=X" in MAJOR_FX_TICKERS


def test_major_fx_universe_includes_xauusd():
    assert "XAUUSD=X" in MAJOR_FX_TICKERS


def test_backtest_defaults_to_oanda_universe_when_tickers_not_provided(monkeypatch):
    idx = pd.date_range("2024-01-01", periods=3, freq="5min", tz="UTC")
    discovered = ["EUR_USD", "GBP_USD"]

    stub_ohlcv = pd.DataFrame(
        {
            "open": [1.0, 1.0, 1.0],
            "high": [1.0, 1.0, 1.0],
            "low": [1.0, 1.0, 1.0],
            "close": [1.0, 1.0, 1.0],
            "volume": [1000, 1000, 1000],
        },
        index=idx,
    )
    mocked_signals = pd.DataFrame(
        {
            "close": [1.0, 1.0, 1.0],
            "signal_event": ["long_entry", "stop_exit", "flat"],
            "position": [1.0, 0.0, 0.0],
            "cumulative_return": [0.0, -0.1, -0.1],
            "drawdown": [0.0, -0.1, -0.1],
        },
        index=idx,
    )

    fetch_calls = []

    monkeypatch.setattr(mtf_module, "get_available_oanda_instruments", lambda only_tradeable=True: discovered)
    monkeypatch.setattr(
        mtf_module,
        "fetch_forex_data_oanda",
        lambda ticker, start, end, config=None: fetch_calls.append(ticker) or {"5m": stub_ohlcv, "4h": stub_ohlcv},
    )
    monkeypatch.setattr(
        mtf_module,
        "build_mtf_signal_frame",
        lambda data_5m, data_4h, config=None: mocked_signals.copy(),
    )
    monkeypatch.setattr(mtf_module, "make_major_pairs_subplots", lambda signals_by_ticker, title: None)

    result = mtf_module.run_mtf_major_pairs_backtest(start="2024-01-01", end="2024-01-02")

    assert fetch_calls == discovered
    assert result["tickers"] == discovered
    assert result["universe_source"] == "oanda_account"


def test_all_oanda_instruments_wrapper_delegates_to_major_runner(monkeypatch):
    captured = {}

    def fake_run(tickers=None, start=None, end=None, config=None):
        captured["tickers"] = tickers
        captured["start"] = start
        captured["end"] = end
        return {"ok": True}

    monkeypatch.setattr(mtf_module, "run_mtf_major_pairs_backtest", fake_run)

    result = mtf_module.run_mtf_all_oanda_instruments_backtest(start="2024-01-01", end="2024-01-02")

    assert result == {"ok": True}
    assert captured["tickers"] is None
    assert captured["start"] == "2024-01-01"
    assert captured["end"] == "2024-01-02"
