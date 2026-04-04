"""Multi-timeframe FX backtest with lag-safe 4H trend filter and 5m execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

try:
    import pandas_ta as ta
except ImportError:  # pragma: no cover - optional runtime dependency
    ta = None

try:
    import yfinance as yf
except ImportError:  # pragma: no cover - optional runtime dependency
    yf = None


@dataclass(frozen=True)
class MTFBacktestConfig:
    ema_periods: Sequence[int] = (8, 20, 50, 200)
    macd_fast: int = 8
    macd_slow: int = 21
    macd_signal: int = 5
    rsi_periods: Sequence[int] = (13, 5)
    stop_lookback: int = 5
    spread_pips: float = 0.8
    pip_size: float = 0.0001


def _require_pandas_ta() -> None:
    if ta is None:
        raise ImportError("pandas_ta is required. Install with: pip install pandas_ta")


def _add_indicators(frame: pd.DataFrame, cfg: MTFBacktestConfig) -> pd.DataFrame:
    _require_pandas_ta()
    out = frame.copy()
    close = out["close"]
    for period in cfg.ema_periods:
        out[f"ema_{period}"] = ta.ema(close, length=period)

    macd = ta.macd(close, fast=cfg.macd_fast, slow=cfg.macd_slow, signal=cfg.macd_signal)
    out["macd_line"] = macd[f"MACD_{cfg.macd_fast}_{cfg.macd_slow}_{cfg.macd_signal}"]
    out["macd_signal"] = macd[f"MACDs_{cfg.macd_fast}_{cfg.macd_slow}_{cfg.macd_signal}"]

    out[f"rsi_{cfg.rsi_periods[0]}"] = ta.rsi(close, length=cfg.rsi_periods[0])
    out[f"rsi_{cfg.rsi_periods[1]}"] = ta.rsi(close, length=cfg.rsi_periods[1])
    return out


def _bullish_filter(ind: pd.DataFrame, cfg: MTFBacktestConfig) -> pd.Series:
    if len(cfg.ema_periods) < 4:
        raise ValueError("ema_periods must contain at least 4 periods")
    fast, mid_fast, mid_slow, slow = cfg.ema_periods[0], cfg.ema_periods[1], cfg.ema_periods[2], cfg.ema_periods[3]
    rsi_a, rsi_b = cfg.rsi_periods
    return (
        (ind[f"ema_{slow}"] < ind[f"ema_{mid_slow}"])
        & (ind[f"ema_{mid_slow}"] < ind[f"ema_{mid_fast}"])
        & (ind[f"ema_{mid_fast}"] < ind[f"ema_{fast}"])
        & (ind["macd_line"] > ind["macd_signal"])
        & (ind[f"rsi_{rsi_a}"] > 50)
        & (ind[f"rsi_{rsi_b}"] > 50)
    )


def _bearish_filter(ind: pd.DataFrame, cfg: MTFBacktestConfig) -> pd.Series:
    if len(cfg.ema_periods) < 4:
        raise ValueError("ema_periods must contain at least 4 periods")
    fast, mid_fast, mid_slow, slow = cfg.ema_periods[0], cfg.ema_periods[1], cfg.ema_periods[2], cfg.ema_periods[3]
    rsi_a, rsi_b = cfg.rsi_periods
    return (
        (ind[f"ema_{slow}"] > ind[f"ema_{mid_slow}"])
        & (ind[f"ema_{mid_slow}"] > ind[f"ema_{mid_fast}"])
        & (ind[f"ema_{mid_fast}"] > ind[f"ema_{fast}"])
        & (ind["macd_line"] < ind["macd_signal"])
        & (ind[f"rsi_{rsi_a}"] < 50)
        & (ind[f"rsi_{rsi_b}"] < 50)
    )


def build_mtf_signal_frame(
    data_5m: pd.DataFrame,
    data_4h: pd.DataFrame,
    config: Optional[MTFBacktestConfig] = None,
) -> pd.DataFrame:
    """
    Build 5m trading signals using lagged 4H trend filter.

    Returns frame with columns including:
    - signal_event: long_entry, short_entry, stop_exit, filter_exit, flat
    - position: {-1, 0, 1}
    - strategy_return, cumulative_return, drawdown
    """
    cfg = config or MTFBacktestConfig()
    df5 = _add_indicators(data_5m.copy(), cfg)
    df4 = _add_indicators(data_4h.copy(), cfg)
    indicator_cols = [f"ema_{p}" for p in cfg.ema_periods] + [
        "macd_line",
        "macd_signal",
        f"rsi_{cfg.rsi_periods[0]}",
        f"rsi_{cfg.rsi_periods[1]}",
    ]
    df4_lag = df4.copy()
    df4_lag[indicator_cols] = df4_lag[indicator_cols].shift(1)
    df4_lag["trend_bull"] = _bullish_filter(df4_lag, cfg)
    df4_lag["trend_bear"] = _bearish_filter(df4_lag, cfg)
    df4_lag["trend_active"] = df4_lag["trend_bull"] | df4_lag["trend_bear"]

    merge_cols = ["trend_bull", "trend_bear", "trend_active", f"ema_{cfg.ema_periods[0]}"]
    df4_lag = df4_lag[merge_cols].rename(columns=lambda c: f"h4_{c}")

    df = df5.join(df4_lag.reindex(df5.index, method="ffill"), how="left")
    for col in ["h4_trend_bull", "h4_trend_bear", "h4_trend_active"]:
        df[col] = df[col].fillna(False).astype(bool)

    fast, mid_fast, mid_slow, slow = cfg.ema_periods[0], cfg.ema_periods[1], cfg.ema_periods[2], cfg.ema_periods[3]
    rsi_a, rsi_b = cfg.rsi_periods
    df["m5_bull_confirm"] = (
        (df[f"ema_{slow}"] < df[f"ema_{mid_slow}"])
        & (df[f"ema_{mid_slow}"] < df[f"ema_{mid_fast}"])
        & (df[f"ema_{mid_fast}"] < df[f"ema_{fast}"])
        & (df[f"rsi_{rsi_a}"] > 50)
        & (df[f"rsi_{rsi_b}"] > 50)
        & (df["close"] > df[f"ema_{fast}"])
    )
    df["m5_bear_confirm"] = (
        (df[f"ema_{slow}"] > df[f"ema_{mid_slow}"])
        & (df[f"ema_{mid_slow}"] > df[f"ema_{mid_fast}"])
        & (df[f"ema_{mid_fast}"] > df[f"ema_{fast}"])
        & (df[f"rsi_{rsi_a}"] < 50)
        & (df[f"rsi_{rsi_b}"] < 50)
        & (df["close"] < df[f"ema_{fast}"])
    )

    spread_cost = (cfg.spread_pips * cfg.pip_size) / df["close"].replace(0, np.nan)
    long_stop = df["low"].shift(1).rolling(cfg.stop_lookback).min()
    short_stop = df["high"].shift(1).rolling(cfg.stop_lookback).max()

    position = np.zeros(len(df), dtype=float)
    event = np.full(len(df), "flat", dtype=object)
    stop_price = np.full(len(df), np.nan, dtype=float)
    trade_id = np.zeros(len(df), dtype=int)

    current_pos = 0.0
    current_trade = 0
    current_stop = np.nan

    for i in range(len(df)):
        row = df.iloc[i]

        if current_pos == 1.0 and (not bool(row["h4_trend_bull"])):
            current_pos = 0.0
            current_stop = np.nan
            event[i] = "filter_exit"
        elif current_pos == -1.0 and (not bool(row["h4_trend_bear"])):
            current_pos = 0.0
            current_stop = np.nan
            event[i] = "filter_exit"
        elif current_pos == 1.0 and pd.notna(current_stop) and float(row["low"]) <= float(current_stop):
            current_pos = 0.0
            current_stop = np.nan
            event[i] = "stop_exit"
        elif current_pos == -1.0 and pd.notna(current_stop) and float(row["high"]) >= float(current_stop):
            current_pos = 0.0
            current_stop = np.nan
            event[i] = "stop_exit"
        elif current_pos == 0.0 and bool(row["h4_trend_bull"]) and bool(row["m5_bull_confirm"]):
            current_pos = 1.0
            current_trade += 1
            current_stop = long_stop.iloc[i]
            event[i] = "long_entry"
        elif current_pos == 0.0 and bool(row["h4_trend_bear"]) and bool(row["m5_bear_confirm"]):
            current_pos = -1.0
            current_trade += 1
            current_stop = short_stop.iloc[i]
            event[i] = "short_entry"

        position[i] = current_pos
        stop_price[i] = current_stop
        trade_id[i] = current_trade if current_pos != 0 else 0

    df["signal_event"] = event
    df["signal"] = np.where(df["signal_event"] == "long_entry", 1, np.where(df["signal_event"] == "short_entry", -1, 0))
    df["signals"] = df["signal_event"]
    df["position"] = position
    df["stop_price"] = stop_price
    df["trade_id"] = trade_id
    df["price_return"] = df["close"].pct_change().fillna(0.0)
    turnover = pd.Series(position, index=df.index).diff().abs().fillna(np.abs(position[0]))
    df["strategy_return"] = pd.Series(position, index=df.index).shift(1).fillna(0.0) * df["price_return"] - turnover * spread_cost.fillna(0.0)
    df["equity_curve"] = (1.0 + df["strategy_return"]).cumprod()
    df["cumulative_return"] = df["equity_curve"] - 1.0
    df["drawdown"] = df["equity_curve"] / df["equity_curve"].cummax() - 1.0
    return df


def fetch_forex_data_yfinance(
    ticker: str,
    start: str,
    end: str,
) -> Dict[str, pd.DataFrame]:
    """Fetch 5m and 4h OHLCV frames from yfinance for a Forex ticker."""
    if yf is None:
        raise ImportError("yfinance is required. Install with: pip install yfinance")

    m5 = yf.download(ticker, interval="5m", start=start, end=end, progress=False, auto_adjust=False)
    h4 = yf.download(ticker, interval="1h", start=start, end=end, progress=False, auto_adjust=False)

    m5 = m5.rename(columns=str.lower)
    h4 = h4.rename(columns=str.lower).resample("4h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    m5 = m5[["open", "high", "low", "close", "volume"]].dropna()
    h4 = h4[["open", "high", "low", "close", "volume"]].dropna()
    return {"5m": m5, "4h": h4}


def make_mtf_plot(signal_frame: pd.DataFrame, title: str = "MTF Forex Backtest") -> go.Figure:
    """Create Plotly chart with close, EMA(8), entries, and active 4H filter shading."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=signal_frame.index, y=signal_frame["close"], name="Close", mode="lines"))
    fig.add_trace(go.Scatter(x=signal_frame.index, y=signal_frame["ema_8"], name="EMA 8", mode="lines"))

    longs = signal_frame[signal_frame["signal_event"] == "long_entry"]
    shorts = signal_frame[signal_frame["signal_event"] == "short_entry"]
    fig.add_trace(
        go.Scatter(
            x=longs.index,
            y=longs["close"],
            mode="markers",
            marker={"symbol": "triangle-up", "size": 9},
            name="Long Entry",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=shorts.index,
            y=shorts["close"],
            mode="markers",
            marker={"symbol": "triangle-down", "size": 9},
            name="Short Entry",
        )
    )

    active = signal_frame["h4_trend_active"].astype(bool)
    starts = signal_frame.index[(active) & (~active.shift(1, fill_value=False))]
    ends = signal_frame.index[(~active) & (active.shift(1, fill_value=False))]
    if active.iloc[-1]:
        ends = ends.append(pd.Index([signal_frame.index[-1]]))

    for start, end in zip(starts, ends):
        fig.add_vrect(
            x0=start,
            x1=end,
            fillcolor="LightGreen",
            opacity=0.15,
            layer="below",
            line_width=0,
        )

    fig.update_layout(title=title, xaxis_title="Time", yaxis_title="Price", template="plotly_white")
    return fig


def run_mtf_forex_backtest(
    ticker: str = "EURUSD=X",
    start: str = "2024-01-01",
    end: str = "2024-03-01",
    config: Optional[MTFBacktestConfig] = None,
) -> Dict[str, object]:
    """Run end-to-end MTF backtest for one FX ticker and return frame + plot."""
    data = fetch_forex_data_yfinance(ticker=ticker, start=start, end=end)
    signals = build_mtf_signal_frame(data_5m=data["5m"], data_4h=data["4h"], config=config)
    fig = make_mtf_plot(signals, title=f"MTF Forex Backtest - {ticker}")
    return {"signals": signals, "figure": fig}
