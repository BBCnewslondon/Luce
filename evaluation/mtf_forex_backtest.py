"""Multi-timeframe FX backtest with lag-safe 4H trend filter and 5m execution."""

from __future__ import annotations

from datetime import date, timedelta
from math import ceil
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    import pandas_ta as ta
except ImportError:  # pragma: no cover - optional runtime dependency
    ta = None


MAJOR_FX_TICKERS: Sequence[str] = (
    "EURUSD=X",
    "GBPUSD=X",
    "USDJPY=X",
    "USDCHF=X",
    "AUDUSD=X",
    "USDCAD=X",
    "NZDUSD=X",
)


@dataclass(frozen=True)
class MTFBacktestConfig:
    ema_periods: Sequence[int] = (8, 20, 50, 200)
    macd_fast: int = 8
    macd_slow: int = 21
    macd_signal: int = 5
    rsi_periods: Sequence[int] = (13, 5)
    stop_lookback: int = 5
    risk_reward_ratio: float = 3.0
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


def _validate_ohlcv(frame: pd.DataFrame, name: str) -> None:
    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(frame.columns):
        raise KeyError(f"{name} must contain columns: {sorted(required)}")
    if frame.empty:
        raise ValueError(f"{name} is empty")
    if (pd.to_numeric(frame["close"], errors="coerce") <= 0).any():
        raise ValueError(f"{name} contains non-positive close values")


def _to_oanda_instrument(symbol: str) -> str:
    """Normalize common FX symbol formats to OANDA instrument format.

    Supported inputs:
    - "EUR_USD"
    - "EURUSD"
    - "EURUSD=X"
    """
    raw = symbol.strip().upper()
    if raw.endswith("=X"):
        raw = raw[:-2]
    if "_" in raw:
        base, quote = raw.split("_", 1)
        if len(base) == 3 and len(quote) == 3:
            return f"{base}_{quote}"
    if len(raw) == 6 and raw.isalpha():
        return f"{raw[:3]}_{raw[3:]}"
    raise ValueError(f"Unsupported symbol format for OANDA: {symbol}")


def _normalize_oanda_ohlcv(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    """Normalize OANDA candle output to index=[timestamp], OHLCV columns."""
    required = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise KeyError(f"{name} missing required candle columns: {missing}")

    out = frame[required].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out = out.dropna(subset=["open", "high", "low", "close", "volume"])
    out = out.set_index("timestamp").sort_index()
    return out[["open", "high", "low", "close", "volume"]]


def _trade_event_points(signal_frame: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Split signal frame into directional entry/exit event points for plotting."""
    exits = signal_frame["signal_event"].isin(["stop_exit", "take_profit_exit", "filter_exit"])
    prev_position = signal_frame["position"].shift(1, fill_value=0.0)
    return {
        "long_entries": signal_frame[signal_frame["signal_event"] == "long_entry"],
        "short_entries": signal_frame[signal_frame["signal_event"] == "short_entry"],
        "long_exits": signal_frame[exits & (prev_position > 0)],
        "short_exits": signal_frame[exits & (prev_position < 0)],
    }


def build_trade_log(signal_frame: pd.DataFrame, ticker: Optional[str] = None) -> pd.DataFrame:
    """Build one-row-per-trade log with open/close timestamps and prices."""
    required = {"signal_event", "position", "close"}
    missing = required.difference(signal_frame.columns)
    if missing:
        raise KeyError(f"signal_frame missing required columns for trade log: {sorted(missing)}")

    records = []
    active_trade_id: Optional[int] = None
    active_side: Optional[str] = None
    active_open_time = None
    active_open_price = np.nan
    active_open_idx = -1
    entry_side = {"long_entry": "long", "short_entry": "short"}
    exit_events = {"filter_exit", "stop_exit", "take_profit_exit"}

    for i, (timestamp, row) in enumerate(signal_frame.iterrows()):
        event = row["signal_event"]

        if event in entry_side:
            if active_side is not None and active_trade_id is not None and active_open_time is not None:
                close_price = float(row["close"])
                hold_bars = i - active_open_idx
                side_sign = 1.0 if active_side == "long" else -1.0
                pnl_pct = side_sign * ((close_price / active_open_price) - 1.0)
                records.append(
                    {
                        "ticker": ticker,
                        "trade_id": active_trade_id,
                        "side": active_side,
                        "open_time": active_open_time,
                        "open_price": active_open_price,
                        "close_time": timestamp,
                        "close_price": close_price,
                        "close_event": "forced_rollover",
                        "bars_held": hold_bars,
                        "pnl_pct": pnl_pct,
                    }
                )

            raw_trade_id = row.get("trade_id", np.nan)
            trade_id = int(raw_trade_id) if pd.notna(raw_trade_id) and float(raw_trade_id) > 0 else len(records) + 1
            active_trade_id = trade_id
            active_side = entry_side[event]
            active_open_time = timestamp
            active_open_price = float(row["close"])
            active_open_idx = i
        elif event in exit_events and active_side is not None and active_trade_id is not None and active_open_time is not None:
            close_price = float(row["close"])
            hold_bars = i - active_open_idx
            side_sign = 1.0 if active_side == "long" else -1.0
            pnl_pct = side_sign * ((close_price / active_open_price) - 1.0)
            records.append(
                {
                    "ticker": ticker,
                    "trade_id": active_trade_id,
                    "side": active_side,
                    "open_time": active_open_time,
                    "open_price": active_open_price,
                    "close_time": timestamp,
                    "close_price": close_price,
                    "close_event": event,
                    "bars_held": hold_bars,
                    "pnl_pct": pnl_pct,
                }
            )
            active_trade_id = None
            active_side = None
            active_open_time = None
            active_open_price = np.nan
            active_open_idx = -1

    if active_side is not None and active_trade_id is not None and active_open_time is not None:
        last_timestamp = signal_frame.index[-1]
        last_close = float(signal_frame["close"].iloc[-1])
        hold_bars = (len(signal_frame) - 1) - active_open_idx
        side_sign = 1.0 if active_side == "long" else -1.0
        pnl_pct = side_sign * ((last_close / active_open_price) - 1.0)
        records.append(
            {
                "ticker": ticker,
                "trade_id": active_trade_id,
                "side": active_side,
                "open_time": active_open_time,
                "open_price": active_open_price,
                "close_time": last_timestamp,
                "close_price": last_close,
                "close_event": "end_of_data",
                "bars_held": hold_bars,
                "pnl_pct": pnl_pct,
            }
        )

    columns = [
        "ticker",
        "trade_id",
        "side",
        "open_time",
        "open_price",
        "close_time",
        "close_price",
        "close_event",
        "bars_held",
        "pnl_pct",
    ]
    return pd.DataFrame(records, columns=columns)


def _bullish_filter(ind: pd.DataFrame, cfg: MTFBacktestConfig) -> pd.Series:
    if len(cfg.ema_periods) < 4:
        raise ValueError("ema_periods must contain at least 4 periods")
    fast, mid_fast, mid_slow, slow = cfg.ema_periods[:4]
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
    fast, mid_fast, mid_slow, slow = cfg.ema_periods[:4]
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

    Entries are triggered only when 4H regime direction is active and
    the 5m timeframe confirms the same regime conditions.

    Returns frame with columns including:
    - signal_event: long_entry, short_entry, stop_exit, take_profit_exit, filter_exit, flat
    - position: {-1, 0, 1}
    - strategy_return, cumulative_return, drawdown
    """
    cfg = config or MTFBacktestConfig()
    _validate_ohlcv(data_5m, "data_5m")
    _validate_ohlcv(data_4h, "data_4h")
    df5 = _add_indicators(data_5m.copy(), cfg)
    df4 = _add_indicators(data_4h.copy(), cfg)
    fast, mid_fast, mid_slow, slow = cfg.ema_periods[0], cfg.ema_periods[1], cfg.ema_periods[2], cfg.ema_periods[3]
    rsi_a, rsi_b = cfg.rsi_periods

    indicator_cols = [f"ema_{p}" for p in cfg.ema_periods] + [
        "macd_line",
        "macd_signal",
        f"rsi_{cfg.rsi_periods[0]}",
        f"rsi_{cfg.rsi_periods[1]}",
    ]
    df4_lag = df4.copy()
    df4_lag[indicator_cols] = df4_lag[indicator_cols].shift(1)
    df4_lag["close"] = df4_lag["close"].shift(1)
    df4_lag["price_above_ema_fast"] = df4_lag["close"] > df4_lag[f"ema_{fast}"]
    df4_lag["price_below_ema_fast"] = df4_lag["close"] < df4_lag[f"ema_{fast}"]
    df4_lag["trend_bull"] = _bullish_filter(df4_lag, cfg) & df4_lag["price_above_ema_fast"]
    df4_lag["trend_bear"] = _bearish_filter(df4_lag, cfg) & df4_lag["price_below_ema_fast"]
    df4_lag["trend_active"] = df4_lag["trend_bull"] | df4_lag["trend_bear"]

    merge_cols = [
        "trend_bull",
        "trend_bear",
        "trend_active",
        "close",
        "price_above_ema_fast",
        "price_below_ema_fast",
        f"ema_{cfg.ema_periods[0]}",
        "macd_line",
        "macd_signal",
        f"rsi_{cfg.rsi_periods[0]}",
        f"rsi_{cfg.rsi_periods[1]}",
    ]
    df4_lag = df4_lag[merge_cols].rename(columns=lambda c: f"h4_{c}")

    df = df5.join(df4_lag.reindex(df5.index).ffill(), how="left")
    for col in ["h4_trend_bull", "h4_trend_bear", "h4_trend_active"]:
        df[col] = df[col].fillna(False).astype(bool)
    for col in ["h4_price_above_ema_fast", "h4_price_below_ema_fast"]:
        df[col] = df[col].fillna(False).astype(bool)

    df["m5_bull_confirm"] = (
        (df[f"ema_{slow}"] < df[f"ema_{mid_slow}"])
        & (df[f"ema_{mid_slow}"] < df[f"ema_{mid_fast}"])
        & (df[f"ema_{mid_fast}"] < df[f"ema_{fast}"])
        & (df["macd_line"] > df["macd_signal"])
        & (df[f"rsi_{rsi_a}"] > 50)
        & (df[f"rsi_{rsi_b}"] > 50)
        & (df["close"] > df[f"ema_{fast}"])
    )
    df["m5_bear_confirm"] = (
        (df[f"ema_{slow}"] > df[f"ema_{mid_slow}"])
        & (df[f"ema_{mid_slow}"] > df[f"ema_{mid_fast}"])
        & (df[f"ema_{mid_fast}"] > df[f"ema_{fast}"])
        & (df["macd_line"] < df["macd_signal"])
        & (df[f"rsi_{rsi_a}"] < 50)
        & (df[f"rsi_{rsi_b}"] < 50)
        & (df["close"] < df[f"ema_{fast}"])
    )

    spread_cost = (cfg.spread_pips * cfg.pip_size) / df["close"]
    long_stop = df["low"].shift(1).rolling(cfg.stop_lookback).min()
    short_stop = df["high"].shift(1).rolling(cfg.stop_lookback).max()

    position = np.zeros(len(df), dtype=float)
    event = np.full(len(df), "flat", dtype=object)
    stop_price = np.full(len(df), np.nan, dtype=float)
    take_profit_price = np.full(len(df), np.nan, dtype=float)
    trade_id = np.zeros(len(df), dtype=int)

    current_pos = 0.0
    current_trade = 0
    current_stop = np.nan
    current_take_profit = np.nan

    for i in range(len(df)):
        row = df.iloc[i]

        if current_pos == 1.0:
            hit_stop = pd.notna(current_stop) and row["low"] <= current_stop
            hit_take_profit = pd.notna(current_take_profit) and row["high"] >= current_take_profit
            if hit_stop:
                current_pos = 0.0
                current_stop = np.nan
                current_take_profit = np.nan
                event[i] = "stop_exit"
            elif hit_take_profit:
                current_pos = 0.0
                current_stop = np.nan
                current_take_profit = np.nan
                event[i] = "take_profit_exit"
            elif not row["h4_trend_bull"]:
                current_pos = 0.0
                current_stop = np.nan
                current_take_profit = np.nan
                event[i] = "filter_exit"
        elif current_pos == -1.0:
            hit_stop = pd.notna(current_stop) and row["high"] >= current_stop
            hit_take_profit = pd.notna(current_take_profit) and row["low"] <= current_take_profit
            if hit_stop:
                current_pos = 0.0
                current_stop = np.nan
                current_take_profit = np.nan
                event[i] = "stop_exit"
            elif hit_take_profit:
                current_pos = 0.0
                current_stop = np.nan
                current_take_profit = np.nan
                event[i] = "take_profit_exit"
            elif not row["h4_trend_bear"]:
                current_pos = 0.0
                current_stop = np.nan
                current_take_profit = np.nan
                event[i] = "filter_exit"
        elif row["h4_trend_bull"] and row["m5_bull_confirm"] and pd.notna(long_stop.iloc[i]):
            entry_price = float(row["close"])
            candidate_stop = float(long_stop.iloc[i])
            risk = entry_price - candidate_stop
            if risk > 0:
                current_pos = 1.0
                current_trade += 1
                current_stop = candidate_stop
                current_take_profit = entry_price + (cfg.risk_reward_ratio * risk)
                event[i] = "long_entry"
        elif row["h4_trend_bear"] and row["m5_bear_confirm"] and pd.notna(short_stop.iloc[i]):
            entry_price = float(row["close"])
            candidate_stop = float(short_stop.iloc[i])
            risk = candidate_stop - entry_price
            if risk > 0:
                current_pos = -1.0
                current_trade += 1
                current_stop = candidate_stop
                current_take_profit = entry_price - (cfg.risk_reward_ratio * risk)
                event[i] = "short_entry"

        position[i] = current_pos
        stop_price[i] = current_stop
        take_profit_price[i] = current_take_profit
        trade_id[i] = current_trade if current_pos != 0 else 0

    df["signal_event"] = event
    signal_map = {"long_entry": 1, "short_entry": -1}
    df["signal"] = df["signal_event"].map(signal_map).fillna(0).astype(int)
    df["signals"] = df["signal_event"]
    df["position"] = position
    df["stop_price"] = stop_price
    df["take_profit_price"] = take_profit_price
    df["trade_id"] = trade_id
    position_series = pd.Series(position, index=df.index)
    df["price_return"] = df["close"].pct_change().fillna(0.0)
    turnover = position_series.diff().abs().fillna(0.0)
    df["strategy_return"] = position_series.shift(1).fillna(0.0) * df["price_return"] - turnover * spread_cost
    df["equity_curve"] = (1.0 + df["strategy_return"]).cumprod()
    df["cumulative_return"] = df["equity_curve"] - 1.0
    df["drawdown"] = df["equity_curve"] / df["equity_curve"].cummax() - 1.0
    return df


def fetch_forex_data_oanda(
    ticker: str,
    start: str,
    end: str,
    config: Optional[MTFBacktestConfig] = None,
) -> Dict[str, pd.DataFrame]:
    """Fetch 5m and 4h OHLCV data from OANDA REST API.

    Raises:
        ValueError: If OANDA returns empty data for either timeframe.
    """
    # Local import keeps indicator-only unit tests independent of API credentials.
    from data_ingestion.oanda_client import OandaClient

    cfg = config or MTFBacktestConfig()
    instrument = _to_oanda_instrument(ticker)
    start_ts = pd.Timestamp(start, tz="UTC").to_pydatetime()
    end_ts = pd.Timestamp(end, tz="UTC").to_pydatetime()

    # Warm up 4h history so slow indicators (e.g., EMA200) can initialize.
    max_ema = max(cfg.ema_periods)
    warmup_h4_bars = max_ema + cfg.macd_slow + cfg.macd_signal + 20
    warmup_days = ceil((warmup_h4_bars * 4) / 24 * 7 / 5)
    h4_from_ts = (pd.Timestamp(start, tz="UTC") - pd.Timedelta(days=warmup_days)).to_pydatetime()

    client = OandaClient()
    m5_raw = client.get_candles_bulk(
        symbol=instrument,
        granularity="M5",
        from_time=start_ts,
        to_time=end_ts,
        include_spread=True,
    )
    h4_raw = client.get_candles_bulk(
        symbol=instrument,
        granularity="H4",
        from_time=h4_from_ts,
        to_time=end_ts,
        include_spread=True,
    )

    if m5_raw.empty or h4_raw.empty:
        raise ValueError(f"Empty OANDA response for ticker={ticker}, start={start}, end={end}")

    m5 = _normalize_oanda_ohlcv(m5_raw, "m5")
    h4 = _normalize_oanda_ohlcv(h4_raw, "h4")

    if m5.empty or h4.empty:
        raise ValueError(
            f"No usable OANDA OHLCV rows after normalization for ticker={ticker}, start={start}, end={end}"
        )
    return {"5m": m5, "4h": h4}


def fetch_forex_data_yfinance(
    ticker: str,
    start: str,
    end: str,
    config: Optional[MTFBacktestConfig] = None,
) -> Dict[str, pd.DataFrame]:
    """Backward-compatible alias retained for existing callers.

    Data source has been migrated to OANDA REST API.
    """
    return fetch_forex_data_oanda(ticker=ticker, start=start, end=end, config=config)


def make_mtf_plot(signal_frame: pd.DataFrame, title: str = "MTF Forex Backtest") -> go.Figure:
    """Create Plotly chart with price/EMAs on top pane and MACD histogram on lower pane."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.72, 0.28],
    )
    fig.add_trace(go.Scatter(x=signal_frame.index, y=signal_frame["close"], name="Close", mode="lines"), row=1, col=1)

    ema_cols = [col for col in signal_frame.columns if col.startswith("ema_") and col[4:].isdigit()]
    for ema_col in sorted(ema_cols, key=lambda col: int(col.split("_", 1)[1])):
        period = ema_col.split("_", 1)[1]
        fig.add_trace(
            go.Scatter(x=signal_frame.index, y=signal_frame[ema_col], name=f"EMA {period}", mode="lines"),
            row=1,
            col=1,
        )

    points = _trade_event_points(signal_frame)
    fig.add_trace(
        go.Scatter(
            x=points["long_entries"].index,
            y=points["long_entries"]["close"],
            mode="markers",
            marker={"symbol": "triangle-up", "size": 9, "color": "#2ca02c"},
            name="Long Entry",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=points["short_entries"].index,
            y=points["short_entries"]["close"],
            mode="markers",
            marker={"symbol": "triangle-down", "size": 9, "color": "#d62728"},
            name="Short Entry",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=points["long_exits"].index,
            y=points["long_exits"]["close"],
            mode="markers",
            marker={"symbol": "x", "size": 9, "color": "#1f77b4"},
            name="Long Exit",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=points["short_exits"].index,
            y=points["short_exits"]["close"],
            mode="markers",
            marker={"symbol": "x", "size": 9, "color": "#9467bd"},
            name="Short Exit",
        ),
        row=1,
        col=1,
    )

    if {"macd_line", "macd_signal"}.issubset(signal_frame.columns):
        macd_hist = signal_frame["macd_line"] - signal_frame["macd_signal"]
        hist_colors = np.where(macd_hist >= 0, "#2ca02c", "#d62728")
        fig.add_trace(
            go.Bar(
                x=signal_frame.index,
                y=macd_hist,
                name="MACD Histogram",
                marker_color=hist_colors,
                opacity=0.35,
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=signal_frame.index, y=signal_frame["macd_line"], name="MACD", mode="lines"),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=signal_frame.index, y=signal_frame["macd_signal"], name="MACD Signal", mode="lines"),
            row=2,
            col=1,
        )
        fig.add_hline(y=0.0, line_width=1, line_dash="dot", line_color="gray", row=2, col=1)

    active = signal_frame["h4_trend_active"].astype(bool)
    starts = signal_frame.index[(active) & (~active.shift(1, fill_value=False))]
    ends = signal_frame.index[(~active) & (active.shift(1, fill_value=False))]
    if active.iloc[-1]:
        ends = ends.union(pd.Index([signal_frame.index[-1]]))

    for start, end in zip(starts, ends):
        fig.add_vrect(
            x0=start,
            x1=end,
            fillcolor="LightGreen",
            opacity=0.15,
            layer="below",
            line_width=0,
            row=1,
            col=1,
        )

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="MACD", row=2, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_layout(title=title, template="plotly_white")
    return fig


def make_major_pairs_subplots(
    signals_by_ticker: Dict[str, pd.DataFrame],
    title: str = "MTF Forex Backtest - Major Pairs",
) -> go.Figure:
    """Create a single figure with one price subplot per FX pair."""
    if not signals_by_ticker:
        raise ValueError("signals_by_ticker is empty")

    tickers = list(signals_by_ticker.keys())
    fig = make_subplots(
        rows=len(tickers),
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.03,
        subplot_titles=tickers,
    )
    legend_shown = {
        "close": False,
        "ema8": False,
        "long_entry": False,
        "short_entry": False,
        "long_exit": False,
        "short_exit": False,
    }

    for row, ticker in enumerate(tickers, start=1):
        frame = signals_by_ticker[ticker]
        points = _trade_event_points(frame)
        fig.add_trace(
            go.Scatter(
                x=frame.index,
                y=frame["close"],
                mode="lines",
                name="Close",
                legendgroup="close",
                showlegend=not legend_shown["close"],
            ),
            row=row,
            col=1,
        )
        legend_shown["close"] = True

        if "ema_8" in frame.columns:
            fig.add_trace(
                go.Scatter(
                    x=frame.index,
                    y=frame["ema_8"],
                    mode="lines",
                    name="EMA 8",
                    legendgroup="ema8",
                    showlegend=not legend_shown["ema8"],
                ),
                row=row,
                col=1,
            )
            legend_shown["ema8"] = True

        if not points["long_entries"].empty:
            fig.add_trace(
                go.Scatter(
                    x=points["long_entries"].index,
                    y=points["long_entries"]["close"],
                    mode="markers",
                    marker={"symbol": "triangle-up", "size": 7, "color": "#2ca02c"},
                    name="Long Entry",
                    legendgroup="long_entry",
                    showlegend=not legend_shown["long_entry"],
                ),
                row=row,
                col=1,
            )
            legend_shown["long_entry"] = True
        if not points["short_entries"].empty:
            fig.add_trace(
                go.Scatter(
                    x=points["short_entries"].index,
                    y=points["short_entries"]["close"],
                    mode="markers",
                    marker={"symbol": "triangle-down", "size": 7, "color": "#d62728"},
                    name="Short Entry",
                    legendgroup="short_entry",
                    showlegend=not legend_shown["short_entry"],
                ),
                row=row,
                col=1,
            )
            legend_shown["short_entry"] = True
        if not points["long_exits"].empty:
            fig.add_trace(
                go.Scatter(
                    x=points["long_exits"].index,
                    y=points["long_exits"]["close"],
                    mode="markers",
                    marker={"symbol": "x", "size": 7, "color": "#1f77b4"},
                    name="Long Exit",
                    legendgroup="long_exit",
                    showlegend=not legend_shown["long_exit"],
                ),
                row=row,
                col=1,
            )
            legend_shown["long_exit"] = True
        if not points["short_exits"].empty:
            fig.add_trace(
                go.Scatter(
                    x=points["short_exits"].index,
                    y=points["short_exits"]["close"],
                    mode="markers",
                    marker={"symbol": "x", "size": 7, "color": "#9467bd"},
                    name="Short Exit",
                    legendgroup="short_exit",
                    showlegend=not legend_shown["short_exit"],
                ),
                row=row,
                col=1,
            )
            legend_shown["short_exit"] = True

        fig.update_yaxes(title_text=ticker, row=row, col=1)

    fig.update_xaxes(title_text="Time", row=len(tickers), col=1)
    fig.update_layout(height=max(340 * len(tickers), 900), title=title, template="plotly_white")
    return fig


def run_mtf_major_pairs_backtest(
    tickers: Sequence[str] = MAJOR_FX_TICKERS,
    start: Optional[str] = None,
    end: Optional[str] = None,
    config: Optional[MTFBacktestConfig] = None,
) -> Dict[str, object]:
    """Run the MTF backtest on major FX pairs and return summary + subplot figure.

    By default uses the most recent 45 days.
    """
    end_date = pd.to_datetime(end).date() if end is not None else date.today()
    start_date = pd.to_datetime(start).date() if start is not None else (end_date - timedelta(days=45))
    start_str = start_date.isoformat()
    end_str = end_date.isoformat()

    cfg = config or MTFBacktestConfig()
    signals_by_ticker: Dict[str, pd.DataFrame] = {}
    trade_logs_by_ticker: Dict[str, pd.DataFrame] = {}
    summary_rows = []

    for ticker in tickers:
        try:
            data = fetch_forex_data_oanda(ticker=ticker, start=start_str, end=end_str, config=cfg)
            signals = build_mtf_signal_frame(data_5m=data["5m"], data_4h=data["4h"], config=cfg)
            signals_by_ticker[ticker] = signals
            trade_log = build_trade_log(signals, ticker=ticker)
            trade_logs_by_ticker[ticker] = trade_log

            entries = int(signals["signal_event"].isin(["long_entry", "short_entry"]).sum())
            summary_rows.append(
                {
                    "ticker": ticker,
                    "rows": int(len(signals)),
                    "entries": entries,
                    "closed_trades": int(len(trade_log)),
                    "total_return": float(signals["cumulative_return"].iloc[-1]),
                    "max_drawdown": float(signals["drawdown"].min()),
                    "error": None,
                }
            )
        except Exception as exc:  # pragma: no cover - runtime/data-path dependent
            summary_rows.append(
                {
                    "ticker": ticker,
                    "rows": 0,
                    "entries": 0,
                    "closed_trades": 0,
                    "total_return": np.nan,
                    "max_drawdown": np.nan,
                    "error": str(exc),
                }
            )

    summary = pd.DataFrame(summary_rows)
    if trade_logs_by_ticker:
        trade_log = pd.concat(trade_logs_by_ticker.values(), ignore_index=True)
    else:
        trade_log = pd.DataFrame(
            columns=[
                "ticker",
                "trade_id",
                "side",
                "open_time",
                "open_price",
                "close_time",
                "close_price",
                "close_event",
                "bars_held",
                "pnl_pct",
            ]
        )
    if signals_by_ticker:
        figure = make_major_pairs_subplots(
            signals_by_ticker,
            title=f"MTF Forex Backtest - Major Pairs ({start_str} to {end_str})",
        )
    else:  # pragma: no cover - runtime/data-path dependent
        figure = go.Figure()
        figure.add_annotation(
            text="No successful pair downloads for selected window.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )
        figure.update_layout(title=f"MTF Forex Backtest - Major Pairs ({start_str} to {end_str})")
    return {
        "signals_by_ticker": signals_by_ticker,
        "trade_logs_by_ticker": trade_logs_by_ticker,
        "trade_log": trade_log,
        "summary": summary,
        "figure": figure,
        "start": start_str,
        "end": end_str,
    }


def run_mtf_forex_backtest(
    ticker: str = "EUR_USD",
    start: str = "2024-01-01",
    end: str = "2024-03-01",
    config: Optional[MTFBacktestConfig] = None,
) -> Dict[str, object]:
    """Run end-to-end MTF backtest for one FX ticker and return frame + plot."""
    data = fetch_forex_data_oanda(ticker=ticker, start=start, end=end, config=config)
    signals = build_mtf_signal_frame(data_5m=data["5m"], data_4h=data["4h"], config=config)
    trade_log = build_trade_log(signals, ticker=ticker)
    fig = make_mtf_plot(signals, title=f"MTF Forex Backtest - {ticker}")
    return {"signals": signals, "trade_log": trade_log, "figure": fig}
