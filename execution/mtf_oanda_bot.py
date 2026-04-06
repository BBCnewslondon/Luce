"""Single-cycle OANDA execution bot driven by the existing MTF strategy."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Callable, Optional

import pandas as pd


def _to_oanda_instrument(symbol: str) -> str:
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


@dataclass(frozen=True)
class MTFOandaBotConfig:
    symbol: str = "EUR_USD"
    order_units: int = 1000
    lookback_days: int = 45
    dry_run: bool = True
    strategy_config: Any = None


@dataclass
class MTFOandaTradingBot:
    config: MTFOandaBotConfig
    oanda_client: Any
    data_fetcher: Optional[Callable[..., dict]] = None
    signal_builder: Optional[Callable[..., pd.DataFrame]] = None

    def __post_init__(self) -> None:
        if self.config.order_units <= 0:
            raise ValueError("order_units must be a positive integer")
        if self.data_fetcher is None or self.signal_builder is None:
            from evaluation.mtf_forex_backtest import build_mtf_signal_frame, fetch_forex_data_oanda

            if self.data_fetcher is None:
                self.data_fetcher = fetch_forex_data_oanda
            if self.signal_builder is None:
                self.signal_builder = build_mtf_signal_frame

    @staticmethod
    def _net_units_for_symbol(open_positions: pd.DataFrame, symbol: str) -> int:
        """Return broker net units for symbol from a point-in-time position snapshot."""
        if open_positions.empty:
            return 0
        scoped = open_positions[open_positions["symbol"] == symbol]
        if scoped.empty:
            return 0
        return int(float(scoped["net_units"].sum()))

    @staticmethod
    def _validate_positions_snapshot_freshness(open_positions: pd.DataFrame, now_ts: pd.Timestamp) -> None:
        """Validate optional position timestamp column if present to reduce stale-state risk."""
        if "timestamp" not in open_positions.columns or open_positions.empty:
            return
        ts = pd.to_datetime(open_positions["timestamp"], utc=True, errors="coerce").dropna()
        if ts.empty:
            return
        latest = ts.max()
        if latest < now_ts - pd.Timedelta(days=1):
            raise ValueError(f"Open position snapshot appears stale: latest timestamp={latest.isoformat()}")

    def run_cycle(self, now: Optional[pd.Timestamp] = None) -> dict:
        """Run one live-trading decision cycle and reconcile broker position to latest MTF signal.

        Args:
            now: Optional UTC timestamp used for deterministic testing.

        Returns:
            Dictionary with decision metadata (signal, target/current units, selected action).

        Raises:
            ValueError: If signal frame is empty or position snapshot appears stale.
        """
        now_ts = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now).tz_convert("UTC")
        start_ts = now_ts - timedelta(days=self.config.lookback_days)
        instrument = _to_oanda_instrument(self.config.symbol)

        data = self.data_fetcher(
            ticker=instrument,
            start=start_ts.isoformat(),
            end=now_ts.isoformat(),
            config=self.config.strategy_config,
        )
        signal_frame = self.signal_builder(
            data_5m=data["5m"],
            data_4h=data["4h"],
            config=self.config.strategy_config,
        )
        if signal_frame.empty:
            raise ValueError(
                "MTF signal frame is empty; cannot execute trading cycle for "
                f"symbol={instrument}, start={start_ts.isoformat()}, end={now_ts.isoformat()}. "
                "Check OANDA connectivity and data availability."
            )

        latest = signal_frame.iloc[-1]
        latest_event = str(latest.get("signal_event", "flat"))
        desired_position = int(latest.get("position", 0))

        current_positions = self.oanda_client.get_open_positions()
        self._validate_positions_snapshot_freshness(current_positions, now_ts)
        current_units = self._net_units_for_symbol(current_positions, instrument)
        target_units = self.config.order_units * desired_position

        action = "hold"
        if target_units == 0 and current_units != 0:
            action = "close_position"
            if not self.config.dry_run:
                self.oanda_client.close_position(symbol=instrument, long_units="ALL", short_units="ALL")
        elif target_units != 0 and current_units != target_units:
            action = "rebalance_position"
            if not self.config.dry_run:
                if current_units != 0:
                    self.oanda_client.close_position(symbol=instrument, long_units="ALL", short_units="ALL")
                self.oanda_client.place_market_order(symbol=instrument, units=target_units)

        return {
            "symbol": instrument,
            "timestamp": now_ts.isoformat(),
            "latest_signal_event": latest_event,
            "desired_position": desired_position,
            "target_units": target_units,
            "current_units": current_units,
            "action": action,
            "dry_run": self.config.dry_run,
        }


def _read_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_oanda_client_cls():
    from data_ingestion.oanda_client import OandaClient

    return OandaClient


def run_from_env() -> dict:
    cfg = MTFOandaBotConfig(
        symbol=os.getenv("BOT_SYMBOL", "EUR_USD"),
        order_units=int(os.getenv("BOT_ORDER_UNITS", "1000")),
        lookback_days=int(os.getenv("BOT_LOOKBACK_DAYS", "45")),
        dry_run=_read_bool_env("BOT_DRY_RUN", True),
    )
    bot = MTFOandaTradingBot(config=cfg, oanda_client=_get_oanda_client_cls()())
    return bot.run_cycle()


if __name__ == "__main__":
    print(run_from_env())
