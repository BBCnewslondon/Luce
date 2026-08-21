"""Practice-only OANDA runner for the multi-timeframe strategy."""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import yaml

from data_ingestion.oanda_client import OandaClient
from evaluation.mtf_forex_backtest import (
    MTFBacktestConfig,
    _to_oanda_instrument,
    build_mtf_signal_frame,
    fetch_forex_data_oanda,
)

logger = logging.getLogger(__name__)
EXIT_EVENTS = {"stop_exit", "take_profit_exit", "filter_exit"}
ENTRY_EVENTS = {"long_entry", "short_entry"}


@dataclass(frozen=True)
class PracticeBotConfig:
    symbols: Sequence[str]
    poll_seconds: int = 60
    lookback_days: int = 10
    units_per_trade: int = 1_000
    max_open_positions: int = 3
    dry_run: bool = True
    environment: str = "practice"

    def __post_init__(self) -> None:
        if self.environment.lower() != "practice":
            raise ValueError("PracticeBotConfig only permits OANDA practice environment")
        if not self.symbols:
            raise ValueError("At least one symbol is required")
        if self.poll_seconds < 1 or self.lookback_days < 1:
            raise ValueError("poll_seconds and lookback_days must be positive")
        if self.units_per_trade < 1 or self.max_open_positions < 1:
            raise ValueError("units_per_trade and max_open_positions must be positive")


class PracticeTradingBot:
    """Evaluate completed M5 candles and optionally submit practice orders."""

    def __init__(
        self,
        config: PracticeBotConfig,
        client: Optional[OandaClient] = None,
        strategy_config: Optional[MTFBacktestConfig] = None,
    ) -> None:
        self.config = config
        self.client = client or OandaClient()
        self.strategy_config = strategy_config or MTFBacktestConfig(
            spread_pips=0.8,
            slippage_pips=0.5,
        )
        self._last_processed_candle: dict[str, pd.Timestamp] = {}

    def _load_signal(self, symbol: str, now: datetime) -> tuple[pd.Timestamp, pd.Series]:
        end = now.replace(second=0, microsecond=0)
        start = end - timedelta(days=self.config.lookback_days)
        data = fetch_forex_data_oanda(
            ticker=symbol,
            start=start.isoformat(),
            end=end.isoformat(),
            config=self.strategy_config,
        )
        signals = build_mtf_signal_frame(
            data_5m=data["5m"],
            data_4h=data["4h"],
            config=self.strategy_config,
        )
        if signals.empty:
            raise ValueError(f"No signal data returned for {symbol}")
        candle_time = pd.Timestamp(signals.index[-1])
        return candle_time, signals.iloc[-1]

    @staticmethod
    def _position_map(positions: pd.DataFrame) -> dict[str, float]:
        if positions.empty:
            return {}
        return {
            str(row.symbol): float(row.net_units)
            for row in positions.itertuples(index=False)
            if float(row.net_units) != 0
        }

    def _submit_entry(self, symbol: str, event: str, signal: pd.Series, execute: bool) -> None:
        units = self.config.units_per_trade if event == "long_entry" else -self.config.units_per_trade
        stop = float(signal["stop_price"])
        target = float(signal["take_profit_price"])
        close = float(signal["close"])
        if not np.isfinite(stop) or not np.isfinite(target):
            logger.warning("Skipping %s: signal has no valid stop/target", symbol)
            return
        if event == "long_entry" and not (stop < close < target):
            logger.warning("Skipping %s: invalid long protection levels", symbol)
            return
        if event == "short_entry" and not (target < close < stop):
            logger.warning("Skipping %s: invalid short protection levels", symbol)
            return

        instrument = _to_oanda_instrument(symbol)
        logger.info(
            "%s %s units=%s close=%.10f stop=%.10f target=%.10f",
            "EXECUTE" if execute else "DRY RUN",
            instrument,
            units,
            close,
            stop,
            target,
        )
        if execute:
            self.client.place_market_order(
                symbol=instrument,
                units=units,
                stop_loss_price=stop,
                take_profit_price=target,
            )

    def _submit_exit(self, symbol: str, net_units: float, execute: bool) -> None:
        instrument = _to_oanda_instrument(symbol)
        logger.info("%s close %s position units=%s", "EXECUTE" if execute else "DRY RUN", instrument, net_units)
        if execute:
            if net_units > 0:
                self.client.close_position(instrument, long_units="ALL", short_units="NONE")
            else:
                self.client.close_position(instrument, long_units="NONE", short_units="ALL")

    def run_once(self, now: Optional[datetime] = None, execute: Optional[bool] = None) -> None:
        """Process each configured symbol once using the latest completed candles."""
        current_time = now or datetime.now(timezone.utc)
        should_execute = not self.config.dry_run if execute is None else execute
        positions = self._position_map(self.client.get_open_positions())
        for symbol in self.config.symbols:
            try:
                candle_time, signal = self._load_signal(symbol, current_time)
                if self._last_processed_candle.get(symbol) == candle_time:
                    continue
                self._last_processed_candle[symbol] = candle_time
                event = str(signal["signal_event"])
                net_units = positions.get(_to_oanda_instrument(symbol), 0.0)
                if event in EXIT_EVENTS and net_units:
                    self._submit_exit(symbol, net_units, should_execute)
                    positions.pop(_to_oanda_instrument(symbol), None)
                elif event in ENTRY_EVENTS and not net_units:
                    if len(positions) >= self.config.max_open_positions:
                        logger.info("Skipping %s: max open positions reached", symbol)
                        continue
                    self._submit_entry(symbol, event, signal, should_execute)
                    if should_execute:
                        positions[_to_oanda_instrument(symbol)] = (
                            self.config.units_per_trade if event == "long_entry" else -self.config.units_per_trade
                        )
            except Exception:
                logger.exception("Failed to process %s", symbol)

    def run_forever(self, execute: Optional[bool] = None) -> None:
        logger.info("Starting practice bot for %s", ", ".join(self.config.symbols))
        while True:
            self.run_once(execute=execute)
            time.sleep(self.config.poll_seconds)


def load_config(path: str = "config/settings.yaml") -> PracticeBotConfig:
    with open(path, encoding="utf-8") as stream:
        settings = yaml.safe_load(stream) or {}
    trading = settings.get("trading", {})
    risk = settings.get("risk", {})
    oanda = settings.get("oanda", {})
    return PracticeBotConfig(
        symbols=tuple(settings.get("symbols", ())),
        poll_seconds=int(trading.get("poll_seconds", 60)),
        lookback_days=int(trading.get("lookback_days", 10)),
        units_per_trade=int(trading.get("units_per_trade", 1_000)),
        max_open_positions=int(risk.get("max_open_positions", 3)),
        dry_run=bool(trading.get("dry_run", True)),
        environment=str(oanda.get("environment", "practice")),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Luce MTF strategy on OANDA practice")
    parser.add_argument("--once", action="store_true", help="process one polling cycle")
    parser.add_argument("--execute", action="store_true", help="submit practice orders; default is dry run")
    parser.add_argument("--config", default="config/settings.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    bot = PracticeTradingBot(config)
    if args.once:
        bot.run_once(execute=args.execute)
    else:
        bot.run_forever(execute=args.execute)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
