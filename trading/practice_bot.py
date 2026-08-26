"""Practice-only OANDA runner for the multi-timeframe strategy."""

from __future__ import annotations

import argparse
import csv
import logging
import logging.handlers
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
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


def _setup_logging(log_dir: str = "logs") -> None:
    """Configure logging to both console and file."""
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Main bot log file
    log_file = log_path / "bot.log"
    
    # Setup file handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=5
    )
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Suppress verbose third-party logs
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("oandapyV20").setLevel(logging.WARNING)


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
        log_dir: str = "logs",
    ) -> None:
        self.config = config
        self.client = client or OandaClient()
        self.strategy_config = strategy_config or MTFBacktestConfig(
            spread_pips=0.8,
            slippage_pips=0.5,
        )
        self._last_processed_candle: dict[str, pd.Timestamp] = {}
        
        # Setup trade logging
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.trades_csv = self.log_dir / "trades.csv"
        self._ensure_trades_csv()

    def _ensure_trades_csv(self) -> None:
        """Ensure trades.csv exists with headers."""
        if not self.trades_csv.exists():
            with open(self.trades_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "symbol", "side", "units", "entry_price",
                    "stop_loss", "take_profit", "status", "mode"
                ])

    def _log_trade(
        self,
        symbol: str,
        side: str,
        units: int,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        status: str,
        mode: str = "dry_run",
    ) -> None:
        """Log a trade to trades.csv."""
        timestamp = datetime.now(timezone.utc).isoformat()
        with open(self.trades_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                symbol,
                side,
                units,
                f"{entry_price:.10f}",
                f"{stop_loss:.10f}",
                f"{take_profit:.10f}",
                status,
                mode,
            ])
        logger.info(
            "Trade logged: %s %s %d units @ %.5f (SL: %.5f TP: %.5f) [%s]",
            symbol, side, units, entry_price, stop_loss, take_profit, status
        )

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
        side = "LONG" if event == "long_entry" else "SHORT"
        mode = "EXECUTE" if execute else "DRY_RUN"
        
        logger.info(
            "%s %s %s units=%s close=%.10f stop=%.10f target=%.10f",
            mode,
            instrument,
            side,
            units,
            close,
            stop,
            target,
        )
        
        # Log trade
        self._log_trade(
            symbol=symbol,
            side=side,
            units=units,
            entry_price=close,
            stop_loss=stop,
            take_profit=target,
            status="SUBMITTED" if execute else "SIGNAL",
            mode="execute" if execute else "dry_run",
        )
        
        if execute:
            self.client.place_market_order(
                symbol=instrument,
                units=units,
                stop_loss_price=stop,
                take_profit_price=target,
            )
            logger.info("Order submitted for %s: %d units", instrument, units)

    def _submit_exit(self, symbol: str, net_units: float, execute: bool) -> None:
        instrument = _to_oanda_instrument(symbol)
        side = "LONG" if net_units > 0 else "SHORT"
        mode = "EXECUTE" if execute else "DRY_RUN"
        
        logger.info(
            "%s close %s position (side=%s units=%s)",
            mode,
            instrument,
            side,
            net_units
        )
        
        # Log exit
        self._log_trade(
            symbol=symbol,
            side=f"{side}_EXIT",
            units=int(net_units),
            entry_price=0.0,  # Exit price not available yet
            stop_loss=0.0,
            take_profit=0.0,
            status="CLOSED" if execute else "EXIT_SIGNAL",
            mode="execute" if execute else "dry_run",
        )
        
        if execute:
            if net_units > 0:
                self.client.close_position(instrument, long_units="ALL", short_units="NONE")
            else:
                self.client.close_position(instrument, long_units="NONE", short_units="ALL")
            logger.info("Position closed for %s", instrument)

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
    parser.add_argument("--log-dir", default="logs", help="directory for logs and trade tracking")
    args = parser.parse_args()
    
    # Setup logging
    _setup_logging(args.log_dir)
    logger.info("Starting Luce Trading Bot (dry_run=%s)", not args.execute)
    
    config = load_config(args.config)
    bot = PracticeTradingBot(config, log_dir=args.log_dir)
    
    logger.info("Configuration loaded: %d symbols, poll_seconds=%d", len(config.symbols), config.poll_seconds)
    
    if args.once:
        logger.info("Running single cycle mode")
        bot.run_once(execute=args.execute)
    else:
        logger.info("Running continuous mode (polling every %d seconds)", config.poll_seconds)
        bot.run_forever(execute=args.execute)


if __name__ == "__main__":
    main()
