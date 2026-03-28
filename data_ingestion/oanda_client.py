"""
OANDA v20 API Client for Forex data ingestion.

Handles authentication, rate limiting, and data normalization
for hourly (H1) candlestick data.
"""

import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from loguru import logger

try:
    import oandapyV20
    from oandapyV20 import API
    from oandapyV20.endpoints import accounts, instruments, orders, positions
    from oandapyV20.exceptions import V20Error
except ImportError:
    raise ImportError("oandapyV20 is required. Install with: pip install oandapyV20")


@dataclass
class OandaConfig:
    """Configuration for OANDA API client."""

    account_id: str
    api_token: str
    environment: str = "practice"  # "practice" or "live"

    @classmethod
    def from_env(cls) -> "OandaConfig":
        """Load configuration from environment variables."""
        account_id = os.getenv("OANDA_ACCOUNT_ID")
        api_token = os.getenv("OANDA_API_TOKEN")
        environment = os.getenv("OANDA_ENVIRONMENT", "practice")

        if not account_id or not api_token:
            raise ValueError(
                "OANDA_ACCOUNT_ID and OANDA_API_TOKEN environment variables are required"
            )

        return cls(
            account_id=account_id, api_token=api_token, environment=environment
        )


class OandaClient:
    """
    OANDA v20 REST API client for Forex trading.

    Provides methods for fetching candlestick data, account info,
    and executing trades via the OANDA REST v20 API.
    """

    # API rate limits
    MAX_CANDLES_PER_REQUEST = 5000
    REQUEST_DELAY_SECONDS = 0.1  # 100ms between requests

    # Granularity mapping
    GRANULARITY_MAP = {
        "M1": "M1",
        "M5": "M5",
        "M15": "M15",
        "M30": "M30",
        "H1": "H1",
        "H4": "H4",
        "D": "D",
        "W": "W",
        "M": "M",
    }

    def __init__(self, config: Optional[OandaConfig] = None):
        """
        Initialize OANDA client.

        Args:
            config: OandaConfig instance. If None, loads from environment.
        """
        self.config = config or OandaConfig.from_env()
        self._api = self._create_api_client()
        self._last_request_time = 0.0
        logger.info(
            f"OANDA client initialized for account {self.config.account_id[:4]}***"
        )

    def _create_api_client(self) -> API:
        """Create OANDA API client instance."""
        return API(
            access_token=self.config.api_token,
            environment=self.config.environment,
        )

    def _rate_limit(self) -> None:
        """Enforce rate limiting between API requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.REQUEST_DELAY_SECONDS:
            time.sleep(self.REQUEST_DELAY_SECONDS - elapsed)
        self._last_request_time = time.time()

    def _make_request(self, endpoint) -> dict:
        """
        Make an API request with rate limiting and error handling.

        Args:
            endpoint: OANDA API endpoint object.

        Returns:
            Response data dictionary.

        Raises:
            V20Error: If the API request fails.
        """
        self._rate_limit()
        try:
            return self._api.request(endpoint)
        except V20Error as e:
            logger.error(f"OANDA API error: {e}")
            raise

    def get_account_summary(self) -> dict:
        """
        Get account summary including balance and NAV.

        Returns:
            Account summary dictionary with balance, NAV, margin info.
        """
        endpoint = accounts.AccountSummary(self.config.account_id)
        response = self._make_request(endpoint)
        return response.get("account", {})

    def get_candles(
        self,
        symbol: str,
        granularity: str = "H1",
        count: Optional[int] = None,
        from_time: Optional[datetime] = None,
        to_time: Optional[datetime] = None,
        include_spread: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch historical candlestick data for a symbol.

        Args:
            symbol: Currency pair (e.g., "EUR_USD").
            granularity: Timeframe ("M1", "M5", "H1", "D", etc.).
            count: Number of candles to fetch (max 5000).
            from_time: Start datetime (UTC).
            to_time: End datetime (UTC).
            include_spread: Whether to include bid-ask spread.

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume, spread.

        Note:
            Uses bar close time as timestamp. All times are UTC.
            Shift(1) should be applied before using in features to avoid look-ahead.
        """
        if granularity not in self.GRANULARITY_MAP:
            raise ValueError(f"Invalid granularity: {granularity}")

        params = {
            "granularity": self.GRANULARITY_MAP[granularity],
            "price": "MBA" if include_spread else "M",  # Mid, Bid, Ask
        }

        if count is not None:
            params["count"] = min(count, self.MAX_CANDLES_PER_REQUEST)
        if from_time is not None:
            params["from"] = from_time.isoformat()
        if to_time is not None:
            params["to"] = to_time.isoformat()

        endpoint = instruments.InstrumentsCandles(instrument=symbol, params=params)
        response = self._make_request(endpoint)

        candles = response.get("candles", [])
        if not candles:
            logger.warning(f"No candles returned for {symbol}")
            return pd.DataFrame()

        return self._parse_candles(candles, symbol, include_spread)

    def _parse_candles(
        self, candles: list, symbol: str, include_spread: bool
    ) -> pd.DataFrame:
        """
        Parse raw candle response into DataFrame.

        Args:
            candles: List of candle dictionaries from API.
            symbol: Currency pair symbol.
            include_spread: Whether spread data is available.

        Returns:
            Normalized DataFrame with OHLCV and spread data.
        """
        records = []
        for candle in candles:
            if not candle.get("complete", False):
                continue  # Skip incomplete candles

            mid = candle.get("mid", {})
            record = {
                "timestamp": pd.Timestamp(candle["time"]).tz_convert("UTC"),
                "symbol": symbol,
                "open": float(mid.get("o", 0)),
                "high": float(mid.get("h", 0)),
                "low": float(mid.get("l", 0)),
                "close": float(mid.get("c", 0)),
                "volume": int(candle.get("volume", 0)),
            }

            if include_spread:
                bid = candle.get("bid", {})
                ask = candle.get("ask", {})
                if bid and ask:
                    # Spread at bar close
                    record["spread"] = float(ask.get("c", 0)) - float(bid.get("c", 0))
                else:
                    record["spread"] = None

            records.append(record)

        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values("timestamp").reset_index(drop=True)

        return df

    def get_candles_range(
        self,
        symbol: str,
        granularity: str,
        from_time: datetime,
        to_time: datetime,
    ) -> pd.DataFrame:
        """
        Fetch candles for a time range, handling pagination.

        For ranges exceeding MAX_CANDLES_PER_REQUEST, makes multiple
        requests and concatenates results.

        Args:
            symbol: Currency pair.
            granularity: Timeframe.
            from_time: Start datetime (UTC).
            to_time: End datetime (UTC).

        Returns:
            Combined DataFrame of all candles in range.
        """
        all_candles = []
        current_from = from_time

        while current_from < to_time:
            df = self.get_candles(
                symbol=symbol,
                granularity=granularity,
                from_time=current_from,
                to_time=to_time,
                count=self.MAX_CANDLES_PER_REQUEST,
            )

            if df.empty:
                break

            all_candles.append(df)
            current_from = df["timestamp"].max() + pd.Timedelta(hours=1)
            logger.debug(f"Fetched {len(df)} candles for {symbol}, continuing from {current_from}")

        if not all_candles:
            return pd.DataFrame()

        result = pd.concat(all_candles, ignore_index=True)
        result = result.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        return result.reset_index(drop=True)

    def get_current_price(self, symbol: str) -> dict:
        """
        Get current bid/ask prices for a symbol.

        Args:
            symbol: Currency pair.

        Returns:
            Dictionary with bid, ask, and spread.
        """
        endpoint = instruments.InstrumentsCandles(
            instrument=symbol, params={"count": 1, "price": "BA"}
        )
        response = self._make_request(endpoint)
        candles = response.get("candles", [])

        if not candles:
            return {}

        latest = candles[-1]
        bid = float(latest.get("bid", {}).get("c", 0))
        ask = float(latest.get("ask", {}).get("c", 0))

        return {"bid": bid, "ask": ask, "spread": ask - bid, "time": latest["time"]}

    def get_open_positions(self) -> pd.DataFrame:
        """
        Get all open positions for the account.

        Returns:
            DataFrame with position details (symbol, units, unrealized P&L).
        """
        endpoint = positions.OpenPositions(self.config.account_id)
        response = self._make_request(endpoint)

        positions_data = response.get("positions", [])
        if not positions_data:
            return pd.DataFrame()

        records = []
        for pos in positions_data:
            long_units = float(pos.get("long", {}).get("units", 0))
            short_units = float(pos.get("short", {}).get("units", 0))
            long_pnl = float(pos.get("long", {}).get("unrealizedPL", 0))
            short_pnl = float(pos.get("short", {}).get("unrealizedPL", 0))

            records.append(
                {
                    "symbol": pos["instrument"],
                    "long_units": long_units,
                    "short_units": short_units,
                    "net_units": long_units + short_units,
                    "unrealized_pnl": long_pnl + short_pnl,
                }
            )

        return pd.DataFrame(records)

    def place_market_order(
        self,
        symbol: str,
        units: int,
        stop_loss_pips: Optional[float] = None,
        take_profit_pips: Optional[float] = None,
    ) -> dict:
        """
        Place a market order.

        Args:
            symbol: Currency pair.
            units: Positive for buy, negative for sell.
            stop_loss_pips: Stop loss in pips from entry.
            take_profit_pips: Take profit in pips from entry.

        Returns:
            Order response dictionary.
        """
        order_data = {
            "order": {
                "type": "MARKET",
                "instrument": symbol,
                "units": str(units),
                "timeInForce": "FOK",  # Fill or Kill
                "positionFill": "DEFAULT",
            }
        }

        # Note: In production, calculate actual price levels from pips
        # This simplified version logs the intent
        if stop_loss_pips or take_profit_pips:
            logger.info(
                f"Order with SL={stop_loss_pips} pips, TP={take_profit_pips} pips"
            )

        endpoint = orders.OrderCreate(self.config.account_id, data=order_data)
        return self._make_request(endpoint)

    def close_position(self, symbol: str, long_units: str = "ALL", short_units: str = "NONE") -> dict:
        """
        Close a position for a symbol.

        Args:
            symbol: Currency pair.
            long_units: Units to close ("ALL" or specific amount).
            short_units: Units to close ("NONE", "ALL", or specific amount).

        Returns:
            Close position response.
        """
        data = {"longUnits": long_units, "shortUnits": short_units}
        endpoint = positions.PositionClose(
            self.config.account_id, instrument=symbol, data=data
        )
        return self._make_request(endpoint)
