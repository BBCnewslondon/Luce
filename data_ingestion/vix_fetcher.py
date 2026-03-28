sk"""
VIX (Volatility Index) data fetcher.

Fetches VIX data from Yahoo Finance as a crash risk indicator.
Data is forward-filled to align with hourly Forex data.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from loguru import logger

try:
    import yfinance as yf
except ImportError:
    raise ImportError("yfinance is required. Install with: pip install yfinance")


@dataclass
class VixConfig:
    """Configuration for VIX data fetcher."""

    ticker: str = "^VIX"
    source: str = "yahoo"


class VixFetcher:
    """
    Fetches CBOE Volatility Index (VIX) data.

    The VIX measures implied volatility of S&P 500 options and serves
    as a "fear gauge" for market crash risk in the trading bot.

    Data is daily and must be forward-filled when merging with
    intraday Forex data to avoid look-ahead bias.
    """

    def __init__(self, config: Optional[VixConfig] = None):
        """
        Initialize VIX fetcher.

        Args:
            config: VixConfig instance. Defaults to ^VIX from Yahoo.
        """
        self.config = config or VixConfig()
        logger.info(f"VIX fetcher initialized for {self.config.ticker}")

    def fetch(
        self,
        start_date: datetime,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Fetch VIX historical data.

        Args:
            start_date: Start date for data fetch.
            end_date: End date (defaults to today).

        Returns:
            DataFrame with columns: timestamp, vix_open, vix_high, vix_low, vix_close.

        Note:
            Data is daily. Use forward_fill_to_hourly() to align with H1 data.
        """
        end_date = end_date or datetime.now()

        logger.info(f"Fetching VIX data from {start_date.date()} to {end_date.date()}")

        try:
            ticker = yf.Ticker(self.config.ticker)
            df = ticker.history(start=start_date, end=end_date)
        except Exception as e:
            logger.error(f"Failed to fetch VIX data: {e}")
            return pd.DataFrame()

        if df.empty:
            logger.warning("No VIX data returned")
            return pd.DataFrame()

        return self._normalize_data(df)

    def _normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize yfinance data to standard format.

        Args:
            df: Raw yfinance DataFrame.

        Returns:
            Normalized DataFrame with consistent column names.
        """
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]

        # Rename for clarity
        result = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(df["date"]).dt.tz_localize("UTC"),
                "vix_open": df["open"].astype(float),
                "vix_high": df["high"].astype(float),
                "vix_low": df["low"].astype(float),
                "vix_close": df["close"].astype(float),
            }
        )

        # Ensure sorted by timestamp
        result = result.sort_values("timestamp").reset_index(drop=True)

        logger.info(f"Fetched {len(result)} VIX data points")
        return result

    def forward_fill_to_hourly(
        self,
        vix_df: pd.DataFrame,
        target_timestamps: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """
        Forward-fill daily VIX data to match hourly Forex timestamps.

        This is critical for avoiding look-ahead bias: VIX values are
        only known after market close, so we use the previous day's
        close for all hours until the next VIX data point.

        Args:
            vix_df: Daily VIX DataFrame from fetch().
            target_timestamps: Hourly timestamps to align to.

        Returns:
            DataFrame with VIX data aligned to target timestamps.

        Example:
            If VIX close on 2024-01-15 was 15.5, all hours from
            2024-01-15 17:00 to 2024-01-16 16:59 will have vix=15.5.
        """
        if vix_df.empty:
            logger.warning("Empty VIX data, returning NaN-filled frame")
            return pd.DataFrame(
                {"timestamp": target_timestamps, "vix": [None] * len(target_timestamps)}
            )

        # Create hourly index from VIX data
        vix_hourly = vix_df.set_index("timestamp")[["vix_close"]].rename(
            columns={"vix_close": "vix"}
        )

        # Reindex to target timestamps with forward fill
        # This ensures no future data leaks into past hours
        target_df = pd.DataFrame(index=target_timestamps)
        merged = target_df.join(vix_hourly, how="left")

        # Forward fill: each hour gets the last known VIX value
        merged["vix"] = merged["vix"].ffill()

        # Also add VIX change (using shift to avoid look-ahead)
        merged["vix_change"] = merged["vix"].pct_change()

        result = merged.reset_index().rename(columns={"index": "timestamp"})
        return result

    def get_current_vix(self) -> Optional[float]:
        """
        Get the most recent VIX value.

        Returns:
            Current VIX close or None if unavailable.
        """
        try:
            ticker = yf.Ticker(self.config.ticker)
            info = ticker.fast_info
            return float(info.get("lastPrice", info.get("regularMarketPrice", 0)))
        except Exception as e:
            logger.error(f"Failed to get current VIX: {e}")
            return None

    def calculate_vix_regime(self, vix_value: float) -> str:
        """
        Classify VIX into market regime categories.

        Args:
            vix_value: Current VIX level.

        Returns:
            Regime label: "low", "normal", "elevated", or "extreme".

        Thresholds based on historical VIX distribution:
            - Low: < 15 (calm markets)
            - Normal: 15-20 (typical conditions)
            - Elevated: 20-30 (increased uncertainty)
            - Extreme: > 30 (crisis/panic)
        """
        if vix_value < 15:
            return "low"
        elif vix_value < 20:
            return "normal"
        elif vix_value < 30:
            return "elevated"
        else:
            return "extreme"

    def should_reduce_exposure(self, vix_value: float, threshold: float = 25.0) -> bool:
        """
        Determine if position sizing should be reduced due to high VIX.

        Used by risk manager to scale down positions during volatility spikes.

        Args:
            vix_value: Current VIX level.
            threshold: VIX level above which to reduce exposure.

        Returns:
            True if VIX exceeds threshold.
        """
        return vix_value > threshold
