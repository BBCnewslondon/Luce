"""
Feature Store with point-in-time semantics.

Stores and retrieves time-series data with strict temporal integrity
to prevent look-ahead bias in backtesting and ML training.
"""

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger


@dataclass
class FeatureStoreConfig:
    """Configuration for feature store."""

    base_path: str = "./data/features"
    partition_by: str = "symbol"  # Partition strategy


class FeatureStore:
    """
    Time-series feature store with point-in-time query support.

    Stores OHLCV data, technical indicators, and alternative data
    in Parquet format with partitioning by symbol. Provides
    point-in-time queries to prevent look-ahead bias.

    Directory structure:
        base_path/
            symbol=EUR_USD/
                data.parquet
            symbol=GBP_USD/
                data.parquet
            ...
            metadata.json
    """

    def __init__(self, config: Optional[FeatureStoreConfig] = None):
        """
        Initialize feature store.

        Args:
            config: FeatureStoreConfig instance.
        """
        self.config = config or FeatureStoreConfig()
        self.base_path = Path(self.config.base_path)
        self._ensure_directory()
        logger.info(f"Feature store initialized at {self.base_path}")

    def _ensure_directory(self) -> None:
        """Create base directory if it doesn't exist."""
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _get_symbol_path(self, symbol: str) -> Path:
        """Get path for symbol's data file."""
        return self.base_path / f"symbol={symbol}" / "data.parquet"

    def store(
        self,
        df: pd.DataFrame,
        symbol: str,
        mode: str = "append",
    ) -> None:
        """
        Store feature data for a symbol.

        Args:
            df: DataFrame with timestamp index or column.
            symbol: Currency pair symbol.
            mode: "append" to add to existing, "overwrite" to replace.

        Raises:
            ValueError: If DataFrame lacks required columns.
        """
        if df.empty:
            logger.warning(f"Empty DataFrame provided for {symbol}, skipping store")
            return

        # Ensure timestamp column exists
        if "timestamp" not in df.columns:
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index().rename(columns={"index": "timestamp"})
            else:
                raise ValueError("DataFrame must have 'timestamp' column or DatetimeIndex")

        # Ensure timestamp is UTC
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_convert("UTC")

        # Sort by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Add symbol column if not present
        if "symbol" not in df.columns:
            df["symbol"] = symbol

        symbol_path = self._get_symbol_path(symbol)
        symbol_path.parent.mkdir(parents=True, exist_ok=True)

        if mode == "append" and symbol_path.exists():
            existing_df = self.load(symbol)
            df = pd.concat([existing_df, df], ignore_index=True)
            df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")

        # Write to parquet
        table = pa.Table.from_pandas(df)
        pq.write_table(table, symbol_path, compression="snappy")

        logger.info(f"Stored {len(df)} rows for {symbol}")

    def load(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Load feature data for a symbol.

        Args:
            symbol: Currency pair symbol.
            start_time: Filter to data after this time.
            end_time: Filter to data before this time.
            columns: Specific columns to load (None = all).

        Returns:
            DataFrame with requested data.
        """
        symbol_path = self._get_symbol_path(symbol)

        if not symbol_path.exists():
            logger.warning(f"No data found for {symbol}")
            return pd.DataFrame()

        # Read parquet with optional column selection
        table = pq.read_table(symbol_path, columns=columns)
        df = table.to_pandas()

        # Apply time filters
        if start_time is not None:
            start_ts = pd.Timestamp(start_time).tz_localize("UTC")
            df = df[df["timestamp"] >= start_ts]

        if end_time is not None:
            end_ts = pd.Timestamp(end_time).tz_localize("UTC")
            df = df[df["timestamp"] <= end_ts]

        return df.sort_values("timestamp").reset_index(drop=True)

    def load_point_in_time(
        self,
        symbol: str,
        as_of_time: datetime,
        lookback_bars: int = 500,
    ) -> pd.DataFrame:
        """
        Load data as it would have been available at a specific time.

        CRITICAL for avoiding look-ahead bias: Only returns data
        with timestamps strictly before as_of_time.

        Args:
            symbol: Currency pair symbol.
            as_of_time: Point in time for the query.
            lookback_bars: Number of bars to return.

        Returns:
            DataFrame with data available as of as_of_time.
        """
        as_of_ts = pd.Timestamp(as_of_time).tz_localize("UTC")

        df = self.load(symbol, end_time=as_of_time)

        if df.empty:
            return df

        # Strictly exclude the as_of_time (current bar is not complete)
        df = df[df["timestamp"] < as_of_ts]

        # Take last N bars
        if len(df) > lookback_bars:
            df = df.tail(lookback_bars)

        return df.reset_index(drop=True)

    def load_multiple_symbols(
        self,
        symbols: List[str],
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Load data for multiple symbols.

        Args:
            symbols: List of currency pairs.
            start_time: Filter to data after this time.
            end_time: Filter to data before this time.

        Returns:
            Dictionary mapping symbol to DataFrame.
        """
        result = {}
        for symbol in symbols:
            df = self.load(symbol, start_time, end_time)
            if not df.empty:
                result[symbol] = df
        return result

    def get_latest_timestamp(self, symbol: str) -> Optional[datetime]:
        """
        Get the most recent timestamp for a symbol.

        Useful for incremental updates.

        Args:
            symbol: Currency pair symbol.

        Returns:
            Latest timestamp or None if no data.
        """
        symbol_path = self._get_symbol_path(symbol)

        if not symbol_path.exists():
            return None

        # Read only timestamp column for efficiency
        table = pq.read_table(symbol_path, columns=["timestamp"])
        df = table.to_pandas()

        if df.empty:
            return None

        return df["timestamp"].max()

    def get_available_symbols(self) -> List[str]:
        """
        Get list of symbols with stored data.

        Returns:
            List of symbol names.
        """
        symbols = []
        for path in self.base_path.iterdir():
            if path.is_dir() and path.name.startswith("symbol="):
                symbol = path.name.replace("symbol=", "")
                symbols.append(symbol)
        return sorted(symbols)

    def get_data_summary(self) -> pd.DataFrame:
        """
        Get summary statistics for all stored data.

        Returns:
            DataFrame with symbol, row count, date range, columns.
        """
        summaries = []
        for symbol in self.get_available_symbols():
            df = self.load(symbol)
            if not df.empty:
                summaries.append({
                    "symbol": symbol,
                    "rows": len(df),
                    "start_date": df["timestamp"].min(),
                    "end_date": df["timestamp"].max(),
                    "columns": len(df.columns),
                    "size_mb": self._get_symbol_path(symbol).stat().st_size / (1024 * 1024),
                })
        return pd.DataFrame(summaries)

    def merge_alternative_data(
        self,
        ohlcv_df: pd.DataFrame,
        vix_df: Optional[pd.DataFrame] = None,
        cot_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Merge OHLCV data with alternative data sources.

        Performs point-in-time correct merging using asof joins
        to prevent look-ahead bias.

        Args:
            ohlcv_df: Base OHLCV DataFrame with timestamp.
            vix_df: VIX data (daily, forward-filled).
            cot_df: COT data (weekly, with 3-day lag already applied).

        Returns:
            Merged DataFrame with all features.
        """
        result = ohlcv_df.copy()

        # Ensure timestamp is the index for merge_asof
        result = result.sort_values("timestamp")

        if vix_df is not None and not vix_df.empty:
            vix_df = vix_df.sort_values("timestamp")
            result = pd.merge_asof(
                result,
                vix_df[["timestamp", "vix", "vix_change"]],
                on="timestamp",
                direction="backward",  # Only use past data
            )
            logger.debug("Merged VIX data")

        if cot_df is not None and not cot_df.empty:
            cot_df = cot_df.sort_values("timestamp")
            cot_cols = ["timestamp", "cot_net_long", "cot_change", "cot_net_pct"]
            cot_cols = [c for c in cot_cols if c in cot_df.columns]
            result = pd.merge_asof(
                result,
                cot_df[cot_cols],
                on="timestamp",
                direction="backward",  # Only use past data
            )
            logger.debug("Merged COT data")

        return result

    def delete_symbol(self, symbol: str) -> bool:
        """
        Delete all data for a symbol.

        Args:
            symbol: Currency pair symbol.

        Returns:
            True if deleted, False if not found.
        """
        symbol_path = self._get_symbol_path(symbol)
        if symbol_path.exists():
            symbol_path.unlink()
            symbol_path.parent.rmdir()
            logger.info(f"Deleted data for {symbol}")
            return True
        return False

    def vacuum(self) -> None:
        """
        Clean up and optimize stored data.

        Removes duplicates and re-compresses parquet files.
        """
        for symbol in self.get_available_symbols():
            df = self.load(symbol)
            if not df.empty:
                # Remove duplicates
                original_len = len(df)
                df = df.drop_duplicates(subset=["timestamp"])

                if len(df) < original_len:
                    self.store(df, symbol, mode="overwrite")
                    logger.info(f"Vacuumed {symbol}: removed {original_len - len(df)} duplicates")
