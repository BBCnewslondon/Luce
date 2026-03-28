"""Data ingestion orchestration for historical feature dataset construction."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import pandas as pd

@dataclass(frozen=True)
class IngestionRequest:
    """Input contract for historical ingestion."""

    symbol: str
    granularity: str
    from_time: datetime
    to_time: datetime


def build_historical_feature_frame(
    request: IngestionRequest,
    oanda_client: Any,
    feature_engine: Any,
    vix_fetcher: Optional[Any] = None,
    cot_fetcher: Optional[Any] = None,
    sentiment_fetcher: Optional[Any] = None,
    include_order_book: bool = True,
) -> pd.DataFrame:
    """
    Fetch market/alternative data and generate lag-safe feature frame.

    Args:
        request: IngestionRequest containing symbol and date range.
        oanda_client: Configured OANDA client.
        feature_engine: Feature engine with TA-Lib indicators.
        vix_fetcher: Optional VIX fetcher.
        cot_fetcher: Optional COT fetcher.
        include_order_book: Include OANDA order-book positioning features.

    Returns:
        Feature DataFrame with timestamp, symbol, indicators, alt-data, and lagged factors.

    Example:
        >>> # Pseudocode usage
        >>> # req = IngestionRequest(symbol="EUR_USD", granularity="H1", from_time=start, to_time=end)
        >>> # features = build_historical_feature_frame(req, oanda, FeatureEngine())
        >>> # "rsi" in features.columns
        >>> # True
    """
    candles = oanda_client.get_candles_bulk(
        symbol=request.symbol,
        granularity=request.granularity,
        from_time=request.from_time,
        to_time=request.to_time,
        include_spread=True,
    )
    if candles.empty:
        return candles

    frame = feature_engine.calculate_all_features(candles)

    if include_order_book:
        order_book = oanda_client.get_order_book_range(request.symbol, pd.DatetimeIndex(frame["timestamp"]))
        frame = frame.merge(order_book, on=["timestamp", "symbol"], how="left")

    if vix_fetcher is not None:
        vix_daily = vix_fetcher.fetch(request.from_time, request.to_time)
        vix_hourly = vix_fetcher.forward_fill_to_hourly(vix_daily, pd.DatetimeIndex(frame["timestamp"]))
        frame = frame.merge(vix_hourly, on="timestamp", how="left")

    if cot_fetcher is not None:
        cot_symbol = cot_fetcher.get_symbol_positioning(request.symbol, request.from_time, request.to_time)
        cot_hourly = cot_fetcher.forward_fill_to_hourly(cot_symbol, pd.DatetimeIndex(frame["timestamp"]))
        frame = frame.merge(cot_hourly, on="timestamp", how="left")

    if sentiment_fetcher is not None:
        sentiment_events = sentiment_fetcher.fetch(
            symbol=request.symbol,
            start_date=request.from_time,
            end_date=request.to_time,
        )
        sentiment_hourly = sentiment_fetcher.aggregate_to_hourly(
            sentiment_events,
            pd.DatetimeIndex(frame["timestamp"]),
        )
        frame = frame.merge(sentiment_hourly, on="timestamp", how="left")

    frame = feature_engine.prepare_ml_features(frame, drop_na=True)
    return frame.sort_values("timestamp").reset_index(drop=True)
