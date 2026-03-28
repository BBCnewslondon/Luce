from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from data_ingestion.pipeline import IngestionRequest, build_historical_feature_frame
from evaluation.metrics import TradingCostModel
from evaluation.walk_forward import WalkForwardConfig, run_walk_forward_backtest


@dataclass
class DummyOanda:
    def get_candles_bulk(self, symbol, granularity, from_time, to_time, include_spread=True):
        n = 1200
        ts = pd.date_range(start=from_time, periods=n, freq="h", tz="UTC")
        base = np.linspace(1.05, 1.15, n)
        return pd.DataFrame(
            {
                "timestamp": ts,
                "symbol": symbol,
                "open": base,
                "high": base + 0.001,
                "low": base - 0.001,
                "close": base + np.sin(np.arange(n) / 15.0) * 0.0005,
                "volume": np.full(n, 1000),
                "spread": np.full(n, 0.00008),
            }
        )

    def get_order_book_range(self, symbol, timestamps):
        return pd.DataFrame(
            {
                "timestamp": timestamps,
                "symbol": symbol,
                "positioning_long_short_skew": np.linspace(-0.2, 0.2, len(timestamps)),
                "positioning_density": np.full(len(timestamps), 10),
            }
        )


class DummyFeatureEngine:
    def calculate_all_features(self, df):
        out = df.copy()
        out["sma_3"] = out["close"].rolling(3).mean()
        out["ema_3"] = out["close"].ewm(span=3, adjust=False).mean()
        out["rsi"] = out["close"].pct_change().rolling(14).mean()
        out["macd_hist"] = out["close"].diff() - out["close"].diff().rolling(5).mean()
        out["target_return_1bar"] = out["close"].pct_change().shift(-1)
        return out

    def prepare_ml_features(self, df, drop_na=True):
        out = df.copy()
        feature_cols = ["sma_3", "ema_3", "rsi", "macd_hist", "positioning_long_short_skew"]
        for c in feature_cols:
            out[c] = out[c].shift(1)
        return out.dropna().reset_index(drop=True) if drop_na else out


class DummyVixFetcher:
    def fetch(self, start_date, end_date):
        ts = pd.date_range(start=start_date, end=end_date, freq="D", tz="UTC")
        return pd.DataFrame(
            {
                "timestamp": ts,
                "vix_open": 18.0,
                "vix_high": 20.0,
                "vix_low": 16.0,
                "vix_close": 19.0,
            }
        )

    def forward_fill_to_hourly(self, vix_df, target_timestamps):
        return pd.DataFrame(
            {
                "timestamp": target_timestamps,
                "vix": np.full(len(target_timestamps), 19.0),
                "vix_change": np.zeros(len(target_timestamps)),
            }
        )


class DummySentimentFetcher:
    def fetch(self, symbol, start_date, end_date):
        ts = pd.date_range(start=start_date, end=end_date, freq="6h", tz="UTC")
        return pd.DataFrame(
            {
                "timestamp": ts,
                "sentiment_score": np.linspace(-0.2, 0.3, len(ts)),
                "relevance": np.full(len(ts), 0.7),
                "novelty": np.linspace(0.5, 0.9, len(ts)),
            }
        )

    def aggregate_to_hourly(self, sentiment_df, target_timestamps):
        sent = sentiment_df.copy().rename(
            columns={"relevance": "sentiment_relevance", "novelty": "sentiment_novelty"}
        )
        sent["timestamp"] = pd.to_datetime(sent["timestamp"], utc=True).dt.floor("h")
        sent["sentiment_buzz"] = 1.0
        merged = pd.DataFrame({"timestamp": target_timestamps}).merge(sent, on="timestamp", how="left")
        for c in ["sentiment_score", "sentiment_relevance", "sentiment_novelty"]:
            merged[c] = merged[c].ffill()
        merged["sentiment_buzz"] = merged["sentiment_buzz"].fillna(0.0)
        return merged


def test_end_to_end_pipeline_and_walk_forward():
    req = IngestionRequest(
        symbol="EUR_USD",
        granularity="H1",
        from_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        to_time=datetime(2024, 3, 1, tzinfo=timezone.utc),
    )
    features = build_historical_feature_frame(
        request=req,
        oanda_client=DummyOanda(),
        feature_engine=DummyFeatureEngine(),
        vix_fetcher=DummyVixFetcher(),
        cot_fetcher=None,
        sentiment_fetcher=DummySentimentFetcher(),
        include_order_book=True,
    )

    assert not features.empty
    assert "target_return_1bar" in features.columns
    assert "sentiment_score" in features.columns

    result = run_walk_forward_backtest(
        frame=features,
        feature_columns=["sma_3", "ema_3", "rsi", "macd_hist", "positioning_long_short_skew", "vix"],
        target_column="target_return_1bar",
        cost_model=TradingCostModel(),
        prediction_threshold=0.0,
        wf_config=WalkForwardConfig(train_size=500, test_size=150, step_size=150),
    )

    assert not result.empty
    assert set(["prediction", "signal", "net_return"]).issubset(result.columns)
