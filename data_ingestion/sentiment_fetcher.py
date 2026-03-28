"""Alternative-data sentiment fetcher with relevance and novelty scoring."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from loguru import logger
except ImportError:  # pragma: no cover
    import logging

    logger = logging.getLogger(__name__)

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:
    class SentimentIntensityAnalyzer:  # pragma: no cover - fallback used in limited envs
        """Tiny fallback polarity estimator when vaderSentiment is unavailable."""

        POS_WORDS = {"rally", "bullish", "upside", "beat", "strong", "growth"}
        NEG_WORDS = {"risk-off", "bearish", "downside", "miss", "weak", "drop"}

        def polarity_scores(self, text: str) -> dict[str, float]:
            words = text.lower().split()
            pos = sum(1 for w in words if w in self.POS_WORDS)
            neg = sum(1 for w in words if w in self.NEG_WORDS)
            denom = max(pos + neg, 1)
            score = (pos - neg) / denom
            return {"compound": float(max(min(score, 1.0), -1.0))}


@dataclass
class SentimentConfig:
    """Configuration for text sentiment feature extraction."""

    symbol_aliases: dict[str, list[str]]
    novelty_window: int = 20


class SentimentFetcher:
    """
    Build alternative-data features from timestamped text events.

    Features:
    - `sentiment_score`: VADER compound sentiment in [-1, 1]
    - `relevance`: weighted mention score for the target symbol
    - `novelty`: 1 - max cosine similarity to recent items (higher is newer)
    """

    def __init__(
        self,
        config: Optional[SentimentConfig] = None,
        event_provider: Optional[Callable[[datetime, datetime], pd.DataFrame]] = None,
    ):
        self.config = config or SentimentConfig(symbol_aliases={})
        self.event_provider = event_provider
        self._sentiment = SentimentIntensityAnalyzer()

    def fetch(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        Fetch raw events via provider and compute NLP features.

        Returns:
            DataFrame with timestamp, sentiment_score, relevance, novelty.
            Empty DataFrame when no provider or no data is available.
        """
        if self.event_provider is None:
            logger.warning("Sentiment event_provider not configured; returning empty frame")
            return pd.DataFrame()

        raw = self.event_provider(start_date, end_date)
        if raw.empty:
            return raw

        return self._from_events(raw=raw, symbol=symbol)

    def _from_events(self, raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
        required = {"timestamp", "text"}
        if not required.issubset(raw.columns):
            raise KeyError(f"Sentiment events must contain {sorted(required)}")

        out = raw.copy()
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
        out = out.dropna(subset=["timestamp", "text"]).sort_values("timestamp").reset_index(drop=True)
        out["text"] = out["text"].astype(str)

        out["sentiment_score"] = out["text"].apply(
            lambda x: float(self._sentiment.polarity_scores(x).get("compound", 0.0))
        )
        out["relevance"] = out["text"].apply(lambda x: self._relevance_score(x, symbol))
        out["novelty"] = self._novelty_score(out["text"])

        return out[["timestamp", "sentiment_score", "relevance", "novelty"]]

    def aggregate_to_hourly(
        self,
        sentiment_df: pd.DataFrame,
        target_timestamps: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """
        Aggregate event-level sentiment features to hourly alignment timestamps.
        """
        if sentiment_df.empty:
            return pd.DataFrame(
                {
                    "timestamp": target_timestamps,
                    "sentiment_score": [None] * len(target_timestamps),
                    "sentiment_relevance": [None] * len(target_timestamps),
                    "sentiment_novelty": [None] * len(target_timestamps),
                    "sentiment_buzz": [0] * len(target_timestamps),
                }
            )

        work = sentiment_df.copy()
        work["hour"] = work["timestamp"].dt.floor("h")
        hourly = (
            work.groupby("hour", as_index=False)
            .agg(
                sentiment_score=("sentiment_score", "mean"),
                sentiment_relevance=("relevance", "mean"),
                sentiment_novelty=("novelty", "mean"),
                sentiment_buzz=("sentiment_score", "size"),
            )
            .rename(columns={"hour": "timestamp"})
        )

        aligned = pd.DataFrame({"timestamp": target_timestamps}).merge(
            hourly,
            on="timestamp",
            how="left",
        )
        aligned[["sentiment_score", "sentiment_relevance", "sentiment_novelty"]] = (
            aligned[["sentiment_score", "sentiment_relevance", "sentiment_novelty"]].ffill()
        )
        aligned["sentiment_buzz"] = aligned["sentiment_buzz"].fillna(0)
        return aligned

    def _relevance_score(self, text: str, symbol: str) -> float:
        aliases = self.config.symbol_aliases.get(symbol, [])
        vocab = [symbol.lower().replace("_", ""), symbol.lower().replace("_", "/"), *[a.lower() for a in aliases]]
        text_lower = text.lower()
        hits = sum(1 for token in vocab if token and token in text_lower)
        return float(min(hits / max(len(vocab), 1), 1.0))

    def _novelty_score(self, text_series: pd.Series) -> pd.Series:
        if text_series.empty:
            return pd.Series(dtype=float)

        vec = TfidfVectorizer(max_features=3000, stop_words="english")
        tfidf = vec.fit_transform(text_series.tolist()).tocsr()
        novelty = np.ones(tfidf.shape[0], dtype=float)

        window = max(1, self.config.novelty_window)
        for idx in range(1, tfidf.shape[0]):
            start = max(0, idx - window)
            prior = tfidf[start:idx]
            current = tfidf.getrow(idx)
            sims = prior.dot(current.T).toarray().ravel()
            max_sim = float(sims.max()) if sims.size else 0.0
            novelty[idx] = 1.0 - max_sim

        return pd.Series(novelty, index=text_series.index)
