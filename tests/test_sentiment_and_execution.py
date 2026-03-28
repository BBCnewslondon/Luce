from datetime import datetime, timezone

import numpy as np
import pandas as pd

from data_ingestion.sentiment_fetcher import SentimentFetcher
from evaluation.metrics import evaluate_execution_shortfall_vs_vwap, implementation_shortfall_bps, vwap
from execution.order_executor import OrderExecutor
from execution.rl_executor import QLearningExecutionPolicy


def test_sentiment_fetcher_aggregates_hourly_features():
    def provider(start_date, end_date):
        return pd.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, 0, 5, tzinfo=timezone.utc),
                    datetime(2024, 1, 1, 0, 25, tzinfo=timezone.utc),
                    datetime(2024, 1, 1, 1, 10, tzinfo=timezone.utc),
                ],
                "text": [
                    "EUR/USD rallies on strong growth surprise",
                    "Traders see EURUSD upside momentum with high buzz",
                    "Risk-off headline weakens euro",
                ],
            }
        )

    fetcher = SentimentFetcher(event_provider=provider)
    events = fetcher.fetch(
        symbol="EUR_USD",
        start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end_date=datetime(2024, 1, 2, tzinfo=timezone.utc),
    )
    target = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    hourly = fetcher.aggregate_to_hourly(events, target)

    assert set(["sentiment_score", "sentiment_relevance", "sentiment_novelty", "sentiment_buzz"]).issubset(hourly.columns)
    assert hourly["sentiment_buzz"].iloc[0] == 2


def test_vwap_and_implementation_shortfall_metrics():
    prices = pd.Series([100.0, 101.0, 99.0])
    volumes = pd.Series([10.0, 20.0, 10.0])
    assert round(vwap(prices, volumes), 4) == 100.25

    buy_shortfall = implementation_shortfall_bps(100.4, 100.0, "buy")
    sell_shortfall = implementation_shortfall_bps(99.6, 100.0, "sell")
    assert buy_shortfall > 0
    assert sell_shortfall > 0

    df = pd.DataFrame(
        {
            "order_id": [1, 1, 1, 2, 2],
            "side": ["buy", "buy", "buy", "sell", "sell"],
            "executed_price": [100.1, 100.2, 100.0, 99.8, 99.7],
            "executed_qty": [10, 15, 5, 20, 10],
            "benchmark_price": [100.0, 100.0, 100.0, 100.0, 100.0],
            "benchmark_volume": [20, 20, 20, 20, 20],
        }
    )
    out = evaluate_execution_shortfall_vs_vwap(df)
    assert set(["executed_vwap", "benchmark_vwap", "implementation_shortfall_bps"]).issubset(out.columns)
    assert len(out) == 2


def test_rl_executor_updates_policy_from_shortfall():
    np.random.seed(11)
    policy = QLearningExecutionPolicy(epsilon=0.0)
    executor = OrderExecutor(dry_run=True, rl_policy=policy)

    state = (1, 2, 1)
    next_state = (1, 1, 2)
    action = executor.choose_execution_action(state, explore=False)
    executor.learn_from_fill(
        state=state,
        action=action,
        next_state=next_state,
        implementation_shortfall_bps=8.0,
    )

    assert state in policy.q_table
    assert np.any(policy.q_table[state] != 0.0)
