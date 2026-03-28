"""Execution realism and pathwise robustness stress testing utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SlippageConfig:
    base_spread_bps: float = 1.0
    latency_bps_per_100ms: float = 0.3
    cancel_risk: float = 0.08


def simulate_slippage_ensemble(
    benchmark_price: float,
    side: str,
    quantity: float,
    order_type: str,
    time_of_day_bucket: str,
    venue: str,
    latency_ms: float,
    n_paths: int = 250,
    seed: int = 42,
    config: SlippageConfig | None = None,
) -> pd.DataFrame:
    """Generate execution outcomes under partial fills, cancellation, and latency."""
    cfg = config or SlippageConfig()
    rng = np.random.default_rng(seed)

    type_adj = {"market": 2.4, "limit": 1.2, "post_only": 0.8}.get(order_type, 1.5)
    tod_adj = {"open": 1.8, "london": 1.1, "ny": 1.2, "overnight": 1.4}.get(time_of_day_bucket, 1.25)
    venue_adj = {"primary": 1.0, "ecnpool": 1.15, "dark": 1.35}.get(venue, 1.2)

    slippage_bps_mean = cfg.base_spread_bps * type_adj * tod_adj * venue_adj + cfg.latency_bps_per_100ms * (latency_ms / 100.0)
    slippage_bps = rng.normal(slippage_bps_mean, max(0.5, slippage_bps_mean * 0.35), size=n_paths)

    canceled = rng.random(n_paths) < cfg.cancel_risk
    partial = rng.uniform(0.35, 1.0, size=n_paths)
    filled_qty = np.where(canceled, 0.0, quantity * partial)

    sign = 1.0 if side.lower() == "buy" else -1.0
    executed = benchmark_price * (1.0 + sign * slippage_bps / 10000.0)

    return pd.DataFrame(
        {
            "path": np.arange(n_paths),
            "benchmark_price": benchmark_price,
            "executed_price": executed,
            "filled_qty": filled_qty,
            "slippage_bps": slippage_bps,
            "canceled": canceled,
        }
    )


def apply_liquidity_budget_penalty(
    expected_returns: pd.Series,
    participation_rate: pd.Series,
    start_penalty: float = 0.05,
    full_penalty: float = 0.10,
) -> pd.Series:
    """Degrade returns progressively as participation approaches impact-heavy levels."""
    r = pd.Series(expected_returns, dtype=float)
    p = pd.Series(participation_rate, dtype=float).reindex(r.index)

    scale = np.where(
        p <= start_penalty,
        1.0,
        np.where(
            p >= full_penalty,
            0.5,
            1.0 - 0.5 * ((p - start_penalty) / (full_penalty - start_penalty + 1e-12)),
        ),
    )
    return r * scale


def inject_adversarial_fast_failure(
    returns: pd.Series,
    cluster_len: int = 3,
    shock_sigma_mult: float = 4.0,
    gap_probability: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """Inject extreme clustered stress moves and gap events into returns."""
    r = pd.Series(returns, dtype=float).reset_index(drop=True)
    rng = np.random.default_rng(seed)

    stressed = r.copy()
    if len(stressed) >= cluster_len:
        start = int(rng.integers(0, len(stressed) - cluster_len + 1))
        sigma = max(stressed.std(), 1e-6)
        stressed.iloc[start : start + cluster_len] += rng.normal(-shock_sigma_mult * sigma, sigma, size=cluster_len)

    gap_mask = rng.random(len(stressed)) < gap_probability
    stressed[gap_mask] += rng.normal(-2.0 * max(r.std(), 1e-6), max(r.std(), 1e-6), size=gap_mask.sum())

    return pd.DataFrame({"base": r, "stressed": stressed})


def summarize_cpcv_path_distribution(path_metrics: pd.DataFrame, n_mc: int = 5000, seed: int = 42) -> pd.Series:
    """Summarize CPCV distributions and Monte Carlo robustness statistics."""
    required = {"sharpe", "max_drawdown"}
    if not required.issubset(path_metrics.columns):
        raise KeyError("path_metrics must contain sharpe and max_drawdown")

    sharpe = pd.to_numeric(path_metrics["sharpe"], errors="coerce").dropna()
    mdd = pd.to_numeric(path_metrics["max_drawdown"], errors="coerce").dropna()

    rng = np.random.default_rng(seed)
    boot = rng.choice(sharpe.values, size=(n_mc, len(sharpe)), replace=True).mean(axis=1)

    return pd.Series(
        {
            "paths": float(len(path_metrics)),
            "sharpe_mean": float(sharpe.mean()),
            "sharpe_p05": float(np.percentile(sharpe, 5)),
            "sharpe_p50": float(np.percentile(sharpe, 50)),
            "sharpe_p95": float(np.percentile(sharpe, 95)),
            "mdd_p95": float(np.percentile(mdd, 95)),
            "mc_sharpe_mean": float(boot.mean()),
            "mc_sharpe_var_1pct": float(np.percentile(boot, 1)),
        }
    )
