"""Genetic optimization of triple-barrier parameters for signal labeling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


@dataclass
class TripleBarrierParams:
    upper_threshold: float
    lower_threshold: float
    max_holding_bars: int


@dataclass
class GAConfig:
    population_size: int = 30
    generations: int = 25
    mutation_rate: float = 0.2
    seed: int = 42


def apply_triple_barrier(prices: pd.Series, params: TripleBarrierParams) -> pd.DataFrame:
    """Apply triple barriers and return labels with realized event returns."""
    px = pd.Series(prices).astype(float).reset_index(drop=True)
    rows = []

    for start in range(0, len(px) - 1):
        p0 = px.iloc[start]
        end = min(len(px) - 1, start + params.max_holding_bars)

        label = 0
        exit_idx = end
        for i in range(start + 1, end + 1):
            ret = px.iloc[i] / p0 - 1.0
            if ret >= params.upper_threshold:
                label = 1
                exit_idx = i
                break
            if ret <= -params.lower_threshold:
                label = -1
                exit_idx = i
                break

        realized = px.iloc[exit_idx] / p0 - 1.0
        rows.append(
            {
                "start": start,
                "end": exit_idx,
                "label": label,
                "event_return": realized,
            }
        )

    return pd.DataFrame(rows)


def optimize_dual_style_barriers(
    prices: pd.Series,
    config: GAConfig | None = None,
) -> dict[str, TripleBarrierParams]:
    """Optimize separate barrier sets for HRHP and LRLP styles."""
    cfg = config or GAConfig()
    return {
        "hrhp": optimize_triple_barriers_ga(prices, style="hrhp", config=cfg),
        "lrlp": optimize_triple_barriers_ga(prices, style="lrlp", config=cfg),
    }


def optimize_triple_barriers_ga(
    prices: pd.Series,
    style: Literal["hrhp", "lrlp"],
    config: GAConfig | None = None,
) -> TripleBarrierParams:
    """Genetic search over upper/lower barriers and holding horizon."""
    cfg = config or GAConfig()
    rng = np.random.default_rng(cfg.seed + (0 if style == "hrhp" else 7))

    pop = [_random_params(rng) for _ in range(cfg.population_size)]

    for _ in range(cfg.generations):
        scored = sorted(
            ((fitness(prices, p, style), p) for p in pop),
            key=lambda x: x[0],
            reverse=True,
        )
        elites = [p for _, p in scored[: max(2, cfg.population_size // 5)]]

        new_pop = elites.copy()
        while len(new_pop) < cfg.population_size:
            p1, p2 = rng.choice(elites, size=2, replace=True)
            child = crossover(p1, p2, rng)
            if rng.random() < cfg.mutation_rate:
                child = mutate(child, rng)
            new_pop.append(child)

        pop = new_pop

    best = max(pop, key=lambda p: fitness(prices, p, style))
    return best


def fitness(prices: pd.Series, params: TripleBarrierParams, style: str) -> float:
    """Risk-adjusted objective supporting distinct HRHP and LRLP trade-offs."""
    labels = apply_triple_barrier(prices, params)
    returns = labels["event_return"]

    if returns.empty:
        return -1e9

    equity = (1.0 + returns).cumprod()
    peak = equity.cummax()
    drawdown = (equity / peak - 1.0).min()

    total_profit = float(equity.iloc[-1] - 1.0)
    mdd = abs(float(drawdown))

    if style == "hrhp":
        w_profit, w_mdd = 1.0, 0.35
    else:
        w_profit, w_mdd = 0.55, 1.0

    return w_profit * total_profit - w_mdd * mdd


def _random_params(rng: np.random.Generator) -> TripleBarrierParams:
    return TripleBarrierParams(
        upper_threshold=float(rng.uniform(0.002, 0.05)),
        lower_threshold=float(rng.uniform(0.002, 0.05)),
        max_holding_bars=int(rng.integers(4, 96)),
    )


def crossover(a: TripleBarrierParams, b: TripleBarrierParams, rng: np.random.Generator) -> TripleBarrierParams:
    alpha = float(rng.uniform(0.25, 0.75))
    return TripleBarrierParams(
        upper_threshold=alpha * a.upper_threshold + (1 - alpha) * b.upper_threshold,
        lower_threshold=alpha * a.lower_threshold + (1 - alpha) * b.lower_threshold,
        max_holding_bars=int(round(alpha * a.max_holding_bars + (1 - alpha) * b.max_holding_bars)),
    )


def mutate(p: TripleBarrierParams, rng: np.random.Generator) -> TripleBarrierParams:
    return TripleBarrierParams(
        upper_threshold=float(np.clip(p.upper_threshold + rng.normal(0.0, 0.003), 0.001, 0.08)),
        lower_threshold=float(np.clip(p.lower_threshold + rng.normal(0.0, 0.003), 0.001, 0.08)),
        max_holding_bars=int(np.clip(p.max_holding_bars + rng.integers(-8, 9), 2, 120)),
    )
