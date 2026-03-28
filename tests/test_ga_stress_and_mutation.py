import numpy as np
import pandas as pd

from evaluation.mutation_smith import MutationSmith, default_trading_mutations
from evaluation.stress_testing import (
    apply_liquidity_budget_penalty,
    inject_adversarial_fast_failure,
    simulate_slippage_ensemble,
    summarize_cpcv_path_distribution,
)
from signal_generation.triple_barrier_ga import (
    GAConfig,
    apply_triple_barrier,
    optimize_dual_style_barriers,
)


def test_ga_barrier_optimizer_produces_distinct_styles():
    rng = np.random.default_rng(4)
    prices = pd.Series(1.10 + np.cumsum(rng.normal(0, 0.0015, size=400)))

    styles = optimize_dual_style_barriers(prices, GAConfig(population_size=12, generations=8, seed=4))
    hrhp = styles["hrhp"]
    lrlp = styles["lrlp"]

    hrhp_labels = apply_triple_barrier(prices, hrhp)
    lrlp_labels = apply_triple_barrier(prices, lrlp)

    assert hrhp.max_holding_bars != lrlp.max_holding_bars or abs(hrhp.upper_threshold - lrlp.upper_threshold) > 1e-6
    assert not hrhp_labels.empty
    assert not lrlp_labels.empty


def test_execution_stress_plumbing_and_cpcv_distribution():
    fills = simulate_slippage_ensemble(
        benchmark_price=100.0,
        side="buy",
        quantity=10_000,
        order_type="market",
        time_of_day_bucket="open",
        venue="primary",
        latency_ms=180,
        n_paths=200,
        seed=7,
    )
    assert len(fills) == 200
    assert {"executed_price", "slippage_bps", "filled_qty", "canceled"}.issubset(fills.columns)

    base_returns = pd.Series(np.full(50, 0.001))
    participation = pd.Series(np.linspace(0.01, 0.12, 50))
    penalized = apply_liquidity_budget_penalty(base_returns, participation)
    assert penalized.iloc[-1] < base_returns.iloc[-1]

    stress = inject_adversarial_fast_failure(pd.Series(np.random.normal(0.0005, 0.01, 200)), seed=7)
    assert {"base", "stressed"}.issubset(stress.columns)

    path_metrics = pd.DataFrame(
        {
            "sharpe": np.random.normal(1.1, 0.5, 120),
            "max_drawdown": np.random.uniform(0.05, 0.35, 120),
        }
    )
    summary = summarize_cpcv_path_distribution(path_metrics, n_mc=2000, seed=7)
    assert {"sharpe_p05", "mdd_p95", "mc_sharpe_var_1pct"}.issubset(summary.index)


def test_mutation_smith_scores_behavioral_blind_spots():
    source = """
def guardrail(vix_value, threshold, position_size, max_position_size, drawdown, max_drawdown):
    if vix_value > threshold:
        position_size = min(position_size, max_position_size)
    if drawdown > max_drawdown:
        return 0
    return position_size
"""

    smith = MutationSmith(default_trading_mutations())

    def test_runner(mutated_source: str) -> bool:
        # Fail tests when critical risk checks are missing/inverted.
        is_bad = (
            "if False:" in mutated_source
            or "return 0" not in mutated_source
            or "if drawdown <= max_drawdown:" in mutated_source
        )
        return not is_bad

    result = smith.run(source, test_runner)
    assert result["total"] >= 2
    assert result["mutation_score"] > 0.5
