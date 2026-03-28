"""Semantic mutation testing helpers for behavioral contract robustness."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List


@dataclass
class Mutation:
    name: str
    old: str
    new: str


class MutationSmith:
    """Apply semantic mutations and score whether tests kill mutated variants."""

    def __init__(self, mutations: List[Mutation] | None = None):
        self.mutations = mutations or []

    def add_mutation(self, name: str, old: str, new: str) -> None:
        self.mutations.append(Mutation(name=name, old=old, new=new))

    def run(self, source: str, test_runner: Callable[[str], bool]) -> dict:
        """
        Run mutations against source and compute mutation score.

        test_runner returns True when tests pass, False when tests fail.
        Mutation is killed if tests fail on mutated source.
        """
        total = 0
        killed = 0
        details = []

        for m in self.mutations:
            if m.old not in source:
                continue

            total += 1
            mutated = source.replace(m.old, m.new, 1)
            passed = bool(test_runner(mutated))
            is_killed = not passed
            killed += int(is_killed)
            details.append({"name": m.name, "killed": is_killed})

        score = (killed / total) if total else 0.0
        return {"total": total, "killed": killed, "mutation_score": score, "details": details}


def default_trading_mutations() -> List[Mutation]:
    """Default semantic guardrail mutations for trading code patterns."""
    return [
        Mutation(
            name="disable_volatility_gate",
            old="if vix_value > threshold:",
            new="if False:",
        ),
        Mutation(
            name="bypass_position_cap",
            old="min(position_size, max_position_size)",
            new="position_size",
        ),
        Mutation(
            name="invert_risk_check",
            old="if drawdown > max_drawdown:",
            new="if drawdown <= max_drawdown:",
        ),
    ]
