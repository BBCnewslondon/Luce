"""Entropy-regulated Lagrangian policy updates for constrained trading RL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Hashable, Optional

import numpy as np


@dataclass
class ERLConfig:
    """Configuration for entropy-regulated constrained policy learning."""

    action_count: int = 3
    learning_rate: float = 0.05
    lambda_lr: float = 0.02
    drawdown_limit: float = 0.12
    leverage_cap: float = 1.5
    entropy_target: float = 0.85


class EntropyRegulatedLagrangianAgent:
    """
    Tabular ERL agent in a CMDP-style Lagrangian formulation.

    Constraints treated as first-class costs:
    - max drawdown
    - leverage cap
    - minimum policy entropy (adaptive to uncertainty)
    """

    def __init__(self, config: Optional[ERLConfig] = None):
        self.config = config or ERLConfig()
        self.logits: Dict[Hashable, np.ndarray] = {}

        self.lambda_drawdown = 0.0
        self.lambda_leverage = 0.0
        self.lambda_entropy = 0.0

    def select_action(
        self,
        state: Hashable,
        rng: Optional[np.random.Generator] = None,
    ) -> int:
        """Sample action from current softmax policy."""
        self._ensure_state(state)
        rng = rng or np.random.default_rng()
        probs = self.policy_probs(state)
        return int(rng.choice(self.config.action_count, p=probs))

    def policy_probs(self, state: Hashable) -> np.ndarray:
        self._ensure_state(state)
        z = self.logits[state]
        z = z - np.max(z)
        exp = np.exp(z)
        return exp / exp.sum()

    def policy_entropy(self, state: Hashable) -> float:
        probs = self.policy_probs(state)
        return float(-(probs * np.log(probs + 1e-12)).sum())

    def update(
        self,
        state: Hashable,
        action: int,
        reward: float,
        drawdown: float,
        leverage: float,
        market_uncertainty: float,
    ) -> Dict[str, float]:
        """
        Perform one ERL policy update with adaptive entropy constraint.
        """
        self._ensure_state(state)

        probs = self.policy_probs(state)
        entropy = self.policy_entropy(state)

        c_drawdown = max(0.0, drawdown - self.config.drawdown_limit)
        c_leverage = max(0.0, abs(leverage) - self.config.leverage_cap)

        unc = float(np.clip(market_uncertainty, 0.0, 1.0))
        entropy_floor = self.config.entropy_target * (0.5 + unc)
        c_entropy = max(0.0, entropy_floor - entropy)

        lagrangian_reward = (
            reward
            - self.lambda_drawdown * c_drawdown
            - self.lambda_leverage * c_leverage
            - self.lambda_entropy * c_entropy
        )

        one_hot = np.zeros(self.config.action_count)
        one_hot[action] = 1.0
        grad = one_hot - probs
        self.logits[state] += self.config.learning_rate * lagrangian_reward * grad

        self.lambda_drawdown = max(
            0.0,
            self.lambda_drawdown + self.config.lambda_lr * c_drawdown,
        )
        self.lambda_leverage = max(
            0.0,
            self.lambda_leverage + self.config.lambda_lr * c_leverage,
        )
        self.lambda_entropy = max(
            0.0,
            self.lambda_entropy + self.config.lambda_lr * c_entropy,
        )

        return {
            "entropy": entropy,
            "entropy_floor": entropy_floor,
            "lagrangian_reward": lagrangian_reward,
            "c_drawdown": c_drawdown,
            "c_leverage": c_leverage,
            "c_entropy": c_entropy,
            "lambda_drawdown": self.lambda_drawdown,
            "lambda_leverage": self.lambda_leverage,
            "lambda_entropy": self.lambda_entropy,
        }

    def _ensure_state(self, state: Hashable) -> None:
        if state not in self.logits:
            self.logits[state] = np.zeros(self.config.action_count, dtype=float)
