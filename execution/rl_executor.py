"""Reinforcement-learning execution policy primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np


State = Tuple[int, int, int]


@dataclass
class QLearningExecutionPolicy:
    """
    Tabular Q-learning policy for microstructure-aware execution decisions.

    State tuple:
    - volatility_bucket
    - queue_bucket
    - urgency_bucket

    Actions:
    - 0: hold (keep resting order)
    - 1: reprice (amend)
    - 2: cross (take liquidity)
    """

    learning_rate: float = 0.1
    discount: float = 0.95
    epsilon: float = 0.1
    action_count: int = 3
    q_table: Dict[State, np.ndarray] = field(default_factory=dict)

    def select_action(self, state: State, explore: bool = True) -> int:
        self._ensure_state(state)
        if explore and np.random.random() < self.epsilon:
            return int(np.random.randint(0, self.action_count))
        return int(np.argmax(self.q_table[state]))

    def update(
        self,
        state: State,
        action: int,
        reward: float,
        next_state: State,
    ) -> None:
        self._ensure_state(state)
        self._ensure_state(next_state)

        old_q = self.q_table[state][action]
        best_next = float(np.max(self.q_table[next_state]))
        td_target = reward + self.discount * best_next
        self.q_table[state][action] = old_q + self.learning_rate * (td_target - old_q)

    @staticmethod
    def reward_from_shortfall(
        implementation_shortfall_bps: float,
        action: int,
        amend_penalty_bps: float = 0.5,
    ) -> float:
        """
        Convert execution outcome to RL reward.

        Higher reward is better. We penalize shortfall and frequent amendments
        to model queue-position opportunity cost.
        """
        penalty = amend_penalty_bps if action == 1 else 0.0
        return float(-implementation_shortfall_bps - penalty)

    def _ensure_state(self, state: State) -> None:
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_count, dtype=float)
