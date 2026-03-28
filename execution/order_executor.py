"""Order execution abstraction."""

from __future__ import annotations

from dataclasses import dataclass

from execution.rl_executor import QLearningExecutionPolicy, State


@dataclass(frozen=True)
class OrderExecutor:
    """Execution abstraction with optional RL action policy."""

    dry_run: bool = True
    rl_policy: QLearningExecutionPolicy | None = None

    def submit(self, symbol: str, units: int) -> dict:
        """Return simulated order payload in dry-run mode."""
        return {"symbol": symbol, "units": units, "status": "simulated" if self.dry_run else "submitted"}

    def choose_execution_action(self, state: State, explore: bool = False) -> int:
        """Choose action from RL policy or default to hold."""
        if self.rl_policy is None:
            return 0
        return self.rl_policy.select_action(state=state, explore=explore)

    def learn_from_fill(
        self,
        state: State,
        action: int,
        next_state: State,
        implementation_shortfall_bps: float,
    ) -> None:
        """Update RL policy from realized shortfall."""
        if self.rl_policy is None:
            return
        reward = self.rl_policy.reward_from_shortfall(
            implementation_shortfall_bps=implementation_shortfall_bps,
            action=action,
        )
        self.rl_policy.update(state=state, action=action, reward=reward, next_state=next_state)
