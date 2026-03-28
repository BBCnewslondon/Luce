"""Order execution abstraction."""

from __future__ import annotations

from dataclasses import dataclass

from execution.erl_cmdp import EntropyRegulatedLagrangianAgent
from execution.rl_executor import QLearningExecutionPolicy, State


@dataclass(frozen=True)
class OrderExecutor:
    """Execution abstraction with optional RL action policy."""

    dry_run: bool = True
    rl_policy: QLearningExecutionPolicy | None = None
    erl_policy: EntropyRegulatedLagrangianAgent | None = None

    def submit(self, symbol: str, units: int) -> dict:
        """Return simulated order payload in dry-run mode."""
        return {"symbol": symbol, "units": units, "status": "simulated" if self.dry_run else "submitted"}

    def choose_execution_action(
        self,
        state: State,
        explore: bool = False,
        market_uncertainty: float = 0.5,
    ) -> int:
        """Choose action from ERL policy, fallback to Q-learning, else hold."""
        if self.erl_policy is not None:
            return self.erl_policy.select_action(state=(state, float(market_uncertainty)))
        if self.rl_policy is not None:
            return self.rl_policy.select_action(state=state, explore=explore)
        return 0

    def learn_from_fill(
        self,
        state: State,
        action: int,
        next_state: State,
        implementation_shortfall_bps: float,
        drawdown: float = 0.0,
        leverage: float = 0.0,
        market_uncertainty: float = 0.5,
    ) -> None:
        """Update active execution policy from realized shortfall and constraints."""
        if self.erl_policy is not None:
            reward = -float(implementation_shortfall_bps)
            self.erl_policy.update(
                state=(state, float(market_uncertainty)),
                action=action,
                reward=reward,
                drawdown=float(drawdown),
                leverage=float(leverage),
                market_uncertainty=float(market_uncertainty),
            )
            return

        if self.rl_policy is not None:
            reward = self.rl_policy.reward_from_shortfall(
                implementation_shortfall_bps=implementation_shortfall_bps,
                action=action,
            )
            self.rl_policy.update(state=state, action=action, reward=reward, next_state=next_state)
