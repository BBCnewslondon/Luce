"""Order execution abstraction."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OrderExecutor:
    """No-op order executor placeholder for testable architecture."""

    dry_run: bool = True

    def submit(self, symbol: str, units: int) -> dict:
        """Return simulated order payload in dry-run mode."""
        return {"symbol": symbol, "units": units, "status": "simulated" if self.dry_run else "submitted"}
