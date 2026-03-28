"""Position state tracker."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class PositionTracker:
    """In-memory position tracker for strategy state."""

    positions: Dict[str, int] = field(default_factory=dict)

    def update(self, symbol: str, units_delta: int) -> int:
        """Apply a delta and return resulting net units."""
        self.positions[symbol] = self.positions.get(symbol, 0) + units_delta
        return self.positions[symbol]
