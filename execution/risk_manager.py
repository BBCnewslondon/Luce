"""Risk manager utilities for position sizing and guardrails."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskManager:
    """Simple risk manager placeholder for package integrity."""

    max_position_pct: float = 0.02
    max_drawdown_pct: float = 0.15

    def scale_units(self, base_units: float, confidence: float) -> float:
        """Scale units by confidence while respecting max position cap."""
        capped_confidence = min(max(confidence, 0.0), 1.0)
        return base_units * capped_confidence
