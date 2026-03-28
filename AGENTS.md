# AGENTS Workflow and Guardrails

## Mission
Ship Luce as a robust, point-in-time correct, cost-aware trading system with disciplined automated quality gates.

## Mandatory Build Sequence
1. Data ingestion contracts first (OHLCV, positioning, VIX, COT, sentiment).
2. Feature engineering and target generation with explicit lag assertions.
3. Signal model training and walk-forward validation.
4. Cost and execution analytics (VWAP, implementation shortfall).
5. Execution policy updates (risk caps + RL reward tuning).
6. Release only after all test and coverage gates pass.

## Decision Logic
- If any ingestion feed is stale or missing, continue with fallback features and log a structured incident.
- If VIX is elevated/extreme, reduce gross exposure and tighten liquidity assumptions.
- If walk-forward IC decays below threshold, cut model weight and retrain.
- If execution shortfall worsens, increase RL amendment penalty and reduce aggressive actions.

## Agent Coding Rules
- Keep features point-in-time safe: no future leakage in joins, fills, or targets.
- Every new model/output must have deterministic tests.
- Prefer pure functions for metrics and feature transforms.
- Avoid hidden side effects in execution/risk code.

## CI Gates
- `pytest` must pass.
- Coverage gate must pass (`--cov-fail-under`).
- End-to-end test must execute ingestion -> signal -> walk-forward path.

## Self-Debugging Loop (Required on Failure)
1. Reproduce with failing test only.
2. Explain failing path line-by-line with concrete values.
3. Generate a minimal patch.
4. Re-run the failing test, then full test suite.
5. If still failing after two attempts, collect trace and isolate root cause before next patch.

## Definition of Done
- No failing tests.
- Coverage gate satisfied.
- New behavior documented in module docstrings and tests.
- No unresolved TODOs tied to production behavior.
