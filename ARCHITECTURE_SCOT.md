<architect_thought>
- Assumptions: We trade liquid major FX pairs at H1; OANDA candles and positioning endpoints are reliable; VIX is daily and must be point-in-time aligned; all features are lagged by at least one bar; labels are forward returns; live and backtest use the same feature code path.
- Sequence: Ingest OANDA candles and positioning plus COT and VIX into point-in-time tables -> compute TA-Lib factors and alternative factors -> apply explicit lagging and target construction -> train ensemble (Gradient Boosted Trees + optional tree peers) with walk-forward splits -> convert predictions to cost-aware signals -> execute constrained position sizing.
- Branch: If data quality checks fail (missing bars, stale alt-data, spread anomalies), skip symbol and log incident; if crash-risk branch triggers (VIX regime elevated/extreme), down-weight gross exposure and tighten liquidity caps; if model confidence below threshold, hold cash.
- Loop: For each retrain cycle, run expanding-window walk-forward, compute rolling Spearman IC, Master's-style profit factor, alpha-decay curve, and transaction-cost-adjusted performance; if IC falls below threshold or decay half-life shortens, trigger retrain/hyperparameter refresh and reduce deployment weight.
- Risks before finalization: Look-ahead leakage from improper timestamp joins, curve-fitting from over-tuned trees, survivorship bias from symbol selection drift, and understated costs (spread/slippage/market impact). Mitigations are mandatory purged splits, embargo gaps, lag assertions, robust OOS regime testing, and explicit cost/liquidity deductions in every simulation path.
</architect_thought>

## Module IO Contracts

### Data Ingestion
- Inputs: symbols list, UTC time range, granularity, OANDA credentials, optional VIX/COT endpoints.
- Outputs: point-in-time DataFrame indexed by timestamp+symbol with OHLCV, spread, order-book positioning, VIX, COT.

### Signal Generation
- Inputs: point-in-time feature frame, feature config, model config, training windows.
- Outputs: pure function signal frame with prediction, confidence, and signed target horizon score.

### Rigorous Evaluation
- Inputs: predicted signals, realized returns, trade cost model, liquidity limits.
- Outputs: rolling IC series, Master's-style profit factor series, alpha decay table, and net performance stats after costs.
