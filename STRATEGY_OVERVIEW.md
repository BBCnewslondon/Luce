# Current MTF Strategy Overview

## Strategy objective
Trade EURUSD intraday on 5-minute candles while only taking trades in the direction of a lag-safe 4-hour trend filter.

## Data and timeframes
- Data source: OANDA REST v20 candles API
- Execution timeframe: 5-minute OHLCV
- Regime timeframe: 4-hour OHLCV
- 4-hour data is shifted by 1 full 4-hour bar before being used in live 5-minute decisions to avoid look-ahead leakage.
- Required credentials: OANDA_ACCOUNT_ID and OANDA_API_TOKEN (optional OANDA_ENVIRONMENT).

## Indicators in use
On both 5-minute and 4-hour frames:
- EMAs: 8, 20, 50, 200
- MACD: fast 8, slow 21, signal 5
- RSI: 13 and 5

## 4-hour trend filter (global regime gate)
Bullish regime requires all of the following on lagged 4-hour indicators:
- EMA200 < EMA50 < EMA20 < EMA8
- MACD line > MACD signal
- RSI13 > 50 and RSI5 > 50
- 4-hour close > EMA8

Bearish regime requires all of the following on lagged 4-hour indicators:
- EMA200 > EMA50 > EMA20 > EMA8
- MACD line < MACD signal
- RSI13 < 50 and RSI5 < 50
- 4-hour close < EMA8

Only when one of these regimes is active can 5-minute entries occur.

## 5-minute execution role
- 5-minute indicators must confirm the same directional regime as 4-hour before entry.
- Directional entries require both timeframes to align:
  - Long entry: 4-hour bullish regime AND 5-minute bullish regime.
  - Short entry: 4-hour bearish regime AND 5-minute bearish regime.
- 5-minute regime uses the same condition set as 4-hour (EMA alignment, MACD direction, RSI thresholds, and close vs EMA8).

## Entry and exit logic
- Position states: long (1), flat (0), short (-1)
- Entry occurs only from flat state.
- Entry requires a valid stop level from the rolling lookback window.
- Initial stop-loss is set from a 5-candle lookback on 5-minute candles using prior candles:
  - Long stop: rolling minimum of prior lows
  - Short stop: rolling maximum of prior highs
- Take-profit is set at a fixed 3:1 reward-to-risk ratio from the entry price (risk:reward = 1:3):
  - Long take-profit: entry + 3 * (entry - stop)
  - Short take-profit: entry - 3 * (stop - entry)
- Exits:
  - stop_exit: stop-loss is hit intrabar
  - take_profit_exit: 3:1 take-profit target is hit intrabar
  - filter_exit: 4-hour trend regime loses any required condition for the open direction

## Cost and performance model
- Spread cost model: spread_pips * pip_size / close
- Turnover cost applied on position changes
- Strategy return per bar:
  - previous position * close-to-close return - turnover * spread cost
- Reported series:
  - strategy_return
  - equity_curve
  - cumulative_return
  - drawdown

## Plot output
- Top pane: Close, EMA 8/20/50/200, long/short markers, active 4-hour regime shading
- Bottom pane: MACD histogram, MACD line, MACD signal line
