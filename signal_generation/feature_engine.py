"""
Feature Engine using TA-Lib for technical indicator calculation.

Calculates 50+ technical indicators with strict point-in-time semantics
to prevent look-ahead bias in ML training and backtesting.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
from loguru import logger

try:
    import talib
except ImportError:
    raise ImportError(
        "TA-Lib is required. Install with: pip install TA-Lib "
        "(requires TA-Lib C library to be installed first)"
    )


@dataclass
class FeatureConfig:
    """Configuration for technical indicators."""

    # Trend indicators
    sma_periods: List[int] = field(default_factory=lambda: [20, 50, 200])
    ema_periods: List[int] = field(default_factory=lambda: [12, 26])
    adx_period: int = 14
    aroon_period: int = 25

    # Momentum indicators
    rsi_period: int = 14
    stoch_k_period: int = 14
    stoch_d_period: int = 3
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    roc_period: int = 10
    williams_period: int = 14
    cci_period: int = 20
    mfi_period: int = 14

    # Volatility indicators
    atr_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0
    keltner_period: int = 20
    keltner_atr_mult: float = 1.5


class FeatureEngine:
    """
    Calculates technical indicators from OHLCV price data.

    All indicators are calculated using TA-Lib for numerical stability
    and performance. Features are computed point-in-time to prevent
    look-ahead bias.

    IMPORTANT: The resulting features should still be shifted by 1 bar
    before use in ML training to ensure we're predicting future returns
    using only past information.
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        """
        Initialize feature engine.

        Args:
            config: FeatureConfig instance.
        """
        self.config = config or FeatureConfig()
        logger.info("Feature engine initialized")

    def calculate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators for a DataFrame.

        Args:
            df: DataFrame with columns: open, high, low, close, volume.

        Returns:
            DataFrame with original columns plus all indicator columns.

        Note:
            Returns NaN for initial bars where indicators cannot be
            calculated (warmup period). These should be dropped before
            ML training.
        """
        if df.empty:
            return df

        result = df.copy()

        # Validate required columns
        required = ["open", "high", "low", "close"]
        missing = [c for c in required if c not in result.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Extract price arrays
        open_arr = result["open"].values.astype(float)
        high = result["high"].values.astype(float)
        low = result["low"].values.astype(float)
        close = result["close"].values.astype(float)
        volume = result.get("volume", pd.Series([0] * len(result))).values.astype(float)

        # Calculate all feature categories
        result = self._add_trend_features(result, open_arr, high, low, close)
        result = self._add_momentum_features(result, high, low, close, volume)
        result = self._add_volatility_features(result, high, low, close)
        result = self._add_price_features(result, open_arr, high, low, close)

        logger.info(f"Calculated {len(result.columns) - len(df.columns)} features")
        return result

    def _add_trend_features(
        self,
        df: pd.DataFrame,
        open_arr: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
    ) -> pd.DataFrame:
        """Add trend-following indicators."""

        # Simple Moving Averages
        for period in self.config.sma_periods:
            df[f"sma_{period}"] = talib.SMA(close, timeperiod=period)
            # Price relative to SMA (normalized)
            df[f"close_sma_{period}_ratio"] = close / df[f"sma_{period}"].values

        # Exponential Moving Averages
        for period in self.config.ema_periods:
            df[f"ema_{period}"] = talib.EMA(close, timeperiod=period)

        # EMA crossover signal
        if len(self.config.ema_periods) >= 2:
            fast_ema = self.config.ema_periods[0]
            slow_ema = self.config.ema_periods[1]
            df["ema_crossover"] = df[f"ema_{fast_ema}"] - df[f"ema_{slow_ema}"]

        # ADX (Average Directional Index) - trend strength
        df["adx"] = talib.ADX(high, low, close, timeperiod=self.config.adx_period)
        df["plus_di"] = talib.PLUS_DI(high, low, close, timeperiod=self.config.adx_period)
        df["minus_di"] = talib.MINUS_DI(high, low, close, timeperiod=self.config.adx_period)
        df["di_diff"] = df["plus_di"] - df["minus_di"]

        # Aroon indicators
        df["aroon_up"], df["aroon_down"] = talib.AROON(high, low, timeperiod=self.config.aroon_period)
        df["aroon_osc"] = df["aroon_up"] - df["aroon_down"]

        # Parabolic SAR
        df["sar"] = talib.SAR(high, low)
        df["sar_signal"] = np.where(close > df["sar"].values, 1, -1)

        return df

    def _add_momentum_features(
        self,
        df: pd.DataFrame,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
    ) -> pd.DataFrame:
        """Add momentum indicators."""

        # RSI
        df["rsi"] = talib.RSI(close, timeperiod=self.config.rsi_period)
        # RSI overbought/oversold zones
        df["rsi_overbought"] = (df["rsi"] > 70).astype(int)
        df["rsi_oversold"] = (df["rsi"] < 30).astype(int)

        # Stochastic
        df["stoch_k"], df["stoch_d"] = talib.STOCH(
            high, low, close,
            fastk_period=self.config.stoch_k_period,
            slowk_period=self.config.stoch_d_period,
            slowd_period=self.config.stoch_d_period,
        )
        df["stoch_signal"] = df["stoch_k"] - df["stoch_d"]

        # MACD
        df["macd"], df["macd_signal"], df["macd_hist"] = talib.MACD(
            close,
            fastperiod=self.config.macd_fast,
            slowperiod=self.config.macd_slow,
            signalperiod=self.config.macd_signal,
        )

        # Rate of Change
        df["roc"] = talib.ROC(close, timeperiod=self.config.roc_period)

        # Williams %R
        df["williams_r"] = talib.WILLR(high, low, close, timeperiod=self.config.williams_period)

        # CCI (Commodity Channel Index)
        df["cci"] = talib.CCI(high, low, close, timeperiod=self.config.cci_period)

        # MFI (Money Flow Index) - volume-weighted RSI
        if volume.sum() > 0:
            df["mfi"] = talib.MFI(high, low, close, volume, timeperiod=self.config.mfi_period)
        else:
            df["mfi"] = np.nan

        # Ultimate Oscillator
        df["ultosc"] = talib.ULTOSC(high, low, close)

        return df

    def _add_volatility_features(
        self,
        df: pd.DataFrame,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
    ) -> pd.DataFrame:
        """Add volatility indicators."""

        # ATR (Average True Range)
        df["atr"] = talib.ATR(high, low, close, timeperiod=self.config.atr_period)
        # Normalized ATR (ATR / Close)
        df["atr_pct"] = df["atr"] / close

        # Bollinger Bands
        df["bb_upper"], df["bb_middle"], df["bb_lower"] = talib.BBANDS(
            close,
            timeperiod=self.config.bb_period,
            nbdevup=self.config.bb_std,
            nbdevdn=self.config.bb_std,
        )
        # BB width (normalized)
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
        # BB position (where price is within bands, 0-1)
        df["bb_position"] = (close - df["bb_lower"].values) / (
            df["bb_upper"].values - df["bb_lower"].values + 1e-10
        )

        # Keltner Channels
        keltner_ma = talib.EMA(close, timeperiod=self.config.keltner_period)
        keltner_atr = talib.ATR(high, low, close, timeperiod=self.config.keltner_period)
        df["keltner_upper"] = keltner_ma + self.config.keltner_atr_mult * keltner_atr
        df["keltner_lower"] = keltner_ma - self.config.keltner_atr_mult * keltner_atr
        df["keltner_width"] = (df["keltner_upper"] - df["keltner_lower"]) / keltner_ma

        # True Range
        df["true_range"] = talib.TRANGE(high, low, close)

        # NATR (Normalized ATR)
        df["natr"] = talib.NATR(high, low, close, timeperiod=self.config.atr_period)

        return df

    def _add_price_features(
        self,
        df: pd.DataFrame,
        open_arr: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
    ) -> pd.DataFrame:
        """Add price-based features."""

        # Returns at various horizons
        df["return_1"] = pd.Series(close).pct_change(1).values
        df["return_5"] = pd.Series(close).pct_change(5).values
        df["return_10"] = pd.Series(close).pct_change(10).values
        df["return_20"] = pd.Series(close).pct_change(20).values

        # Log returns
        df["log_return_1"] = np.log(close / np.roll(close, 1))
        df["log_return_1"] = np.where(np.isinf(df["log_return_1"]), np.nan, df["log_return_1"])

        # Realized volatility (rolling std of returns)
        df["realized_vol_20"] = pd.Series(df["return_1"]).rolling(20).std().values * np.sqrt(252 * 24)

        # High-Low range
        df["hl_range"] = (high - low) / close
        df["hl_range_avg_20"] = pd.Series(df["hl_range"]).rolling(20).mean().values

        # Close position in day's range
        df["close_location"] = (close - low) / (high - low + 1e-10)

        # Gap (open vs previous close)
        prev_close = np.roll(close, 1)
        df["gap"] = (open_arr - prev_close) / prev_close
        df["gap"] = np.where(np.isinf(df["gap"]), np.nan, df["gap"])

        # Candle patterns (simplified)
        body = close - open_arr
        df["candle_body"] = body / (high - low + 1e-10)
        df["candle_upper_shadow"] = (high - np.maximum(open_arr, close)) / (high - low + 1e-10)
        df["candle_lower_shadow"] = (np.minimum(open_arr, close) - low) / (high - low + 1e-10)

        # Higher highs / lower lows
        df["higher_high"] = (high > np.roll(high, 1)).astype(int)
        df["lower_low"] = (low < np.roll(low, 1)).astype(int)

        return df

    def add_target_variable(
        self,
        df: pd.DataFrame,
        horizon: int = 1,
        target_type: str = "return",
    ) -> pd.DataFrame:
        """
        Add target variable for ML training.

        CRITICAL: Target is the FUTURE return, so it MUST be computed
        using shift(-horizon) to get forward returns. This is the
        value we're trying to predict.

        Args:
            df: DataFrame with close prices.
            horizon: Number of bars ahead for target.
            target_type: "return" for percentage return, "direction" for sign.

        Returns:
            DataFrame with target column added.
        """
        result = df.copy()

        # Forward return (what we're trying to predict)
        forward_return = result["close"].pct_change(horizon).shift(-horizon)
        result[f"target_return_{horizon}bar"] = forward_return

        if target_type == "direction":
            result[f"target_direction_{horizon}bar"] = np.sign(forward_return)

        logger.info(f"Added target variable: {horizon}-bar forward {target_type}")
        return result

    def prepare_ml_features(
        self,
        df: pd.DataFrame,
        drop_na: bool = True,
    ) -> pd.DataFrame:
        """
        Prepare features for ML training.

        Shifts all features by 1 bar to ensure we only use past
        information when predicting. Drops warmup period NaNs.

        Args:
            df: DataFrame with all features calculated.
            drop_na: Whether to drop rows with NaN values.

        Returns:
            ML-ready DataFrame.
        """
        result = df.copy()

        # Identify feature columns (exclude target, timestamp, symbol)
        exclude_cols = ["timestamp", "symbol", "open", "high", "low", "close", "volume", "spread"]
        exclude_cols += [c for c in result.columns if c.startswith("target_")]
        feature_cols = [c for c in result.columns if c not in exclude_cols]

        # Shift features by 1 to avoid look-ahead
        # (features at time t should predict returns at time t+1)
        for col in feature_cols:
            result[col] = result[col].shift(1)

        if drop_na:
            original_len = len(result)
            result = result.dropna()
            logger.info(f"Dropped {original_len - len(result)} rows with NaN (warmup period)")

        return result

    def get_feature_names(self) -> List[str]:
        """
        Get list of feature names that will be generated.

        Returns:
            List of feature column names.
        """
        # Generate a dummy DataFrame to get actual feature names
        dummy = pd.DataFrame({
            "open": [1.0] * 300,
            "high": [1.01] * 300,
            "low": [0.99] * 300,
            "close": [1.0] * 300,
            "volume": [1000] * 300,
        })

        features_df = self.calculate_all_features(dummy)

        exclude = ["open", "high", "low", "close", "volume"]
        return [c for c in features_df.columns if c not in exclude]
