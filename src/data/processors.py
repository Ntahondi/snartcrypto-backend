"""
src/data/processors.py

Data processing and feature engineering utilities for SmartCrypto AI v3.1.0.

Supports:
    - Continuous Return Regression AI
    - 6-Head Smart Trader AI
    - Market GPT World Model
    - Model 4 Strategy Detector Ensemble

The processor remains a shared feature-engineering layer.
Model-specific feature selection is performed by the consuming model.
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pandas_ta as ta

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Central feature-engineering engine for the SmartCrypto AI system.

    Produces:
        - Price/return features
        - Momentum features
        - Volatility features
        - Volume features
        - Derivatives features
        - Market microstructure features
        - Multi-timeframe features
        - Technical indicators
        - Stationary features
        - Model target features
    """

    def __init__(self):
        self.feature_columns: List[str] = []

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # NON-STATIONARY / EXCLUDED FEATURES
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        self.non_stationary_cols = [
            # Raw prices
            "open",
            "high",
            "low",
            "close",
            "log_close",

            # Price-level moving averages
            "sma20",
            "sma50",
            "ema12",
            "ema26",

            # Price-level volume features
            "vwap",
            "typical_price",
            "money_flow",

            # Multi-timeframe price levels
            "open_4h",
            "high_4h",
            "low_4h",
            "close_4h",

            "open_1d",
            "high_1d",
            "low_1d",
            "close_1d",

            # Bollinger price levels
            "BBL_20_2.0_2.0",
            "BBM_20_2.0_2.0",
            "BBU_20_2.0_2.0",

            # Ichimoku price levels
            "ichi_ICS_26",
            "ichi_IKS_26",
            "ichi_ISA_9",
            "ichi_ISB_26",
            "ichi_ITS_9",

            # Keltner price levels
            "KCBe_20_2",
            "KCLe_20_2",
            "KCUe_20_2",

            # Pivot price levels
            "pivot",
            "support1",
            "resistance1",

            # Raw derivatives
            "funding_rate",
            "open_interest",
            "open_interest_usd",
            "funding_high",
            "funding_low",

            # Target leakage
            "confidence_score",
            "market_regime",
            "risk_level",
            "volatility_regime",

            # Identifiers
            "timestamp",
            "symbol",
            "interval",
            "date",
        ]

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # MAIN FEATURE ENGINEERING
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer the complete SmartCrypto feature set.

        No future information is intentionally introduced.
        """

        try:
            if df is None or df.empty:
                logger.warning("Empty dataframe supplied to DataProcessor.")
                return pd.DataFrame()

            df = df.copy()

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # 0. NORMALIZE INPUT
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            df.columns = [
                str(col).strip()
                for col in df.columns
            ]

            required_columns = [
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]

            missing = [
                col
                for col in required_columns
                if col not in df.columns
            ]

            if missing:
                logger.error(
                    "Missing required market columns: %s",
                    missing,
                )
                return pd.DataFrame()

            for column in required_columns:
                df[column] = pd.to_numeric(
                    df[column],
                    errors="coerce",
                )

            # Preserve chronological order when timestamp exists.
            if "timestamp" in df.columns:
                try:
                    df["timestamp"] = pd.to_datetime(
                        df["timestamp"],
                        errors="coerce",
                    )

                    df = (
                        df.sort_values("timestamp")
                        .drop_duplicates(
                            subset=["timestamp"],
                            keep="last",
                        )
                        .reset_index(drop=True)
                    )
                except Exception:
                    pass

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # 1. BASIC PRICE / RETURN FEATURES
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            df["log_close"] = np.log(
                df["close"].clip(lower=1e-12)
            )

            df["ret_1"] = df["close"].pct_change()
            df["ret_3"] = df["close"].pct_change(3)

            df["range"] = (
                (df["high"] - df["low"])
                / (df["close"] + 1e-8)
            )

            df["body"] = (
                (df["close"] - df["open"])
                / (df["close"] + 1e-8)
            )

            df["upper_wick"] = (
                df["high"]
                - df[["open", "close"]].max(axis=1)
            ) / (df["close"] + 1e-8)

            df["lower_wick"] = (
                df[["open", "close"]].min(axis=1)
                - df["low"]
            ) / (df["close"] + 1e-8)

            df["close_position"] = (
                (df["close"] - df["low"])
                / (
                    df["high"]
                    - df["low"]
                    + 1e-8
                )
            )

            df["typical_price"] = (
                df["high"]
                + df["low"]
                + df["close"]
            ) / 3.0

            df["money_flow"] = (
                df["typical_price"]
                * df["volume"]
            )

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # 2. ROLLING VWAP
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            rolling_volume = (
                df["volume"]
                .rolling(24, min_periods=1)
                .sum()
            )

            rolling_money_flow = (
                df["money_flow"]
                .rolling(24, min_periods=1)
                .sum()
            )

            df["vwap"] = (
                rolling_money_flow
                / (rolling_volume + 1e-8)
            )

            df["price_vwap_ratio"] = (
                df["close"]
                / (df["vwap"] + 1e-8)
            )

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # 3. MOVING AVERAGES
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            df["sma20"] = (
                df["close"]
                .rolling(20, min_periods=1)
                .mean()
            )

            df["sma50"] = (
                df["close"]
                .rolling(50, min_periods=1)
                .mean()
            )

            df["ema12"] = (
                df["close"]
                .ewm(span=12, adjust=False)
                .mean()
            )

            df["ema26"] = (
                df["close"]
                .ewm(span=26, adjust=False)
                .mean()
            )

            df["ema12_ratio"] = (
                df["close"]
                / (df["ema12"] + 1e-8)
            )

            df["ema26_ratio"] = (
                df["close"]
                / (df["ema26"] + 1e-8)
            )

            df["ema_spread"] = (
                (df["ema12"] - df["ema26"])
                / (df["close"] + 1e-8)
            )

            df["sma20_ratio"] = (
                df["close"]
                / (df["sma20"] + 1e-8)
            )

            df["sma50_ratio"] = (
                df["close"]
                / (df["sma50"] + 1e-8)
            )

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # 4. MOMENTUM
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            df["rsi_14"] = ta.rsi(
                df["close"],
                length=14,
            )

            df["mom_1"] = df["close"].pct_change(1)
            df["mom_3"] = df["close"].pct_change(3)
            df["mom_6"] = df["close"].pct_change(6)
            df["mom_12"] = df["close"].pct_change(12)
            df["mom_24"] = df["close"].pct_change(24)

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # 5. STOCHASTIC
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            stoch = ta.stoch(
                df["high"],
                df["low"],
                df["close"],
                k=14,
                d=3,
            )

            if stoch is not None and not stoch.empty:
                df = pd.concat(
                    [df, stoch],
                    axis=1,
                )

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # 6. MACD
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            macd = ta.macd(
                df["close"],
                fast=12,
                slow=26,
                signal=9,
            )

            if macd is not None and not macd.empty:
                df = pd.concat(
                    [df, macd],
                    axis=1,
                )

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # 7. ATR / VOLATILITY
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            df["atr14"] = ta.atr(
                df["high"],
                df["low"],
                df["close"],
                length=14,
            )

            if (
                "atr14" not in df.columns
                or df["atr14"].isna().all()
                or (df["atr14"].fillna(0) == 0).all()
            ):
                logger.warning(
                    "ATR14 unavailable. Using manual ATR calculation."
                )

                high_low = (
                    df["high"] - df["low"]
                )

                high_close = np.abs(
                    df["high"]
                    - df["close"].shift(1)
                )

                low_close = np.abs(
                    df["low"]
                    - df["close"].shift(1)
                )

                true_range = pd.concat(
                    [
                        high_low,
                        high_close,
                        low_close,
                    ],
                    axis=1,
                ).max(axis=1)

                df["atr14"] = (
                    true_range
                    .rolling(14, min_periods=1)
                    .mean()
                )

            df["atr_pct"] = (
                df["atr14"]
                / (df["close"] + 1e-8)
            )

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # 8. BOLLINGER BANDS
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            bb = ta.bbands(
                df["close"],
                length=20,
                std=2,
            )

            if bb is not None and not bb.empty:
                df = pd.concat(
                    [df, bb],
                    axis=1,
                )

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # 9. KELTNER CHANNEL
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            kc = ta.kc(
                df["high"],
                df["low"],
                df["close"],
                length=20,
            )

            if kc is not None and not kc.empty:
                df = pd.concat(
                    [df, kc],
                    axis=1,
                )

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # 10. ADX / TREND
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            adx = ta.adx(
                df["high"],
                df["low"],
                df["close"],
                length=14,
            )

            if adx is not None and not adx.empty:
                df = pd.concat(
                    [df, adx],
                    axis=1,
                )

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # 11. DERIVATIVES
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            if "funding_rate" in df.columns:

                df["funding_rate"] = pd.to_numeric(
                    df["funding_rate"],
                    errors="coerce",
                )

                df["funding_8h"] = (
                    df["funding_rate"]
                    .rolling(8, min_periods=1)
                    .sum()
                )

                df["funding_24h"] = (
                    df["funding_rate"]
                    .rolling(24, min_periods=1)
                    .sum()
                )

                df["funding_change_8h"] = (
                    df["funding_rate"]
                    .diff(8)
                )

                df["funding_change_24h"] = (
                    df["funding_rate"]
                    .diff(24)
                )

                mean_90d = (
                    df["funding_rate"]
                    .rolling(90, min_periods=20)
                    .mean()
                )

                std_90d = (
                    df["funding_rate"]
                    .rolling(90, min_periods=20)
                    .std()
                )

                df["funding_zscore"] = (
                    df["funding_rate"] - mean_90d
                ) / (std_90d + 1e-8)

                df["funding_high"] = (
                    df["funding_rate"]
                    .rolling(90, min_periods=1)
                    .max()
                )

                df["funding_low"] = (
                    df["funding_rate"]
                    .rolling(90, min_periods=1)
                    .min()
                )

                df["funding_percentile"] = (
                    (
                        df["funding_rate"]
                        - df["funding_low"]
                    )
                    / (
                        df["funding_high"]
                        - df["funding_low"]
                        + 1e-8
                    )
                )

            if "open_interest" in df.columns:

                df["open_interest"] = pd.to_numeric(
                    df["open_interest"],
                    errors="coerce",
                )

                df["oi_change_1h"] = (
                    df["open_interest"]
                    .pct_change(1)
                )

                df["oi_change_24h"] = (
                    df["open_interest"]
                    .pct_change(24)
                )

                df["oi_momentum_7d"] = (
                    df["open_interest"]
                    .pct_change(168)
                )

                oi_high_24h = (
                    df["open_interest"]
                    .rolling(24, min_periods=1)
                    .max()
                )

                oi_low_24h = (
                    df["open_interest"]
                    .rolling(24, min_periods=1)
                    .min()
                )

                df["oi_position_24h"] = (
                    (
                        df["open_interest"]
                        - oi_low_24h
                    )
                    / (
                        oi_high_24h
                        - oi_low_24h
                        + 1e-8
                    )
                )

                if "open_interest_usd" in df.columns:

                    df["oi_volume_ratio"] = (
                        df["open_interest_usd"]
                        / (
                            df["volume"]
                            * df["close"]
                            + 1e-8
                        )
                    )

                else:

                    df["oi_volume_ratio"] = (
                        df["open_interest"]
                        / (df["volume"] + 1e-8)
                    )

                price_direction = np.sign(
                    df["close"].pct_change(24)
                )

                oi_direction = np.sign(
                    df["oi_change_24h"]
                )

                df["price_oi_divergence"] = (
                    price_direction
                    * oi_direction
                )

                df["oi_turnover_ratio"] = (
                    df["open_interest"]
                    / (
                        df["volume"]
                        .rolling(24, min_periods=1)
                        .mean()
                        + 1e-8
                    )
                )

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # 12. VOLUME
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            df["volume_ema_short"] = (
                df["volume"]
                .ewm(span=5, adjust=False)
                .mean()
            )

            df["volume_ema_long"] = (
                df["volume"]
                .ewm(span=20, adjust=False)
                .mean()
            )

            df["volume_oscillator"] = (
                df["volume_ema_short"]
                - df["volume_ema_long"]
            ) / (
                df["volume_ema_long"]
                + 1e-8
            )

            df["vol_12"] = (
                df["volume"]
                .rolling(12, min_periods=1)
                .mean()
            )

            df["vol_sma20"] = (
                df["volume"]
                .rolling(20, min_periods=1)
                .mean()
            )

            df["vol_ratio"] = (
                df["volume"]
                / (df["vol_sma20"] + 1e-8)
            )

            df["volume_change"] = (
                df["volume"].pct_change()
            )

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # 13. MULTI-TIMEFRAME FEATURES
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            df["open_4h"] = (
                df["open"]
                .rolling(4, min_periods=4)
                .apply(
                    lambda x: x[0],
                    raw=True,
                )
            )

            df["high_4h"] = (
                df["high"]
                .rolling(4, min_periods=4)
                .max()
            )

            df["low_4h"] = (
                df["low"]
                .rolling(4, min_periods=4)
                .min()
            )

            df["close_4h"] = (
                df["close"]
                .rolling(4, min_periods=4)
                .apply(
                    lambda x: x[-1],
                    raw=True,
                )
            )

            df["volume_4h"] = (
                df["volume"]
                .rolling(4, min_periods=4)
                .sum()
            )

            df["open_1d"] = (
                df["open"]
                .rolling(24, min_periods=24)
                .apply(
                    lambda x: x[0],
                    raw=True,
                )
            )

            df["high_1d"] = (
                df["high"]
                .rolling(24, min_periods=24)
                .max()
            )

            df["low_1d"] = (
                df["low"]
                .rolling(24, min_periods=24)
                .min()
            )

            df["close_1d"] = (
                df["close"]
                .rolling(24, min_periods=24)
                .apply(
                    lambda x: x[-1],
                    raw=True,
                )
            )

            df["volume_1d"] = (
                df["volume"]
                .rolling(24, min_periods=24)
                .sum()
            )

            df["price_pos_4h"] = (
                df["close"] - df["low_4h"]
            ) / (
                df["high_4h"]
                - df["low_4h"]
                + 1e-8
            )

            df["price_pos_1d"] = (
                df["close"] - df["low_1d"]
            ) / (
                df["high_1d"]
                - df["low_1d"]
                + 1e-8
            )

            df["return_4h"] = (
                df["close"]
                / (df["close"].shift(4) + 1e-8)
                - 1.0
            )

            df["return_1d"] = (
                df["close"]
                / (df["close"].shift(24) + 1e-8)
                - 1.0
            )

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # 14. PRICE DISTANCES / Z-SCORES
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            high_20 = (
                df["high"]
                .rolling(20, min_periods=1)
                .max()
            )

            low_20 = (
                df["low"]
                .rolling(20, min_periods=1)
                .min()
            )

            close_mean_20 = (
                df["close"]
                .rolling(20, min_periods=1)
                .mean()
            )

            close_std_20 = (
                df["close"]
                .rolling(20, min_periods=1)
                .std()
            )

            df["distance_from_20h_high"] = (
                df["close"] - high_20
            ) / (df["close"] + 1e-8)

            df["distance_from_20h_low"] = (
                df["close"] - low_20
            ) / (df["close"] + 1e-8)

            df["price_zscore_20"] = (
                df["close"] - close_mean_20
            ) / (close_std_20 + 1e-8)

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # 15. MARKET MICROSTRUCTURE
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            if (
                "taker_buy_quote_volume" in df.columns
                and "quote_asset_volume" in df.columns
            ):

                df["buy_pressure"] = (
                    df["taker_buy_quote_volume"]
                    / (
                        df["quote_asset_volume"]
                        + 1e-8
                    )
                )

                df["order_imbalance"] = (
                    2.0
                    * df["taker_buy_quote_volume"]
                    - df["quote_asset_volume"]
                ) / (
                    df["quote_asset_volume"]
                    + 1e-8
                )

            else:

                df["buy_pressure"] = 0.5
                df["order_imbalance"] = 0.0

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # 16. CANDLE MICROSTRUCTURE
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            candle_range = (
                df["high"]
                - df["low"]
                + 1e-8
            )

            df["body_to_range"] = (
                abs(df["close"] - df["open"])
                / candle_range
            )

            df["upper_wick_ratio"] = (
                df["high"]
                - df[["open", "close"]].max(axis=1)
            ) / candle_range

            df["lower_wick_ratio"] = (
                df[["open", "close"]].min(axis=1)
                - df["low"]
            ) / candle_range

            df["close_to_high"] = (
                df["high"] - df["close"]
            ) / candle_range

            df["close_to_low"] = (
                df["close"] - df["low"]
            ) / candle_range

            df["candle_direction"] = np.sign(
                df["close"] - df["open"]
            )

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # 17. SEASONALITY
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            if "timestamp" in df.columns:

                timestamp = pd.to_datetime(
                    df["timestamp"],
                    errors="coerce",
                )

                df["hour"] = timestamp.dt.hour
                df["day_of_week"] = (
                    timestamp.dt.dayofweek
                )

            else:

                df["hour"] = 0
                df["day_of_week"] = 0

            df["hour"] = df["hour"].fillna(0)
            df["day_of_week"] = (
                df["day_of_week"].fillna(0)
            )

            df["hour_sin"] = np.sin(
                2.0
                * np.pi
                * df["hour"]
                / 24.0
            )

            df["hour_cos"] = np.cos(
                2.0
                * np.pi
                * df["hour"]
                / 24.0
            )

            df["day_sin"] = np.sin(
                2.0
                * np.pi
                * df["day_of_week"]
                / 7.0
            )

            df["day_cos"] = np.cos(
                2.0
                * np.pi
                * df["day_of_week"]
                / 7.0
            )

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # 18. VOLATILITY FEATURES
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            df["volatility_rolling"] = (
                df["close"]
                .pct_change()
                .rolling(20, min_periods=2)
                .std()
            )

            if (
                "atr14" in df.columns
                and not df["atr14"].isna().all()
            ):

                df["volatility_pct"] = (
                    df["atr14"]
                    / (df["close"] + 1e-8)
                )

            else:

                df["volatility_pct"] = (
                    df["volatility_rolling"]
                )

            df["volatility_change"] = (
                df["volatility_pct"].pct_change()
            )

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # 19. TREND STRUCTURE
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            df["trend_strength_20"] = (
                df["close"]
                / (
                    df["close"]
                    .rolling(20, min_periods=1)
                    .mean()
                    + 1e-8
                )
                - 1.0
            )

            df["trend_strength_50"] = (
                df["close"]
                / (
                    df["close"]
                    .rolling(50, min_periods=1)
                    .mean()
                    + 1e-8
                )
                - 1.0
            )

            df["momentum_acceleration"] = (
                df["mom_6"]
                - df["mom_6"].shift(6)
            )

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # 20. NORMALIZE KNOWN PANDAS-TA NAMES
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            rename_map = {
                "EMA_12": "ema12",
                "EMA_26": "ema26",

                "RSI_14": "rsi_14",

                "ATRr_14": "atr14",

                "ADXR_14": "ADXR_14_2",

                "BBP_20_2.0": "BBP_20_2.0_2.0",
                "BBB_20_2.0": "BBB_20_2.0_2.0",
            }

            df = df.rename(
                columns=rename_map
            )

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # 21. FINAL CLEANING
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            numeric_columns = (
                df.select_dtypes(
                    include=[np.number]
                ).columns
            )

            df[numeric_columns] = (
                df[numeric_columns]
                .replace(
                    [np.inf, -np.inf],
                    np.nan,
                )
            )

            df[numeric_columns] = (
                df[numeric_columns]
                .ffill()
                .bfill()
                .fillna(0.0)
            )

            self.feature_columns = (
                self.get_stationary_features(df)
            )

            if "atr14" in df.columns:

                logger.info(
                    "ATR14 available: mean=%.8f, non-zero=%d",
                    float(df["atr14"].mean()),
                    int(
                        (
                            df["atr14"] > 0
                        ).sum()
                    ),
                )

            logger.info(
                "Feature engineering completed: "
                "%d rows, %d numeric features, "
                "%d stationary features",
                len(df),
                len(
                    df.select_dtypes(
                        include=[np.number]
                    ).columns
                ),
                len(self.feature_columns),
            )

            return df

        except Exception as exc:

            logger.error(
                "Error engineering features: %s",
                exc,
                exc_info=True,
            )

            return pd.DataFrame()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # STATIONARY FEATURES
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def get_stationary_features(
        self,
        df: pd.DataFrame,
    ) -> List[str]:

        if df is None or df.empty:
            return []

        all_columns = (
            df.select_dtypes(
                include=[np.number]
            ).columns.tolist()
        )

        stationary_features = []

        for column in all_columns:

            if column in self.non_stationary_cols:
                continue

            if column.startswith("future_"):
                continue

            if column.startswith("target_"):
                continue

            stationary_features.append(column)

        return stationary_features

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # MODEL 4 FEATURE ACCESS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def get_model4_features(
        self,
        df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Return features for Model 4 using the exact feature
        column names supplied by the trained detector package.

        This prevents the processor from inventing or changing
        the feature contract of the trained Model 4 detectors.
        """

        if df is None or df.empty:
            return pd.DataFrame()

        if feature_columns is None:
            feature_columns = self.feature_columns

        if not feature_columns:
            return pd.DataFrame()

        result = pd.DataFrame(index=df.index)

        for column in feature_columns:

            if column in df.columns:
                result[column] = df[column]

            else:
                result[column] = 0.0

        result = (
            result
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .ffill()
            .bfill()
            .fillna(0.0)
        )

        return result

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # MODEL 4 SINGLE-ROW INPUT
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def prepare_model4_input(
        self,
        df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
    ) -> Optional[np.ndarray]:
        """
        Prepare the latest observation for a Model 4 detector.

        The detector's trained feature_columns should be passed
        when available.
        """

        try:

            features = self.get_model4_features(
                df,
                feature_columns,
            )

            if features.empty:
                return None

            return (
                features
                .iloc[-1:]
                .to_numpy(
                    dtype=np.float32
                )
            )

        except Exception as exc:

            logger.error(
                "Error preparing Model 4 input: %s",
                exc,
                exc_info=True,
            )

            return None

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TARGET CREATION
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def create_targets(
        self,
        df: pd.DataFrame,
        thresholds: Optional[Dict] = None,
    ) -> pd.DataFrame:

        if df is None or df.empty:
            return pd.DataFrame()

        if thresholds is None:
            thresholds = {
                "1h": 0.003,
                "4h": 0.008,
                "1d": 0.015,
            }

        df = df.copy()

        shifts = {
            "1h": 1,
            "4h": 4,
            "1d": 24,
        }

        for horizon, threshold in thresholds.items():

            ret_column = (
                f"future_ret_{horizon}"
            )

            target_column = (
                f"target_{horizon}"
            )

            if ret_column not in df.columns:

                shift_value = shifts[horizon]

                df[ret_column] = (
                    df["close"].shift(-shift_value)
                    / (df["close"] + 1e-8)
                    - 1.0
                )

            df[target_column] = np.where(
                df[ret_column] > threshold,
                2,
                np.where(
                    df[ret_column] < -threshold,
                    0,
                    1,
                ),
            )

        return df

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # RISK TARGETS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def create_risk_targets(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        if df is None or df.empty:
            return pd.DataFrame()

        df = df.copy()

        if "atr14" not in df.columns:
            raise ValueError(
                "atr14 is required to create risk targets."
            )

        df["volatility_pct"] = (
            df["atr14"]
            / (df["close"] + 1e-8)
        )

        train_end = max(
            int(len(df) * 0.7),
            1,
        )

        train_vol = (
            df["volatility_pct"]
            .iloc[:train_end]
        )

        q33 = train_vol.quantile(0.33)
        q67 = train_vol.quantile(0.67)

        df["risk_level"] = pd.cut(
            df["volatility_pct"],
            bins=[
                -np.inf,
                q33,
                q67,
                np.inf,
            ],
            labels=[0, 1, 2],
        ).astype(int)

        return df

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # MARKET REGIME TARGETS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def create_regime_targets(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        if df is None or df.empty:
            return pd.DataFrame()

        df = df.copy()

        if "atr14" not in df.columns:
            raise ValueError(
                "atr14 is required to create regime targets."
            )

        df["volatility_pct"] = (
            df["atr14"]
            / (df["close"] + 1e-8)
        )

        adx_col = None

        for candidate in [
            "ADX_14",
            "adx",
        ]:
            if candidate in df.columns:
                adx_col = candidate
                break

        if adx_col is None:
            df["market_regime"] = 3
            return df

        train_end = max(
            int(len(df) * 0.7),
            1,
        )

        train_vol = (
            df["volatility_pct"]
            .iloc[:train_end]
        )

        q67 = train_vol.quantile(0.67)

        conditions = [
            df[adx_col] < 20,

            (
                (df[adx_col] > 25)
                & (
                    df["volatility_pct"]
                    < q67
                )
            ),

            df["volatility_pct"] >= q67,
        ]

        df["market_regime"] = np.select(
            conditions,
            [
                1,  # RANGING
                0,  # TRENDING
                2,  # VOLATILE
            ],
            default=3,  # TRANSITION
        )

        return df

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # CONFIDENCE TARGET
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def create_confidence_target(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        if df is None or df.empty:
            return pd.DataFrame()

        df = df.copy()

        required = [
            "target_1h",
            "target_4h",
            "target_1d",
        ]

        missing = [
            column
            for column in required
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                "Missing target columns: "
                f"{missing}"
            )

        agreement = (
            (df["target_1h"] == df["target_4h"])
            &
            (df["target_4h"] == df["target_1d"])
        ).astype(float)

        df["confidence_score"] = (
            agreement * 0.7
            + 0.3
        )

        return df

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # FEATURE ACCESS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def get_feature_columns(self) -> List[str]:
        return list(self.feature_columns)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # LATEST FEATURE VECTOR
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def get_latest_features(
        self,
        df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
    ) -> Optional[np.ndarray]:

        try:

            if df is None or df.empty:
                return None

            columns = (
                feature_columns
                if feature_columns is not None
                else self.feature_columns
            )

            if not columns:
                return None

            result = pd.DataFrame(index=df.index)

            for column in columns:

                if column in df.columns:
                    result[column] = df[column]

                else:
                    result[column] = 0.0

            result = (
                result
                .replace(
                    [np.inf, -np.inf],
                    np.nan,
                )
                .ffill()
                .bfill()
                .fillna(0.0)
            )

            return (
                result
                .iloc[-1:]
                .to_numpy(
                    dtype=np.float32
                )
            )

        except Exception as exc:

            logger.error(
                "Error extracting latest features: %s",
                exc,
                exc_info=True,
            )

            return None