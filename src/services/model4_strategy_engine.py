"""
src/services/model4_strategy_engine.py

Model 4 Strategy Intelligence Layer for SmartCrypto AI v3.

Evaluates all 9 pattern & setup detectors:
    1. momentum_reversal
    2. ma_crossover
    3. heikin_ashi
    4. swing_trading
    5. candlestick
    6. role_reversal
    7. bollinger_squeeze
    8. narrow_range
    9. rsi_2

Provides:
    - Independent probability & active state for each detector
    - Strategy directional bias (BUY / SELL / NEUTRAL)
    - Strategy agreement, conflict, and confirmation scores
    - Confidence boost/penalty calculations for SignalGenerator
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

logger = logging.getLogger("model4_strategy_engine")


class Model4StrategyEngine:
    """
    Model 4 Strategy Intelligence Engine.
    Loads and runs inference across all 9 strategy detectors.
    """

    STRATEGY_NAMES: List[str] = [
        "momentum_reversal",
        "ma_crossover",
        "heikin_ashi",
        "swing_trading",
        "candlestick",
        "role_reversal",
        "bollinger_squeeze",
        "narrow_range",
        "rsi_2",
    ]

    def __init__(
        self,
        package_path: Union[str, Path] = "data/model4/MODEL4_STRATEGY_DETECTOR_V1/MODEL4_STRATEGY_DETECTOR_V1_PACKAGE.joblib",
        models_dir: Union[str, Path] = "data/model4/MODEL4_STRATEGY_DETECTOR_V1/models",
    ):
        self.package_path = Path(package_path)
        self.models_dir = Path(models_dir)
        self.detectors: Dict[str, Dict[str, Any]] = {}
        self.feature_columns: List[str] = []
        self.is_loaded: bool = False
        self._load_detectors()

    def _load_detectors(self) -> None:
        """Load all 9 strategy detectors from the package or individual files."""
        # Try loading consolidated package first
        if self.package_path.exists():
            try:
                pkg = joblib.load(self.package_path)
                if isinstance(pkg, dict) and "detectors" in pkg:
                    self.detectors = pkg["detectors"]
                    self.feature_columns = pkg.get("feature_columns", [])
                    self.is_loaded = len(self.detectors) >= len(self.STRATEGY_NAMES)
                    logger.info(
                        f"Loaded {len(self.detectors)} Model 4 detectors from package {self.package_path}"
                    )
                    return
            except Exception as exc:
                logger.warning(
                    f"Failed to load Model 4 package ({exc}), falling back to individual model files"
                )

        # Fallback: load individual .joblib files
        if self.models_dir.exists():
            for strategy in self.STRATEGY_NAMES:
                model_file = self.models_dir / f"{strategy}_detector.joblib"
                if model_file.exists():
                    try:
                        loaded = joblib.load(model_file)
                        if isinstance(loaded, dict):
                            self.detectors[strategy] = loaded
                        else:
                            self.detectors[strategy] = {
                                "strategy": strategy,
                                "model": loaded,
                                "threshold": 0.50,
                                "feature_columns": [],
                            }
                    except Exception as exc:
                        logger.error(f"Error loading detector {model_file}: {exc}")

            self.is_loaded = len(self.detectors) > 0
            logger.info(
                f"Loaded {len(self.detectors)} individual Model 4 detectors from {self.models_dir}"
            )
        else:
            logger.warning(f"Model 4 directory not found: {self.models_dir}")

    @staticmethod
    def extract_model4_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the 94 Model 4 strategy features from OHLCV candles.
        Optimized vectorized implementation.
        """
        if len(df) < 30:
            return pd.DataFrame()

        df_out = pd.DataFrame(index=df.index)

        # Base series
        c = df["close"].astype(float)
        o = df["open"].astype(float)
        h = df["high"].astype(float)
        l = df["low"].astype(float)
        v = df["volume"].astype(float) if "volume" in df.columns else pd.Series(0.0, index=df.index)

        trades = df["trades_count"].astype(float) if "trades_count" in df.columns else v * 0.1
        tb_base = df["taker_buy_base_volume"].astype(float) if "taker_buy_base_volume" in df.columns else v * 0.5
        tb_quote = df["taker_buy_quote_volume"].astype(float) if "taker_buy_quote_volume" in df.columns else tb_base * c

        df_out["open"] = o
        df_out["high"] = h
        df_out["low"] = l
        df_out["close"] = c
        df_out["volume"] = v
        df_out["trades_count"] = trades
        df_out["taker_buy_base_volume"] = tb_base
        df_out["taker_buy_quote_volume"] = tb_quote

        # 1. Momentum features
        df_out["return_3"] = (c - c.shift(3)) / (c.shift(3) + 1e-12)
        df_out["log_return_1"] = np.log(np.maximum(c, 1e-12) / np.maximum(c.shift(1), 1e-12))
        m6 = c - c.shift(6)
        m12 = c - c.shift(12)
        df_out["momentum_6"] = m6
        df_out["momentum_12"] = m12
        df_out["momentum_acceleration"] = m6 - m6.shift(3)
        df_out["momentum_acceleration_12"] = m12 - m12.shift(6)

        # 2. Moving averages
        ema12 = c.ewm(span=12, adjust=False).mean()
        ema26 = c.ewm(span=26, adjust=False).mean()
        df_out["price_vs_ema12"] = (c - ema12) / (ema12 + 1e-12)
        df_out["price_vs_ema26"] = (c - ema26) / (ema26 + 1e-12)
        df_out["ema12_minus_ema26"] = (ema12 - ema26) / (c + 1e-12)
        df_out["ema_bull_cross"] = ((ema12 > ema26) & (ema12.shift(1) <= ema26.shift(1))).astype(float)
        df_out["ema_bear_cross"] = ((ema12 < ema26) & (ema12.shift(1) >= ema26.shift(1))).astype(float)

        # 3. Candlestick anatomy
        c_range = np.maximum(h - l, 1e-12)
        body = np.abs(c - o)
        df_out["candle_range"] = c_range
        df_out["body"] = body
        df_out["body_pct"] = body / c_range
        u_wick = h - np.maximum(o, c)
        l_wick = np.minimum(o, c) - l
        df_out["upper_wick"] = u_wick
        df_out["lower_wick"] = l_wick
        df_out["upper_wick_pct"] = u_wick / c_range
        df_out["lower_wick_pct"] = l_wick / c_range
        df_out["candle_direction"] = np.where(c > o, 1.0, np.where(c < o, -1.0, 0.0))
        df_out["doji"] = (df_out["body_pct"] < 0.1).astype(float)
        df_out["long_body"] = (df_out["body_pct"] > 0.6).astype(float)
        df_out["hammer"] = ((df_out["lower_wick_pct"] > 0.6) & (df_out["upper_wick_pct"] < 0.15) & (df_out["body_pct"] > 0.15)).astype(float)
        df_out["shooting_star"] = ((df_out["upper_wick_pct"] > 0.6) & (df_out["lower_wick_pct"] < 0.15) & (df_out["body_pct"] > 0.15)).astype(float)
        df_out["inside_bar"] = ((h < h.shift(1)) & (l > l.shift(1))).astype(float)
        df_out["outside_bar"] = ((h > h.shift(1)) & (l < l.shift(1))).astype(float)

        # 4. ATR & Volatility
        tr = np.maximum(h - l, np.maximum(np.abs(h - c.shift(1)), np.abs(l - c.shift(1))))
        atr14 = tr.rolling(14).mean()
        atr28 = tr.rolling(28).mean()
        atr7 = tr.rolling(7).mean()
        df_out["atr_14"] = atr14
        df_out["atr_14_pct"] = atr14 / (c + 1e-12)
        df_out["atr_ratio_short_long"] = atr7 / (atr28 + 1e-12)
        df_out["range_vs_atr"] = c_range / (atr14 + 1e-12)
        vol_mean20 = v.rolling(20).mean()
        df_out["volume_ratio_20"] = v / (vol_mean20 + 1e-12)
        df_out["volume_change_1"] = v.pct_change(1).fillna(0.0)
        df_out["volume_change_6"] = v.pct_change(6).fillna(0.0)

        # 5. RSI (2 & 14)
        def _calc_rsi(series: pd.Series, period: int) -> pd.Series:
            delta = series.diff()
            gain = (delta.where(delta > 0, 0.0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
            rs = gain / (loss + 1e-12)
            return 100.0 - (100.0 / (1.0 + rs))

        rsi2 = _calc_rsi(c, 2)
        rsi14 = _calc_rsi(c, 14)
        df_out["rsi_2"] = rsi2
        df_out["rsi_14"] = rsi14
        df_out["rsi_2_oversold"] = (rsi2 < 10.0).astype(float)
        df_out["rsi_2_overbought"] = (rsi2 > 90.0).astype(float)
        df_out["rsi_2_extreme"] = ((rsi2 < 5.0) | (rsi2 > 95.0)).astype(float)
        df_out["rsi_2_change"] = rsi2 - rsi2.shift(1)

        # 6. Heikin Ashi
        ha_close = (o + h + l + c) / 4.0
        ha_open = (o.shift(1) + c.shift(1)) / 2.0
        ha_open.iloc[0] = (o.iloc[0] + c.iloc[0]) / 2.0
        for i in range(1, min(len(df), 50)):
            ha_open.iloc[i] = (ha_open.iloc[i - 1] + ha_close.iloc[i - 1]) / 2.0

        ha_high = np.maximum(h, np.maximum(ha_open, ha_close))
        ha_low = np.minimum(l, np.minimum(ha_open, ha_close))
        ha_range = np.maximum(ha_high - ha_low, 1e-12)
        ha_body = np.abs(ha_close - ha_open)
        df_out["ha_body_pct"] = ha_body / ha_range
        ha_u_wick = ha_high - np.maximum(ha_open, ha_close)
        ha_l_wick = np.minimum(ha_open, ha_close) - ha_low
        df_out["ha_upper_wick_pct"] = ha_u_wick / ha_range
        df_out["ha_lower_wick_pct"] = ha_l_wick / ha_range
        df_out["ha_direction"] = np.where(ha_close > ha_open, 1.0, -1.0)
        df_out["ha_wick_balance"] = df_out["ha_upper_wick_pct"] - df_out["ha_lower_wick_pct"]
        df_out["ha_trend_strength"] = np.abs(df_out["ha_wick_balance"]) * df_out["ha_body_pct"]
        df_out["ha_reversal"] = (df_out["ha_direction"] != df_out["ha_direction"].shift(1)).astype(float)

        # 7. Bollinger Bands
        bb_mid20 = c.rolling(20).mean()
        bb_std20 = c.rolling(20).std()
        bb_upper20 = bb_mid20 + 2.0 * bb_std20
        bb_lower20 = bb_mid20 - 2.0 * bb_std20
        bb_width20 = (bb_upper20 - bb_lower20) / (bb_mid20 + 1e-12)
        df_out["bb_width_20"] = bb_width20
        df_out["bb_position"] = (c - bb_lower20) / (bb_upper20 - bb_lower20 + 1e-12)
        # Fast rolling percentile
        bb_width_p50 = (
            bb_width20.rolling(50, min_periods=10)
            .apply(lambda x: (x < x[-1]).mean() if len(x) > 0 else 0.5, raw=True)
            .fillna(0.5)
        )
        df_out["bb_width_percentile_50"] = bb_width_p50
        df_out["bb_squeeze_adaptive"] = (bb_width_p50 < 0.20).astype(float)
        df_out["bb_squeeze_absolute"] = (bb_width20 < 0.03).astype(float)
        df_out["bb_squeeze"] = np.maximum(df_out["bb_squeeze_adaptive"], df_out["bb_squeeze_absolute"])
        df_out["bb_breakout_up"] = ((c > bb_upper20.shift(1)) & (c.shift(1) <= bb_upper20.shift(1))).astype(float)
        df_out["bb_breakout_down"] = ((c < bb_lower20.shift(1)) & (c.shift(1) >= bb_lower20.shift(1))).astype(float)

        # 8. Narrow Range
        r_min4 = c_range.rolling(4).min()
        r_min7 = c_range.rolling(7).min()
        r_min14 = c_range.rolling(14).min()
        df_out["nr4"] = (c_range <= r_min4).astype(float)
        df_out["nr7"] = (c_range <= r_min7).astype(float)
        df_out["nr14"] = (c_range <= r_min14).astype(float)
        r_p20 = (
            c_range.rolling(20, min_periods=5)
            .apply(lambda x: (x < x[-1]).mean() if len(x) > 0 else 0.5, raw=True)
            .fillna(0.5)
        )
        df_out["range_percentile_20"] = r_p20
        df_out["narrow_range_score"] = (df_out["nr4"] + df_out["nr7"] + df_out["nr14"]) / 3.0
        df_out["range_expansion"] = (c_range > 1.5 * c_range.shift(1)).astype(float)

        # 9. Support / Resistance / Swing / Role Reversal
        r_high20 = h.rolling(20).max().shift(1)
        r_low20 = l.rolling(20).min().shift(1)
        df_out["distance_to_confirmed_resistance"] = (r_high20 - c) / (c + 1e-12)
        df_out["distance_to_confirmed_support"] = (c - r_low20) / (c + 1e-12)
        df_out["higher_high"] = (h > h.shift(1)).astype(float)
        df_out["lower_high"] = (h < h.shift(1)).astype(float)
        df_out["higher_low"] = (l > l.shift(1)).astype(float)
        df_out["lower_low"] = (l < l.shift(1)).astype(float)
        res_break = (c > r_high20).astype(float)
        sup_break = (c < r_low20).astype(float)
        df_out["resistance_break"] = res_break
        df_out["support_break"] = sup_break
        df_out["breakout_distance"] = np.where(
            res_break > 0,
            (c - r_high20) / (c + 1e-12),
            np.where(sup_break > 0, (r_low20 - c) / (c + 1e-12), 0.0),
        )
        df_out["distance_to_resistance"] = df_out["distance_to_confirmed_resistance"]
        df_out["distance_to_support"] = df_out["distance_to_confirmed_support"]
        prev_res_break = res_break.shift(1).fillna(0.0)
        prev_sup_break = sup_break.shift(1).fillna(0.0)
        df_out["previous_resistance_break"] = prev_res_break
        df_out["previous_support_break"] = prev_sup_break
        df_out["resistance_role_reversal"] = ((prev_res_break > 0) & (l <= r_high20) & (c >= r_high20)).astype(float)
        df_out["support_role_reversal"] = ((prev_sup_break > 0) & (h >= r_low20) & (c <= r_low20)).astype(float)
        df_out["resistance_rejection"] = ((h >= r_high20) & (c < r_high20) & (df_out["upper_wick_pct"] > 0.3)).astype(float)
        df_out["support_rejection"] = ((l <= r_low20) & (c > r_low20) & (df_out["lower_wick_pct"] > 0.3)).astype(float)
        df_out["role_reversal_strength"] = (df_out["resistance_role_reversal"] + df_out["support_role_reversal"]) * df_out["body_pct"]
        df_out["role_rejection_strength"] = (df_out["resistance_rejection"] + df_out["support_rejection"]) * (
            df_out["upper_wick_pct"] + df_out["lower_wick_pct"]
        )

        # 10. Multi-timeframe context
        df_out["ctx4h_volume"] = v.rolling(4).sum() / ((v.rolling(24).sum() / 6.0) + 1e-12)
        df_out["ctx4h_return"] = (c - c.shift(4)) / (c.shift(4) + 1e-12)
        h4 = h.rolling(4).max()
        l4 = l.rolling(4).min()
        df_out["ctx4h_range_pct"] = (h4 - l4) / (c + 1e-12)
        df_out["ctx4h_price_position"] = (c - l4) / (h4 - l4 + 1e-12)

        df_out["ctx1d_volume"] = v.rolling(24).sum() / ((v.rolling(120).sum() / 5.0) + 1e-12)
        df_out["ctx1d_return"] = (c - c.shift(24)) / (c.shift(24) + 1e-12)
        h24 = h.rolling(24).max()
        l24 = l.rolling(24).min()
        df_out["ctx1d_range_pct"] = (h24 - l24) / (c + 1e-12)
        df_out["ctx1d_price_position"] = (c - l24) / (h24 - l24 + 1e-12)

        # Clean NaNs and Infs
        df_out = df_out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return df_out

    def evaluate(self, market_df: pd.DataFrame, committee_direction: str = "BUY") -> Dict[str, Any]:
        """
        Evaluate market conditions across all 9 strategy detectors.

        Args:
            market_df: Recent candle DataFrame (at least 30 candles)
            committee_direction: Direction voted by the 2/3 AI Committee ("BUY" or "SELL")

        Returns:
            Comprehensive Model 4 strategy intelligence payload.
        """
        if not self.is_loaded or len(market_df) < 30:
            return self._empty_response()

        try:
            feat_df = self.extract_model4_features(market_df)
            if feat_df.empty:
                return self._empty_response()

            latest_row = feat_df.iloc[[-1]]
            results: Dict[str, Dict[str, Any]] = {}
            bullish_count = 0
            bearish_count = 0
            neutral_count = 0
            active_count = 0

            # Evaluate each of the 9 detectors
            for strategy in self.STRATEGY_NAMES:
                det = self.detectors.get(strategy)
                if not det:
                    results[strategy] = {
                        "name": strategy,
                        "active": False,
                        "probability": 0.0,
                        "threshold": 0.50,
                        "direction": "NEUTRAL",
                    }
                    neutral_count += 1
                    continue

                model = det.get("model")
                imputer = det.get("imputer")
                scaler = det.get("scaler")
                threshold = float(det.get("threshold", 0.50))
                feat_cols = det.get("feature_columns", self.feature_columns)

                # Prepare input vector
                if feat_cols:
                    X_df = pd.DataFrame(
                        [[latest_row[col].values[0] if col in latest_row.columns else 0.0 for col in feat_cols]],
                        columns=feat_cols,
                    )
                else:
                    X_df = latest_row.copy()

                # Transform with feature names preserved
                X_vals = X_df
                if imputer is not None:
                    try:
                        X_vals = imputer.transform(X_vals)
                        if isinstance(X_vals, np.ndarray) and feat_cols:
                            X_vals = pd.DataFrame(X_vals, columns=feat_cols)
                    except Exception:
                        pass

                if scaler is not None:
                    try:
                        X_vals = scaler.transform(X_vals)
                    except Exception:
                        pass

                # Predict probability
                try:
                    if hasattr(model, "predict_proba"):
                        proba = float(model.predict_proba(X_vals)[0, 1])
                    elif hasattr(model, "decision_function"):
                        raw = float(model.decision_function(X_vals)[0])
                        proba = float(1.0 / (1.0 + np.exp(-raw)))
                    elif hasattr(model, "predict"):
                        proba = float(model.predict(X_vals)[0])
                    else:
                        proba = 0.0
                except Exception as exc:
                    logger.debug(f"Error predicting for {strategy}: {exc}")
                    proba = 0.0

                is_active = proba >= threshold
                if is_active:
                    active_count += 1

                # Determine strategy directional intent
                direction = self._infer_strategy_direction(strategy, latest_row, proba, is_active)
                if direction == "BUY":
                    bullish_count += 1
                elif direction == "SELL":
                    bearish_count += 1
                else:
                    neutral_count += 1

                results[strategy] = {
                    "name": strategy,
                    "active": bool(is_active),
                    "probability": round(float(proba), 4),
                    "threshold": round(float(threshold), 4),
                    "direction": direction,
                }

            # Aggregated metrics
            total_active = active_count
            bias = "BUY" if bullish_count > bearish_count else ("SELL" if bearish_count > bullish_count else "NEUTRAL")

            # Agreement with 2/3 Committee
            if committee_direction == "BUY":
                aligned_count = bullish_count
                conflicted_count = bearish_count
            elif committee_direction == "SELL":
                aligned_count = bearish_count
                conflicted_count = bullish_count
            else:
                aligned_count = 0
                conflicted_count = 0

            agreement_score = round(aligned_count / 9.0, 4)
            conflict_score = round(conflicted_count / 9.0, 4)
            confirmation_score = round((aligned_count - conflicted_count + 9.0) / 18.0, 4)

            # Confidence adjustment (-0.20 to +0.15)
            if agreement_score >= 0.55:
                confidence_delta = min(0.15, (agreement_score - 0.50) * 0.35)
            elif conflict_score >= 0.45:
                confidence_delta = max(-0.20, -(conflict_score - 0.35) * 0.40)
            else:
                confidence_delta = 0.0

            return {
                "bias": bias,
                "strategy_bias": bias,
                "active_count": total_active,
                "total_count": 9,
                "bullish_count": bullish_count,
                "bearish_count": bearish_count,
                "neutral_count": neutral_count,
                "agreement_score": agreement_score,
                "conflict_score": conflict_score,
                "confirmation_score": confirmation_score,
                "confidence_delta": round(confidence_delta, 4),
                "is_aligned": (bias == committee_direction) if committee_direction in ["BUY", "SELL"] else False,
                "strategies": results,
                "active_strategies": [s for s, d in results.items() if d["active"]],
            }

        except Exception as exc:
            logger.error(f"Error evaluating Model 4 strategies: {exc}", exc_info=True)
            return self._empty_response()

    def _infer_strategy_direction(
        self, strategy: str, row: pd.DataFrame, proba: float, is_active: bool
    ) -> str:
        """Infer directional bias (BUY, SELL, NEUTRAL) for each strategy."""
        if not is_active:
            return "HOLD"

        try:
            if strategy == "rsi_2":
                # RSI 2 oversold = BUY bounce; overbought = SELL
                rsi2 = float(row["rsi_2"].values[0]) if "rsi_2" in row else 50.0
                return "BUY" if rsi2 < 30.0 else ("SELL" if rsi2 > 70.0 else "HOLD")

            elif strategy == "momentum_reversal":
                # Reversal from negative momentum = BUY; reversal from positive = SELL
                ret3 = float(row["return_3"].values[0]) if "return_3" in row else 0.0
                return "BUY" if ret3 < 0 else "SELL"

            elif strategy == "ma_crossover":
                bull = float(row["ema_bull_cross"].values[0]) if "ema_bull_cross" in row else 0.0
                diff = float(row["ema12_minus_ema26"].values[0]) if "ema12_minus_ema26" in row else 0.0
                return "BUY" if (bull > 0 or diff > 0) else "SELL"

            elif strategy == "heikin_ashi":
                ha_dir = float(row["ha_direction"].values[0]) if "ha_direction" in row else 0.0
                return "BUY" if ha_dir > 0 else "SELL"

            elif strategy == "candlestick":
                hammer = float(row["hammer"].values[0]) if "hammer" in row else 0.0
                star = float(row["shooting_star"].values[0]) if "shooting_star" in row else 0.0
                c_dir = float(row["candle_direction"].values[0]) if "candle_direction" in row else 0.0
                if hammer > 0 or c_dir > 0:
                    return "BUY"
                elif star > 0 or c_dir < 0:
                    return "SELL"
                return "HOLD"

            elif strategy == "role_reversal":
                res_rr = float(row["resistance_role_reversal"].values[0]) if "resistance_role_reversal" in row else 0.0
                sup_rr = float(row["support_role_reversal"].values[0]) if "support_role_reversal" in row else 0.0
                if res_rr > 0:
                    return "BUY"
                elif sup_rr > 0:
                    return "SELL"
                return "HOLD"

            elif strategy == "bollinger_squeeze":
                pos = float(row["bb_position"].values[0]) if "bb_position" in row else 0.5
                return "BUY" if pos > 0.5 else "SELL"

            elif strategy == "narrow_range":
                c_dir = float(row["candle_direction"].values[0]) if "candle_direction" in row else 0.0
                return "BUY" if c_dir > 0 else ("SELL" if c_dir < 0 else "HOLD")

            elif strategy == "swing_trading":
                hh = float(row["higher_high"].values[0]) if "higher_high" in row else 0.0
                hl = float(row["higher_low"].values[0]) if "higher_low" in row else 0.0
                return "BUY" if (hh > 0 or hl > 0) else "SELL"

        except Exception:
            pass

        return "HOLD"

    def _empty_response(self) -> Dict[str, Any]:
        """Return fallback response when detectors are unavailable."""
        return {
            "bias": "NEUTRAL",
            "strategy_bias": "NEUTRAL",
            "active_count": 0,
            "total_count": 9,
            "bullish_count": 0,
            "bearish_count": 0,
            "neutral_count": 9,
            "agreement_score": 0.0,
            "conflict_score": 0.0,
            "confirmation_score": 0.5,
            "confidence_delta": 0.0,
            "is_aligned": False,
            "strategies": {
                s: {"name": s, "active": False, "probability": 0.0, "threshold": 0.50, "direction": "HOLD"}
                for s in self.STRATEGY_NAMES
            },
            "active_strategies": [],
        }
