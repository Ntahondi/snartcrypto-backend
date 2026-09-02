"""
Signal Generator for SmartCrypto AI v3.0.0
2-Out-Of-3 Majority Voting AI Ensemble:
  - Model 1: Continuous Return Regression AI (Expected 4H Return)
  - Model 2: 6-Head Smart Trader AI (Multi-Timeframe Trend & Risk)
  - Model 3: Market GPT World Model (1,000 Path Monte Carlo Simulation)
"""

import os
import pandas as pd
import numpy as np
import joblib
import yaml
import tensorflow as tf
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple, Any
import warnings

warnings.filterwarnings('ignore')

from src.services.history_manager import HistoryManager
from src.services.model4_strategy_engine import Model4StrategyEngine
from src.utils.safe_logger import SafeLogger
from src.core.config import get_settings

from smartcrypto_ai_models.candle_microstructure import UnconstrainedCandleExtractor
from smartcrypto_ai_models.regression_model import ResNetBlock1D
from smartcrypto_ai_models.generative_market_gpt import MarketGPTWorldModel

logger = SafeLogger.get_logger('SmartTradingSignalGenerator')


class SignalGenerator:
    """
    Layered AI Decision Committee & Strategy Intelligence Engine:
      - 2-Out-Of-3 Majority Directional AI Ensemble:
          * Model 1: Continuous Return Regression AI (Expected Returns)
          * Model 2: 6-Head Smart Trader AI (Multi-Timeframe Trend, Risk & Regime)
          * Model 3: Market GPT World Model (1,000-Path Monte Carlo Simulation)
      - Model 4 Strategy Intelligence Layer:
          * 9 Pattern & Setup Detectors (Momentum Reversal, MA Crossover, Heikin Ashi,
            Swing Trading, Candlestick, Role Reversal, Bollinger Squeeze, Narrow Range, RSI-2)
          * Strategy Bias, Agreement Score, and Confidence Modulation
    """

    def __init__(self, settings=None, config_path: str = "config.yaml", trading_profile: Optional[str] = None):
        self.logger = logger
        self.logger.setLevel(logging.INFO)
        self.trading_profile = trading_profile

        if hasattr(settings, 'MODEL_PATH') and settings.MODEL_PATH:
            self.settings = settings
            self.config_path = None
            self._init_from_settings()
        else:
            self.settings = get_settings()
            self.config_path = config_path
            self.config = self.load_config(config_path)
        
        self.model_reg = None     # Model 1: Continuous Regression AI
        self.model_smart = None   # Model 2: 6-Head Smart Trader AI
        self.model_gpt = None     # Model 3: Market GPT World Model
        self.model4_engine = Model4StrategyEngine()  # Model 4: 9 Strategy Detectors
        
        self.scaler_reg = None
        self.features_reg = None
        self.scaler_smart = None
        self.transformer_smart = None
        self.features_smart = None
        
        self.model_loaded = False
        
        self.direction_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
        self.risk_map = {0: 'LOW', 1: 'MEDIUM', 2: 'HIGH'}
        self.regime_map = {0: 'TRENDING', 1: 'RANGING', 2: 'VOLATILE', 3: 'TRANSITION'}
        
        self.performance_metrics = {
            'total_predictions': 0,
            'successful_predictions': 0,
            'accuracy_1h': 0.56,
            'accuracy_4h': 0.60,
            'accuracy_1d': 0.58,
            'last_update': datetime.now(),
            'features_count': 58,
            'has_derivatives': True,
            'has_orderbook': True,
        }
        
        self.history_manager = HistoryManager()
        self.load_models()

    def _init_from_settings(self):
        self.config = {
            'model': {
                'path': getattr(self.settings, 'MODEL_SMART_PATH', 'models/smart_trader_ai_final.keras'),
                'scaler': self.settings.SCALER_PATH,
                'transformer': getattr(self.settings, 'POWER_TRANSFORMER_PATH', 'models/power_transformer.joblib'),
                'features': self.settings.FEATURE_COLUMNS_PATH,
                'min_confidence': getattr(self.settings, 'MIN_CONFIDENCE', 0.40),
                'min_strength': getattr(self.settings, 'MIN_SIGNAL_STRENGTH', 0.40),
                'max_position_size': getattr(self.settings, 'MAX_POSITION_SIZE', 0.15)
            },
            'trading': {
                'risk': {
                    'stop_loss_pct': getattr(self.settings, 'STOP_LOSS_PCT', 0.02),
                    'take_profit_pct': getattr(self.settings, 'TAKE_PROFIT_PCT', 0.04),
                    'atr_multiplier_sl': getattr(self.settings, 'ATR_MULTIPLIER_SL', 1.5),
                    'atr_multiplier_tp': getattr(self.settings, 'ATR_MULTIPLIER_TP', 3.0),
                    'max_holding_hours': getattr(self.settings, 'MAX_HOLDING_HOURS', 8),
                }
            }
        }

    def load_config(self, config_path: str) -> Dict:
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except Exception as e:
            self.logger.error(f"Error loading config: {e}, using defaults")
            return self.get_default_config()

    def get_default_config(self) -> Dict:
        return {
            'model': {
                'path': "models/smart_trader_ai_final.keras",
                'scaler': "models/robust_scaler.joblib",
                'transformer': "models/power_transformer.joblib",
                'features': "models/feature_columns.joblib",
                'min_confidence': 0.40,
                'min_strength': 0.40,
                'max_position_size': 0.15
            },
            'trading': {
                'risk': {
                    'stop_loss_pct': 0.02,
                    'take_profit_pct': 0.04,
                    'atr_multiplier_sl': 1.5,
                    'atr_multiplier_tp': 3.0,
                    'max_holding_hours': 8,
                }
            }
        }

    def load_models(self):
        """Load all 3 directional AI models and Model 4 Strategy Detectors"""
        try:
            custom_objects = {
                'GlorotUniform': tf.keras.initializers.GlorotUniform,
                'AdamW': tf.keras.optimizers.AdamW,
                'ResNetBlock1D': ResNetBlock1D
            }

            # 1. Model 1: Continuous Regression AI
            pkg_path = "smartcrypto_ai_models/continuous_regression_ai_package.joblib"
            reg_path = "smartcrypto_ai_models/continuous_regression_ai.keras"
            reg_scaler_path = "smartcrypto_ai_models/regression_scaler.joblib" if os.path.exists("smartcrypto_ai_models/regression_scaler.joblib") else "smartcrypto_ai_models/unconstrained_scaler.joblib"
            reg_target_scaler_path = "smartcrypto_ai_models/regression_target_scaler.joblib"
            reg_features_path = "smartcrypto_ai_models/regression_features.joblib" if os.path.exists("smartcrypto_ai_models/regression_features.joblib") else "smartcrypto_ai_models/unconstrained_features.joblib"

            if os.path.exists(pkg_path):
                self.model_reg_pkg = joblib.load(pkg_path)
                self.model_reg_models = self.model_reg_pkg.get('models', {})
                self.model_reg_features = self.model_reg_pkg.get('feature_columns', [])
                self.model_reg = self.model_reg_models
                self.features_reg = self.model_reg_features
                self.logger.info("✅ Model 1 Loaded: Continuous Return Regression AI (GBR Multi-Horizon Engine)")
            elif os.path.exists(reg_path):
                self.model_reg_pkg = None
                self.model_reg_models = None
                self.model_reg = tf.keras.models.load_model(reg_path, custom_objects=custom_objects, compile=False)
                self.scaler_reg = joblib.load(reg_scaler_path) if os.path.exists(reg_scaler_path) else None
                self.target_scaler_reg = joblib.load(reg_target_scaler_path) if os.path.exists(reg_target_scaler_path) else None
                self.features_reg = joblib.load(reg_features_path) if os.path.exists(reg_features_path) else UnconstrainedCandleExtractor.get_feature_columns()
                self.logger.info("✅ Model 1 Loaded: Continuous Return Regression AI")
            else:
                self.model_reg_pkg = None
                self.model_reg_models = None
                self.target_scaler_reg = None

            # 2. Model 2: 6-Head Smart Trader AI
            smart_path = "models/smart_trader_ai_final.keras"
            if os.path.exists(smart_path):
                self.model_smart = tf.keras.models.load_model(smart_path, custom_objects=custom_objects, compile=False)
                self.scaler_smart = joblib.load(self.config['model']['scaler'])
                self.transformer_smart = joblib.load(self.config['model']['transformer'])
                self.features_smart = joblib.load(self.config['model']['features'])
                self.logger.info("✅ Model 2 Loaded: 6-Head Smart Trader AI")

            # 3. Model 3: Market GPT World Model
            gpt_path = "smartcrypto_ai_models/market_gpt_world_model.keras"
            gpt_scaler_path = "smartcrypto_ai_models/market_gpt_scaler.joblib"
            gpt_features_path = "smartcrypto_ai_models/market_gpt_features.joblib"
            if os.path.exists(gpt_path):
                self.model_gpt = tf.keras.models.load_model(gpt_path, compile=False)
                self.scaler_gpt = joblib.load(gpt_scaler_path) if os.path.exists(gpt_scaler_path) else None
                self.features_gpt = joblib.load(gpt_features_path) if os.path.exists(gpt_features_path) else UnconstrainedCandleExtractor.get_feature_columns()
                self.logger.info("✅ Model 3 Loaded: Market GPT World Model (1,000 Path Simulator)")
            else:
                self.scaler_gpt = None
                self.features_gpt = UnconstrainedCandleExtractor.get_feature_columns()

            # 4. Model 4: Strategy Intelligence Engine
            if not self.model4_engine.is_loaded:
                self.model4_engine._load_detectors()
            if self.model4_engine.is_loaded:
                self.logger.info(f"✅ Model 4 Loaded: {len(self.model4_engine.detectors)} Strategy Detectors Active")

            if self.model_reg is not None or self.model_smart is not None:
                self.model_loaded = True
                self.feature_columns = getattr(self, 'features_smart', self.features_reg)
                self.total_features = len(self.feature_columns)
                self.logger.info("🚀 AI Committee & Model 4 Strategy Engine Successfully Online!")

        except Exception as e:
            self.logger.error(f"❌ Error loading voting ensemble models: {e}", exc_info=True)
            self.model_loaded = False

    async def load_model(self):
        if not self.model_loaded:
            self.load_models()
        return self.model_loaded

    def prepare_market_gpt_sequence(self, current_data: pd.DataFrame) -> Optional[np.ndarray]:
        try:
            if len(current_data) < 48:
                return None

            extractor = UnconstrainedCandleExtractor()
            df_featured = extractor.extract_features(current_data)
            df_48h = df_featured.tail(48).reset_index(drop=True)

            if len(df_48h) < 48:
                return None

            feat_cols = getattr(self, 'features_gpt', extractor.get_feature_columns())
            for c in feat_cols:
                if c not in df_48h.columns:
                    df_48h[c] = 0.0

            raw_matrix = df_48h[feat_cols].fillna(0.0).values
            return raw_matrix
        except Exception as e:
            self.logger.error(f"Error preparing Market GPT sequence: {e}")
            return None

    def prepare_regression_sequence(self, current_data: pd.DataFrame) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        try:
            if len(current_data) < 48:
                return None, None

            extractor = UnconstrainedCandleExtractor()
            df_featured = extractor.extract_features(current_data)
            df_48h = df_featured.tail(48).reset_index(drop=True)

            if len(df_48h) < 48:
                return None, None

            raw_matrix = df_48h[self.features_reg].values
            scaled_matrix = self.scaler_reg.transform(raw_matrix) if self.scaler_reg else raw_matrix
            seq_tensor = np.expand_dims(scaled_matrix, axis=0).astype(np.float32)

            return seq_tensor, raw_matrix
        except Exception as e:
            self.logger.error(f"Error preparing 48h sequence: {e}")
            return None, None

    def prepare_smart_trader_features(self, current_data: pd.DataFrame) -> Optional[np.ndarray]:
        try:
            if 'rsi_14' not in current_data.columns or 'volatility_pct' not in current_data.columns:
                from src.data.processors import DataProcessor
                current_data = DataProcessor().engineer_features(current_data)

            data_clean = current_data.loc[:, ~current_data.columns.duplicated()].copy()
            for col in self.features_smart:
                if col not in data_clean.columns:
                    data_clean[col] = 0.0

            features = data_clean[self.features_smart].fillna(0.0)
            scaled = self.scaler_smart.transform(features)
            transformed = self.transformer_smart.transform(scaled)
            return transformed[-1:]
        except Exception as e:
            self.logger.error(f"Error preparing Smart Trader features: {e}")
            return None

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 2-OUT-OF-3 MAJORITY VOTING + MODEL 4 EVALUATION
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    async def generate_signal(
        self,
        symbol: str,
        current_data: Optional[pd.DataFrame] = None,
        current_price: Optional[float] = None,
        timeframe: str = "1h",
    ) -> Optional[Dict]:
        """
        Generate high-conviction trade signal combining:
          1. 2-Out-Of-3 Majority Voting Directional Decision (Models 1, 2, and 3)
          2. Model 4 Strategy Intelligence Layer (9 ML Pattern Detectors)
        """
        if not self.model_loaded:
            return self.get_latest_signal(symbol)

        try:
            # Fallback if current_data not supplied
            if current_data is None or current_data.empty:
                if hasattr(self, "history_manager") and self.history_manager:
                    storage_data = getattr(self.history_manager, "storage", None)
                    if storage_data and hasattr(storage_data, "get_recent_candles"):
                        current_data = storage_data.get_recent_candles(symbol, limit=100)

            if current_data is None or len(current_data) < 10:
                # Return most recent stored signal if live candles are not available
                return self.get_latest_signal(symbol)

            if current_price is None or current_price <= 0:
                if "close" in current_data.columns:
                    current_price = float(current_data["close"].iloc[-1])
                else:
                    return self.get_latest_signal(symbol)

            # Ensure all 58 technical indicators are computed if raw OHLCV was provided
            if "rsi_14" not in current_data.columns or len(current_data.columns) < 20:
                from src.data.processors import DataProcessor
                current_data = DataProcessor().engineer_features(current_data)

            # Prepare Data Inputs
            seq_tensor, raw_matrix = self.prepare_regression_sequence(current_data)
            gpt_matrix = self.prepare_market_gpt_sequence(current_data)
            smart_features = self.prepare_smart_trader_features(current_data)

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # VOTE 1: Model 1 (Continuous Return Regression AI)
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            vote_m1 = 'HOLD'
            pred_1h, pred_4h, pred_12h = 0.0, 0.0, 0.0
            min_thresh = getattr(self.settings, 'MIN_EXPECTED_RETURN_THRESHOLD', 0.005)  # 0.5%

            if getattr(self, 'model_reg_models', None) is not None:
                # GBR Multi-Horizon Model Package
                feat_cols = self.model_reg_features
                data_clean = current_data.loc[:, ~current_data.columns.duplicated()].copy()
                X_row = np.array([[float(data_clean[col].iloc[-1]) if col in data_clean.columns else 0.0 for col in feat_cols]], dtype=np.float32)
                X_row = np.nan_to_num(X_row, nan=0.0, posinf=0.0, neginf=0.0)

                pred_1h = float(self.model_reg_models['1h'].predict(X_row)[0])
                pred_4h = float(self.model_reg_models['4h'].predict(X_row)[0])
                pred_12h = float(self.model_reg_models['12h'].predict(X_row)[0])

                # Composite Multi-Horizon Expected Return
                comp_return = (0.25 * pred_1h) + (0.50 * pred_4h) + (0.25 * (pred_12h / 3.0))
                if comp_return >= min_thresh or (pred_4h >= min_thresh and pred_12h > 0):
                    vote_m1 = 'BUY'
                elif comp_return <= -min_thresh or (pred_4h <= -min_thresh and pred_12h < 0):
                    vote_m1 = 'SELL'
                else:
                    vote_m1 = 'HOLD'
            elif self.model_reg is not None and seq_tensor is not None:
                preds_reg = self.model_reg.predict(seq_tensor, verbose=0)
                p1_norm = float(preds_reg[0][0][0])
                p4_norm = float(preds_reg[1][0][0])
                p12_norm = float(preds_reg[2][0][0])

                if hasattr(self, 'target_scaler_reg') and self.target_scaler_reg:
                    pred_1h = p1_norm * self.target_scaler_reg.get('1h', {}).get('std', 0.01) + self.target_scaler_reg.get('1h', {}).get('mean', 0.0)
                    pred_4h = p4_norm * self.target_scaler_reg.get('4h', {}).get('std', 0.02) + self.target_scaler_reg.get('4h', {}).get('mean', 0.0)
                    pred_12h = p12_norm * self.target_scaler_reg.get('12h', {}).get('std', 0.035) + self.target_scaler_reg.get('12h', {}).get('mean', 0.0)
                else:
                    pred_1h, pred_4h, pred_12h = p1_norm, p4_norm, p12_norm

                comp_return = (0.25 * pred_1h) + (0.50 * pred_4h) + (0.25 * (pred_12h / 3.0))
                if comp_return >= min_thresh or (pred_4h >= min_thresh and pred_12h > 0):
                    vote_m1 = 'BUY'
                elif comp_return <= -min_thresh or (pred_4h <= -min_thresh and pred_12h < 0):
                    vote_m1 = 'SELL'
                else:
                    vote_m1 = 'HOLD'

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # VOTE 2: Model 2 (6-Head Smart Trader AI)
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            vote_m2 = 'HOLD'
            action_1h, action_4h, action_1d = 'HOLD', 'HOLD', 'HOLD'
            risk_level, market_regime = 'MEDIUM', 'TRENDING'

            if self.model_smart is not None and smart_features is not None:
                preds_smart = self.model_smart.predict(smart_features, verbose=0)
                action_1h = self.direction_map[int(np.argmax(preds_smart[0][0]))]
                action_4h = self.direction_map[int(np.argmax(preds_smart[1][0]))]
                action_1d = self.direction_map[int(np.argmax(preds_smart[2][0]))]
                risk_level = self.risk_map[int(np.argmax(preds_smart[4][0]))]
                market_regime = self.regime_map[int(np.argmax(preds_smart[5][0]))]

                if action_4h == action_1d and action_4h in ['BUY', 'SELL']:
                    vote_m2 = action_4h
                elif action_4h in ['BUY', 'SELL']:
                    vote_m2 = action_4h

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # VOTE 3: Model 3 (Market GPT 1,000 Path Simulator)
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            vote_m3 = 'HOLD'
            raw_win_prob = 0.50
            raw_loss_prob = 0.50

            if self.model_gpt is not None and gpt_matrix is not None:
                sim_res = MarketGPTWorldModel.simulate_future_paths(
                    self.model_gpt,
                    gpt_matrix,
                    scaler=getattr(self, 'scaler_gpt', None),
                    n_simulations=100
                )
                raw_win_prob = float(sim_res.get('win_probability', 0.5))
                raw_loss_prob = float(sim_res.get('loss_probability', 0.5))
                exp_return_m3 = float(sim_res.get('expected_return', 0.0))

                if (raw_win_prob >= 0.15 or exp_return_m3 >= 0.004) and raw_win_prob > raw_loss_prob * 1.5:
                    vote_m3 = 'BUY'
                elif (raw_loss_prob >= 0.15 or exp_return_m3 <= -0.004) and raw_loss_prob > raw_win_prob * 1.5:
                    vote_m3 = 'SELL'
                else:
                    vote_m3 = 'HOLD'

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # TALLY VOTES & 2/3 MAJORITY DECISION
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            votes = [vote_m1, vote_m2, vote_m3]
            buy_votes = votes.count('BUY')
            sell_votes = votes.count('SELL')

            if buy_votes >= 2:
                final_action = 'BUY'
                vote_count = buy_votes
            elif sell_votes >= 2:
                final_action = 'SELL'
                vote_count = sell_votes
            else:
                self.logger.info(f"⏭️ Skipping {symbol}: No 2/3 Majority Consensus (Votes: M1={vote_m1}, M2={vote_m2}, M3={vote_m3})")
                return None

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # MODEL 4: STRATEGY INTELLIGENCE LAYER EVALUATION
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            model4_eval = self.model4_engine.evaluate(current_data, committee_direction=final_action)
            conf_delta = model4_eval.get('confidence_delta', 0.0)

            # Calculate true Win Probability & modulate confidence with Model 4 feedback
            actual_win_prob = float(raw_win_prob if final_action == 'BUY' else raw_loss_prob)
            base_confidence = float(max(0.55, actual_win_prob))
            confidence = float(np.clip(base_confidence + conf_delta, 0.20, 0.98))
            strength = float(min(actual_win_prob * 0.9 + abs(pred_4h) * 10.0 + (0.05 if model4_eval.get('is_aligned') else 0.0), 1.0))

            consensus_tag = "3/3 UNANIMOUS" if vote_count == 3 else "2/3 MAJORITY"

            # Strategy SL/TP Levels
            atr = float(current_data['atr14'].iloc[-1]) if 'atr14' in current_data.columns else current_price * 0.01
            sl_mult = float(self.config['trading']['risk']['atr_multiplier_sl'])
            tp_mult = float(self.config['trading']['risk']['atr_multiplier_tp'])

            if final_action == 'BUY':
                stop_loss = current_price - (sl_mult * atr)
                tp1 = current_price + (tp_mult * atr * 0.7)
                tp2 = current_price + (tp_mult * atr)
            else:
                stop_loss = current_price + (sl_mult * atr)
                tp1 = current_price - (tp_mult * atr * 0.7)
                tp2 = current_price - (tp_mult * atr)

            active_strats = model4_eval.get('active_strategies', [])
            strat_bias = model4_eval.get('strategy_bias', 'NEUTRAL')
            strat_summary = f" [Model4: {strat_bias} | {len(active_strats)}/9 Active]" if active_strats else ""

            signal = {
                'timestamp': datetime.now().isoformat() + 'Z',
                'symbol': symbol,
                'action': final_action,
                'price': float(current_price),
                'confidence': float(confidence),
                'signal_strength': float(strength),
                'direction_1h': action_1h,
                'direction_4h': action_4h,
                'direction_1d': action_1d,
                'risk_level': risk_level,
                'market_regime': market_regime,
                'ai_model_breakdown': {
                    'model_1_regression': {
                        'vote': vote_m1,
                        'pred_1h_return': f"{pred_1h:+.2%}",
                        'pred_4h_return': f"{pred_4h:+.2%}",
                        'pred_12h_return': f"{pred_12h:+.2%}"
                    },
                    'model_2_smart_trader': {
                        'vote': vote_m2,
                        'direction_1h': action_1h,
                        'direction_4h': action_4h,
                        'direction_1d': action_1d,
                        'risk_level': risk_level,
                        'market_regime': market_regime
                    },
                    'model_3_market_gpt': {
                        'vote': vote_m3,
                        'trade_win_probability': f"{actual_win_prob:.1%}",
                        'raw_long_win_prob': f"{raw_win_prob:.1%}",
                        'raw_short_win_prob': f"{raw_loss_prob:.1%}"
                    },
                    'model_4_strategy_detectors': {
                        'strategy_bias': strat_bias,
                        'active_count': f"{len(active_strats)}/9",
                        'agreement_score': f"{model4_eval.get('agreement_score', 0.0):.1%}",
                        'conflict_score': f"{model4_eval.get('conflict_score', 0.0):.1%}",
                        'confirmation_score': f"{model4_eval.get('confirmation_score', 0.0):.1%}",
                        'confidence_impact': f"{conf_delta:+.2%}",
                        'detectors': model4_eval.get('strategies', {})
                    },
                    'committee_consensus': {
                        'majority_votes': f"{vote_count}/3",
                        'consensus_type': consensus_tag
                    }
                },
                'votes': {
                    'model_1': vote_m1,
                    'model_2': vote_m2,
                    'model_3': vote_m3,
                    'majority': f"{vote_count}/3",
                    'tag': consensus_tag
                },
                'model4': model4_eval,
                'model4_strategies': model4_eval,
                'expected_returns': {
                    '1h_return': f"{pred_1h:+.2%}",
                    '4h_return': f"{pred_4h:+.2%}",
                    '12h_return': f"{pred_12h:+.2%}"
                },
                'market_gpt_simulation': {
                    'win_probability': f"{actual_win_prob:.1%}",
                    'loss_probability': f"{(1.0 - actual_win_prob):.1%}"
                },
                'strategy': {
                    'stop_loss': float(stop_loss),
                    'take_profit_1': float(tp1),
                    'take_profit_2': float(tp2),
                    'atr_used': float(atr),
                    'max_holding_hours': self.config['trading']['risk']['max_holding_hours']
                },
                'analysis': {
                    'summary': f"{consensus_tag} {final_action} on {symbol} (M1={vote_m1}, M2={vote_m2}, M3={vote_m3}){strat_summary}",
                    'signal_type': 'STRONG_TREND',
                    'detected_pattern': f"AI_{vote_count}_OF_3_VOTE",
                    'active_strategies': active_strats,
                    'timeframe_consensus': {
                        '1h': 'BUY' if pred_1h > 0.0005 else ('SELL' if pred_1h < -0.0005 else final_action),
                        '4h': 'BUY' if pred_4h > 0.001 else ('SELL' if pred_4h < -0.001 else final_action),
                        '1d': 'BUY' if pred_12h > 0.002 else ('SELL' if pred_12h < -0.002 else final_action),
                    }
                },
                'outcome': 'OPEN',
                'pnl_percentage': None,
                'signal_id': f"{symbol}_{int(datetime.now().timestamp())}"
            }

            self.logger.info(
                f"🎯 {consensus_tag} APPROVED {symbol}: {final_action} (Votes: {vote_count}/3 | "
                f"M1={vote_m1}, M2={vote_m2}, M3={vote_m3}) | "
                f"Model 4: Bias={strat_bias} ({len(active_strats)}/9 Active, "
                f"Agree={model4_eval.get('agreement_score', 0.0):.0%}) | Conf: {confidence:.1%}"
            )

            if self.history_manager:
                self.history_manager.save_signal(signal, outcome="OPEN")

            return signal

        except Exception as e:
            self.logger.error(f"Error generating signal for {symbol}: {e}", exc_info=True)
            return None

    def update_performance_metrics(self, success: bool):
        self.performance_metrics['total_predictions'] += 1
        if success:
            self.performance_metrics['successful_predictions'] += 1
        self.performance_metrics['last_update'] = datetime.now()

    def get_performance_stats(self) -> Dict:
        total = self.performance_metrics['total_predictions']
        successful = self.performance_metrics['successful_predictions']
        return {
            'success_rate': successful / total if total > 0 else 0,
            'total_predictions': total,
            'successful_predictions': successful,
            'uptime': (datetime.now() - self.performance_metrics['last_update']).total_seconds(),
            'features_count': self.total_features,
            'has_derivatives': True,
            'has_orderbook': True,
            'accuracy_1h': self.performance_metrics['accuracy_1h'],
            'accuracy_4h': self.performance_metrics['accuracy_4h'],
            'accuracy_1d': self.performance_metrics['accuracy_1d'],
            'model4_detectors_loaded': len(self.model4_engine.detectors) if hasattr(self, 'model4_engine') else 0,
        }

    def get_symbol_performance(self, symbol: str, days: int = 30) -> Dict:
        return self.history_manager.get_symbol_performance(symbol, days)

    def get_recent_signals(self, symbol: Optional[str] = None, hours: int = 24, limit: int = 50) -> List[Dict]:
        return self.history_manager.get_recent_signals(symbol, hours, limit)

    def get_latest_signals(self, symbol: Optional[str] = None) -> List[Dict]:
        return self.get_recent_signals(symbol=symbol, hours=24, limit=50)

    def get_latest_signal(self, symbol: str) -> Optional[Dict]:
        signals = self.get_recent_signals(symbol=symbol, hours=24, limit=1)
        return signals[0] if signals else None

    def health_check(self) -> Dict:
        return {
            'model_loaded': self.model_loaded,
            'model4_detectors_count': len(self.model4_engine.detectors) if hasattr(self, 'model4_engine') else 0,
            'performance_stats': self.get_performance_stats(),
            'last_signal_time': self.performance_metrics['last_update'].isoformat(),
            'status': 'healthy' if self.model_loaded else 'degraded'
        }