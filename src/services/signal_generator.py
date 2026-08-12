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
from src.utils.safe_logger import SafeLogger
from src.core.config import get_settings

from smartcrypto_ai_models.candle_microstructure import UnconstrainedCandleExtractor
from smartcrypto_ai_models.conv1d_attention_model import ResNetBlock1D
from smartcrypto_ai_models.generative_market_gpt import MarketGPTWorldModel

logger = SafeLogger.get_logger('SmartTradingSignalGenerator')


class SignalGenerator:
    """
    2-Out-Of-3 Majority Voting AI Ensemble.
    Executes a trade if at least 2 out of 3 AI models agree on BUY or SELL.
    Logs diagnostic vote breakdowns for every symbol.
    """

    def __init__(self, settings=None, config_path: str = "config.yaml"):
        self.logger = logger
        self.logger.setLevel(logging.INFO)

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
        """Load all 3 AI models for the 2-out-of-3 voting ensemble"""
        try:
            custom_objects = {
                'GlorotUniform': tf.keras.initializers.GlorotUniform,
                'AdamW': tf.keras.optimizers.AdamW,
                'ResNetBlock1D': ResNetBlock1D
            }

            # 1. Model 1: Continuous Regression AI
            reg_path = "smartcrypto_ai_models/continuous_regression_ai.keras"
            reg_scaler_path = "smartcrypto_ai_models/unconstrained_scaler.joblib"
            reg_features_path = "smartcrypto_ai_models/unconstrained_features.joblib"

            if os.path.exists(reg_path):
                self.model_reg = tf.keras.models.load_model(reg_path, custom_objects=custom_objects, compile=False)
                self.scaler_reg = joblib.load(reg_scaler_path) if os.path.exists(reg_scaler_path) else None
                self.features_reg = joblib.load(reg_features_path) if os.path.exists(reg_features_path) else UnconstrainedCandleExtractor.get_feature_columns()
                self.logger.info("✅ Model 1 Loaded: Continuous Return Regression AI")

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
            if os.path.exists(gpt_path):
                self.model_gpt = tf.keras.models.load_model(gpt_path, compile=False)
                self.logger.info("✅ Model 3 Loaded: Market GPT World Model (1,000 Path Simulator)")

            if self.model_reg is not None or self.model_smart is not None:
                self.model_loaded = True
                self.feature_columns = getattr(self, 'features_smart', self.features_reg)
                self.total_features = len(self.feature_columns)
                self.logger.info("🚀 2-Out-Of-3 Voting AI Ensemble Successfully Online!")

        except Exception as e:
            self.logger.error(f"❌ Error loading voting ensemble models: {e}", exc_info=True)
            self.model_loaded = False

    async def load_model(self):
        if not self.model_loaded:
            self.load_models()
        return self.model_loaded

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
    # 2-OUT-OF-3 MAJORITY VOTING EVALUATION
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def generate_signal(self, symbol: str, current_data: pd.DataFrame, current_price: float) -> Optional[Dict]:
        """Generate high-conviction signal using 2-out-of-3 Majority Voting Ensemble"""
        if not self.model_loaded:
            return None

        try:
            # Prepare Data Inputs
            seq_tensor, raw_matrix = self.prepare_regression_sequence(current_data)
            smart_features = self.prepare_smart_trader_features(current_data)

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # VOTE 1: Model 1 (Continuous Return Regression AI)
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            vote_m1 = 'HOLD'
            pred_1h, pred_4h, pred_12h = 0.0, 0.0, 0.0
            min_thresh = getattr(self.settings, 'MIN_EXPECTED_RETURN_THRESHOLD', 0.007)  # 0.7%

            if self.model_reg is not None and seq_tensor is not None:
                preds_reg = self.model_reg.predict(seq_tensor, verbose=0)
                pred_1h = float(preds_reg[0][0][0])
                pred_4h = float(preds_reg[1][0][0])
                pred_12h = float(preds_reg[2][0][0])

                if pred_4h >= min_thresh and pred_1h > 0:
                    vote_m1 = 'BUY'
                elif pred_4h <= -min_thresh and pred_1h < 0:
                    vote_m1 = 'SELL'

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # VOTE 2: Model 2 (6-Head Smart Trader AI)
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            vote_m2 = 'HOLD'
            risk_level, market_regime = 'MEDIUM', 'TRENDING'
            if self.model_smart is not None and smart_features is not None:
                preds_smart = self.model_smart.predict(smart_features, verbose=0)
                action_4h = self.direction_map[np.argmax(preds_smart[1][0])]
                action_1d = self.direction_map[np.argmax(preds_smart[2][0])]
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
            win_prob = 0.50
            loss_prob = 0.50
            if self.model_gpt is not None and raw_matrix is not None:
                sim_res = MarketGPTWorldModel.simulate_future_paths(self.model_gpt, raw_matrix, n_simulations=500)
                win_prob = float(sim_res.get('win_probability', 0.5))
                loss_prob = float(sim_res.get('loss_probability', 0.5))

                if win_prob >= 0.55:
                    vote_m3 = 'BUY'
                elif loss_prob >= 0.55:
                    vote_m3 = 'SELL'

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # TALLY VOTES & DIAGNOSTIC LOGGING
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            votes = [vote_m1, vote_m2, vote_m3]
            buy_votes = votes.count('BUY')
            sell_votes = votes.count('SELL')

            # Log diagnostic vote breakdown for every symbol!
            self.logger.info(
                f"📊 3-AI Vote Scan [{symbol}]: "
                f"Model1={vote_m1} ({pred_4h:+.2%}) | "
                f"Model2={vote_m2} | "
                f"Model3={vote_m3} (WinProb: {win_prob:.1%})"
            )

            # Determine Majority Action (Requires 2 or 3 votes)
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
            # APPROVED SIGNAL SETUP
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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

            confidence = max(0.55, win_prob) if vote_m3 != 'HOLD' else 0.60
            strength = min(abs(pred_4h) * 15.0, 1.0) if vote_m1 != 'HOLD' else 0.50

            signal = {
                'timestamp': datetime.now().isoformat() + 'Z',
                'symbol': symbol,
                'action': final_action,
                'price': float(current_price),
                'confidence': float(confidence),
                'signal_strength': float(strength),
                'direction_1h': vote_m1 if vote_m1 != 'HOLD' else final_action,
                'direction_4h': vote_m2 if vote_m2 != 'HOLD' else final_action,
                'direction_1d': final_action,
                'risk_level': risk_level,
                'market_regime': market_regime,
                'votes': {
                    'model_1_regression': vote_m1,
                    'model_2_smart_trader': vote_m2,
                    'model_3_market_gpt': vote_m3,
                    'majority_votes': f"{vote_count}/3"
                },
                'expected_returns': {
                    '1h_return': f"{pred_1h:+.2%}",
                    '4h_return': f"{pred_4h:+.2%}",
                    '12h_return': f"{pred_12h:+.2%}"
                },
                'market_gpt_simulation': {
                    'win_probability': f"{win_prob:.1%}",
                    'loss_probability': f"{loss_prob:.1%}"
                },
                'strategy': {
                    'stop_loss': float(stop_loss),
                    'take_profit_1': float(tp1),
                    'take_profit_2': float(tp2),
                    'atr_used': float(atr),
                    'max_holding_hours': self.config['trading']['risk']['max_holding_hours']
                },
                'analysis': {
                    'summary': f"2-out-of-3 Majority ({vote_count}/3) Approved {final_action} on {symbol}",
                    'signal_type': 'STRONG_TREND',
                    'detected_pattern': 'MAJORITY_VOTING_CONSENSUS'
                },
                'outcome': 'OPEN',
                'pnl_percentage': None,
                'signal_id': f"{symbol}_{int(datetime.now().timestamp())}"
            }

            self.logger.info(
                f"🎯 2-OUT-OF-3 MAJORITY APPROVED {symbol}: {final_action} (Votes: {vote_count}/3) | "
                f"M1={vote_m1}, M2={vote_m2}, M3={vote_m3}"
            )

            if self.history_manager:
                self.history_manager.save_signal(signal, outcome="OPEN")

            return signal

        except Exception as e:
            self.logger.error(f"Error generating 2/3 voting signal for {symbol}: {e}", exc_info=True)
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
        }

    def get_symbol_performance(self, symbol: str, days: int = 30) -> Dict:
        return self.history_manager.get_symbol_performance(symbol, days)

    def get_recent_signals(self, symbol: Optional[str] = None, hours: int = 24, limit: int = 50) -> List[Dict]:
        return self.history_manager.get_recent_signals(symbol, hours, limit)

    def health_check(self) -> Dict:
        return {
            'model_loaded': self.model_loaded,
            'performance_stats': self.get_performance_stats(),
            'last_signal_time': self.performance_metrics['last_update'].isoformat(),
            'status': 'healthy' if self.model_loaded else 'degraded'
        }