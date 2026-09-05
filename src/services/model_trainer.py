"""
Enhanced Model Training Service for SmartCrypto AI v3.0.0
Supports both FINE-TUNING and FULL RETRAINING modes
Compatible with new AI model (derivatives, order book, stationary features)
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, losses, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import RobustScaler, PowerTransformer
from sklearn.metrics import accuracy_score
import joblib
import logging
import asyncio
from datetime import datetime, timedelta
import ccxt.async_support as ccxt
from typing import Dict, List, Optional, Tuple
import warnings
import shutil
import os
import requests
import uuid
import json
import pandas_ta as ta

warnings.filterwarnings('ignore')

from src.utils.logger import get_logger
from src.core.config import get_settings

from src.utils.safe_logger import SafeLogger
logger = SafeLogger.get_logger(__name__)


class ModelTrainer:
    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        exchange_cls = getattr(ccxt, 'binanceusdm', ccxt.binance)
        self.exchange = exchange_cls({
            'enableRateLimit': True,
            'options': {'defaultType': 'future', 'adjustForTimeDifference': True}
        })
        self.is_training = False
        self.last_training_time = None
        self.training_history = []
        
        self.mode = getattr(self.settings, 'TRAINING_MODE', 'fine_tune')
        
        # FINE-TUNING PARAMETERS
        self.fine_tune_days = getattr(self.settings, 'FINE_TUNE_DAYS', 60)
        self.fine_tune_lr = getattr(self.settings, 'FINE_TUNE_LEARNING_RATE', 0.0001)
        self.fine_tune_epochs = getattr(self.settings, 'FINE_TUNE_EPOCHS', 5)
        self.fine_tune_min_improvement = getattr(self.settings, 'FINE_TUNE_MIN_IMPROVEMENT', 0.015)
        
        # FULL RETRAIN PARAMETERS
        self.full_retrain_days = getattr(self.settings, 'FULL_RETRAIN_DAYS', 730)
        self.full_retrain_interval_months = getattr(self.settings, 'FULL_RETRAIN_INTERVAL_MONTHS', 6)
        self.last_full_retrain = None
        
        # MODEL QUALITY THRESHOLDS
        self.min_accuracy_1h = 0.55
        self.min_accuracy_4h = 0.58
        self.min_accuracy_1d = 0.62
        self.min_improvement = 0.02
        self.current_model_performance = None
        
        # Stationary features exclusion
        self.non_stationary_cols = [
            'open', 'high', 'low', 'close', 'log_close',
            'sma20', 'sma50', 'ema12', 'ema26',
            'vwap', 'typical_price', 'money_flow',
            'open_4h', 'high_4h', 'low_4h', 'close_4h',
            'open_1d', 'high_1d', 'low_1d', 'close_1d',
            'pivot', 'support1', 'resistance1',
            'BBL_20_2.0_2.0', 'BBM_20_2.0_2.0', 'BBU_20_2.0_2.0',
            'ichi_ICS_26', 'ichi_IKS_26', 'ichi_ISA_9', 'ichi_ISB_26', 'ichi_ITS_9',
            'KCBe_20_2', 'KCLe_20_2', 'KCUe_20_2',
            'funding_rate', 'open_interest', 'open_interest_usd',
            'funding_high', 'funding_low',
            'confidence_score', 'market_regime', 'risk_level',
            'volatility_regime',
            'timestamp', 'symbol', 'interval', 'date',
        ]
        
        logger.info(f"🔧 ModelTrainer initialized with mode: {self.mode}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # FINE-TUNING METHODS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _freeze_early_layers(self, model, freeze_ratio: float = 0.7):
        """Freeze early layers to preserve base knowledge"""
        total_layers = len(model.layers)
        freeze_until = int(total_layers * freeze_ratio)
        
        frozen_count = 0
        for i, layer in enumerate(model.layers):
            if i < freeze_until:
                layer.trainable = False
                frozen_count += 1
            else:
                layer.trainable = True
        
        logger.info(f"🔒 Frozen {frozen_count}/{total_layers} layers (preserving base knowledge)")
        return model

    async def _fine_tune_model(self) -> bool:
            """Fine-tune existing 6-head model on recent data without blocking asyncio event loop"""
            logger.info("🔧 Starting fine-tuning in background thread...")
            try:
                from src.services.signal_generator import SignalGenerator
                generator = SignalGenerator(self.settings)
                await generator.load_model()
                
                # Safely reference the 6-Head Smart Trader AI model from SignalGenerator
                target_model = getattr(generator, 'model_smart', getattr(generator, 'model', None))

                if target_model is None or not generator.model_loaded:
                    logger.error("❌ No valid model available to fine-tune")
                    return False

                logger.info("✅ Current 6-Head Smart Trader model loaded for fine-tuning")

                # Fetch recent 60 days of training data
                all_data = await self.fetch_training_data(self.settings.SYMBOLS, days=self.fine_tune_days)

                if not all_data:
                    return False

                result = self.prepare_training_data(all_data, feature_cols=generator.feature_columns)
                if result[0] is None:
                    return False

                (X_train, X_test, y1_train, y1_test, y4_train, y4_test,
                y24_train, y24_test, y_conf_train, y_conf_test,
                y_risk_train, y_risk_test, y_regime_train, y_regime_test,
                feature_cols) = result

                # Safe scaler & transformer reference from SignalGenerator
                scaler = getattr(generator, 'scaler_smart', getattr(generator, 'scaler', None))
                power_transformer = getattr(generator, 'transformer_smart', getattr(generator, 'power_transformer', None))

                if scaler is None:
                    scaler, power_transformer, X_train_t, X_test_t = self.scale_features(X_train, X_test)
                else:
                    X_train_scaled = scaler.transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    X_train_t = power_transformer.transform(X_train_scaled) if power_transformer else X_train_scaled
                    X_test_t = power_transformer.transform(X_test_scaled) if power_transformer else X_test_scaled

                # Evaluate baseline performance BEFORE fine-tuning
                current_perf = await self._evaluate_model(target_model, X_test_t, y1_test, y4_test, y24_test)
                old_score = current_perf.get('overall_score', 0.5)

                # Freeze early layers on target_model
                model = self._freeze_early_layers(target_model, freeze_ratio=0.7)
                model.compile(
                    optimizer=optimizers.AdamW(learning_rate=self.fine_tune_lr, weight_decay=0.001),
                    loss={
                        'direction_1h': 'sparse_categorical_crossentropy',
                        'direction_4h': 'sparse_categorical_crossentropy',
                        'direction_1d': 'sparse_categorical_crossentropy',
                        'confidence_score': 'mse',
                        'risk_level': 'sparse_categorical_crossentropy',
                        'market_regime': 'sparse_categorical_crossentropy'
                    },
                    loss_weights={'direction_1h': 0.25, 'direction_4h': 0.25, 'direction_1d': 0.25, 'confidence_score': 0.1, 'risk_level': 0.075, 'market_regime': 0.075},
                    metrics={'direction_1h': ['accuracy'], 'direction_4h': ['accuracy'], 'direction_1d': ['accuracy'], 'risk_level': ['accuracy'], 'market_regime': ['accuracy'], 'confidence_score': ['mae']}
                )

                train_targets = {'direction_1h': y1_train, 'direction_4h': y4_train, 'direction_1d': y24_train, 'confidence_score': y_conf_train, 'risk_level': y_risk_train, 'market_regime': y_regime_train}
                val_targets = {'direction_1h': y1_test, 'direction_4h': y4_test, 'direction_1d': y24_test, 'confidence_score': y_conf_test, 'risk_level': y_risk_test, 'market_regime': y_regime_test}

                temp_dir = "models/temp"
                os.makedirs(temp_dir, exist_ok=True)
                temp_model_path = os.path.join(temp_dir, f"fine_tune_{uuid.uuid4().hex[:8]}.keras")

                callbacks = [
                    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=0),
                    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7, verbose=0),
                    ModelCheckpoint(temp_model_path, monitor='val_direction_1h_accuracy', save_best_only=True, mode='max', verbose=0)
                ]

                # Run model.fit in background thread
                def _sync_fit():
                    return model.fit(
                        X_train_t, train_targets,
                        validation_data=(X_test_t, val_targets),
                        epochs=self.fine_tune_epochs,
                        batch_size=256,
                        callbacks=callbacks,
                        verbose=0
                    )

                logger.info(f"🏋️ Fine-tuning model weights in background thread...")
                await asyncio.to_thread(_sync_fit)

                if os.path.exists(temp_model_path):
                    try:
                        model = tf.keras.models.load_model(temp_model_path)
                        os.remove(temp_model_path)
                    except Exception as e:
                        logger.warning(f"Could not load checkpoint: {e}")

                new_perf = await self._evaluate_model(model, X_test_t, y1_test, y4_test, y24_test)
                improvement = new_perf['overall_score'] - old_score

                if improvement >= self.fine_tune_min_improvement:
                    if self.deploy_new_model(model, scaler, power_transformer, feature_cols):
                        logger.info(f"✅ Fine-tuned model deployed successfully! (Gain: +{improvement*100:.2f}%)")
                        return True
                else:
                    logger.info(f"⏭️ Fine-tuning gain (+{improvement*100:.2f}%) below threshold - preserving base model")
                    return False

            except Exception as e:
                logger.error(f"❌ Fine-tuning error: {e}", exc_info=True)
                return False

    async def _evaluate_model(self, model, X_test, y1_test, y4_test, y24_test) -> Dict:
        """Evaluate model performance"""
        try:
            predictions = model.predict(X_test, verbose=0)
            
            acc_1h = accuracy_score(y1_test, np.argmax(predictions[0], axis=1))
            acc_4h = accuracy_score(y4_test, np.argmax(predictions[1], axis=1))
            acc_1d = accuracy_score(y24_test, np.argmax(predictions[2], axis=1))
            
            return {
                'accuracy_1h': acc_1h,
                'accuracy_4h': acc_4h,
                'accuracy_1d': acc_1d,
                'overall_score': (acc_1h * 0.5 + acc_4h * 0.3 + acc_1d * 0.2)
            }
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {'overall_score': 0}

    def _should_full_retrain(self) -> bool:
        """Check if full retrain is needed based on schedule"""
        if self.mode != 'full_retrain':
            return False
        
        if self.last_full_retrain is None:
            return True
        
        months_since = (datetime.now() - self.last_full_retrain).days / 30
        return months_since >= self.full_retrain_interval_months

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # COMPLETE FEATURE ENGINEERING
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Complete feature engineering with derivatives support"""
        try:
            df = df.copy()
            
            # Ensure optional volume columns exist
            for col in ['quote_asset_volume', 'trades_count', 'taker_buy_quote_volume']:
                if col not in df.columns:
                    df[col] = 0.0
                else:
                    df[col] = df[col].fillna(0.0)

            # 1. BASIC CALCULATIONS
            df['log_close'] = np.log(df['close'])
            df['ret_1'] = df['close'].pct_change()
            df['ret_3'] = df['close'].pct_change(3)
            df['range'] = (df['high'] - df['low']) / df['close']
            df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
            df['money_flow'] = df['typical_price'] * df['volume']
            
            # 2. ROLLING VWAP
            df['vwap'] = (
                df['money_flow'].rolling(24).sum() / 
                (df['volume'].rolling(24).sum() + 1e-8)
            ).ffill()
            df['price_vwap_ratio'] = df['close'] / df['vwap']
            
            # 3. TECHNICAL INDICATORS (pandas_ta)
            df['rsi_14'] = ta.rsi(df['close'], length=14)
            
            stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3)
            df = pd.concat([df, stoch], axis=1)
            
            macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
            df = pd.concat([df, macd], axis=1)
            
            # 4. ATR
            try:
                df['atr14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
            except Exception:
                pass
            
            if 'atr14' not in df.columns or df['atr14'].isna().all() or (df['atr14'] == 0).all():
                logger.warning("ATR14 not calculated properly, computing manually...")
                high_low = df['high'] - df['low']
                high_close = np.abs(df['high'] - df['close'].shift())
                low_close = np.abs(df['low'] - df['close'].shift())
                tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                df['atr14'] = tr.rolling(14).mean()
            
            # 5. BOLLINGER BANDS & KELTNER / ADX
            bb_2 = ta.bbands(df['close'], length=20, std=2)
            df = pd.concat([df, bb_2], axis=1)
            
            kc = ta.kc(df['high'], df['low'], df['close'], length=20)
            df = pd.concat([df, kc], axis=1)
            
            adx = ta.adx(df['high'], df['low'], df['close'], length=14)
            df = pd.concat([df, adx], axis=1)
            
            # 6. DERIVATIVES FEATURES
            if 'funding_rate' in df.columns and 'open_interest' in df.columns:
                df['funding_8h'] = df['funding_rate'].rolling(8).sum()
                df['funding_24h'] = df['funding_rate'].rolling(24).sum()
                df['funding_change_8h'] = df['funding_rate'].diff(8).fillna(0)
                df['funding_change_24h'] = df['funding_rate'].diff(24).fillna(0)
                
                mean_90d = df['funding_rate'].rolling(90).mean()
                std_90d = df['funding_rate'].rolling(90).std()
                df['funding_zscore'] = (df['funding_rate'] - mean_90d) / (std_90d + 1e-8)
                
                df['funding_high'] = df['funding_rate'].rolling(90).max()
                df['funding_low'] = df['funding_rate'].rolling(90).min()
                df['funding_percentile'] = (
                    (df['funding_rate'] - df['funding_low']) / 
                    (df['funding_high'] - df['funding_low'] + 1e-8)
                )
                
                df['oi_change_1h'] = df['open_interest'].pct_change(1).fillna(0)
                df['oi_change_24h'] = df['open_interest'].pct_change(24).fillna(0)
                df['oi_momentum_7d'] = df['open_interest'].pct_change(168).fillna(0)
                
                oi_high_24h = df['open_interest'].rolling(24).max()
                oi_low_24h = df['open_interest'].rolling(24).min()
                df['oi_position_24h'] = (df['open_interest'] - oi_low_24h) / (oi_high_24h - oi_low_24h + 1e-8)
                
                if 'open_interest_usd' in df.columns:
                    df['oi_volume_ratio'] = df['open_interest_usd'] / (df['volume'] * df['close'] + 1e-8)
                else:
                    df['oi_volume_ratio'] = df['open_interest'] / (df['volume'] + 1e-8)
                
                price_dir = np.sign(df['close'].pct_change(24).fillna(0))
                oi_dir = np.sign(df['oi_change_24h'])
                df['price_oi_divergence'] = price_dir * oi_dir
                
                df['oi_turnover_ratio'] = df['open_interest'] / (df['volume'].rolling(24).mean() + 1e-8)
            
            # 7. VOLUME FEATURES
            df['volume_ema_short'] = df['volume'].ewm(span=5).mean()
            df['volume_ema_long'] = df['volume'].ewm(span=20).mean()
            df['volume_oscillator'] = (df['volume_ema_short'] - df['volume_ema_long']) / (df['volume_ema_long'] + 1e-8)
            
            df['vol_12'] = df['volume'].rolling(12).mean()
            df['vol_sma20'] = df['volume'].rolling(20).mean()
            df['vol_ratio'] = df['volume'] / (df['vol_sma20'] + 1e-8)
            
            # 8. MOMENTUM
            df['mom_6'] = df['close'].pct_change(6)
            df['mom_12'] = df['close'].pct_change(12)
            
            # 9. MULTI-TIMEFRAME
            df['open_4h'] = df['open'].rolling(4).apply(lambda x: x[0], raw=True)
            df['high_4h'] = df['high'].rolling(4).max()
            df['low_4h'] = df['low'].rolling(4).min()
            df['close_4h'] = df['close'].rolling(4).apply(lambda x: x[-1], raw=True)
            df['volume_4h'] = df['volume'].rolling(4).sum()
            
            df['open_1d'] = df['open'].rolling(24).apply(lambda x: x[0], raw=True)
            df['high_1d'] = df['high'].rolling(24).max()
            df['low_1d'] = df['low'].rolling(24).min()
            df['close_1d'] = df['close'].rolling(24).apply(lambda x: x[-1], raw=True)
            df['volume_1d'] = df['volume'].rolling(24).sum()
            
            df['price_pos_4h'] = (df['close'] - df['low_4h']) / (df['high_4h'] - df['low_4h'] + 1e-8)
            df['price_pos_1d'] = (df['close'] - df['low_1d']) / (df['high_1d'] - df['low_1d'] + 1e-8)
            
            # 10. STATIONARY DISTANCES & Z-SCORES
            df['distance_from_20h_high'] = (df['close'] - df['high'].rolling(20).max()) / (df['close'] + 1e-8)
            df['distance_from_20h_low'] = (df['close'] - df['low'].rolling(20).min()) / (df['close'] + 1e-8)
            df['price_zscore_20'] = (df['close'] - df['close'].rolling(20).mean()) / (df['close'].rolling(20).std() + 1e-8)
            
            # 11. MARKET MICROSTRUCTURE
            if 'taker_buy_quote_volume' in df.columns and 'quote_asset_volume' in df.columns:
                df['buy_pressure'] = df['taker_buy_quote_volume'] / (df['quote_asset_volume'] + 1e-8)
                df['order_imbalance'] = (
                    2 * df['taker_buy_quote_volume'] - df['quote_asset_volume']
                ) / (df['quote_asset_volume'] + 1e-8)
            else:
                df['buy_pressure'] = 0.5
                df['order_imbalance'] = 0.0
            
            # 12. SEASONALITY
            if 'timestamp' in df.columns:
                df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
                df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
            else:
                df['hour'] = 0
                df['day_of_week'] = 0
            
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            
            # 13. VOLATILITY
            df['volatility_rolling'] = df['close'].rolling(20).std()
            
            if 'atr14' in df.columns and not df['atr14'].isna().all():
                df['volatility_pct'] = df['atr14'] / (df['close'] + 1e-8)
            else:
                logger.warning("ATR14 not available, using price std as volatility proxy")
                df['volatility_pct'] = df['volatility_rolling'] / (df['close'] + 1e-8)
            
            # 14. RENAME COLUMNS TO MATCH MODEL
            rename_map = {
                'EMA_12': 'ema12',
                'EMA_26': 'ema26',
                'RSI_14': 'rsi_14',
                'ATRr_14': 'atr14',
                'MACD_12_26_9': 'MACD_12_26_9',
                'MACDh_12_26_9': 'MACDh_12_26_9',
                'MACDs_12_26_9': 'MACDs_12_26_9',
                'ADX_14': 'ADX_14',
                'ADXR_14': 'ADXR_14_2',
                'DMP_14': 'DMP_14',
                'DMN_14': 'DMN_14',
                'BBP_20_2.0': 'BBP_20_2.0_2.0',
                'BBB_20_2.0': 'BBB_20_2.0_2.0',
            }
            df = df.rename(columns=rename_map)
            
            # 15. CLEAN UP
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.ffill().bfill().fillna(0)
            
            if 'atr14' in df.columns:
                logger.info(f"✅ ATR14 available: mean={df['atr14'].mean():.6f}, non-zero={(df['atr14'] > 0).sum()}")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error in feature engineering: {e}", exc_info=True)
            return pd.DataFrame()

    def generate_targets(self, df: pd.DataFrame) -> pd.DataFrame:
            """Generate balanced targets using Quantile Binning (33% BUY, 33% HOLD, 33% SELL)"""
            df = df.copy()
            horizons = {'1h': 1, '4h': 4, '1d': 24}
            
            for horizon, shift in horizons.items():
                ret_col = f'future_ret_{horizon}'
                df[ret_col] = df['close'].shift(-shift) / df['close'] - 1.0
                
                valid_returns = df[ret_col].dropna()
                if len(valid_returns) > 50:
                    q33 = valid_returns.quantile(0.33)
                    q67 = valid_returns.quantile(0.67)
                else:
                    q33, q67 = -0.003, 0.003

                df[f'target_{horizon}'] = np.where(
                    df[ret_col] > q67, 2,
                    np.where(df[ret_col] < q33, 0, 1)
                )
            
            df['volatility_pct'] = df['atr14'] / (df['close'] + 1e-8)
            valid_vol = df['volatility_pct'].dropna()
            vq33 = valid_vol.quantile(0.33) if len(valid_vol) > 50 else 0.01
            vq67 = valid_vol.quantile(0.67) if len(valid_vol) > 50 else 0.02

            df['risk_level'] = pd.cut(
                df['volatility_pct'],
                bins=[-np.inf, vq33, vq67, np.inf],
                labels=[0, 1, 2]
            ).astype(int)

            adx_col = 'ADX_14' if 'ADX_14' in df.columns else 'adx'
            conditions = [
                (df[adx_col] < 20),
                (df[adx_col] > 25) & (df['volatility_pct'] < vq67),
                (df['volatility_pct'] >= vq67)
            ]
            df['market_regime'] = np.select(conditions, [1, 0, 2], default=3)

            agreement = (
                (df['target_1h'] == df['target_4h']) & 
                (df['target_4h'] == df['target_1d'])
            ).astype(float)
            df['confidence_score'] = agreement * 0.5 + 0.5

            return df

    def get_stationary_features(self, df: pd.DataFrame) -> List[str]:
        """Get stationary features (returns 58 features)"""
        all_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        exclude_cols = [
            'open', 'high', 'low', 'close', 'log_close',
            'sma20', 'sma50', 'ema12', 'ema26',
            'vwap', 'typical_price', 'money_flow',
            'open_4h', 'high_4h', 'low_4h', 'close_4h',
            'open_1d', 'high_1d', 'low_1d', 'close_1d',
            'pivot', 'support1', 'resistance1',
            'BBL_20_2.0_2.0', 'BBM_20_2.0_2.0', 'BBU_20_2.0_2.0',
            'ichi_ICS_26', 'ichi_IKS_26', 'ichi_ISA_9', 'ichi_ISB_26', 'ichi_ITS_9',
            'KCBe_20_2', 'KCLe_20_2', 'KCUe_20_2',
            'funding_rate', 'open_interest', 'open_interest_usd',
            'funding_high', 'funding_low',
            'confidence_score', 'market_regime', 'risk_level',
            'volatility_regime',
            'timestamp', 'symbol', 'interval', 'date',
            'future_ret_1h', 'future_ret_4h', 'future_ret_1d',
            'target_1h', 'target_4h', 'target_1d',
        ]
        
        keep_features = [
            'BBP_20_2.0_2.0',
            'BBB_20_2.0_2.0', 
            'volatility_pct',
            'quote_asset_volume',  
            'trades_count',         
        ]
        
        stationary = [
            col for col in all_cols 
            if col not in exclude_cols
            and not col.startswith('future_')
            and not col.startswith('target_')
        ]
        
        for feat in keep_features:
            if feat not in stationary and feat in all_cols:
                stationary.append(feat)
        
        logger.info(f"📊 Features count: {len(stationary)}")
        return stationary

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # FETCH TRAINING DATA
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def fetch_training_data(self, symbols: List[str], days: int = 90) -> Dict[str, pd.DataFrame]:
        """Fetch comprehensive training data including all Binance fields and derivatives"""
        logger.info(f"📥 Fetching {days} days of training data for {len(symbols)} symbols...")
        
        all_data = {}
        expected_candles = days * 24
        
        for symbol in symbols:
            try:
                logger.info(f"📊 Fetching {symbol}...")
                
                all_ohlcv = []
                start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
                
                while start_time < int(datetime.now().timestamp() * 1000):
                    url = f"{self.settings.BINANCE_API_BASE}/api/v3/klines"
                    params = {
                        'symbol': symbol,
                        'interval': '1h',
                        'limit': 1000,
                        'startTime': start_time
                    }
                    
                    response = requests.get(url, params=params, timeout=30, verify=False)
                    if response.status_code != 200:
                        break
                    
                    data = response.json()
                    if not data:
                        break
                    
                    # Length-safe extraction of optional fields
                    for kline in data:
                        all_ohlcv.append([
                            kline[0], 
                            float(kline[1]), float(kline[2]), float(kline[3]), float(kline[4]), float(kline[5]),
                            float(kline[7]) if len(kline) > 7 else 0.0,   # quote_asset_volume
                            float(kline[8]) if len(kline) > 8 else 0.0,   # trades_count
                            float(kline[9]) if len(kline) > 9 else 0.0,   # taker_buy_base_volume  <-- ADD THIS
                            float(kline[10]) if len(kline) > 10 else 0.0 # taker_buy_quote_volume
                        ])
                    
                    start_time = data[-1][0] + 1
                    if len(all_ohlcv) >= expected_candles:
                        break
                
                if not all_ohlcv:
                    logger.warning(f"⚠️ No data for {symbol}")
                    continue
                
                df = pd.DataFrame(all_ohlcv, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'quote_asset_volume', 'trades_count', 'taker_buy_base_volume', 'taker_buy_quote_volume'
                ])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
                
                symbol_ccxt = symbol.replace('USDT', '/USDT')
                
                try:
                    funding_raw = await self.exchange.fetch_funding_rate_history(
                        symbol=symbol_ccxt, limit=1000
                    )
                    if funding_raw:
                        df_funding = pd.DataFrame(funding_raw)
                        df_funding['timestamp'] = pd.to_datetime(df_funding['timestamp'], unit='ms')
                        df_funding = df_funding[['timestamp', 'fundingRate']].rename(
                            columns={'fundingRate': 'funding_rate'}
                        )
                        df = pd.merge(df, df_funding, on='timestamp', how='left')
                    else:
                        df['funding_rate'] = 0.0
                except Exception as e:
                    logger.warning(f"Funding rate fetch failed for {symbol}: {e}")
                    df['funding_rate'] = 0.0
                
                try:
                    oi = await self.exchange.fetch_open_interest(symbol_ccxt)
                    if oi:
                        df['open_interest'] = oi.get('openInterest', 0)
                        df['open_interest_usd'] = oi.get('openInterestValue', 0)
                    else:
                        df['open_interest'] = 0.0
                        df['open_interest_usd'] = 0.0
                except Exception:
                    df['open_interest'] = 0.0
                    df['open_interest_usd'] = 0.0
                
                df = self.engineer_features(df)
                df = self.generate_targets(df)
                
                if 'atr14' in df.columns and not df['atr14'].isna().all():
                    all_data[symbol] = df
                    logger.info(f"✅ {symbol}: {len(df)} records, ATR mean: {df['atr14'].mean():.6f}")
                else:
                    logger.warning(f"⚠️ Feature validation failed for {symbol}")
                    
            except Exception as e:
                logger.error(f"❌ Error fetching data for {symbol}: {e}")
        
        try:
            await self.exchange.close()
        except Exception:
            pass
        
        logger.info(f"✅ Completed: {len(all_data)} symbols")
        return all_data

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PREPARE TRAINING DATA (STRICT 58 FEATURE GUARANTEE)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def prepare_training_data(self, all_data: Dict[str, pd.DataFrame], feature_cols: Optional[List[str]] = None):
        """Prepare training data and strictly guarantee feature matrix shape matching model requirement"""
        try:
            combined_data = []
            for symbol, df in all_data.items():
                if len(df) > 100:
                    # Fill optional volume columns with zeros instead of letting dropna remove them
                    for col in ['quote_asset_volume', 'trades_count', 'taker_buy_quote_volume']:
                        if col in df.columns:
                            df[col] = df[col].fillna(0.0)
                        else:
                            df[col] = 0.0

                    df_clean = df.dropna(subset=['target_1h', 'target_4h', 'target_1d', 'confidence_score', 'risk_level', 'market_regime'])
                    if len(df_clean) > 50:
                        df_clean['symbol'] = symbol
                        combined_data.append(df_clean)
            
            if not combined_data:
                logger.error("❌ No valid data for training")
                return None, None, None, None, None, None, None, None, None, None, None, None, None, None, None

            full_dataset = pd.concat(combined_data, ignore_index=True)
            
            # Load exact feature columns from saved model if feature_cols is not passed
            if feature_cols is None:
                try:
                    from src.services.signal_generator import SignalGenerator
                    gen = SignalGenerator(self.settings)
                    gen.load_models()
                    if gen.model_loaded and gen.feature_columns:
                        feature_cols = gen.feature_columns
                        logger.info(f"📊 Using model's saved {len(feature_cols)} feature columns")
                    else:
                        feature_cols = self.get_stationary_features(full_dataset)
                except Exception as e:
                    logger.warning(f"Could not load feature columns from model: {e}")
                    feature_cols = self.get_stationary_features(full_dataset)
            
            # Add missing feature columns with zeros if missing in dataset
            missing_cols = set(feature_cols) - set(full_dataset.columns)
            if missing_cols:
                logger.warning(f"⚠️ Adding {len(missing_cols)} missing feature columns filled with 0.0: {missing_cols}")
                for col in missing_cols:
                    full_dataset[col] = 0.0
            
            # Extract exact feature matrix
            X = full_dataset[feature_cols].fillna(0.0).replace([np.inf, -np.inf], 0.0)
            
            logger.info(f"📊 Feature matrix shape: {X.shape} (strictly matched to {len(feature_cols)} features)")
            
            y1 = full_dataset['target_1h'].values
            y4 = full_dataset['target_4h'].values
            y24 = full_dataset['target_1d'].values
            y_conf = full_dataset['confidence_score'].values
            y_risk = full_dataset['risk_level'].values
            y_regime = full_dataset['market_regime'].values

            split_idx = int(len(X) * 0.8)
            
            X_train, X_test = X[:split_idx], X[split_idx:]
            y1_train, y1_test = y1[:split_idx], y1[split_idx:]
            y4_train, y4_test = y4[:split_idx], y4[split_idx:]
            y24_train, y24_test = y24[:split_idx], y24[split_idx:]
            y_conf_train, y_conf_test = y_conf[:split_idx], y_conf[split_idx:]
            y_risk_train, y_risk_test = y_risk[:split_idx], y_risk[split_idx:]
            y_regime_train, y_regime_test = y_regime[:split_idx], y_regime[split_idx:]

            return (X_train, X_test, y1_train, y1_test, y4_train, y4_test, 
                    y24_train, y24_test, y_conf_train, y_conf_test, 
                    y_risk_train, y_risk_test, y_regime_train, y_regime_test, 
                    feature_cols)
            
        except Exception as e:
            logger.error(f"❌ Error preparing training data: {e}", exc_info=True)
            return None, None, None, None, None, None, None, None, None, None, None, None, None, None, None

    def scale_features(self, X_train, X_test):
        """Scale features using RobustScaler and PowerTransformer"""
        try:
            scaler = RobustScaler()
            power_transformer = PowerTransformer(method='yeo-johnson')

            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            X_train_transformed = power_transformer.fit_transform(X_train_scaled)
            X_test_transformed = power_transformer.transform(X_test_scaled)

            return scaler, power_transformer, X_train_transformed, X_test_transformed
            
        except Exception as e:
            logger.error(f"❌ Error scaling features: {e}")
            return None, None, None, None

    def create_trader_smart_model(self, input_shape: int, n_features: int):
        """Create multi-head model"""
        inputs = layers.Input(shape=(input_shape,), name='market_features')

        attention_weights = layers.Dense(n_features, activation='softmax', name='feature_attention')(inputs)
        weighted_features = layers.Multiply()([inputs, attention_weights])

        x = layers.Dense(512, activation='swish')(weighted_features)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)

        x = layers.Dense(256, activation='swish')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.25)(x)

        x = layers.Dense(128, activation='swish')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)

        shared = layers.Dense(64, activation='swish', name='shared_representation')(x)

        def create_dir_head(name):
            h = layers.Dense(32, activation='swish')(shared)
            h = layers.Dropout(0.2)(h)
            return layers.Dense(3, activation='softmax', name=name)(h)

        conf_output = layers.Dense(1, activation='sigmoid', name='confidence_score')(
            layers.Dense(16, activation='swish')(shared)
        )
        risk_output = layers.Dense(3, activation='softmax', name='risk_level')(
            layers.Dense(16, activation='swish')(shared)
        )
        regime_output = layers.Dense(4, activation='softmax', name='market_regime')(
            layers.Dense(16, activation='swish')(shared)
        )

        return models.Model(
            inputs=inputs,
            outputs=[
                create_dir_head('direction_1h'),
                create_dir_head('direction_4h'),
                create_dir_head('direction_1d'),
                conf_output,
                risk_output,
                regime_output
            ],
            name='TraderSmartAI'
        )

    def create_and_train_model(self, X_train, X_test, y1_train, y4_train, y24_train,
                               y1_test, y4_test, y24_test, y_conf_train, y_conf_test,
                               y_risk_train, y_risk_test, y_regime_train, y_regime_test):
        """Create and train the model"""
        try:
            input_shape = X_train.shape[1]
            n_features = X_train.shape[1]
            
            model = self.create_trader_smart_model(input_shape, n_features)
            
            optimizer = optimizers.AdamW(learning_rate=0.001, weight_decay=0.01)
            model.compile(
                optimizer=optimizer,
                loss={
                    'direction_1h': 'sparse_categorical_crossentropy',
                    'direction_4h': 'sparse_categorical_crossentropy',
                    'direction_1d': 'sparse_categorical_crossentropy',
                    'confidence_score': 'mse',
                    'risk_level': 'sparse_categorical_crossentropy',
                    'market_regime': 'sparse_categorical_crossentropy'
                },
                loss_weights={
                    'direction_1h': 0.25,
                    'direction_4h': 0.25,
                    'direction_1d': 0.25,
                    'confidence_score': 0.1,
                    'risk_level': 0.075,
                    'market_regime': 0.075
                },
                metrics={
                    'direction_1h': ['accuracy'],
                    'direction_4h': ['accuracy'], 
                    'direction_1d': ['accuracy'],
                    'risk_level': ['accuracy'],
                    'market_regime': ['accuracy'],
                    'confidence_score': ['mae']
                }
            )

            train_targets = {
                'direction_1h': y1_train,
                'direction_4h': y4_train,
                'direction_1d': y24_train,
                'confidence_score': y_conf_train,
                'risk_level': y_risk_train,
                'market_regime': y_regime_train
            }

            val_targets = {
                'direction_1h': y1_test,
                'direction_4h': y4_test,
                'direction_1d': y24_test,
                'confidence_score': y_conf_test,
                'risk_level': y_risk_test,
                'market_regime': y_regime_test
            }

            temp_dir = "models/temp"
            os.makedirs(temp_dir, exist_ok=True)
            temp_model_path = os.path.join(temp_dir, f"model_{uuid.uuid4().hex[:8]}.keras")
            
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
                ModelCheckpoint(temp_model_path, monitor='val_direction_1h_accuracy', 
                               save_best_only=True, mode='max', verbose=1)
            ]

            logger.info("🏋️ Starting model training with REAL targets...")
            history = model.fit(
                X_train, train_targets,
                validation_data=(X_test, val_targets),
                epochs=50,
                batch_size=512,
                callbacks=callbacks,
                verbose=1
            )

            if os.path.exists(temp_model_path):
                try:
                    best_model = tf.keras.models.load_model(temp_model_path)
                    model = best_model
                    os.remove(temp_model_path)
                    logger.info("✅ Loaded best model from checkpoint")
                except Exception as e:
                    logger.warning(f"Could not load checkpoint: {e}")

            logger.info("✅ Model training completed successfully")
            return model
            
        except Exception as e:
            logger.error(f"❌ Error training model: {e}")
            return None

    async def validate_model_quality(self, model, X_test, y1_test, y4_test, y24_test) -> Tuple[bool, Dict]:
        """Validate if the new model meets quality standards"""
        try:
            predictions = model.predict(X_test, verbose=0)
            
            acc_1h = accuracy_score(y1_test, np.argmax(predictions[0], axis=1))
            acc_4h = accuracy_score(y4_test, np.argmax(predictions[1], axis=1))
            acc_1d = accuracy_score(y24_test, np.argmax(predictions[2], axis=1))
            
            conf_scores = predictions[3].flatten()
            conf_quality = float(np.mean(conf_scores))
            
            performance_metrics = {
                'accuracy_1h': acc_1h,
                'accuracy_4h': acc_4h,
                'accuracy_1d': acc_1d,
                'confidence_quality': conf_quality,
                'overall_score': (acc_1h * 0.5 + acc_4h * 0.3 + acc_1d * 0.2),
                'timestamp': datetime.now().isoformat()
            }
            
            meets_standards = (
                acc_1h >= self.min_accuracy_1h and
                acc_4h >= self.min_accuracy_4h and
                acc_1d >= self.min_accuracy_1d and
                performance_metrics['overall_score'] >= 0.55
            )
            
            improvement_achieved = True
            if self.current_model_performance:
                improvement_achieved = (
                    performance_metrics['overall_score'] > 
                    self.current_model_performance['overall_score'] + self.min_improvement
                )
            
            is_usable = meets_standards and improvement_achieved
            
            logger.info(f"🔍 Validation: 1h={acc_1h:.3f}, 4h={acc_4h:.3f}, 1d={acc_1d:.3f}")
            logger.info(f"🔍 Overall: {performance_metrics['overall_score']:.3f} - {'✅ USABLE' if is_usable else '❌ REJECTED'}")
            
            return is_usable, performance_metrics
            
        except Exception as e:
            logger.error(f"❌ Validation failed: {e}")
            return False, {}

    def backup_current_model(self):
        """Backup current model files"""
        try:
            backup_dir = "models/backup"
            os.makedirs(backup_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            files = [
                self.settings.MODEL_PATH,
                self.settings.SCALER_PATH,
                self.settings.FEATURE_COLUMNS_PATH,
                self.settings.FEATURE_COLUMNS_PATH.replace('feature_columns', 'power_transformer')
            ]
            
            for f in files:
                if os.path.exists(f):
                    name = os.path.basename(f)
                    shutil.copy2(f, f"{backup_dir}/{timestamp}_{name}")
            
            logger.info(f"💾 Backup created: {timestamp}")
            return True
        except Exception as e:
            logger.error(f"❌ Backup failed: {e}")
            return False

    def deploy_new_model(self, model, scaler, power_transformer, feature_columns) -> bool:
        """Deploy new model"""
        try:
            if not self.backup_current_model():
                return False
            
            temp_dir = "models/temp"
            os.makedirs(temp_dir, exist_ok=True)
            
            temp_id = uuid.uuid4().hex[:8]
            temp_model_path = os.path.join(temp_dir, f"deploy_{temp_id}.keras")
            temp_scaler_path = os.path.join(temp_dir, f"scaler_{temp_id}.joblib")
            temp_features_path = os.path.join(temp_dir, f"features_{temp_id}.joblib")
            temp_transformer_path = os.path.join(temp_dir, f"transformer_{temp_id}.joblib")
            
            model.save(temp_model_path)
            joblib.dump(scaler, temp_scaler_path)
            joblib.dump(power_transformer, temp_transformer_path)
            joblib.dump(feature_columns, temp_features_path)
            
            os.replace(temp_model_path, self.settings.MODEL_PATH)
            os.replace(temp_scaler_path, self.settings.SCALER_PATH)
            os.replace(temp_features_path, self.settings.FEATURE_COLUMNS_PATH)
            os.replace(temp_transformer_path, self.settings.FEATURE_COLUMNS_PATH.replace('feature_columns', 'power_transformer'))
            
            logger.info("🚀 New model deployed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Deployment failed: {e}")
            return False

    def rollback_model(self):
        """Rollback to previous model"""
        try:
            backup_dir = "models/backup"
            if not os.path.exists(backup_dir):
                return False
            
            backups = [f for f in os.listdir(backup_dir) if f.endswith('.keras') or f.endswith('.joblib')]
            if not backups:
                return False
            
            latest_ts = sorted(backups)[-1].split('_')[0]
            
            for f in os.listdir(backup_dir):
                if f.startswith(latest_ts):
                    shutil.copy2(f"{backup_dir}/{f}", f"models/{'_'.join(f.split('_')[1:])}")
            
            logger.info("🔄 Rollback completed")
            return True
        except Exception as e:
            logger.error(f"❌ Rollback failed: {e}")
            return False

    async def retrain_model(self, force_retrain: bool = False) -> bool:
        """Main retraining entrypoint"""
        if self.is_training:
            logger.warning("⚠️ Training already in progress")
            return False

        if not force_retrain and not self.settings.AUTO_RETRAIN:
            return False

        self.is_training = True
        logger.info(f"🔄 Starting model training (mode: {self.mode})...")

        try:
            if self.mode == 'fine_tune':
                if self._should_full_retrain():
                    logger.info("📅 Time for scheduled full retrain - switching mode")
                    result = await self._full_retrain_model()
                else:
                    result = await self._fine_tune_model()
            else:
                result = await self._full_retrain_model()
            
            return result

        except Exception as e:
            logger.error(f"❌ Retraining failed: {e}", exc_info=True)
            return False
        finally:
            self.is_training = False

    async def _full_retrain_model(self) -> bool:
        """Full retraining with 2+ years of data"""
        logger.info(f"🔄 Starting full retraining with {self.full_retrain_days} days of data...")
        
        try:
            all_data = await self.fetch_training_data(
                self.settings.SYMBOLS, 
                days=self.full_retrain_days
            )
            
            if not all_data:
                logger.error("❌ No data for full retraining")
                return False
            
            result = self.prepare_training_data(all_data)
            if result[0] is None:
                return False
            
            (X_train, X_test, y1_train, y1_test, y4_train, y4_test,
             y24_train, y24_test, y_conf_train, y_conf_test,
             y_risk_train, y_risk_test, y_regime_train, y_regime_test,
             feature_cols) = result
            
            scaler, power_transformer, X_train_t, X_test_t = self.scale_features(X_train, X_test)
            if scaler is None:
                return False
            
            model = self.create_and_train_model(
                X_train_t, X_test_t,
                y1_train, y4_train, y24_train,
                y1_test, y4_test, y24_test,
                y_conf_train, y_conf_test,
                y_risk_train, y_risk_test,
                y_regime_train, y_regime_test
            )
            
            if model is None:
                return False
            
            is_usable, perf = await self.validate_model_quality(model, X_test_t, y1_test, y4_test, y24_test)
            if not is_usable:
                logger.warning("❌ New model rejected - keeping current")
                return False
            
            if self.deploy_new_model(model, scaler, power_transformer, feature_cols):
                self.current_model_performance = perf
                self.last_training_time = datetime.now()
                self.last_full_retrain = datetime.now()
                self.training_history.append({
                    'timestamp': datetime.now(),
                    'mode': 'full_retrain',
                    'performance': perf,
                    'status': 'success'
                })
                logger.info("✅ Full retraining completed successfully!")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Full retraining failed: {e}")
            return False

    async def start_auto_retraining(self):
        """Start auto-retraining loop"""
        logger.info(f"🔄 Auto-retraining started (mode: {self.mode}, interval: {self.settings.RETRAIN_INTERVAL_HOURS}h)")
        while True:
            try:
                await self.retrain_model()
                await asyncio.sleep(self.settings.RETRAIN_INTERVAL_HOURS * 3600)
            except Exception as e:
                logger.error(f"❌ Auto-retraining error: {e}")
                await asyncio.sleep(3600)

    def get_training_status(self) -> Dict:
        """Get current training status"""
        return {
            'is_training': self.is_training,
            'mode': self.mode,
            'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None,
            'last_full_retrain': self.last_full_retrain.isoformat() if self.last_full_retrain else None,
            'training_history_count': len(self.training_history),
            'auto_retraining_enabled': self.settings.AUTO_RETRAIN,
            'retrain_interval_hours': self.settings.RETRAIN_INTERVAL_HOURS,
            'current_model_performance': self.current_model_performance,
            'fine_tune_params': {
                'days': self.fine_tune_days,
                'learning_rate': self.fine_tune_lr,
                'epochs': self.fine_tune_epochs,
                'min_improvement': self.fine_tune_min_improvement
            },
            'full_retrain_params': {
                'days': self.full_retrain_days,
                'interval_months': self.full_retrain_interval_months
            },
            'min_accuracy_1h': self.min_accuracy_1h,
            'min_accuracy_4h': self.min_accuracy_4h,
            'min_accuracy_1d': self.min_accuracy_1d
        }