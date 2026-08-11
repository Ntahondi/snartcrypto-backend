"""
Data processing utilities for SmartCrypto v3.0.0
Fully compatible with new AI model (derivatives, order book, stationary features)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
import pandas_ta as ta
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Process and engineer features for trading data.
    Supports derivatives (funding rates, open interest) and stationary features.
    """
    
    def __init__(self):
        self.feature_columns = []
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 🔥 NEW: Stationary feature exclusion list
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self.non_stationary_cols = [
            # Raw prices
            'open', 'high', 'low', 'close', 'log_close',
            # Moving averages (price levels)
            'sma20', 'sma50', 'ema12', 'ema26',
            # Volume weighted averages
            'vwap', 'typical_price', 'money_flow',
            # Multi-timeframe prices
            'open_4h', 'high_4h', 'low_4h', 'close_4h',
            'open_1d', 'high_1d', 'low_1d', 'close_1d',
            # Bollinger price levels (NON-STATIONARY - EXCLUDE!)
            'BBL_20_2.0_2.0', 'BBM_20_2.0_2.0', 'BBU_20_2.0_2.0',
            # Ichimoku price levels
            'ichi_ICS_26', 'ichi_IKS_26', 'ichi_ISA_9', 'ichi_ISB_26', 'ichi_ITS_9',
            # Keltner price levels
            'KCBe_20_2', 'KCLe_20_2', 'KCUe_20_2',
            # Pivot levels
            'pivot', 'support1', 'resistance1',
            # Derivatives raw values (keep derived features)
            'funding_rate', 'open_interest', 'open_interest_usd',
            'funding_high', 'funding_low',
            # Target leakage features
            'confidence_score', 'market_regime', 'risk_level',
            'volatility_regime',
            # Identifiers
            'timestamp', 'symbol', 'interval', 'date',
        ]
        
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Complete feature engineering with derivatives support.
        All outputs are stationary (ratios, z-scores, percentages).
        """
        try:
            df = df.copy()
            
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # 1. BASIC CALCULATIONS
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            df['log_close'] = np.log(df['close'])
            df['ret_1'] = df['close'].pct_change()
            df['ret_3'] = df['close'].pct_change(3)
            df['range'] = (df['high'] - df['low']) / df['close']
            df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
            df['money_flow'] = df['typical_price'] * df['volume']
            
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # 2. FIXED: ROLLING VWAP (NO LOOK-AHEAD)
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            df['vwap'] = (
                df['money_flow'].rolling(24).sum() / 
                (df['volume'].rolling(24).sum() + 1e-8)
            ).ffill()
            df['price_vwap_ratio'] = df['close'] / df['vwap']
            
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # 3. TECHNICAL INDICATORS (pandas_ta)
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # Momentum (RSI 14 only)
            df['rsi_14'] = ta.rsi(df['close'], length=14)
            
            # Stochastic
            stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3)
            df = pd.concat([df, stoch], axis=1)
            
            # MACD
            macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
            df = pd.concat([df, macd], axis=1)
            
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # 4. ATR - WITH FALLBACK
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            df['atr14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
            
            # ✅ FIX: If ATR is missing or all zeros, calculate manually
            if 'atr14' not in df.columns or df['atr14'].isna().all() or (df['atr14'] == 0).all():
                logger.warning("ATR14 not calculated properly, computing manually...")
                # Manual ATR calculation
                high_low = df['high'] - df['low']
                high_close = np.abs(df['high'] - df['close'].shift())
                low_close = np.abs(df['low'] - df['close'].shift())
                tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                df['atr14'] = tr.rolling(14).mean()
            
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # 5. BOLLINGER BANDS (2-std only)
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            bb_2 = ta.bbands(df['close'], length=20, std=2)
            df = pd.concat([df, bb_2], axis=1)
            
            # Keltner Channels
            kc = ta.kc(df['high'], df['low'], df['close'], length=20)
            df = pd.concat([df, kc], axis=1)
            
            # Trend (ADX)
            adx = ta.adx(df['high'], df['low'], df['close'], length=14)
            df = pd.concat([df, adx], axis=1)
            
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # 6. DERIVATIVES FEATURES (if data available)
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            if 'funding_rate' in df.columns:
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
            
            if 'open_interest' in df.columns:
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
            
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # 7. VOLUME FEATURES
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            df['volume_ema_short'] = df['volume'].ewm(span=5).mean()
            df['volume_ema_long'] = df['volume'].ewm(span=20).mean()
            df['volume_oscillator'] = (df['volume_ema_short'] - df['volume_ema_long']) / (df['volume_ema_long'] + 1e-8)
            
            df['vol_12'] = df['volume'].rolling(12).mean()
            df['vol_sma20'] = df['volume'].rolling(20).mean()
            df['vol_ratio'] = df['volume'] / (df['vol_sma20'] + 1e-8)
            
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # 8. MOMENTUM FEATURES
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            df['mom_6'] = df['close'].pct_change(6)
            df['mom_12'] = df['close'].pct_change(12)
            
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # 9. MULTI-TIMEFRAME FEATURES
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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
            
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # 10. STATIONARY DISTANCES & Z-SCORES
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            df['distance_from_20h_high'] = (df['close'] - df['high'].rolling(20).max()) / (df['close'] + 1e-8)
            df['distance_from_20h_low'] = (df['close'] - df['low'].rolling(20).min()) / (df['close'] + 1e-8)
            df['price_zscore_20'] = (df['close'] - df['close'].rolling(20).mean()) / (df['close'].rolling(20).std() + 1e-8)
            
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # 11. MARKET MICROSTRUCTURE
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            if 'taker_buy_quote_volume' in df.columns and 'quote_asset_volume' in df.columns:
                df['buy_pressure'] = df['taker_buy_quote_volume'] / (df['quote_asset_volume'] + 1e-8)
                df['order_imbalance'] = (
                    2 * df['taker_buy_quote_volume'] - df['quote_asset_volume']
                ) / (df['quote_asset_volume'] + 1e-8)
            else:
                df['buy_pressure'] = 0.5
                df['order_imbalance'] = 0.0
            
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # 12. SEASONALITY
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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
            
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # 13. VOLATILITY (WITH SAFETY CHECK)
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            df['volatility_rolling'] = df['close'].rolling(20).std()
            
            # ✅ FIX: Check if atr14 exists before using it
            if 'atr14' in df.columns and not df['atr14'].isna().all():
                df['volatility_pct'] = df['atr14'] / (df['close'] + 1e-8)
            else:
                logger.warning("ATR14 not available, using price std as volatility proxy")
                df['volatility_pct'] = df['volatility_rolling'] / (df['close'] + 1e-8)
            
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # 14. RENAME COLUMNS TO MATCH MODEL
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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
            
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # 15. CLEAN UP
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            df = df.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
            
            # Store feature columns
            self.feature_columns = self.get_stationary_features(df)
            
            # ✅ Log ATR status
            if 'atr14' in df.columns:
                logger.info(f"✅ ATR14 available: mean={df['atr14'].mean():.6f}, non-zero={ (df['atr14'] > 0).sum() }")
            
            return df
            
        except Exception as e:
            logger.error(f"Error engineering features: {e}", exc_info=True)
            return pd.DataFrame()
    
    def get_stationary_features(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of stationary features only.
        Excludes non-stationary features (raw prices, price levels).
        """
        all_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Filter out non-stationary features
        stationary_features = [
            col for col in all_cols 
            if col not in self.non_stationary_cols
            and not col.startswith('future_')
            and not col.startswith('target_')
        ]
        
        return stationary_features
    
    def create_targets(self, df: pd.DataFrame, thresholds: Dict = None) -> pd.DataFrame:
        """
        Create 3-class targets for each timeframe.
        
        Args:
            df: DataFrame with future returns
            thresholds: Dict with thresholds for each timeframe
                e.g., {'1h': 0.003, '4h': 0.008, '1d': 0.015}
        """
        if thresholds is None:
            thresholds = {'1h': 0.003, '4h': 0.008, '1d': 0.015}
        
        df = df.copy()
        
        for horizon, thresh in thresholds.items():
            ret_col = f'future_ret_{horizon}'
            
            if ret_col not in df.columns:
                shifts = {'1h': 1, '4h': 4, '1d': 24}
                df[ret_col] = df['close'].shift(-shifts[horizon]) / df['close'] - 1
            
            df[f'target_{horizon}'] = np.where(
                df[ret_col] > thresh, 2,
                np.where(df[ret_col] < -thresh, 0, 1)
            )
        
        return df
    
    def create_risk_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create risk level targets from volatility quantiles"""
        df = df.copy()
        df['volatility_pct'] = df['atr14'] / (df['close'] + 1e-8)
        
        # Fit quantiles on training data (first 70%)
        train_vol = df['volatility_pct'].iloc[:int(len(df)*0.7)]
        q33, q67 = train_vol.quantile(0.33), train_vol.quantile(0.67)
        
        df['risk_level'] = pd.cut(
            df['volatility_pct'],
            bins=[-np.inf, q33, q67, np.inf],
            labels=[0, 1, 2]  # 0=Low, 1=Medium, 2=High
        ).astype(int)
        
        return df
    
    def create_regime_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create market regime targets from ADX and volatility"""
        df = df.copy()
        df['volatility_pct'] = df['atr14'] / (df['close'] + 1e-8)
        
        adx_col = 'ADX_14' if 'ADX_14' in df.columns else 'adx'
        
        # Use training data for quantile
        train_vol = df['volatility_pct'].iloc[:int(len(df)*0.7)]
        q67 = train_vol.quantile(0.67)
        
        conditions = [
            (df[adx_col] < 20),                                                           # Ranging
            (df[adx_col] > 25) & (df['volatility_pct'] < q67),                           # Trending
            (df['volatility_pct'] >= q67),                                               # Volatile
        ]
        df['market_regime'] = np.select(conditions, [1, 0, 2], default=3)  # 3=Transition
        
        return df
    
    def create_confidence_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create confidence targets from multi-timeframe agreement"""
        df = df.copy()
        
        agreement = (
            (df['target_1h'] == df['target_4h']) & 
            (df['target_4h'] == df['target_1d'])
        ).astype(float)
        
        df['confidence_score'] = agreement * 0.7 + 0.3
        return df
    
    def get_feature_columns(self) -> List[str]:
        """Get list of feature columns"""
        return self.feature_columns