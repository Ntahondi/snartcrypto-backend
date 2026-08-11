# smartcrypto_ai_models/candle_microstructure.py

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class UnconstrainedCandleExtractor:
    """
    Extracts complete, unconstrained 30-channel dynamic market matrix
    combining multi-period returns, full candle anatomy, volume microstructure,
    and volatility expansion channels.
    """

    @staticmethod
    def extract_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Guarantee basic volume columns exist
        for col in ['quote_asset_volume', 'trades_count', 'taker_buy_quote_volume', 'taker_buy_base_volume']:
            if col not in df.columns:
                df[col] = 0.0
            else:
                df[col] = df[col].fillna(0.0)

        # 1. Base Volatility (ATR14 & ATR50)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        df['atr14'] = tr.rolling(14).mean().ffill().fillna(1e-4)
        df['atr50'] = tr.rolling(50).mean().ffill().fillna(1e-4)

        bar_range = (df['high'] - df['low']).replace(0.0, 1e-8)
        body_size = np.abs(df['close'] - df['open'])

        # 2. Multi-Period Returns (Short, Medium, Long-Term Dynamics)
        df['ret_1'] = df['close'].pct_change(1).fillna(0.0)
        df['ret_2'] = df['close'].pct_change(2).fillna(0.0)
        df['ret_3'] = df['close'].pct_change(3).fillna(0.0)
        df['ret_6'] = df['close'].pct_change(6).fillna(0.0)
        df['ret_12'] = df['close'].pct_change(12).fillna(0.0)
        df['ret_24'] = df['close'].pct_change(24).fillna(0.0)
        df['ret_48'] = df['close'].pct_change(48).fillna(0.0)

        # 3. Candle Anatomy & Geometry Ratios
        df['body_ratio'] = body_size / bar_range
        upper_wick = df['high'] - np.maximum(df['open'], df['close'])
        lower_wick = np.minimum(df['open'], df['close']) - df['low']
        
        df['upper_wick_ratio'] = upper_wick / bar_range
        df['lower_wick_ratio'] = lower_wick / bar_range
        df['candle_dir'] = np.sign(df['close'] - df['open'])

        # 4. Volatility Expansion Ratios
        df['bar_expansion_ratio'] = bar_range / (df['atr14'] + 1e-8)
        df['atr_ratio'] = df['atr14'] / (df['atr50'] + 1e-8)

        # 5. Volume Microstructure & Taker Force
        vol_sma20 = df['volume'].rolling(20).mean().replace(0.0, 1e-8)
        df['volume_force'] = (df['volume'] * df['candle_dir']) / vol_sma20

        trade_sma20 = df['trades_count'].rolling(20).mean().replace(0.0, 1e-8)
        df['trade_count_ratio'] = df['trades_count'] / trade_sma20

        if (df['taker_buy_quote_volume'] > 0).any() and (df['quote_asset_volume'] > 0).any():
            df['buy_pressure'] = df['taker_buy_quote_volume'] / (df['quote_asset_volume'] + 1e-8)
            df['order_imbalance'] = (
                2.0 * df['taker_buy_quote_volume'] - df['quote_asset_volume']
            ) / (df['quote_asset_volume'] + 1e-8)
        else:
            df['buy_pressure'] = 0.5
            df['order_imbalance'] = 0.0

        # 6. Stationary Z-Scores (Clipped for NN Stability)
        price_std_20 = df['close'].rolling(20).std().replace(0.0, 1e-8)
        price_std_50 = df['close'].rolling(50).std().replace(0.0, 1e-8)
        
        df['price_zscore_20'] = ((df['close'] - df['close'].rolling(20).mean()) / price_std_20).clip(-5.0, 5.0)
        df['price_zscore_50'] = ((df['close'] - df['close'].rolling(50).mean()) / price_std_50).clip(-5.0, 5.0)

        # 7. Cyclical Time Encoding
        if 'timestamp' in df.columns:
            ts = pd.to_datetime(df['timestamp'])
            df['hour_sin'] = np.sin(2.0 * np.pi * ts.dt.hour / 24.0)
            df['hour_cos'] = np.cos(2.0 * np.pi * ts.dt.hour / 24.0)
            df['day_sin'] = np.sin(2.0 * np.pi * ts.dt.dayofweek / 7.0)
            df['day_cos'] = np.cos(2.0 * np.pi * ts.dt.dayofweek / 7.0)
        else:
            df['hour_sin'] = 0.0
            df['hour_cos'] = 0.0
            df['day_sin'] = 0.0
            df['day_cos'] = 0.0

        df = df.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
        return df

    @staticmethod
    def get_feature_columns() -> list:
        """Returns full unconstrained 21-channel feature list"""
        return [
            'ret_1', 'ret_2', 'ret_3', 'ret_6', 'ret_12', 'ret_24', 'ret_48',
            'body_ratio', 'upper_wick_ratio', 'lower_wick_ratio', 'candle_dir',
            'bar_expansion_ratio', 'atr_ratio', 'volume_force', 'trade_count_ratio',
            'buy_pressure', 'order_imbalance', 'price_zscore_20', 'price_zscore_50',
            'hour_sin', 'hour_cos'
        ]