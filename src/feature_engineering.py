"""
Feature engineering with technical indicators and derivatives
"""
import numpy as np
import pandas as pd
import pandas_ta as ta
import ccxt
from datetime import datetime
from src.config import config
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Advanced feature engineering for crypto trading"""
    
    def __init__(self):
        self.exchange = ccxt.binanceusdm({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
    
    def fetch_derivatives_data(self, symbol_ccxt='BTC/USDT'):
        """Fetch funding rates and open interest"""
        try:
            # Funding Rates
            funding_raw = self.exchange.fetch_funding_rate_history(
                symbol=symbol_ccxt, limit=1000
            )
            df_funding = pd.DataFrame(funding_raw)
            df_funding['timestamp'] = pd.to_datetime(df_funding['timestamp'], unit='ms')
            df_funding = df_funding[['timestamp', 'fundingRate']].rename(
                columns={'fundingRate': 'funding_rate'}
            )
            
            # Open Interest - FIXED
            try:
                oi = self.exchange.fetch_open_interest(symbol_ccxt)
                df_oi = pd.DataFrame([{
                    'timestamp': pd.Timestamp.now(),
                    'open_interest': oi.get('openInterest', 0),
                    'open_interest_usd': oi.get('openInterestValue', 0)
                }])
            except:
                try:
                    symbol_raw = symbol_ccxt.replace('/', '')
                    oi_raw = self.exchange.public_get_futures_data_openinteresthist({
                        'symbol': symbol_raw,
                        'period': '1h',
                        'limit': 1000
                    })
                    df_oi = pd.DataFrame(oi_raw)
                    df_oi['timestamp'] = pd.to_datetime(df_oi['timestamp'].astype(int), unit='ms')
                    df_oi['open_interest'] = df_oi['sumOpenInterest'].astype(float)
                    df_oi['open_interest_usd'] = df_oi['sumOpenInterestValue'].astype(float)
                    df_oi = df_oi[['timestamp', 'open_interest', 'open_interest_usd']]
                except:
                    df_oi = pd.DataFrame(columns=['timestamp', 'open_interest', 'open_interest_usd'])
                    logger.warning("Using zero values for Open Interest")
            
            # Merge
            if len(df_oi) > 0 and len(df_funding) > 0:
                df_deriv = pd.merge(df_funding, df_oi, on='timestamp', how='outer')
                df_deriv = df_deriv.sort_values('timestamp').ffill().bfill().fillna(0)
            else:
                df_deriv = df_funding.copy()
                df_deriv['open_interest'] = 0
                df_deriv['open_interest_usd'] = 0
            
            return df_deriv
            
        except Exception as e:
            logger.warning(f"Could not fetch derivatives data: {e}")
            return None
    
    def add_technical_indicators(self, df):
        """Add all technical indicators"""
        df = df.copy()
        
        # Price features
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['money_flow'] = df['typical_price'] * df['volume']
        
        # VWAP (rolling 24h)
        df['vwap'] = (
            df['money_flow'].rolling(config.VWAP_WINDOW).sum() / 
            (df['volume'].rolling(config.VWAP_WINDOW).sum() + 1e-8)
        ).ffill()
        df['price_vwap_ratio'] = df['close'] / df['vwap']
        
        # Volume features
        df['volume_ema_short'] = df['volume'].ewm(span=5).mean()
        df['volume_ema_long'] = df['volume'].ewm(span=20).mean()
        df['volume_oscillator'] = (
            df['volume_ema_short'] - df['volume_ema_long']
        ) / (df['volume_ema_long'] + 1e-8)
        
        # Momentum
        df['rsi_14'] = ta.rsi(df['close'], length=14)
        stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3)
        macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
        df = pd.concat([df, stoch, macd], axis=1)
        
        # Volatility
        bb_2 = ta.bbands(df['close'], length=20, std=2)
        kc = ta.kc(df['high'], df['low'], df['close'], length=20)
        df = pd.concat([df, bb_2, kc], axis=1)
        
        # Trend
        adx = ta.adx(df['high'], df['low'], df['close'], length=14)
        df = pd.concat([df, adx], axis=1)
        
        # Stationary features
        df['distance_from_20h_high'] = (
            df['close'] - df['high'].rolling(20).max()
        ) / (df['close'] + 1e-8)
        df['distance_from_20h_low'] = (
            df['close'] - df['low'].rolling(20).min()
        ) / (df['close'] + 1e-8)
        df['price_zscore_20'] = (
            df['close'] - df['close'].rolling(20).mean()
        ) / (df['close'].rolling(20).std() + 1e-8)
        
        # Market microstructure
        if 'taker_buy_quote_volume' in df.columns and 'quote_asset_volume' in df.columns:
            df['buy_pressure'] = df['taker_buy_quote_volume'] / (df['quote_asset_volume'] + 1e-8)
            df['order_imbalance'] = (
                2 * df['taker_buy_quote_volume'] - df['quote_asset_volume']
            ) / (df['quote_asset_volume'] + 1e-8)
        
        # Seasonality
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Clean
        df = df.ffill().bfill().fillna(0).replace([np.inf, -np.inf], 0)
        
        return df
    
    def add_derivatives_features(self, df, df_deriv):
        """Add derivatives features"""
        if df_deriv is None:
            return df
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = pd.merge(df, df_deriv, on='timestamp', how='left')
        
        # Funding rate features
        df['funding_8h'] = df['funding_rate'].rolling(8).sum()
        df['funding_24h'] = df['funding_rate'].rolling(24).sum()
        df['funding_change_8h'] = df['funding_rate'].diff(8).fillna(0)
        df['funding_change_24h'] = df['funding_rate'].diff(24).fillna(0)
        
        # Funding Z-score
        mean_funding = df['funding_rate'].rolling(config.FUNDING_WINDOW).mean()
        std_funding = df['funding_rate'].rolling(config.FUNDING_WINDOW).std()
        df['funding_zscore'] = (df['funding_rate'] - mean_funding) / (std_funding + 1e-8)
        
        # Funding extremes
        df['funding_percentile'] = (
            (df['funding_rate'] - df['funding_rate'].rolling(90).min()) /
            (df['funding_rate'].rolling(90).max() - df['funding_rate'].rolling(90).min() + 1e-8)
        )
        
        # Open interest features
        df['oi_change_1h'] = df['open_interest'].pct_change(1).fillna(0)
        df['oi_change_24h'] = df['open_interest'].pct_change(24).fillna(0)
        df['oi_position_24h'] = (
            (df['open_interest'] - df['open_interest'].rolling(24).min()) /
            (df['open_interest'].rolling(24).max() - df['open_interest'].rolling(24).min() + 1e-8)
        )
        
        # Leverage features
        df['oi_volume_ratio'] = df['open_interest_usd'] / (df['volume'] * df['close'] + 1e-8)
        df['oi_turnover_ratio'] = df['open_interest'] / (df['volume'].rolling(24).mean() + 1e-8)
        
        # Price-OI divergence
        price_dir = np.sign(df['close'].pct_change(24).fillna(0))
        oi_dir = np.sign(df['oi_change_24h'])
        df['price_oi_divergence'] = price_dir * oi_dir
        
        # Clean
        df = df.ffill().bfill().fillna(0).replace([np.inf, -np.inf], 0)
        
        return df
    
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
    
    def create_risk_targets(self, df):
        """Create risk level targets"""
        df['volatility_pct'] = df['atr14'] / (df['close'] + 1e-8)
        train_vol = df['volatility_pct'].iloc[:int(len(df)*0.7)]
        q33, q67 = train_vol.quantile(0.33), train_vol.quantile(0.67)
        
        df['risk_level'] = pd.cut(
            df['volatility_pct'],
            bins=[-np.inf, q33, q67, np.inf],
            labels=[0, 1, 2]
        ).astype(int)
        
        return df
    
    def create_regime_targets(self, df):
        """Create market regime targets"""
        adx_col = 'ADX_14' if 'ADX_14' in df.columns else 'adx'
        df['volatility_pct'] = df.get('volatility_pct', df['atr14'] / (df['close'] + 1e-8))
        
        conditions = [
            (df[adx_col] < 20),
            (df[adx_col] > 25) & (df['volatility_pct'] < df['volatility_pct'].quantile(0.67)),
            (df['volatility_pct'] >= df['volatility_pct'].quantile(0.67))
        ]
        df['market_regime'] = np.select(conditions, [1, 0, 2], default=3)
        
        return df
    
    def create_confidence_target(self, df):
        """Create confidence targets from multi-timeframe agreement"""
        agreement = (
            (df['target_1h'] == df['target_4h']) & 
            (df['target_4h'] == df['target_1d'])
        ).astype(float)
        df['confidence_score'] = agreement * 0.7 + 0.3
        return df
    
    def get_stationary_features(self, df):
        """Get list of strictly stationary features - BULLETPROOF VERSION"""
        
        # 1. Exact column names to exclude
        non_stationary = [
            # Raw prices
            'open', 'high', 'low', 'close', 'log_close',
            # Moving averages
            'sma20', 'sma50', 'ema12', 'ema26',
            # Volume weighted averages & raw price values
            'vwap', 'typical_price', 'money_flow',
            # Multi-timeframe raw prices
            'open_4h', 'high_4h', 'low_4h', 'close_4h',
            'open_1d', 'high_1d', 'low_1d', 'close_1d',
            # Pivot levels
            'pivot', 'support1', 'resistance1',
            # Derivatives raw values
            'funding_rate', 'open_interest', 'open_interest_usd',
            'funding_high', 'funding_low',
            # Target & auxiliary target leakage features
            'confidence_score', 'market_regime', 'risk_level', 'volatility_regime',
            # Identifiers
            'timestamp', 'symbol', 'interval', 'date',
        ]
        
        # Get all numeric columns
        all_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        stationary_cols = []
        for col in all_cols:
            # Skip exact non-stationary columns
            if col in non_stationary:
                continue
                
            # Skip dynamic target/future columns
            if col.startswith('target_') or col.startswith('future_'):
                continue
                
            # Skip raw Bollinger Band price levels (BBL, BBM, BBU), but KEEP BBB (bandwidth) and BBP (%B)!
            if ('BBL_' in col or 'BBM_' in col or 'BBU_' in col) and not ('BBB_' in col or 'BBP_' in col):
                continue
                
            # Skip Ichimoku & Keltner raw price level prefixes
            if col.startswith('ichi_') or col.startswith('KC'):
                continue
                
            stationary_cols.append(col)
            
        return stationary_cols