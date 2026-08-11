"""
Seamless Parquet Dataset Sync & Merger (2019 to Present)
Appends missing 320.8 days (Sept 2025 to Present) to master Parquet dataset.
Run: python sync_parquet.py
"""

import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime, timedelta

parquet_path = "data/raw/combined_multi_horizon_1h.parquet"

if not os.path.exists(parquet_path):
    print(f"❌ File not found at '{parquet_path}'")
    exit()

print("=" * 75)
print("🚀 SMARTCRYPTO PARQUET SYNC & MERGER (2019 - PRESENT)")
print("=" * 75)

# 1. Load Master Dataset
df_master = pd.read_parquet(parquet_path)
ts_col = 'timestamp' if 'timestamp' in df_master.columns else 'open_time'
df_master[ts_col] = pd.to_datetime(df_master[ts_col])

symbols = df_master['symbol'].unique().tolist()
max_ts = df_master[ts_col].max()
now = pd.Timestamp.now()

print(f"📊 Master Dataset: {len(df_master):,} records across {len(symbols)} symbols")
print(f"📋 Columns ({len(df_master.columns)}): {list(df_master.columns)}")
print(f"📅 Current End Date: {max_ts}")
print(f"📥 Fetching missing candles from {max_ts} to present...\n")

def process_raw_candles(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer all 47 features and quantile targets matching the master dataset schema"""
    df = df.copy()
    
    # Basic Features
    df['interval'] = '1h'
    df['log_close'] = np.log(df['close'])
    df['ret_1'] = df['close'].pct_change()
    df['ret_3'] = df['close'].pct_change(3)
    df['range'] = (df['high'] - df['low']) / df['close']
    
    # Moving Averages & Momentum
    df['sma20'] = df['close'].rolling(20).mean()
    df['sma50'] = df['close'].rolling(50).mean()
    df['ema12'] = df['close'].ewm(span=12).mean()
    df['ema26'] = df['close'].ewm(span=26).mean()
    df['mom_6'] = df['close'].pct_change(6)
    df['mom_12'] = df['close'].pct_change(12)
    
    # ATR14
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr14'] = tr.rolling(14).mean()

    # Volume Indicators
    df['vol_12'] = df['volume'].rolling(12).mean()
    df['vol_sma20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / (df['vol_sma20'] + 1e-8)

    # 4H Multi-Timeframe Resampling
    df['open_4h'] = df['open'].rolling(4).apply(lambda x: x[0], raw=True)
    df['high_4h'] = df['high'].rolling(4).max()
    df['low_4h'] = df['low'].rolling(4).min()
    df['close_4h'] = df['close'].rolling(4).apply(lambda x: x[-1], raw=True)
    df['volume_4h'] = df['volume'].rolling(4).sum()

    # 1D Multi-Timeframe Resampling
    df['open_1d'] = df['open'].rolling(24).apply(lambda x: x[0], raw=True)
    df['high_1d'] = df['high'].rolling(24).max()
    df['low_1d'] = df['low'].rolling(24).min()
    df['close_1d'] = df['close'].rolling(24).apply(lambda x: x[-1], raw=True)
    df['volume_1d'] = df['volume'].rolling(24).sum()

    # Price Position Indicators
    df['price_pos_4h'] = (df['close'] - df['low_4h']) / (df['high_4h'] - df['low_4h'] + 1e-8)
    df['price_pos_1d'] = (df['close'] - df['low_1d']) / (df['high_1d'] - df['low_1d'] + 1e-8)

    # Multi-Horizon Futures & Quantile Targets
    horizons = {'1h': 1, '4h': 4, '1d': 24}
    for horizon, shift in horizons.items():
        df[f'future_close_{horizon}'] = df['close'].shift(-shift)
        df[f'future_ret_{horizon}'] = df[f'future_close_{horizon}'] / df['close'] - 1.0

        valid_rets = df[f'future_ret_{horizon}'].dropna()
        if len(valid_rets) > 50:
            q33 = valid_rets.quantile(0.33)
            q67 = valid_rets.quantile(0.67)
        else:
            q33, q67 = -0.003, 0.003

        df[f'target_{horizon}'] = np.where(
            df[f'future_ret_{horizon}'] > q67, 2,
            np.where(df[f'future_ret_{horizon}'] < q33, 0, 1)
        )

    # Clean infinities and fill NaNs
    df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
    return df

new_data_frames = []

for symbol in symbols:
    symbol_master = df_master[df_master['symbol'] == symbol]
    last_symbol_ts = symbol_master[ts_col].max()
    start_time_ms = int(last_symbol_ts.timestamp() * 1000) + 1
    now_ms = int(datetime.utcnow().timestamp() * 1000)

    print(f"🪙 Fetching {symbol} from {last_symbol_ts}...")
    
    all_klines = []
    while start_time_ms < now_ms:
        url = "https://api.binance.com/api/v3/klines"
        params = {
            'symbol': symbol,
            'interval': '1h',
            'limit': 1000,
            'startTime': start_time_ms
        }
        try:
            res = requests.get(url, params=params, timeout=15)
            if res.status_code != 200:
                break
            data = res.json()
            if not data:
                break
                
            for kline in data:
                all_klines.append([
                    symbol,
                    pd.to_datetime(kline[0], unit='ms'),
                    float(kline[1]), float(kline[2]), float(kline[3]), float(kline[4]), float(kline[5]),
                    float(kline[7]) if len(kline) > 7 else 0.0,
                    float(kline[8]) if len(kline) > 8 else 0.0,
                    float(kline[9]) if len(kline) > 9 else 0.0,
                    float(kline[10]) if len(kline) > 10 else 0.0
                ])
                
            start_time_ms = data[-1][0] + 1
            if len(data) < 1000:
                break
        except Exception as e:
            print(f"❌ Error fetching {symbol}: {e}")
            break

    if all_klines:
        df_raw = pd.DataFrame(all_klines, columns=[
            'symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'quote_asset_volume', 'trades_count', 'taker_buy_base_volume', 'taker_buy_quote_volume'
        ])
        
        df_featured = process_raw_candles(df_raw)
        new_data_frames.append(df_featured)
        print(f"   ✅ {symbol}: {len(df_featured):,} new candles fetched and engineered.")

if new_data_frames:
    print("\n🔗 Merging new records into Master Parquet Dataset...")
    df_new_all = pd.concat(new_data_frames, ignore_index=True)
    
    # Ensure exact 47 column alignment with master schema
    df_combined = pd.concat([df_master, df_new_all[df_master.columns]], ignore_index=True)
    df_combined = df_combined.drop_duplicates(subset=['symbol', ts_col]).sort_values(['symbol', ts_col]).reset_index(drop=True)
    
    # Overwrite master Parquet file on disk
    df_combined.to_parquet(parquet_path, index=False)
    
    print("=" * 75)
    print(f"🎉 SUCCESS! Master Parquet Dataset updated on disk!")
    print(f"📊 Total Records: {len(df_combined):,} (Added {len(df_combined) - len(df_master):,} new rows)")
    print(f"📅 New Date Range: {df_combined[ts_col].min()} ➔ {df_combined[ts_col].max()}")
    print("=" * 75)
else:
    print("Dataset is already up to date!")