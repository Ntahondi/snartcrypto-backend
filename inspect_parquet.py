"""
Parquet Dataset Inspector
Run: python inspect_parquet.py
"""

import pandas as pd
import os
from datetime import datetime

parquet_path = "data/raw/combined_multi_horizon_1h.parquet"

if not os.path.exists(parquet_path):
    print(f"❌ File not found at '{parquet_path}'")
    # Search for alternative parquet files in project
    found = []
    for root, dirs, files in os.walk("."):
        for f in files:
            if f.endswith(".parquet"):
                found.append(os.path.join(root, f))
    if found:
        print(f"Found parquet files in project: {found}")
    exit()

print("=" * 70)
print(f"📦 PARQUET DATASET INSPECTION: {parquet_path}")
print("=" * 70)

df = pd.read_parquet(parquet_path)

print(f"📊 Total Records: {len(df):,}")
print(f"📐 File Size on Disk: {os.path.getsize(parquet_path) / (1024*1024):.2f} MB")
print(f"📋 Total Columns: {len(df.columns)}")

if 'symbol' in df.columns:
    print(f"🪙 Symbols ({df['symbol'].nunique()}): {df['symbol'].unique().tolist()}")

# Determine timestamp column
ts_col = None
for col in ['timestamp', 'open_time', 'time', 'date']:
    if col in df.columns:
        ts_col = col
        break

if ts_col:
    df[ts_col] = pd.to_datetime(df[ts_col])
    min_ts = df[ts_col].min()
    max_ts = df[ts_col].max()
    now = pd.Timestamp.now(tz='UTC') if max_ts.tzinfo else pd.Timestamp.now()
    
    gap_hours = (now - max_ts).total_seconds() / 3600
    gap_days = gap_hours / 24
    
    print(f"\n📅 Start Date: {min_ts}")
    print(f"📅 End Date:   {max_ts}")
    print(f"⏰ Outdated Gap: {gap_hours:.1f} hours ({gap_days:.1f} days behind present time)")

print("=" * 70)