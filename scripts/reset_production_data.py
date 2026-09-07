#!/usr/bin/env python3
"""
Production Data Cleanup & Reset Utility for SmartCrypto.
Clears signals, live positions, and historical trade logs while preserving
users, subscriptions, exchange API keys, and billing records.

Usage:
  Direct Python:
    python scripts/reset_production_data.py --all --force
  Inside Docker Container:
    docker exec -it smartcrypto-ai-v3 python scripts/reset_production_data.py --all --force
"""

import os
import sys
import glob
import sqlite3
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def reset_sqlite_data(db_paths: list[str], clear_signals: bool = True, clear_positions: bool = True):
    """Purge signals and positions tables from SQLite databases without touching users/billing."""
    print("\n📦 [1/3] Scanning and cleaning SQLite databases...")
    
    for db_path in db_paths:
        if not os.path.exists(db_path):
            continue
            
        print(f"  🔍 Connecting to database: {db_path}")
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Get existing tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = {row[0] for row in cursor.fetchall()}
            
            if clear_signals and "signals" in tables:
                cursor.execute("SELECT COUNT(*) FROM signals;")
                count = cursor.fetchone()[0]
                cursor.execute("DELETE FROM signals;")
                print(f"    🗑️ Deleted {count} records from 'signals'")
                
            if clear_signals and "pattern_drawings" in tables:
                cursor.execute("SELECT COUNT(*) FROM pattern_drawings;")
                count = cursor.fetchone()[0]
                cursor.execute("DELETE FROM pattern_drawings;")
                print(f"    🗑️ Deleted {count} records from 'pattern_drawings'")

            if clear_positions:
                for pos_table in ["closed_trades", "closed_positions", "positions", "trades", "performance_metrics"]:
                    if pos_table in tables:
                        cursor.execute(f"SELECT COUNT(*) FROM {pos_table};")
                        count = cursor.fetchone()[0]
                        cursor.execute(f"DELETE FROM {pos_table};")
                        print(f"    🗑️ Deleted {count} records from '{pos_table}'")

            conn.commit()
            # Vacuum to reclaim disk space
            cursor.execute("VACUUM;")
            conn.close()
            print(f"  ✅ Successfully cleaned and vacuumed {db_path}")
        except Exception as e:
            print(f"  ❌ Error processing {db_path}: {e}")


def reset_file_storage(clear_signals: bool = True, clear_positions: bool = True):
    """Purge persistent JSON and JSONL position and signal history files."""
    print("\n📁 [2/3] Cleaning persistent JSON/JSONL volume files...")
    
    dirs_to_clean = []
    if clear_positions:
        dirs_to_clean.append("positions")
    if clear_signals:
        dirs_to_clean.append("signal_history")
    
    # Also clean files in data/
    data_files = []
    if clear_signals:
        data_files.extend(["data/signals.jsonl", "data/pattern_drawings.jsonl"])
    if clear_positions:
        data_files.extend(["data/performance.json", "data/patterns.json"])

    # Clean directories
    for dir_name in dirs_to_clean:
        p = Path(dir_name)
        if p.exists() and p.is_dir():
            files = list(p.glob("*.*"))
            for f in files:
                try:
                    f.unlink()
                    print(f"    🗑️ Deleted file: {f}")
                except Exception as e:
                    print(f"    ⚠️ Could not delete {f}: {e}")
            print(f"  ✅ Cleared folder '{dir_name}/' ({len(files)} files removed)")
        else:
            p.mkdir(parents=True, exist_ok=True)
            print(f"  ℹ️ Created clean folder '{dir_name}/'")

    # Clean data files
    for df in data_files:
        p = Path(df)
        if p.exists():
            try:
                p.unlink()
                print(f"    🗑️ Deleted data file: {df}")
            except Exception as e:
                print(f"    ⚠️ Could not delete {df}: {e}")


def reset_redis_cache():
    """Flush signals and portfolio cache keys from Redis."""
    print("\n⚡ [3/3] Clearing Redis memory cache...")
    try:
        from src.core.config import get_settings
        import redis

        settings = get_settings()
        redis_url = getattr(settings, "REDIS_URL", "redis://localhost:6379/0")
        r = redis.from_url(redis_url)
        
        # Scan and delete signals, portfolio, positions, and market caches
        keys = r.keys("signals:*") + r.keys("portfolio:*") + r.keys("positions:*") + r.keys("market:*")
        if keys:
            r.delete(*keys)
            print(f"  ✅ Flushed {len(keys)} Redis cache keys matching signals/portfolio")
        else:
            print("  ℹ️ No active Redis cache keys found matching signals/portfolio")
    except Exception as e:
        print(f"  ℹ️ Redis flush skipped ({e})")


def main():
    parser = argparse.ArgumentParser(description="SmartCrypto Production Data Reset Tool")
    parser.add_argument("--all", action="store_true", help="Delete both signals and positions data")
    parser.add_argument("--signals-only", action="store_true", help="Delete only signals data")
    parser.add_argument("--positions-only", action="store_true", help="Delete only positions/trades data")
    parser.add_argument("--force", "-f", action="store_true", help="Skip confirmation prompt")
    
    args = parser.parse_args()

    clear_signals = args.all or args.signals_only or (not args.positions_only)
    clear_positions = args.all or args.positions_only or (not args.signals_only)

    print("=" * 65)
    print("🚨 SMARTCRYPTO PRODUCTION DATABASE & VOLUME RESET UTILITY")
    print("=" * 65)
    print(f"Target: Signals = {clear_signals} | Positions/Trades = {clear_positions}")
    print("Security: User accounts, subscriptions, & exchange API keys are PROTECTED.")
    print("=" * 65)

    if not args.force:
        confirm = input("\n⚠️ Are you sure you want to proceed? (yes/no): ").strip().lower()
        if confirm != "yes":
            print("Operation aborted.")
            return

    # Potential database paths
    possible_dbs = [
        "data/smartcrypto.db",
        "data/app.db",
        "storage/data.db",
        "smartcrypto.db",
    ]

    reset_sqlite_data(possible_dbs, clear_signals=clear_signals, clear_positions=clear_positions)
    reset_file_storage(clear_signals=clear_signals, clear_positions=clear_positions)
    reset_redis_cache()

    print("\n" + "=" * 65)
    print("🎉 RESET COMPLETE! All signals and positions have been safely purged.")
    print("💡 Next Step: If your container is running, restart it to load clean state:")
    print("   docker compose restart smartcrypto-bot")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
