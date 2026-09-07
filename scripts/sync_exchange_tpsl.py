#!/usr/bin/env python3
"""
scripts/sync_exchange_tpsl.py

Installs exchange-side Stop Loss and Take Profit orders on Bitget for currently
active positions in PortfolioManager.
"""

import asyncio
import json
import os
import sys
import uuid
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.config import get_settings
from src.utils.safe_logger import SafeLogger
import ccxt.async_support as ccxt

logger = SafeLogger.get_logger(__name__)


async def sync_tpsl():
    settings = get_settings()

    api_key = settings.BITGET_API_KEY
    secret = settings.BITGET_API_SECRET
    password = settings.BITGET_PASSPHRASE

    if not (api_key and secret and password):
        print("❌ Bitget API credentials missing in .env")
        return

    # 1. Load active positions from trade_history.jsonl
    journal_path = Path("positions/trade_history.jsonl")
    if not journal_path.exists():
        print("ℹ️ No positions/trade_history.jsonl file found.")
        return

    latest_positions = {}
    with open(journal_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                pos_id = data.get("id")
                if pos_id:
                    latest_positions[pos_id] = data
            except Exception:
                continue

    open_positions = [p for p in latest_positions.values() if p.get("status") == "OPEN"]
    if not open_positions:
        print("ℹ️ No active OPEN positions found in journal.")
        return

    print(f"🔍 Found {len(open_positions)} active positions in journal:")
    for p in open_positions:
        print(f"   - {p.get('symbol')} {p.get('action')} | Entry: {p.get('entry_price')} | SL: {p.get('stop_loss')} | TP: {p.get('take_profit')}")

    # 2. Connect to Bitget
    exchange = ccxt.bitget({
        "apiKey": api_key,
        "secret": secret,
        "password": password,
        "enableRateLimit": True,
        "options": {
            "defaultType": "swap",
        },
    })

    try:
        await exchange.load_markets()
        print("✅ Bitget markets loaded successfully.")

        for pos in open_positions:
            raw_symbol = pos.get("symbol", "").upper()
            base = raw_symbol.replace("USDT", "")
            ccxt_symbol = f"{base}/USDT:USDT"

            if ccxt_symbol not in exchange.markets:
                print(f"⚠️ Market {ccxt_symbol} not found on Bitget.")
                continue

            action = pos.get("action", "BUY").upper()
            close_side = "sell" if action in ("BUY", "LONG") else "buy"
            sl_price = float(pos.get("stop_loss", 0.0) or 0.0)
            tp_price = float(pos.get("take_profit", 0.0) or 0.0)

            # Query live position size on Bitget
            try:
                live_positions = await exchange.fetch_positions([ccxt_symbol], {"productType": "USDT-FUTURES", "marginCoin": "USDT"})
                wanted_side = "long" if action in ("BUY", "LONG") else "short"
                live_qty = 0.0
                for lp in live_positions:
                    side = str(lp.get("side", "")).lower()
                    if side == wanted_side:
                        contracts = float(lp.get("contracts", lp.get("contractSize", 0.0)) or 0.0)
                        live_qty = max(live_qty, abs(contracts))

                if live_qty <= 0:
                    live_qty = float(pos.get("quantity", 0.0) or 0.0)
                
                live_qty = float(exchange.amount_to_precision(ccxt_symbol, live_qty))
                print(f"\n📊 Processing {ccxt_symbol} (Live Qty: {live_qty}):")
            except Exception as e:
                print(f"⚠️ Could not fetch live position for {ccxt_symbol}: {e}")
                live_qty = float(exchange.amount_to_precision(ccxt_symbol, float(pos.get("quantity", 0.0) or 0.0)))

            if live_qty <= 0:
                print(f"   ⏭️ Skipping {ccxt_symbol}: quantity is zero.")
                continue

            # Install Stop Loss on Bitget
            if sl_price > 0:
                try:
                    sl_params = {
                        "stopLossPrice": sl_price,
                        "oneWayMode": True,
                        "reduceOnly": True,
                        "clientOid": f"sl-{uuid.uuid4().hex[:20]}",
                    }
                    sl_order = await exchange.create_order(
                        ccxt_symbol,
                        "market",
                        close_side,
                        live_qty,
                        None,
                        sl_params,
                    )
                    print(f"   🛡️ Bitget Stop Loss installed at ${sl_price:.4f} (Order ID: {sl_order.get('id')})")
                except Exception as e:
                    print(f"   ❌ Failed to install Stop Loss for {ccxt_symbol}: {e}")

            # Install Take Profit on Bitget
            if tp_price > 0:
                try:
                    tp_params = {
                        "takeProfitPrice": tp_price,
                        "oneWayMode": True,
                        "reduceOnly": True,
                        "clientOid": f"tp-{uuid.uuid4().hex[:20]}",
                    }
                    tp_order = await exchange.create_order(
                        ccxt_symbol,
                        "market",
                        close_side,
                        live_qty,
                        None,
                        tp_params,
                    )
                    print(f"   🎯 Bitget Take Profit installed at ${tp_price:.4f} (Order ID: {tp_order.get('id')})")
                except Exception as e:
                    print(f"   ❌ Failed to install Take Profit for {ccxt_symbol}: {e}")

    finally:
        await exchange.close()
        print("\n✅ Bitget session closed.")


if __name__ == "__main__":
    asyncio.run(sync_tpsl())
