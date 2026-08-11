"""
Telegram Signal & Notification Service for SmartCrypto AI v3.0.0
Broadcasts high-conviction AI signals and trade outcome reports to Telegram VIP Channel.
"""

import httpx
import logging
import asyncio
from typing import Dict, Optional, Any
from datetime import datetime

from src.core.config import get_settings
from src.utils.safe_logger import SafeLogger

logger = SafeLogger.get_logger(__name__)


class TelegramService:
    """Async Telegram Service for VIP Signal Channels & Admin Alerts"""
    
    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.bot_token = getattr(self.settings, 'TELEGRAM_BOT_TOKEN', '')
        self.channel_id = getattr(self.settings, 'TELEGRAM_CHANNEL_ID', '')
        self.admin_chat_id = getattr(self.settings, 'TELEGRAM_ADMIN_CHAT_ID', '')
        self.enable_telegram = getattr(self.settings, 'ENABLE_TELEGRAM', False)
        
        self.api_base = f"https://api.telegram.org/bot{self.bot_token}" if self.bot_token else ""

    async def _send_message(self, chat_id: str, text: str, parse_mode: str = "HTML") -> bool:
        """Send message via Telegram Bot API with detailed error logging"""
        if not self.enable_telegram or not self.bot_token or not chat_id:
            return False

        try:
            url = f"{self.api_base}/sendMessage"
            payload = {
                "chat_id": chat_id,
                "text": text,
                "parse_mode": parse_mode,
                "disable_web_page_preview": True
            }
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(url, json=payload)
                if response.status_code == 200:
                    return True
                else:
                    logger.error(f"❌ Telegram API Error ({response.status_code}): {response.text}")
                    return False
        except Exception as e:
            logger.error(f"❌ Exception sending Telegram message: {e}")
            return False

    async def broadcast_signal(self, signal: Dict) -> bool:
        """Format and broadcast an accepted 3-Tier AI trading signal to VIP Channel"""
        if not self.enable_telegram or not self.channel_id:
            return False

        symbol = signal.get('symbol', 'BTCUSDT').replace('USDT', '')
        action = signal.get('action', 'BUY')
        price = float(signal.get('price', 0.0))
        confidence = float(signal.get('confidence', 0.5))
        strength = float(signal.get('signal_strength', 0.5))
        
        strategy = signal.get('strategy', {})
        stop_loss = float(strategy.get('stop_loss', 0.0))
        tp1 = float(strategy.get('take_profit_1', 0.0))
        tp2 = float(strategy.get('take_profit_2', 0.0))
        max_hold = strategy.get('max_holding_hours', 8)

        direction_icon = "🟢 BUY (LONG)" if action == "BUY" else "🔴 SELL (SHORT)"
        
        # Calculate percentage offsets
        sl_pct = abs((stop_loss - price) / price * 100) if price > 0 else 0
        tp1_pct = abs((tp1 - price) / price * 100) if price > 0 else 0
        tp2_pct = abs((tp2 - price) / price * 100) if price > 0 else 0

        # Extract 3-Tier AI specific predictions
        expected_returns = signal.get('expected_returns', {})
        gpt_sim = signal.get('market_gpt_simulation', {})
        exp_4h = expected_returns.get('4h_return', 'N/A')
        gpt_win_prob = gpt_sim.get('win_probability', f"{confidence:.1%}")

        # Institutional HTML-formatted message
        message = (
            f"<b>🚀 SMARTCRYPTO AI v3.0 VIP SIGNAL</b>\n\n"
            f"<b>Symbol:</b> #{symbol}USDT\n"
            f"<b>Direction:</b> {direction_icon}\n"
            f"<b>Entry Price:</b> ${price:,.4f}\n\n"
            f"<b>🎯 Take Profit Targets:</b>\n"
            f"• TP1: <code>${tp1:,.4f}</code> (+{tp1_pct:.2f}%)\n"
            f"• TP2: <code>${tp2:,.4f}</code> (+{tp2_pct:.2f}%)\n\n"
            f"<b>🛡️ Stop Loss:</b> <code>${stop_loss:,.4f}</code> (-{sl_pct:.2f}%)\n\n"
            f"<b>📊 AI 3-Tier Ensemble Analysis:</b>\n"
            f"• Expected 4H Return: <b>{exp_4h}</b>\n"
            f"• Monte Carlo Win Prob: <b>{gpt_win_prob}</b>\n"
            f"• Signal Strength: <b>{strength:.1%}</b>\n"
            f"• Timeframe Alignment: 1H={signal.get('direction_1h')} | 4H={signal.get('direction_4h')} | 1D={signal.get('direction_1d')}\n"
            f"• Risk Level: <b>{signal.get('risk_level', 'MEDIUM')}</b>\n"
            f"• Market Regime: <b>{signal.get('market_regime', 'TRENDING')}</b>\n\n"
            f"⏰ <b>Max Holding Time:</b> {max_hold} Hours\n\n"
            f"<i>⚠️ SmartCrypto AI Automated Signal • Manage your risk responsibly.</i>"
        )

        success = await self._send_message(self.channel_id, message)
        if success:
            logger.info(f"📣 Broadcasted VIP Telegram Signal for #{symbol}USDT")
        return success

    async def broadcast_trade_closed(self, position: Any) -> bool:
        """Broadcast closed trade outcome (WIN / LOSS) to VIP Channel for social proof"""
        if not self.enable_telegram or not self.channel_id:
            return False

        symbol = position.symbol.replace('USDT', '')
        pnl = float(position.pnl or 0.0)
        pnl_pct = float(position.pnl_percentage or 0.0)
        result_icon = "🎉 WIN" if pnl > 0 else "🛑 CLOSED"
        profile_display = str(getattr(position, 'profile_name', 'Day Trader') or 'Day Trader').title()
        
        message = (
            f"<b>{result_icon}: #{symbol}USDT Position Closed</b>\n\n"
            f"<b>Direction:</b> {position.action}\n"
            f"<b>Entry Price:</b> ${float(position.entry_price):,.4f}\n"
            f"<b>Exit Price:</b> ${float(position.exit_price or position.current_price):,.4f}\n"
            f"<b>Profit/Loss:</b> <code>{'+' if pnl >= 0 else ''}${pnl:,.2f} ({'+' if pnl_pct >= 0 else ''}{pnl_pct:.2f}%)</code>\n"
            f"<b>Profile:</b> {profile_display}\n\n"
            f"<i>SmartCrypto AI Automated Execution</i>"
        )

        return await self._send_message(self.channel_id, message)

    async def send_admin_alert(self, text: str) -> bool:
        """Send admin system alerts to your private chat"""
        if self.admin_chat_id:
            return await self._send_message(self.admin_chat_id, f"<b>🔔 ADMIN ALERT:</b>\n{text}")
        return False