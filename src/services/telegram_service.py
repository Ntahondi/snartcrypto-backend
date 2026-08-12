"""
Telegram Signal & Notification Service for SnartCrypto AI v3.0.0
Broadcasts VIP Signals, Trade Outcomes, Daily Market Intelligence Briefings,
Weekly Performance Audits, and handles Interactive Subscriber Commands (/predict).
"""

import httpx
import logging
import asyncio
from typing import Dict, Optional, List, Any
from datetime import datetime

from src.core.config import get_settings
from src.utils.safe_logger import SafeLogger

logger = SafeLogger.get_logger(__name__)


class TelegramService:
    """Async Telegram Service for VIP Channels, Admin Alerts, and Interactive Commands"""
    
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

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 1. VIP CHANNEL SIGNAL BROADCASTING
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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
        
        sl_pct = abs((stop_loss - price) / price * 100) if price > 0 else 0
        tp1_pct = abs((tp1 - price) / price * 100) if price > 0 else 0
        tp2_pct = abs((tp2 - price) / price * 100) if price > 0 else 0

        expected_returns = signal.get('expected_returns', {})
        gpt_sim = signal.get('market_gpt_simulation', {})
        exp_4h = expected_returns.get('4h_return', 'N/A')
        gpt_win_prob = gpt_sim.get('win_probability', f"{confidence:.1%}")

        message = (
            f"<b>🚀 SNARTCRYPTO VIP SIGNAL</b>\n\n"
            f"<b>📌 ASSET</b>\n"
            f"├ <b>Symbol:</b> #{symbol}USDT\n"
            f"├ <b>Direction:</b> {direction_icon}\n"
            f"└ <b>Entry:</b> ${price:,.4f}\n\n"
            f"<b>🎯 TAKE PROFIT TARGETS</b>\n"
            f"├ TP1: ${tp1:,.4f}  (+{tp1_pct:.2f}%)\n"
            f"└ TP2: ${tp2:,.4f}  (+{tp2_pct:.2f}%)\n\n"
            f"<b>🛡️ STOP LOSS</b>\n"
            f"└ ${stop_loss:,.4f}  (-{sl_pct:.2f}%)\n\n"
            f"<b>📊 AI ENSEMBLE ANALYSIS</b>\n"
            f"├ 4H Expected Return: {exp_4h}\n"
            f"├ Win Probability: {gpt_win_prob}\n"
            f"├ Signal Strength: {strength:.1%}\n"
            f"├ Timeframes: 1H={signal.get('direction_1h')} • 4H={signal.get('direction_4h')} • 1D={signal.get('direction_1d')}\n"
            f"├ Risk Level: {signal.get('risk_level', 'MEDIUM')}\n"
            f"└ Market Regime: {signal.get('market_regime', 'TRENDING')}\n\n"
            f"<b>⏰ MAX HOLDING TIME</b>\n"
            f"└ {max_hold} Hours\n\n"
            f"<i>⚠️ SnartCrypto Automated Signal • Manage your risk responsibly.</i>\n\n"
            f"<pre>━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📋 COPY SIGNAL DETAILS\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"SYMBOL: #{symbol}USDT\n"
            f"DIRECTION: {direction_icon}\n"
            f"ENTRY: ${price:,.4f}\n"
            f"TP1: ${tp1:,.4f} (+{tp1_pct:.2f}%)\n"
            f"TP2: ${tp2:,.4f} (+{tp2_pct:.2f}%)\n"
            f"SL: ${stop_loss:,.4f} (-{sl_pct:.2f}%)\n"
            f"RISK: {signal.get('risk_level', 'MEDIUM')}\n"
            f"REGIME: {signal.get('market_regime', 'TRENDING')}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━</pre>"
        )

        success = await self._send_message(self.channel_id, message)
        if success:
            logger.info(f"📣 Broadcasted VIP Telegram Signal for #{symbol}USDT")
        return success

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 2. CLOSED POSITION WIN / LOSS BANNERS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def broadcast_trade_closed(self, position: Any) -> bool:
        """Broadcast closed trade outcome (WIN / LOSS) to VIP Channel for social proof"""
        if not self.enable_telegram or not self.channel_id:
            return False

        symbol = position.symbol.replace('USDT', '')
        pnl = float(position.pnl or 0.0)
        pnl_pct = float(position.pnl_percentage or 0.0)
        result_icon = "🎉 WIN ALERT" if pnl > 0 else "🛑 POSITION CLOSED"
        profile_display = str(getattr(position, 'profile_name', 'Day Trader') or 'Day Trader').title()
        
        message = (
            f"<b>{result_icon}: #{symbol}USDT</b>\n\n"
            f"<b>Direction:</b> {position.action}\n"
            f"<b>Entry Price:</b> ${float(position.entry_price):,.4f}\n"
            f"<b>Exit Price:</b> ${float(position.exit_price or position.current_price):,.4f}\n"
            f"<b>Profit/Loss:</b> <code>{'+' if pnl >= 0 else ''}${pnl:,.2f} ({'+' if pnl_pct >= 0 else ''}{pnl_pct:.2f}%)</code>\n"
            f"<b>Profile:</b> {profile_display}\n\n"
            f"<i>SnartCrypto AI Automated Execution</i>"
        )

        return await self._send_message(self.channel_id, message)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 3. DAILY MORNING MARKET BRIEFING
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def broadcast_daily_briefing(self, market_summary: Dict) -> bool:
        """Broadcast morning market intelligence briefing to VIP channel"""
        if not self.enable_telegram or not self.channel_id:
            return False

        date_str = datetime.utcnow().strftime("%Y-%m-%d")
        regime = market_summary.get('market_regime', 'TRENDING')
        top_coins = market_summary.get('top_momentum_symbols', ['BTC', 'ETH', 'SOL'])

        message = (
            f"<b>🌅 SNARTCRYPTO AI DAILY MARKET BRIEFING ({date_str})</b>\n\n"
            f"<b>Market Regime:</b> <b>{regime}</b>\n"
            f"<b>Top AI Momentum Assets:</b> #{' #'.join(top_coins)}\n\n"
            f"<b>📊 Market Overview:</b>\n"
            f"• 3-Tier AI Ensemble is actively scanning 1H candle closes 24/7.\n"
            f"• High-conviction threshold: <b>±0.7% expected return</b> required.\n"
            f"• Monte Carlo simulation filter: <b>≥55% win probability</b>.\n\n"
            f"<i>Stay tuned for automated real-time VIP trade signals!</i>"
        )

        return await self._send_message(self.channel_id, message)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 4. WEEKLY SUNDAY PERFORMANCE AUDIT
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def broadcast_weekly_performance_summary(self, summary: Dict) -> bool:
        """Broadcast transparent weekly performance recap to VIP channel"""
        if not self.enable_telegram or not self.channel_id:
            return False

        total_signals = summary.get('total_signals', 0)
        closed_signals = summary.get('closed_signals', 0)
        win_rate = summary.get('overall_win_rate', 0.0)
        total_pnl = summary.get('total_pnl', 0.0)

        message = (
            f"<b>📊 SNARTCRYPTO AI WEEKLY PERFORMANCE AUDIT</b>\n\n"
            f"• Total Signals Evaluated: <b>{total_signals:,}</b>\n"
            f"• Total Closed Trades: <b>{closed_signals:,}</b>\n"
            f"• <b>Weekly Win Rate:</b> <code>{win_rate:.1%}</code>\n"
            f"• <b>Total Net PnL:</b> <code>{'+' if total_pnl >= 0 else ''}{total_pnl:.2f}%</code>\n\n"
            f"<i>Verified by SnartCrypto AI Database • 100% Transparent Trading.</i>"
        )

        return await self._send_message(self.channel_id, message)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 5. ON-DEMAND COMMAND RESPONSES (/predict)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def handle_predict_response(self, user_chat_id: str, symbol: str, signal: Optional[Dict]) -> bool:
        """Format and send instant /predict response to user direct message"""
        symbol_clean = symbol.replace('USDT', '')
        
        if not signal:
            text = f"ℹ️ <b>#{symbol_clean}USDT</b> is currently in a low-volatility sideways range. No high-conviction trade setup."
        else:
            action = signal.get('action', 'HOLD')
            exp_ret = signal.get('expected_returns', {}).get('4h_return', '0.0%')
            gpt_win_prob = signal.get('market_gpt_simulation', {}).get('win_probability', '50.0%')
            regime = signal.get('market_regime', 'TRENDING')
            risk = signal.get('risk_level', 'MEDIUM')

            text = (
                f"<b>📊 INSTANT AI FORECAST: #{symbol_clean}USDT</b>\n\n"
                f"• AI Action: <b>{action}</b>\n"
                f"• Expected 4H Return: <b>{exp_ret}</b>\n"
                f"• Monte Carlo Win Prob: <b>{gpt_win_prob}</b>\n"
                f"• Market Regime: <b>{regime}</b>\n"
                f"• Risk Level: <b>{risk}</b>\n"
                f"• Entry Price: <code>${signal.get('price', 0.0):,.4f}</code>"
            )

        return await self._send_message(user_chat_id, text)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 6. ADMIN SYSTEM ALERTS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def send_admin_alert(self, text: str) -> bool:
        """Send admin system alerts to your private chat"""
        if self.admin_chat_id:
            return await self._send_message(self.admin_chat_id, f"<b>🔔 ADMIN ALERT:</b>\n{text}")
        return False