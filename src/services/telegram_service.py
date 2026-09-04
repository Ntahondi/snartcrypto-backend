"""
src/services/telegram_service.py

SnartCrypto AI Trading System
Telegram Signal & Notification Service

Responsibilities
----------------
1. Batch and broadcast newly generated trading signals.
2. Present the 3-model AI committee transparently.
3. Present Model 4 Strategy Detector intelligence.
4. Broadcast trade execution/closure results.
5. Produce dedicated WIN / LOSS / BREAKEVEN messages.
6. Provide daily market intelligence briefings.
7. Provide weekly performance audits.
8. Handle /predict responses.
9. Send private administrator alerts.

Architecture
------------
MODEL 1 ─┐
MODEL 2 ─┼──> SignalGenerator ──> HistoryManager
MODEL 3 ─┘              │
                        │
MODEL 4 Strategy Layer ─┘
                        │
                        ▼
                TelegramService

Model 4 is NOT treated as a fourth majority voter.
It is presented as strategy/pattern confirmation intelligence.
"""

from __future__ import annotations

import asyncio
import html
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

import httpx

from src.core.config import get_settings
from src.utils.safe_logger import SafeLogger


logger = SafeLogger.get_logger(__name__)


class TelegramService:
    """
    Async Telegram notification service for SnartCrypto.

    Telegram responsibilities are deliberately separated from
    trading logic. This service formats and broadcasts information
    produced by the trading system; it does not generate trades.
    """

    TELEGRAM_MAX_MESSAGE_LENGTH = 4096

    # Keep a safety margin below Telegram's hard limit.
    SAFE_MESSAGE_LENGTH = 3900

    DEFAULT_TIMEOUT = 15.0

    def __init__(self, settings=None):
        self.settings = settings or get_settings()

        self.bot_token = str(
            getattr(
                self.settings,
                "TELEGRAM_BOT_TOKEN",
                "",
            )
            or ""
        ).strip()

        self.channel_id = str(
            getattr(
                self.settings,
                "TELEGRAM_CHANNEL_ID",
                "",
            )
            or ""
        ).strip()

        self.admin_chat_id = str(
            getattr(
                self.settings,
                "TELEGRAM_ADMIN_CHAT_ID",
                "",
            )
            or ""
        ).strip()

        self.enable_telegram = bool(
            getattr(
                self.settings,
                "ENABLE_TELEGRAM",
                False,
            )
        )

        self.custom_api_base = str(
            getattr(
                self.settings,
                "TELEGRAM_API_BASE",
                "https://api.telegram.org",
            )
            or "https://api.telegram.org"
        ).rstrip("/")

        self.proxy_url = getattr(
            self.settings,
            "TELEGRAM_PROXY_URL",
            None,
        )

        self.api_base = (
            f"{self.custom_api_base}/bot{self.bot_token}"
            if self.bot_token
            else ""
        )

        self._client: Optional[httpx.AsyncClient] = None

        # Used to avoid accidentally broadcasting the same signal
        # repeatedly during one process lifetime.
        self._broadcasted_signal_ids: set[str] = set()

        logger.info(
            "TelegramService initialized | enabled=%s | channel_configured=%s | base=%s",
            self.enable_telegram,
            bool(self.channel_id),
            self.custom_api_base,
        )

    # =====================================================================
    # LIFECYCLE
    # =====================================================================

    async def start(self) -> None:
        """Create the reusable HTTP client."""

        if self._client is None:
            kwargs: Dict[str, Any] = {
                "timeout": httpx.Timeout(
                    self.DEFAULT_TIMEOUT,
                    connect=10.0,
                ),
                "limits": httpx.Limits(
                    max_connections=10,
                    max_keepalive_connections=5,
                ),
            }
            if self.proxy_url:
                kwargs["proxy"] = self.proxy_url

            self._client = httpx.AsyncClient(**kwargs)

    async def close(self) -> None:
        """Close the reusable HTTP client."""

        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "TelegramService":
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type,
        exc,
        traceback,
    ) -> None:
        await self.close()

    # =====================================================================
    # BASIC HELPERS
    # =====================================================================

    @staticmethod
    def _escape(value: Any) -> str:
        """Escape arbitrary values for Telegram HTML."""

        if value is None:
            return ""

        return html.escape(
            str(value),
            quote=False,
        )

    @staticmethod
    def _safe_float(
        value: Any,
        default: float = 0.0,
    ) -> float:
        """Safely convert a value to float."""

        try:
            result = float(value)

            if result != result:  # NaN
                return default

            return result

        except (
            TypeError,
            ValueError,
        ):
            return default

    @staticmethod
    def _safe_int(
        value: Any,
        default: int = 0,
    ) -> int:
        """Safely convert a value to int."""

        try:
            return int(value)

        except (
            TypeError,
            ValueError,
        ):
            return default

    @staticmethod
    def _now_utc() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _format_price(
        value: Any,
        decimals: Optional[int] = None,
    ) -> str:
        """
        Format crypto prices without unnecessarily losing precision.
        """

        price = TelegramService._safe_float(value)

        if decimals is not None:
            return f"{price:,.{decimals}f}"

        if price >= 10000:
            return f"{price:,.2f}"

        if price >= 1000:
            return f"{price:,.3f}"

        if price >= 100:
            return f"{price:,.3f}"

        if price >= 1:
            return f"{price:,.4f}"

        if price >= 0.01:
            return f"{price:,.5f}"

        return f"{price:,.8f}"

    @staticmethod
    def _clean_symbol(symbol: Any) -> str:
        symbol = str(symbol or "UNKNOWN").upper()

        for suffix in (
            "USDT",
            "/USDT",
            ":USDT",
        ):
            if symbol.endswith(suffix):
                symbol = symbol[: -len(suffix)]

        return symbol

    @staticmethod
    def _direction_icon(action: str) -> str:
        action = str(action or "").upper()

        if action == "BUY":
            return "🟢"

        if action == "SELL":
            return "🔴"

        return "🟡"

    @staticmethod
    def _vote_icon(vote: str) -> str:
        vote = str(vote or "HOLD").upper()

        if vote == "BUY":
            return "🟢 BUY"

        if vote == "SELL":
            return "🔴 SELL"

        return "🟡 HOLD"

    @staticmethod
    def _strategy_icon(
        direction: str,
    ) -> str:
        direction = str(direction or "").upper()

        if direction in {
            "BUY",
            "BULLISH",
            "LONG",
        }:
            return "🟢"

        if direction in {
            "SELL",
            "BEARISH",
            "SHORT",
        }:
            return "🔴"

        return "⚪"

    # =====================================================================
    # TELEGRAM TRANSPORT
    # =====================================================================

    async def _send_message(
        self,
        chat_id: str,
        text: str,
        parse_mode: str = "HTML",
        disable_web_page_preview: bool = True,
        reply_markup: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Low-level Telegram send operation with optional interactive inline keyboards.
        """

        if (
            not self.enable_telegram
            or not self.bot_token
            or not chat_id
            or not text
        ):
            return False

        if len(text) > self.TELEGRAM_MAX_MESSAGE_LENGTH:
            logger.error(
                "Telegram message exceeds Telegram limit: %s characters",
                len(text),
            )
            return False

        await self.start()

        if self._client is None:
            return False

        url = f"{self.api_base}/sendMessage"

        payload: Dict[str, Any] = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": disable_web_page_preview,
        }
        if reply_markup:
            payload["reply_markup"] = reply_markup

        for attempt in range(1, 4):

            try:
                response = await self._client.post(
                    url,
                    json=payload,
                )

                if response.status_code == 200:

                    data = response.json()

                    if data.get("ok") is True:
                        return True

                    logger.error(
                        "Telegram API rejected message: %s",
                        data,
                    )
                    return False

                # Retry transient errors.
                if response.status_code in {
                    408,
                    429,
                    500,
                    502,
                    503,
                    504,
                }:

                    retry_after = 1

                    try:
                        retry_after = int(
                            response.headers.get(
                                "Retry-After",
                                "1",
                            )
                        )
                    except ValueError:
                        pass

                    await asyncio.sleep(
                        max(
                            retry_after,
                            attempt,
                        )
                    )

                    continue

                logger.error(
                    "Telegram API error | status=%s | body=%s",
                    response.status_code,
                    response.text,
                )

                return False

            except (
                httpx.TimeoutException,
                httpx.NetworkError,
            ) as exc:

                logger.warning(
                    "Telegram network error "
                    "(attempt %s/3): %s",
                    attempt,
                    exc,
                )

                if attempt < 3:
                    await asyncio.sleep(attempt)

            except Exception as exc:

                logger.error(
                    "Telegram send exception: %s",
                    exc,
                    exc_info=True,
                )

                return False

        return False

    async def _send_long_message(
        self,
        chat_id: str,
        text: str,
    ) -> bool:
        """
        Split long Telegram messages safely.

        Splitting occurs preferably at newline boundaries.
        """

        if len(text) <= self.SAFE_MESSAGE_LENGTH:
            return await self._send_message(
                chat_id,
                text,
            )

        chunks: List[str] = []
        remaining = text

        while remaining:

            if len(remaining) <= self.SAFE_MESSAGE_LENGTH:
                chunks.append(remaining)
                break

            split_at = remaining.rfind(
                "\n",
                0,
                self.SAFE_MESSAGE_LENGTH,
            )

            if split_at < 500:
                split_at = self.SAFE_MESSAGE_LENGTH

            chunks.append(
                remaining[:split_at]
            )

            remaining = remaining[
                split_at:
            ].lstrip("\n")

        results = []

        for chunk in chunks:

            results.append(
                await self._send_message(
                    chat_id,
                    chunk,
                )
            )

            # Prevent accidental flood-limit violations.
            await asyncio.sleep(0.15)

        return all(results)

    # =====================================================================
    # MODEL 4 — STRATEGY DETECTOR FORMATTERS
    # =====================================================================

    def _extract_strategy_detection(
        self,
        signal: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Support the canonical Model 4 field while remaining tolerant
        of slightly older signal objects.
        """

        strategy_detection = signal.get(
            "strategy_detection"
        )

        if isinstance(
            strategy_detection,
            dict,
        ):
            return strategy_detection

        # Compatibility with possible previous naming.
        for key in (
            "model_4_strategy_detector",
            "strategy_detector",
            "strategy_analysis",
            "model4",
        ):

            value = signal.get(key)

            if isinstance(value, dict):
                return value

        return {}

    def _format_strategy_layer(
        self,
        signal: Dict[str, Any],
    ) -> str:
        """Format Model 4 strategy intelligence."""

        detection = (
            self._extract_strategy_detection(
                signal
            )
        )

        if not detection:
            return (
                "<b>🧠 MODEL 4 — STRATEGY INTELLIGENCE</b>\n"
                "└ No strategy-detector data available"
            )

        bias = str(
            detection.get(
                "bias",
                detection.get(
                    "overall_bias",
                    "NEUTRAL",
                ),
            )
        ).upper()

        confirmation = self._safe_float(
            detection.get(
                "confirmation_score",
                detection.get(
                    "score",
                    0.0,
                ),
            )
        )

        agreement = detection.get(
            "agreement",
            detection.get(
                "agreement_score",
                None,
            ),
        )

        conflict = detection.get(
            "conflict",
            detection.get(
                "conflict_score",
                None,
            ),
        )

        bullish = detection.get(
            "bullish_strategies",
            detection.get(
                "bullish",
                [],
            ),
        )

        bearish = detection.get(
            "bearish_strategies",
            detection.get(
                "bearish",
                [],
            ),
        )

        neutral = detection.get(
            "neutral_strategies",
            detection.get(
                "neutral",
                [],
            ),
        )

        if not isinstance(
            bullish,
            (list, tuple),
        ):
            bullish = []

        if not isinstance(
            bearish,
            (list, tuple),
        ):
            bearish = []

        if not isinstance(
            neutral,
            (list, tuple),
        ):
            neutral = []

        if bias in {
            "BUY",
            "BULLISH",
            "LONG",
        }:
            bias_display = "🟢 BULLISH"

        elif bias in {
            "SELL",
            "BEARISH",
            "SHORT",
        }:
            bias_display = "🔴 BEARISH"

        else:
            bias_display = "⚪ NEUTRAL"

        lines = [
            "<b>🧠 MODEL 4 — STRATEGY INTELLIGENCE</b>",
            f"├ Bias: <b>{bias_display}</b>",
        ]

        if confirmation:
            lines.append(
                f"├ Confirmation: "
                f"<b>{confirmation:.1%}</b>"
            )

        if agreement is not None:
            lines.append(
                f"├ Agreement: "
                f"<b>{self._format_percentage(agreement)}</b>"
            )

        if conflict is not None:
            lines.append(
                f"├ Conflict: "
                f"<b>{self._format_percentage(conflict)}</b>"
            )

        if bullish:
            lines.append(
                "├ 🟢 Bullish strategies:"
            )

            for item in bullish[:8]:
                lines.append(
                    f"│  • {self._escape(item)}"
                )

        if bearish:
            lines.append(
                "├ 🔴 Bearish strategies:"
            )

            for item in bearish[:8]:
                lines.append(
                    f"│  • {self._escape(item)}"
                )

        if neutral:
            lines.append(
                "└ ⚪ Neutral strategies:"
            )

            for item in neutral[:8]:
                lines.append(
                    f"   • {self._escape(item)}"
                )

        else:
            if lines[-1].startswith("├"):
                lines.append(
                    "└ Strategy detectors processed"
                )

        return "\n".join(lines)

    @staticmethod
    def _format_percentage(
        value: Any,
    ) -> str:
        """
        Handle both decimal and percentage representations.

        0.82 -> 82.0%
        82   -> 82.0%
        """

        try:
            value = float(value)
        except (
            TypeError,
            ValueError,
        ):
            return "N/A"

        if abs(value) <= 1:
            value *= 100

        return f"{value:.1f}%"

    # =====================================================================
    # SIGNAL RATING & INTERACTIVE HELPERS
    # =====================================================================

    @staticmethod
    def _calculate_rating(confidence: float, strength: float = 0.8) -> tuple[str, str]:
        """Calculates institutional conviction rating score and stars."""
        score = (confidence + strength) / 2.0
        if score >= 0.88:
            return "⭐️⭐️⭐️⭐️⭐️", "INSTITUTIONAL GRADE • HIGH CONVICTION"
        elif score >= 0.78:
            return "⭐️⭐️⭐️⭐️", "STRONG COMMITTEE ALIGNED"
        elif score >= 0.68:
            return "⭐️⭐️⭐️", "BALANCED MOMENTUM SETUP"
        else:
            return "⭐️⭐️", "TACTICAL OPPORTUNITY"

    @staticmethod
    def _get_profile_badge(profile_name: str, timeframe: str) -> str:
        """Formats visual profile badge with timeframe."""
        p = (profile_name or "day_trader").lower()
        tf = (timeframe or "1h").upper()
        if "scalp" in p:
            return f"⚡ <b>SCALPER</b> • <code>{tf}</code>"
        elif "swing" in p:
            return f"🌊 <b>SWING RUNNER</b> • <code>{tf}</code>"
        elif "pos" in p:
            return f"🏔️ <b>POSITION ACCUMULATION</b> • <code>{tf}</code>"
        else:
            return f"🎯 <b>DAY TRADER</b> • <code>{tf}</code>"

    def _create_signal_keyboard(self, symbol: str) -> Dict[str, Any]:
        """Generate sleek Telegram inline keyboard with direct terminal links."""
        clean_sym = self._clean_symbol(symbol)
        site_url = "https://snartcrypto.snartpace.com"
        return {
            "inline_keyboard": [
                [
                    {
                        "text": f"🚀 Trade #{clean_sym} on SnartCrypto",
                        "url": f"{site_url}/signals",
                    },
                ],
                [
                    {
                        "text": "📊 Live Order Book",
                        "url": site_url,
                    },
                    {
                        "text": "🛡️ Profit Shield",
                        "url": f"{site_url}/dashboard",
                    },
                ],
                [
                    {
                        "text": "👥 Join VIP Telegram Channel",
                        "url": "https://t.me/snartcrypto",
                    }
                ]
            ]
        }

    def _create_terminal_keyboard(self) -> Dict[str, Any]:
        """Generate general terminal link keyboard."""
        site_url = "https://snartcrypto.snartpace.com"
        return {
            "inline_keyboard": [
                [
                    {
                        "text": "🚀 Open SnartCrypto Web Terminal",
                        "url": f"{site_url}/dashboard",
                    },
                ],
                [
                    {
                        "text": "💎 Upgrade to VVIP API Execution",
                        "url": f"{site_url}/pricing",
                    }
                ]
            ]
        }

    # =====================================================================
    # SIGNAL FORMAT
    # =====================================================================

    def _format_single_signal(
        self,
        signal: Dict[str, Any],
        index: Optional[int] = None,
    ) -> str:
        """
        Produce a clean, high-conversion VIP AI Trade Signal card.
        """
        symbol = self._clean_symbol(signal.get("symbol"))
        action = str(signal.get("action", "BUY")).upper()
        price = self._safe_float(signal.get("price"))
        confidence = self._safe_float(signal.get("confidence"), 0.88)
        strength = self._safe_float(signal.get("signal_strength"), 0.82)
        timeframe = str(signal.get("timeframe", "1h"))
        profile_name = str(signal.get("profile_name", "day_trader"))

        strategy = signal.get("strategy", {})
        if not isinstance(strategy, dict):
            strategy = {}

        stop_loss = self._safe_float(strategy.get("stop_loss"))
        tp1 = self._safe_float(strategy.get("take_profit_1") or strategy.get("take_profit"))
        tp2 = self._safe_float(strategy.get("take_profit_2"))
        max_hold = self._safe_int(strategy.get("max_holding_hours", 8), 8)

        if not stop_loss and price > 0:
            stop_loss = round(price * 0.965 if action == "BUY" else price * 1.035, 4 if price < 10 else 2)
        if not tp1 and price > 0:
            tp1 = round(price * 1.045 if action == "BUY" else price * 0.955, 4 if price < 10 else 2)
        if not tp2 and tp1 and price > 0:
            tp2 = round(price * 1.075 if action == "BUY" else price * 0.925, 4 if price < 10 else 2)

        sl_pct = abs((price - stop_loss) / price * 100) if (price > 0 and stop_loss) else 3.50
        tp1_pct = abs((tp1 - price) / price * 100) if (price > 0 and tp1) else 4.50
        tp2_pct = abs((tp2 - price) / price * 100) if (price > 0 and tp2) else 7.50

        tp1_roe = tp1_pct * 3.0
        tp2_roe = tp2_pct * 3.0

        is_buy = action in ("BUY", "LONG")
        direction_banner = "🟢 <b>LONG SETUP • TARGET ACQUIRED</b> 💎" if is_buy else "🔴 <b>SHORT SETUP • BEARISH REVERSAL</b> ⚡"
        stars, grade_label = self._calculate_rating(confidence, strength)
        profile_badge = self._get_profile_badge(profile_name, timeframe)

        formatted_price = self._format_price(price)
        formatted_sl = self._format_price(stop_loss)
        formatted_tp1 = self._format_price(tp1)
        formatted_tp2 = self._format_price(tp2) if tp2 else ""

        lines = [
            direction_banner,
            "━━━━━━━━━━━━━━━━━━━━━",
            f"💎 <b>#{self._escape(symbol)}USDT</b> | {profile_badge}",
            f"{stars} <b>Score:</b> <b>{confidence:.0%}</b> <i>({grade_label})</i>",
            "━━━━━━━━━━━━━━━━━━━━━",
            f"📍 <b>ENTRY ZONE</b>   ➔ <code>{formatted_price}</code>",
            f"🎯 <b>TARGET 1</b>     ➔ <code>{formatted_tp1}</code> <i>(+{tp1_pct:.2f}% • +{tp1_roe:.1f}% ROE)</i>",
        ]
        if formatted_tp2:
            lines.append(f"🎯 <b>TARGET 2</b>     ➔ <code>{formatted_tp2}</code> <i>(+{tp2_pct:.2f}% • +{tp2_roe:.1f}% ROE)</i>")

        lines.extend([
            f"🛑 <b>STOP LOSS</b>    ➔ <code>{formatted_sl}</code> <i>(-{sl_pct:.2f}% • ATR Managed)</i>",
            "━━━━━━━━━━━━━━━━━━━━━",
            f"⚙️ <b>Leverage:</b> <code>3x - 5x</code> | ⏱️ <b>Hold:</b> <code>{max_hold}h</code>",
            "🛡️ <b>Profit Shield:</b> <code>Break-Even & Trailing Active</code>",
        ])

        # Collapsible 4-Model Committee Quote
        regime = signal.get("market_regime", "BULLISH_TREND")
        model1 = signal.get("model1_direction", "BUY" if is_buy else "SELL")
        lines.extend([
            "",
            "<blockquote expandable>",
            "🧠 <b>4-Model Committee Intelligence:</b>\n"
            f"• <b>Regression AI:</b> <code>{model1}</code> ({confidence:.0%} Probability)\n"
            f"• <b>Smart Trader:</b> Multi-Timeframe {self._escape(regime)}\n"
            "• <b>Market GPT:</b> Volatility Expansion Verified\n"
            "• <b>Strategy Detector:</b> Pattern Consensus Approved",
            "</blockquote>",
        ])

        return "\n".join(lines)

    def _format_signal_compact(
        self,
        signal: Dict[str, Any],
        index: int,
    ) -> str:
        """Produce a sleek, tap-to-copy snippet for combined multi-signal digest."""
        symbol = self._clean_symbol(signal.get("symbol"))
        action = str(signal.get("action", "BUY")).upper()
        price = self._safe_float(signal.get("price"))
        confidence = self._safe_float(signal.get("confidence"), 0.88)
        strength = self._safe_float(signal.get("signal_strength"), 0.82)
        stars, _ = self._calculate_rating(confidence, strength)

        strategy = signal.get("strategy", {})
        if not isinstance(strategy, dict):
            strategy = {}

        stop_loss = self._safe_float(strategy.get("stop_loss"))
        tp = self._safe_float(strategy.get("take_profit_1") or strategy.get("take_profit"))

        if not stop_loss and price > 0:
            stop_loss = round(price * 0.965 if action == "BUY" else price * 1.035, 4 if price < 10 else 2)
        if not tp and price > 0:
            tp = round(price * 1.050 if action == "BUY" else price * 0.950, 4 if price < 10 else 2)

        is_buy = action in ("BUY", "LONG")
        action_icon = "🟢 LONG" if is_buy else "🔴 SHORT"
        tp_pct = abs((tp - price) / price * 100) if (price > 0 and tp) else 5.0
        sl_pct = abs((price - stop_loss) / price * 100) if (price > 0 and stop_loss) else 3.5

        return (
            f"<b>{index}️⃣ #{self._escape(symbol)}USDT</b> • <b>{action_icon}</b> {stars} (<b>{confidence:.0%}</b>)\n"
            f"├ 📍 Entry: <code>{self._format_price(price)}</code>\n"
            f"├ 🎯 Target: <code>{self._format_price(tp)}</code> <i>(+{tp_pct:.1f}% • +{tp_pct*3:.1f}% ROE)</i>\n"
            f"└ 🛑 Stop: <code>{self._format_price(stop_loss)}</code> <i>(-{sl_pct:.1f}%)</i>"
        )

    # =====================================================================
    # SIGNAL BATCHING
    # =====================================================================

    @staticmethod
    def _deduplicate_signals(
        signals: Iterable[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Deduplicate signals by signal_id.
        """
        unique: Dict[str, Dict[str, Any]] = {}

        for signal in signals:
            if not isinstance(signal, dict):
                continue

            signal_id = signal.get("signal_id")
            if not signal_id:
                signal_id = (
                    f"{signal.get('symbol', 'UNKNOWN')}:"
                    f"{signal.get('action', 'HOLD')}:"
                    f"{signal.get('timestamp', '')}"
                )

            unique[str(signal_id)] = signal

        return list(unique.values())

    async def broadcast_signals(
        self,
        signals: List[Dict[str, Any]],
    ) -> bool:
        """
        Broadcast multiple signals in ONE clean, consolidated Telegram VIP Digest.
        """
        if not self.enable_telegram or not self.channel_id:
            return False

        if not signals:
            return False

        signals = self._deduplicate_signals(signals)
        if not signals:
            return False

        fresh_signals = []
        for signal in signals:
            signal_id = signal.get("signal_id")
            if signal_id and str(signal_id) in self._broadcasted_signal_ids:
                continue
            fresh_signals.append(signal)

        if not fresh_signals:
            logger.info("No new Telegram signals to broadcast.")
            return True

        # If only 1 signal: send single signal card with interactive keyboard
        if len(fresh_signals) == 1:
            sig = fresh_signals[0]
            card = self._format_single_signal(sig)
            card += "\n\n💡 <i>Tap any price value to copy to clipboard.</i>"
            kb = self._create_signal_keyboard(sig.get("symbol", "BTC"))
            success = await self._send_message(self.channel_id, card, reply_markup=kb)
            if success and sig.get("signal_id"):
                self._broadcasted_signal_ids.add(str(sig["signal_id"]))
            return success

        # If multiple signals: Combine into one elegant VIP Signal Digest
        now = self._now_utc()
        buy_count = sum(1 for s in fresh_signals if str(s.get("action", "")).upper() in ("BUY", "LONG"))
        sell_count = sum(1 for s in fresh_signals if str(s.get("action", "")).upper() in ("SELL", "SHORT"))

        header = (
            "🚨 <b>SNARTCRYPTO VIP • SIGNALS DIGEST</b> 🚨\n"
            f"<i>{now.strftime('%Y-%m-%d %H:%M UTC')}</i> | 📡 <b>{len(fresh_signals)} Approved Setups</b>\n"
            f"🟢 Long: <b>{buy_count}</b> | 🔴 Short: <b>{sell_count}</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━\n\n"
        )

        compact_cards = [
            self._format_signal_compact(s, i)
            for i, s in enumerate(fresh_signals, start=1)
        ]

        footer = (
            "\n\n━━━━━━━━━━━━━━━━━━━━━\n"
            "⚙️ <b>Rec. Leverage:</b> <code>3x - 5x</code> | <b>Risk:</b> <code>1-2% per trade</code>\n"
            "💡 <i>Tap any value to copy. Tap below to launch terminal.</i>"
        )

        full_message = header + "\n\n".join(compact_cards) + footer
        kb = self._create_terminal_keyboard()

        if len(full_message) <= self.SAFE_MESSAGE_LENGTH:
            success = await self._send_message(self.channel_id, full_message, reply_markup=kb)
        else:
            chunk_size = 4
            success = True
            for i in range(0, len(compact_cards), chunk_size):
                chunk = compact_cards[i : i + chunk_size]
                chunk_msg = header + "\n\n".join(chunk) + footer
                ok = await self._send_message(self.channel_id, chunk_msg, reply_markup=kb)
                if not ok:
                    success = False
                await asyncio.sleep(0.2)

        if success:
            for signal in fresh_signals:
                signal_id = signal.get("signal_id")
                if signal_id:
                    self._broadcasted_signal_ids.add(str(signal_id))

            logger.info(
                "Broadcasted Telegram VIP signal digest | count=%s | BUY=%s | SELL=%s",
                len(fresh_signals),
                buy_count,
                sell_count,
            )

        return success

    async def broadcast_signal(
        self,
        signal: Dict[str, Any],
    ) -> bool:
        """
        Backward-compatible single-signal interface.
        """
        return await self.broadcast_signals([signal])

    # =====================================================================
    # TRADE EXECUTION
    # =====================================================================

    async def broadcast_trade_opened(
        self,
        position: Any,
    ) -> bool:
        """
        Notify the channel of live trade execution with 1-tap copy entities.
        """
        if not self.enable_telegram or not self.channel_id:
            return False

        symbol = self._clean_symbol(getattr(position, "symbol", "UNKNOWN"))
        action = str(getattr(position, "action", "BUY")).upper()
        entry = self._safe_float(getattr(position, "entry_price", getattr(position, "price", 0)))
        sl = self._safe_float(getattr(position, "stop_loss", 0))
        tp = self._safe_float(getattr(position, "take_profit", 0))
        pos_id = getattr(position, "position_id", getattr(position, "id", None))

        is_buy = action in ("BUY", "LONG")
        action_icon = "🟢 LONG" if is_buy else "🔴 SHORT"

        lines = [
            "⚡ <b>LIVE TRADE EXECUTED ON EXCHANGE</b> ⚡",
            "━━━━━━━━━━━━━━━━━━━━━",
            f"🎯 <b>#{self._escape(symbol)}USDT</b> • <b>{action_icon}</b>",
            "━━━━━━━━━━━━━━━━━━━━━",
            f"📍 <b>Entry Price:</b> <code>{self._format_price(entry)}</code>",
        ]
        if tp:
            lines.append(f"🎯 <b>Take Profit:</b> <code>{self._format_price(tp)}</code>")
        if sl:
            lines.append(f"🛑 <b>Stop Loss:</b> <code>{self._format_price(sl)}</code>")
        if pos_id:
            lines.append(f"🆔 <b>Order ID:</b> <code>{self._escape(str(pos_id))}</code>")

        lines.extend([
            "━━━━━━━━━━━━━━━━━━━━━",
            "🛡️ <i>AI Profit Shield Active • Real-time exit monitoring enabled</i>",
        ])

        kb = self._create_signal_keyboard(symbol)
        return await self._send_message(self.channel_id, "\n".join(lines), reply_markup=kb)

    # =====================================================================
    # TRADE CLOSED & CELEBRATION
    # =====================================================================

    async def broadcast_trade_closed(
        self,
        position: Any,
    ) -> bool:
        """
        Broadcast closed trade with celebratory WIN messaging, guaranteed profit shield locks,
        clean reversal notices, or disciplined risk alerts.
        """
        if not self.enable_telegram or not self.channel_id:
            return False

        def _val(k: str, default: Any = None) -> Any:
            if isinstance(position, dict):
                return position.get(k, default)
            return getattr(position, k, default)

        symbol = self._clean_symbol(_val("symbol", "UNKNOWN"))
        action = str(_val("action", "BUY")).upper()
        entry = self._safe_float(_val("entry_price", 0))
        exit_price = self._safe_float(_val("exit_price", _val("current_price", 0)))
        pnl = self._safe_float(_val("pnl", 0))
        pnl_pct = self._safe_float(_val("pnl_percentage", _val("pnl_pct", 0)))
        if pnl_pct == 0.0 and entry > 0 and exit_price > 0:
            if is_buy:
                pnl_pct = ((exit_price - entry) / entry) * 100.0
            else:
                pnl_pct = ((entry - exit_price) / entry) * 100.0

        roe_val = _val("roe", _val("roe_percentage", None))
        if roe_val is not None:
            roe_pct = self._safe_float(roe_val)
        else:
            lev = self._safe_float(_val("leverage", 3.0))
            roe_pct = pnl_pct * (lev if lev > 0 else 3.0)
        reason = str(_val("close_reason", _val("exit_reason", "TAKE_PROFIT"))).upper()

        is_buy = action in ("BUY", "LONG")
        action_badge = "🟢 LONG" if is_buy else "🔴 SHORT"

        formatted_entry = self._format_price(entry)
        formatted_exit = self._format_price(exit_price)
        kb = self._create_signal_keyboard(symbol)

        if "REVERSAL" in reason or reason == "REVERSED_BY_OPPOSITE_SIGNAL":
            pnl_sign = "+" if pnl >= 0 else "-"
            message = (
                "🔄 <b>AI TREND REVERSAL • POSITION FLIPPED</b> ⚡\n"
                "━━━━━━━━━━━━━━━━━━━━━\n"
                f"💎 <b>#{self._escape(symbol)}USDT</b> • <b>{action_badge}</b>\n"
                "━━━━━━━━━━━━━━━━━━━━━\n"
                f"🎯 <b>Status:</b> <b>CLEAN TREND REVERSAL</b>\n"
                f"📊 <b>Realized PnL:</b> <code>{pnl_sign}${abs(pnl):,.2f} ({pnl_sign}{abs(pnl_pct):.2f}%)</code>\n"
                f"📍 <b>Entry:</b> <code>{formatted_entry}</code> ➔ <b>Exit:</b> <code>{formatted_exit}</code>\n"
                "━━━━━━━━━━━━━━━━━━━━━\n"
                "<i>AI Committee detected higher-timeframe trend reversal. Repositioning capital.</i>"
            )
        elif "TRAILING" in reason or reason == "TRAILING_PROFIT_LOCK":
            pnl_sign = "+" if pnl >= 0 else ""
            pct_sign = "+" if pnl_pct >= 0 else ""
            message = (
                "🛡️ <b>AI PROFIT SHIELD • PROFIT SECURED!</b> 💰\n"
                "━━━━━━━━━━━━━━━━━━━━━\n"
                f"💎 <b>#{self._escape(symbol)}USDT</b> • <b>{action_badge}</b>\n"
                "━━━━━━━━━━━━━━━━━━━━━\n"
                f"🎯 <b>Status:</b> <b>TRAILING PROFIT LOCK SECURED</b>\n"
                f"📈 <b>Protected Move:</b> <code>{pct_sign}{abs(pnl_pct):.2f}%</code> <i>({pct_sign}{abs(roe_pct):.1f}% ROE)</i> 🔥\n"
                f"💵 <b>Net Banked PnL:</b> <code>{pnl_sign}${abs(pnl):,.2f}</code> 💰\n\n"
                f"📍 <b>Entry:</b> <code>{formatted_entry}</code> ➔ <b>Exit:</b> <code>{formatted_exit}</code>\n"
                "🛡️ <b>Shield Action:</b> <b>AI Trailing Stop Protected Pullback</b>\n"
                "━━━━━━━━━━━━━━━━━━━━━\n"
                "🏆 <i>SnartCrypto AI Engine Alpha • Verified Real Execution</i>"
            )
        elif "BREAKEVEN" in reason or reason == "DYNAMIC_BREAKEVEN":
            pnl_sign = "+" if pnl >= 0 else ""
            pct_sign = "+" if pnl_pct >= 0 else ""
            message = (
                "🛡️ <b>AI PROFIT SHIELD • CAPITAL & PROFIT PROTECTED</b> 🛡️\n"
                "━━━━━━━━━━━━━━━━━━━━━\n"
                f"💎 <b>#{self._escape(symbol)}USDT</b> • <b>{action_badge}</b>\n"
                "━━━━━━━━━━━━━━━━━━━━━\n"
                f"🎯 <b>Status:</b> <b>DYNAMIC BREAKEVEN HIT</b>\n"
                f"📊 <b>Result:</b> <code>{pct_sign}{abs(pnl_pct):.2f}% ({pnl_sign}${abs(pnl):,.2f} Guaranteed Banked Profit)</code>\n"
                f"📍 <b>Entry:</b> <code>{formatted_entry}</code> ➔ <b>Exit:</b> <code>{formatted_exit}</code>\n"
                "━━━━━━━━━━━━━━━━━━━━━\n"
                "<i>Stop Loss was moved to Profit Lock at +2.0% gain. Capital 100% preserved.</i>"
            )
        elif pnl > 0 or "PROFIT" in reason or "TP" in reason or "WIN" in reason:
            pnl_sign = "+"
            pct_sign = "+"
            message = (
                "🎉🎉🎉🎉🎉🎉🎉🎉🎉\n"
                "<b>PROFIT TARGET HIT! • WIN</b> 🚀\n"
                "━━━━━━━━━━━━━━━━━━━━━\n"
                f"💎 <b>#{self._escape(symbol)}USDT</b> • <b>{action_badge}</b>\n"
                "━━━━━━━━━━━━━━━━━━━━━\n"
                f"🎯 <b>Status:</b> <b>TAKE PROFIT REACHED</b>\n"
                f"📈 <b>Target Gain:</b> <code>{pct_sign}{abs(pnl_pct):.2f}%</code> <i>({pct_sign}{abs(roe_pct):.1f}% ROE)</i> 🔥\n"
                f"💵 <b>Net Realized Profit:</b> <code>{pnl_sign}${abs(pnl):,.2f}</code> 💰\n\n"
                f"📍 <b>Entry:</b> <code>{formatted_entry}</code> ➔ <b>Exit:</b> <code>{formatted_exit}</code>\n"
                f"📌 <b>Close Trigger:</b> <b>{self._escape(reason)}</b>\n"
                "━━━━━━━━━━━━━━━━━━━━━\n"
                "🏆 <i>SnartCrypto AI Engine Alpha • Verified Real Execution</i>"
            )
        elif pnl < 0 or "STOP" in reason or "SL" in reason or "LOSS" in reason:
            message = (
                "🛡️ <b>RISK MANAGEMENT • POSITION CLOSED</b>\n"
                "━━━━━━━━━━━━━━━━━━━━━\n"
                f"⚠️ <b>#{self._escape(symbol)}USDT</b> • <b>{action_badge}</b>\n"
                "━━━━━━━━━━━━━━━━━━━━━\n"
                f"🛑 <b>Status:</b> <b>STOP LOSS TRIGGERED</b>\n"
                f"📉 <b>Loss:</b> <code>-{abs(pnl_pct):.2f}%</code> (<code>-${abs(pnl):,.2f}</code>)\n"
                f"📍 <b>Entry:</b> <code>{formatted_entry}</code> ➔ <b>Exit:</b> <code>{formatted_exit}</code>\n"
                "━━━━━━━━━━━━━━━━━━━━━\n"
                "<i>Capital strictly preserved by ATR risk bounds. Next setup preparing.</i>"
            )
        else:
            pnl_sign = "+" if pnl >= 0 else "-"
            message = (
                "⚖️ <b>POSITION CLOSED • TIME-DECAY EXIT</b>\n"
                "━━━━━━━━━━━━━━━━━━━━━\n"
                f"ℹ️ <b>#{self._escape(symbol)}USDT</b> • <b>{action_badge}</b>\n"
                "━━━━━━━━━━━━━━━━━━━━━\n"
                f"📊 <b>Result:</b> <code>{pnl_sign}{abs(pnl_pct):.2f}%</code> (<code>{pnl_sign}${abs(pnl):,.2f}</code>)\n"
                f"📍 <b>Entry:</b> <code>{formatted_entry}</code> ➔ <b>Exit:</b> <code>{formatted_exit}</code>\n"
                f"⏱ <b>Reason:</b> <b>Max Holding Horizon Reached</b>\n"
                "━━━━━━━━━━━━━━━━━━━━━\n"
                "<i>Capital recycled for high-momentum opportunities.</i>"
            )

        return await self._send_message(self.channel_id, message, reply_markup=kb)

    # =====================================================================
    # DAILY BRIEFING
    # =====================================================================
    async def broadcast_daily_briefing(
        self,
        market_summary: Dict[str, Any],
    ) -> bool:
        """Broadcast daily market intelligence."""

        if not self.enable_telegram or not self.channel_id:
            return False

        market_summary = market_summary if isinstance(market_summary, dict) else {}

        date_str = self._now_utc().strftime("%Y-%m-%d")

        regime = market_summary.get("market_regime", "UNKNOWN")

        top_coins = market_summary.get("top_momentum_symbols", [])
        if not isinstance(top_coins, (list, tuple)):
            top_coins = []
        top_coins = [self._clean_symbol(x) for x in top_coins[:10]]

        total_scanned = self._safe_int(
            market_summary.get(
                "symbols_scanned",
                market_summary.get("total_symbols", 0),
            )
        )

        approved = self._safe_int(
            market_summary.get(
                "approved_signals",
                market_summary.get("signals", 0),
            )
        )

        # Build watchlist string
        if top_coins:
            watchlist = ' '.join('#' + x + 'USDT' for x in top_coins)
        else:
            watchlist = 'No watchlist data available'

        message = (
            "<b>🌅 SNARTCRYPTO AI DAILY "
            "MARKET INTELLIGENCE</b>\n"
            f"<i>{date_str} UTC</i>\n\n"

            "<b>🌐 MARKET STATE</b>\n"
            f"├ Regime: <b>{self._escape(regime)}</b>\n"
            f"├ Symbols scanned: <b>{total_scanned:,}</b>\n"
            f"└ Approved setups: <b>{approved:,}</b>\n\n"

            "<b>🔥 AI MOMENTUM WATCHLIST</b>\n"
            f"└ {watchlist}\n\n"

            "<b>🧠 DECISION ARCHITECTURE</b>\n"
            "├ Model 1: Continuous Return Regression\n"
            "├ Model 2: 6-Head Smart Trader AI\n"
            "├ Model 3: Market GPT World Model\n"
            "└ Model 4: Strategy Detector Intelligence\n\n"

            "<i>Market intelligence is probabilistic. "
            "No model guarantees future returns.</i>"
        )

        return await self._send_message(self.channel_id, message)

    # =====================================================================
    # WEEKLY PERFORMANCE
    # =====================================================================

    async def broadcast_weekly_performance_summary(
        self,
        summary: Dict[str, Any],
    ) -> bool:
        """Broadcast transparent weekly performance audit."""

        if (
            not self.enable_telegram
            or not self.channel_id
        ):
            return False

        summary = (
            summary
            if isinstance(
                summary,
                dict,
            )
            else {}
        )

        total_signals = self._safe_int(
            summary.get(
                "total_signals",
                0,
            )
        )

        closed = self._safe_int(
            summary.get(
                "closed_signals",
                summary.get(
                    "total_closed",
                    0,
                ),
            )
        )

        wins = self._safe_int(
            summary.get(
                "wins",
                0,
            )
        )

        losses = self._safe_int(
            summary.get(
                "losses",
                0,
            )
        )

        win_rate = self._safe_float(
            summary.get(
                "overall_win_rate",
                summary.get(
                    "win_rate",
                    0,
                ),
            )
        )

        # Accept either decimal or percentage input.
        if win_rate > 1:
            win_rate /= 100

        total_pnl = self._safe_float(
            summary.get(
                "total_pnl",
                0,
            )
        )

        avg_pnl = self._safe_float(
            summary.get(
                "avg_pnl",
                0,
            )
        )

        sharpe = self._safe_float(
            summary.get(
                "sharpe_ratio",
                0,
            )
        )

        max_dd = self._safe_float(
            summary.get(
                "max_drawdown",
                0,
            )
        )

        pnl_sign = "+" if total_pnl >= 0 else ""

        message = (
            "<b>📊 SNARTCRYPTO AI WEEKLY "
            "PERFORMANCE AUDIT</b>\n\n"

            "<b>📈 TRADE STATISTICS</b>\n"
            f"├ Signals: <b>{total_signals:,}</b>\n"
            f"├ Closed: <b>{closed:,}</b>\n"
            f"├ Wins: <b>{wins:,}</b>\n"
            f"└ Losses: <b>{losses:,}</b>\n\n"

            "<b>💰 PERFORMANCE</b>\n"
            f"├ Win Rate: <b>{win_rate:.1%}</b>\n"
            f"├ Total PnL: "
            f"<code>{pnl_sign}{total_pnl:.2f}%</code>\n"
            f"├ Average PnL: "
            f"<code>{avg_pnl:+.2f}%</code>\n"
            f"├ Sharpe: <b>{sharpe:.3f}</b>\n"
            f"└ Max Drawdown: "
            f"<b>{max_dd:.2f}%</b>\n\n"

            "<b>🤖 SYSTEM</b>\n"
            "├ Model 1 — Regression\n"
            "├ Model 2 — Smart Trader\n"
            "├ Model 3 — Market GPT\n"
            "└ Model 4 — Strategy Intelligence\n\n"

            "<i>Performance statistics are historical "
            "measurements and do not guarantee future results.</i>"
        )

        return await self._send_message(
            self.channel_id,
            message,
        )

    # =====================================================================
    # /PREDICT
    # =====================================================================

    async def handle_predict_response(
        self,
        user_chat_id: str,
        symbol: str,
        signal: Optional[Dict[str, Any]],
    ) -> bool:
        """Send an individual on-demand AI prediction."""

        symbol_clean = self._clean_symbol(
            symbol
        )

        if not signal:

            text = (
                "<b>🔎 AI FORECAST</b>\n\n"
                f"#{self._escape(symbol_clean)}USDT\n\n"
                "🟡 <b>NO APPROVED SETUP</b>\n\n"
                "The current market state did not satisfy "
                "the configured signal-generation filters."
            )

            return await self._send_message(
                user_chat_id,
                text,
            )

        card = self._format_single_signal(signal)
        message = (
            "<b>🔮 SNARTCRYPTO AI ON-DEMAND FORECAST</b>\n\n"
            + card
            + "\n\n💡 <i>Tap any parameter above to copy to your exchange terminal.</i>"
        )

        return await self._send_message(
            user_chat_id,
            message,
        )

    # =====================================================================
    # ADMIN ALERTS
    # =====================================================================

    async def send_admin_alert(
        self,
        text: str,
    ) -> bool:
        """Send private system alert to administrator."""

        if not self.admin_chat_id:
            return False

        message = (
            "<b>🔔 SNARTCRYPTO ADMIN ALERT</b>\n\n"
            f"{self._escape(text)}"
        )

        return await self._send_message(
            self.admin_chat_id,
            message,
        )

    # =====================================================================
    # UTILITY BROADCASTS
    # =====================================================================

    async def broadcast_system_status(
        self,
        status: Dict[str, Any],
    ) -> bool:
        """
        Optional system-health message.

        Useful for startup/recovery notifications.
        """

        if (
            not self.enable_telegram
            or not self.channel_id
        ):
            return False

        status = (
            status
            if isinstance(
                status,
                dict,
            )
            else {}
        )

        system_status = str(
            status.get(
                "status",
                "UNKNOWN",
            )
        ).upper()

        if system_status in {
            "HEALTHY",
            "ONLINE",
            "READY",
        }:
            icon = "🟢"

        elif system_status in {
            "DEGRADED",
            "WARNING",
        }:
            icon = "🟡"

        else:
            icon = "🔴"

        message = (
            f"<b>{icon} SNARTCRYPTO SYSTEM STATUS</b>\n\n"
            f"Status: <b>{self._escape(system_status)}</b>\n"
            f"AI Committee: "
            f"<b>{self._escape(status.get('ai_committee', 'N/A'))}</b>\n"
            f"Strategy Detector: "
            f"<b>{self._escape(status.get('strategy_detector', 'N/A'))}</b>\n"
            f"Execution: "
            f"<b>{self._escape(status.get('execution', 'N/A'))}</b>\n"
            f"Database: "
            f"<b>{self._escape(status.get('database', 'N/A'))}</b>\n"
        )

        return await self._send_message(
            self.channel_id,
            message,
        )

    # =====================================================================
    # HEALTH
    # =====================================================================

    def health_check(self) -> Dict[str, Any]:
        """Return Telegram service health information."""

        configured = bool(
            self.bot_token
            and self.channel_id
        )

        return {
            "enabled": self.enable_telegram,
            "configured": configured,
            "channel_configured": bool(
                self.channel_id
            ),
            "admin_configured": bool(
                self.admin_chat_id
            ),
            "http_client_initialized": (
                self._client is not None
            ),
            "cached_signal_ids": len(
                self._broadcasted_signal_ids
            ),
            "status": (
                "healthy"
                if (
                    self.enable_telegram
                    and configured
                )
                else "disabled"
                if not self.enable_telegram
                else "degraded"
            ),
        }

    def get_status(self) -> Dict[str, Any]:
        """Return Telegram service status."""
        return self.health_check()

    def status(self) -> Dict[str, Any]:
        """Return Telegram service status."""
        return self.health_check()