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
    ) -> bool:
        """
        Low-level Telegram send operation.

        Includes:
        - reusable HTTP client
        - retry handling
        - Telegram API validation
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

        payload = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": disable_web_page_preview,
        }

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
    # SIGNAL FORMAT
    # =====================================================================

    def _format_single_signal(
        self,
        signal: Dict[str, Any],
        index: Optional[int] = None,
    ) -> str:
        """
        Produce a clean, professional, 1-tap copy-ready VIP AI Trade Signal card.
        """
        symbol = self._clean_symbol(signal.get("symbol"))
        action = str(signal.get("action", "BUY")).upper()
        price = self._safe_float(signal.get("price"))
        confidence = self._safe_float(signal.get("confidence"), 0.88)
        strength = self._safe_float(signal.get("signal_strength"), 0.82)

        strategy = signal.get("strategy", {})
        if not isinstance(strategy, dict):
            strategy = {}

        stop_loss = self._safe_float(strategy.get("stop_loss"))
        tp1 = self._safe_float(strategy.get("take_profit_1") or strategy.get("take_profit"))
        tp2 = self._safe_float(strategy.get("take_profit_2"))
        max_hold = self._safe_int(strategy.get("max_holding_hours", 4), 4)

        if not stop_loss and price > 0:
            stop_loss = round(price * 0.965 if action == "BUY" else price * 1.035, 4 if price < 10 else 2)
        if not tp1 and price > 0:
            tp1 = round(price * 1.045 if action == "BUY" else price * 0.955, 4 if price < 10 else 2)
        if not tp2 and tp1 and price > 0:
            tp2 = round(price * 1.075 if action == "BUY" else price * 0.925, 4 if price < 10 else 2)

        # Correct directional percentage calculations
        sl_pct = abs((price - stop_loss) / price * 100) if (price > 0 and stop_loss) else 3.50
        tp1_pct = abs((tp1 - price) / price * 100) if (price > 0 and tp1) else 4.50
        tp2_pct = abs((tp2 - price) / price * 100) if (price > 0 and tp2) else 7.50

        is_buy = action in ("BUY", "LONG")
        action_badge = "🟢 LONG / BUY" if is_buy else "🔴 SHORT / SELL"
        num_str = f"<b>#{index} </b>" if index is not None else ""

        formatted_price = self._format_price(price)
        formatted_sl = self._format_price(stop_loss)
        formatted_tp1 = self._format_price(tp1)
        formatted_tp2 = self._format_price(tp2) if tp2 else ""

        lines = [
            f"⚡ {num_str}<b>#{self._escape(symbol)}USDT</b> • {action_badge} ⚡",
            "━━━━━━━━━━━━━━━━━━━━",
            f"📍 <b>Entry:</b> <code>{formatted_price}</code>",
            f"🎯 <b>Take Profit 1:</b> <code>{formatted_tp1}</code> <i>(+{tp1_pct:.2f}%)</i>",
        ]
        if formatted_tp2:
            lines.append(f"🎯 <b>Take Profit 2:</b> <code>{formatted_tp2}</code> <i>(+{tp2_pct:.2f}%)</i>")

        lines.extend([
            f"🛑 <b>Stop Loss:</b> <code>{formatted_sl}</code> <i>(-{sl_pct:.2f}%)</i>",
            "",
            f"⚙️ <b>Leverage:</b> <code>3x - 5x</code> | <b>Risk:</b> <code>1-2%</code>",
            f"🤖 <b>AI Confidence:</b> <b>{confidence:.1%}</b> | Hold: <b>{max_hold}h</b>",
        ])

        return "\n".join(lines)

    def _format_signal_compact(
        self,
        signal: Dict[str, Any],
        index: int,
    ) -> str:
        """Produce a high-density, tap-to-copy snippet for combined multi-signal digest."""
        symbol = self._clean_symbol(signal.get("symbol"))
        action = str(signal.get("action", "BUY")).upper()
        price = self._safe_float(signal.get("price"))
        confidence = self._safe_float(signal.get("confidence"), 0.88)

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
            f"<b>{index}️⃣ #{self._escape(symbol)}USDT</b> • <b>{action_icon}</b> (AI: <b>{confidence:.0%}</b>)\n"
            f"├ 📍 Entry: <code>{self._format_price(price)}</code>\n"
            f"├ 🎯 TP: <code>{self._format_price(tp)}</code> <i>(+{tp_pct:.1f}%)</i>\n"
            f"└ 🛑 SL: <code>{self._format_price(stop_loss)}</code> <i>(-{sl_pct:.1f}%)</i>"
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

        # If only 1 signal: send single signal card with quick copy tips
        if len(fresh_signals) == 1:
            card = self._format_single_signal(fresh_signals[0])
            card += "\n\n💡 <i>Tap any number above to copy directly to Binance/Bybit.</i>"
            success = await self._send_message(self.channel_id, card)
            if success and fresh_signals[0].get("signal_id"):
                self._broadcasted_signal_ids.add(str(fresh_signals[0]["signal_id"]))
            return success

        # If multiple signals: Combine into one elegant VIP Signal Digest
        now = self._now_utc()
        buy_count = sum(1 for s in fresh_signals if str(s.get("action", "")).upper() in ("BUY", "LONG"))
        sell_count = sum(1 for s in fresh_signals if str(s.get("action", "")).upper() in ("SELL", "SHORT"))

        header = (
            "🚨 <b>SNARTCRYPTO VIP • SIGNALS DIGEST</b> 🚨\n"
            f"<i>{now.strftime('%Y-%m-%d %H:%M UTC')}</i> | 📡 <b>{len(fresh_signals)} Approved Setups</b>\n"
            f"🟢 Long: <b>{buy_count}</b> | 🔴 Short: <b>{sell_count}</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n\n"
        )

        compact_cards = [
            self._format_signal_compact(s, i)
            for i, s in enumerate(fresh_signals, start=1)
        ]

        footer = (
            "\n\n━━━━━━━━━━━━━━━━━━━━\n"
            "⚙️ <b>Rec. Leverage:</b> <code>3x - 5x</code> | <b>Risk:</b> <code>1-2% per trade</code>\n"
            "💡 <i>Tap any value above to copy directly to your trading terminal.</i>"
        )

        full_message = header + "\n\n".join(compact_cards) + footer

        if len(full_message) <= self.SAFE_MESSAGE_LENGTH:
            success = await self._send_message(self.channel_id, full_message)
        else:
            chunk_size = 4
            success = True
            for i in range(0, len(compact_cards), chunk_size):
                chunk = compact_cards[i : i + chunk_size]
                chunk_msg = header + "\n\n".join(chunk) + footer
                ok = await self._send_message(self.channel_id, chunk_msg)
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
        action_icon = "🟢 LONG / BUY" if is_buy else "🔴 SHORT / SELL"

        lines = [
            "⚡ <b>LIVE TRADE EXECUTED</b> ⚡",
            "━━━━━━━━━━━━━━━━━━━━",
            f"🎯 <b>#{self._escape(symbol)}USDT</b> • <b>{action_icon}</b>",
            "━━━━━━━━━━━━━━━━━━━━",
            f"📍 <b>Entry:</b> <code>{self._format_price(entry)}</code>",
        ]
        if tp:
            lines.append(f"🎯 <b>Take Profit:</b> <code>{self._format_price(tp)}</code>")
        if sl:
            lines.append(f"🛑 <b>Stop Loss:</b> <code>{self._format_price(sl)}</code>")
        if pos_id:
            lines.append(f"🆔 <b>Order ID:</b> <code>{self._escape(str(pos_id))}</code>")

        lines.extend([
            "━━━━━━━━━━━━━━━━━━━━",
            "<i>Automated trade live on exchange • Live exit monitoring active</i>",
        ])

        return await self._send_message(self.channel_id, "\n".join(lines))

    # =====================================================================
    # TRADE CLOSED & CELEBRATION
    # =====================================================================

    async def broadcast_trade_closed(
        self,
        position: Any,
    ) -> bool:
        """
        Broadcast closed trade with celebratory WIN messaging or disciplined risk alerts.
        """
        if not self.enable_telegram or not self.channel_id:
            return False

        symbol = self._clean_symbol(getattr(position, "symbol", "UNKNOWN"))
        action = str(getattr(position, "action", "BUY")).upper()
        entry = self._safe_float(getattr(position, "entry_price", 0))
        exit_price = self._safe_float(getattr(position, "exit_price", getattr(position, "current_price", 0)))
        pnl = self._safe_float(getattr(position, "pnl", 0))
        pnl_pct = self._safe_float(getattr(position, "pnl_percentage", 0))
        reason = str(getattr(position, "close_reason", getattr(position, "exit_reason", "TAKE_PROFIT"))).upper()

        is_buy = action in ("BUY", "LONG")
        action_label = "LONG" if is_buy else "SHORT"

        formatted_entry = self._format_price(entry)
        formatted_exit = self._format_price(exit_price)

        if pnl > 0 or "PROFIT" in reason or "TP" in reason or "WIN" in reason:
            # 🎉 CELEBRATORY WIN PRESENTATION 🎉
            pnl_sign = "+"
            pct_sign = "+"
            message = (
                "🎉 <b>PROFIT TARGET HIT! • WIN</b> 🚀\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                f"💎 <b>#{self._escape(symbol)}USDT</b> • 🟢 <b>{action_label}</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                f"🎯 <b>Status:</b> <b>TAKE PROFIT REACHED</b>\n"
                f"📈 <b>Gain:</b> <code>{pct_sign}{abs(pnl_pct):.2f}%</code> 🔥\n"
                f"💵 <b>Net Realized PnL:</b> <code>{pnl_sign}${abs(pnl):,.2f}</code> 💰\n\n"
                f"📍 <b>Entry:</b> <code>{formatted_entry}</code>\n"
                f"🏁 <b>Exit:</b> <code>{formatted_exit}</code>\n"
                f"📌 <b>Close Trigger:</b> <b>{self._escape(reason)}</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                "🏆 <i>SnartCrypto AI Engine Alpha • Verified Execution</i>"
            )
        elif pnl < 0 or "STOP" in reason or "SL" in reason or "LOSS" in reason:
            # 🛡️ DISCIPLINED RISK MANAGEMENT 🛡️
            message = (
                "🛡️ <b>RISK MANAGEMENT • POSITION CLOSED</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                f"⚠️ <b>#{self._escape(symbol)}USDT</b> • 🔴 <b>{action_label}</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                f"🛑 <b>Status:</b> <b>STOP LOSS TRIGGERED</b>\n"
                f"📉 <b>Loss:</b> <code>-{abs(pnl_pct):.2f}%</code> (<code>-${abs(pnl):,.2f}</code>)\n"
                f"📍 <b>Entry:</b> <code>{formatted_entry}</code> ➔ <b>Exit:</b> <code>{formatted_exit}</code>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                "<i>Capital preserved by automated risk protocol. Next setup ready.</i>"
            )
        else:
            # ⚖️ TIMEOUT / BREAKEVEN ⚖️
            pnl_sign = "+" if pnl >= 0 else "-"
            message = (
                "⚖️ <b>POSITION CLOSED • TIMEOUT</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                f"ℹ️ <b>#{self._escape(symbol)}USDT</b> • 🟡 <b>{action_label}</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                f"📊 <b>Result:</b> <code>{pnl_sign}{abs(pnl_pct):.2f}%</code> (<code>{pnl_sign}${abs(pnl):,.2f}</code>)\n"
                f"📍 <b>Entry:</b> <code>{formatted_entry}</code> ➔ <b>Exit:</b> <code>{formatted_exit}</code>\n"
                f"⏱ <b>Reason:</b> <b>Max Holding Duration Reached</b>\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                "<i>Position closed to recycle capital for higher-conviction opportunities.</i>"
            )

        return await self._send_message(self.channel_id, message)

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