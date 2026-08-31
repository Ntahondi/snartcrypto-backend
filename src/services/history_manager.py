"""
src/services/history_manager.py

SmartCrypto AI v3.2.0
=====================

High-level business logic for:

    - Trading signal history
    - Signal lifecycle management
    - Model 1/2/3 prediction history
    - Model 4 strategy-detector history
    - Pattern visualization
    - Historical pattern similarity
    - Performance calculations
    - Symbol performance
    - Overall performance summaries
    - Cache management
    - Storage health

Architecture:

    SignalGenerator
          |
          v
    HistoryManager
          |
          v
    DataStorage
          |
          v
       SQLite

Important:
    HistoryManager NEVER generates or modifies trading decisions.

It records what SignalGenerator decided.

Model 4 is stored as a strategy-confirmation layer and
is therefore preserved separately from the core AI committee.
"""

from __future__ import annotations

import copy
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from src.data.storage import DataStorage
from src.utils.safe_logger import SafeLogger


logger = SafeLogger.get_logger(__name__)


class HistoryManager:
    """
    History Manager Service.

    Responsibilities:

        1. Save generated signals.
        2. Track open/closed signal lifecycle.
        3. Persist Model 1/2/3 predictions.
        4. Persist Model 4 strategy detections.
        5. Maintain in-memory signal cache.
        6. Calculate historical performance.
        7. Find historically similar signals.
        8. Provide frontend-ready history objects.
        9. Report storage/cache health.

    It does NOT:
        - generate signals
        - alter model predictions
        - approve trades
        - execute orders
        - calculate position sizing
    """

    # ============================================================
    # CONSTANTS
    # ============================================================

    CLOSED_OUTCOMES = {
        "WIN",
        "LOSS",
        "BREAKEVEN",
        "CANCELLED",
        "EXPIRED",
        "CLOSED",
    }

    PROFITABLE_OUTCOMES = {
        "WIN",
    }

    LOSS_OUTCOMES = {
        "LOSS",
    }

    MODEL4_STRATEGIES = (
        "bollinger_squeeze",
        "candlestick",
        "heikin_ashi",
        "ma_crossover",
        "momentum_reversal",
        "narrow_range",
        "role_reversal",
        "rsi_2",
    )

    # ============================================================
    # INITIALIZATION
    # ============================================================

    def __init__(
        self,
        storage_path: str = "data/",
        use_db: bool = True,
    ):
        """
        Initialize HistoryManager.

        Parameters
        ----------
        storage_path:
            Root path used by DataStorage.

        use_db:
            True  -> SQLite/database mode.
            False -> file-backed storage mode.
        """

        self.storage = DataStorage(
            storage_path,
            use_db,
        )

        self.signals_cache: Dict[str, Dict[str, Any]] = {}

        self.pattern_drawings_cache: Dict[
            str,
            Dict[str, Any],
        ] = {}

        self._load_caches()

        logger.info(
            "HistoryManager initialized "
            f"(mode: {'DB' if use_db else 'Files'})"
        )

    # ============================================================
    # DATETIME UTILITIES
    # ============================================================

    @staticmethod
    def _utc_now() -> datetime:
        """Return timezone-aware UTC datetime."""

        return datetime.now(timezone.utc)

    @classmethod
    def _utc_iso(cls) -> str:
        """Return current UTC time in ISO-8601 format."""

        return cls._utc_now().isoformat()

    @staticmethod
    def _normalize_datetime(
        value: Any,
    ) -> str:
        """
        Normalize datetime values into timezone-aware ISO-8601.

        Prevents errors such as:

            2026-08-30T10:00:00+00:00Z

        which incorrectly mixes an offset with an extra Z.
        """

        if value is None:
            return datetime.now(
                timezone.utc
            ).isoformat()

        if isinstance(
            value,
            datetime,
        ):

            dt = value

        else:

            text = str(value).strip()

            if not text:
                return datetime.now(
                    timezone.utc
                ).isoformat()

            # Handle trailing Z.
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"

            try:

                dt = datetime.fromisoformat(
                    text
                )

            except ValueError:

                # If parsing fails, preserve the
                # original value instead of inventing
                # a different timestamp.
                return str(value)

        if dt.tzinfo is None:

            dt = dt.replace(
                tzinfo=timezone.utc
            )

        else:

            dt = dt.astimezone(
                timezone.utc
            )

        return dt.isoformat()

    @staticmethod
    def _parse_datetime(
        value: Any,
    ) -> Optional[datetime]:
        """
        Parse stored timestamp safely.
        """

        if value is None:
            return None

        try:

            if isinstance(
                value,
                datetime,
            ):

                dt = value

            else:

                text = str(value).strip()

                if text.endswith("Z"):
                    text = (
                        text[:-1]
                        + "+00:00"
                    )

                dt = datetime.fromisoformat(
                    text
                )

            if dt.tzinfo is None:

                dt = dt.replace(
                    tzinfo=timezone.utc
                )

            return dt.astimezone(
                timezone.utc
            )

        except Exception:

            return None

    # ============================================================
    # CACHE MANAGEMENT
    # ============================================================

    def _load_caches(self) -> None:
        """
        Load signals and pattern drawings from storage.

        The cache is intentionally bounded by the storage
        query limits.
        """

        self.signals_cache = {}

        self.pattern_drawings_cache = {}

        # --------------------------------------------------------
        # Signals
        # --------------------------------------------------------

        try:

            signals = self.storage.get_signals(
                hours=8760,
                limit=10000,
            )

        except Exception as exc:

            logger.error(
                f"Failed loading signal cache: {exc}",
                exc_info=True,
            )

            signals = []

        for signal in signals or []:

            signal_id = signal.get(
                "signal_id"
            )

            if signal_id:

                self.signals_cache[
                    str(signal_id)
                ] = signal

        # --------------------------------------------------------
        # Pattern drawings
        # --------------------------------------------------------

        try:

            drawings = (
                self.storage.get_pattern_drawings(
                    hours=8760,
                    limit=5000,
                )
            )

        except Exception as exc:

            logger.error(
                f"Failed loading drawing cache: {exc}",
                exc_info=True,
            )

            drawings = []

        for drawing in drawings or []:

            pattern_id = drawing.get(
                "pattern_id"
            )

            if pattern_id:

                self.pattern_drawings_cache[
                    str(pattern_id)
                ] = drawing

        logger.info(
            "History cache loaded: "
            f"{len(self.signals_cache)} signals, "
            f"{len(self.pattern_drawings_cache)} drawings"
        )

    def refresh_cache(self) -> None:
        """
        Refresh all in-memory caches from storage.
        """

        self._load_caches()

    # ============================================================
    # SIGNAL NORMALIZATION
    # ============================================================

    def _build_signal_record(
        self,
        signal: Dict[str, Any],
        outcome: Optional[str] = None,
        pnl: Optional[float] = None,
        pnl_percentage: Optional[float] = None,
        existing_signal: Optional[
            Dict[str, Any]
        ] = None,
    ) -> Dict[str, Any]:
        """
        Convert SignalGenerator output into a storage-ready
        history record.

        This preserves the new Model 4 information.
        """

        source = (
            existing_signal
            if existing_signal is not None
            else signal
        )

        strategy = copy.deepcopy(
            signal.get(
                "strategy",
                source.get("strategy", {}),
            )
            or {}
        )

        analysis = copy.deepcopy(
            signal.get(
                "analysis",
                source.get("analysis", {}),
            )
            or {}
        )

        ai_breakdown = copy.deepcopy(
            signal.get(
                "ai_model_breakdown",
                source.get(
                    "ai_model_breakdown",
                    {},
                ),
            )
            or {}
        )

        votes = copy.deepcopy(
            signal.get(
                "votes",
                source.get(
                    "votes",
                    {},
                ),
            )
            or {}
        )

        expected_returns = copy.deepcopy(
            signal.get(
                "expected_returns",
                source.get(
                    "expected_returns",
                    {},
                ),
            )
            or {}
        )

        market_gpt = copy.deepcopy(
            signal.get(
                "market_gpt_simulation",
                source.get(
                    "market_gpt_simulation",
                    {},
                ),
            )
            or {}
        )

        strategy_detection = copy.deepcopy(
            signal.get(
                "strategy_detection",
                source.get(
                    "strategy_detection",
                    {},
                ),
            )
            or {}
        )

        signal_id = (
            signal.get("signal_id")
            or source.get("signal_id")
        )

        if not signal_id:

            symbol = (
                signal.get("symbol")
                or "UNKNOWN"
            )

            timestamp = (
                datetime.now(
                    timezone.utc
                ).strftime(
                    "%Y%m%d_%H%M%S_%f"
                )
            )

            signal_id = (
                f"{symbol}_{timestamp}"
            )

        timestamp = self._normalize_datetime(
            signal.get(
                "timestamp",
                source.get("timestamp"),
            )
        )

        final_outcome = (
            outcome
            if outcome is not None
            else signal.get(
                "outcome",
                source.get(
                    "outcome",
                    "OPEN",
                ),
            )
        )

        final_pnl = (
            pnl
            if pnl is not None
            else signal.get(
                "pnl",
                source.get("pnl"),
            )
        )

        final_pnl_percentage = (
            pnl_percentage
            if pnl_percentage is not None
            else signal.get(
                "pnl_percentage",
                source.get(
                    "pnl_percentage"
                ),
            )
        )

        record = {
            # ----------------------------------------------------
            # Identity
            # ----------------------------------------------------

            "signal_id": str(signal_id),

            "timestamp": timestamp,

            "symbol": signal.get(
                "symbol",
                source.get("symbol"),
            ),

            # ----------------------------------------------------
            # Core decision
            # ----------------------------------------------------

            "action": signal.get(
                "action",
                source.get("action", "HOLD"),
            ),

            "price": signal.get(
                "price",
                source.get("price"),
            ),

            "confidence": float(
                signal.get(
                    "confidence",
                    source.get(
                        "confidence",
                        0.0,
                    ),
                )
                or 0.0
            ),

            "signal_strength": float(
                signal.get(
                    "signal_strength",
                    source.get(
                        "signal_strength",
                        0.0,
                    ),
                )
                or 0.0
            ),

            # ----------------------------------------------------
            # Multi-timeframe intelligence
            # ----------------------------------------------------

            "direction_1h": signal.get(
                "direction_1h",
                source.get(
                    "direction_1h",
                    "HOLD",
                ),
            ),

            "direction_4h": signal.get(
                "direction_4h",
                source.get(
                    "direction_4h",
                    "HOLD",
                ),
            ),

            "direction_1d": signal.get(
                "direction_1d",
                source.get(
                    "direction_1d",
                    "HOLD",
                ),
            ),

            "risk_level": signal.get(
                "risk_level",
                source.get(
                    "risk_level",
                    "MEDIUM",
                ),
            ),

            "market_regime": signal.get(
                "market_regime",
                source.get(
                    "market_regime",
                    "TRENDING",
                ),
            ),

            # ----------------------------------------------------
            # Trading strategy
            # ----------------------------------------------------

            "strategy": strategy,

            "stop_loss": strategy.get(
                "stop_loss"
            ),

            "take_profit": strategy.get(
                "take_profit",
                strategy.get(
                    "take_profit_2"
                ),
            ),

            "take_profit_1": strategy.get(
                "take_profit_1"
            ),

            "take_profit_2": strategy.get(
                "take_profit_2"
            ),

            "atr_used": strategy.get(
                "atr_used"
            ),

            "max_holding_hours": strategy.get(
                "max_holding_hours",
                8,
            ),

            # ----------------------------------------------------
            # Analysis
            # ----------------------------------------------------

            "analysis": analysis,

            # ----------------------------------------------------
            # NEW AI committee history
            # ----------------------------------------------------

            "ai_model_breakdown": ai_breakdown,

            "votes": votes,

            "expected_returns": (
                expected_returns
            ),

            "market_gpt_simulation": (
                market_gpt
            ),

            # ----------------------------------------------------
            # NEW Model 4 history
            # ----------------------------------------------------

            "strategy_detection": (
                strategy_detection
            ),

            # ----------------------------------------------------
            # Position lifecycle
            # ----------------------------------------------------

            "outcome": (
                str(final_outcome).upper()
                if final_outcome
                else "OPEN"
            ),

            "pnl": final_pnl,

            "pnl_percentage": (
                final_pnl_percentage
            ),

            "entry_price": signal.get(
                "entry_price",
                source.get(
                    "entry_price",
                    signal.get(
                        "price",
                        source.get("price"),
                    ),
                ),
            ),

            "exit_price": signal.get(
                "exit_price",
                source.get("exit_price"),
            ),

            "exit_time": signal.get(
                "exit_time",
                source.get("exit_time"),
            ),

            "position_id": signal.get(
                "position_id",
                source.get("position_id"),
            ),

            # ----------------------------------------------------
            # Metadata
            # ----------------------------------------------------

            "signal_type": analysis.get(
                "signal_type",
                source.get(
                    "signal_type",
                    "AI_ENSEMBLE_WITH_"
                    "STRATEGY_CONFIRMATION",
                ),
            ),

            "created_at": (
                self._utc_iso()
            ),

            "updated_at": (
                self._utc_iso()
            ),
        }

        return record

    # ============================================================
    # SIGNAL MANAGEMENT
    # ============================================================

    def save_signal(
        self,
        signal: Dict[str, Any],
        outcome: Optional[str] = None,
        pnl: Optional[float] = None,
        pnl_percentage: Optional[
            float
        ] = None,
    ) -> str:
        """
        Save a new trading signal.

        Returns
        -------
        str
            signal_id on success, empty string on failure.
        """

        try:

            if not isinstance(
                signal,
                dict,
            ):
                logger.error(
                    "Cannot save signal: "
                    "signal must be a dictionary"
                )

                return ""

            signal_record = (
                self._build_signal_record(
                    signal=signal,
                    outcome=outcome,
                    pnl=pnl,
                    pnl_percentage=(
                        pnl_percentage
                    ),
                )
            )

            signal_id = signal_record[
                "signal_id"
            ]

            # ----------------------------------------------------
            # Avoid accidental duplicate insertion.
            # ----------------------------------------------------

            if signal_id in self.signals_cache:

                logger.warning(
                    "Signal already exists in cache: "
                    f"{signal_id}"
                )

                return signal_id

            saved = self.storage.save_signal(
                signal_record
            )

            if not saved:

                logger.error(
                    "DataStorage rejected signal: "
                    f"{signal_id}"
                )

                return ""

            self.signals_cache[
                signal_id
            ] = signal_record

            logger.info(
                "Signal saved: "
                f"{signal_record.get('symbol')} "
                f"{signal_record.get('action')} | "
                f"Model4="
                f"{self._model4_status(signal_record)} | "
                f"Outcome="
                f"{signal_record.get('outcome')}"
            )

            # ----------------------------------------------------
            # Only closed profitable/loss outcomes affect
            # performance statistics.
            # ----------------------------------------------------

            if self._is_performance_outcome(
                signal_record.get("outcome")
            ):

                self._update_performance(
                    signal_record
                )

            return signal_id

        except Exception as exc:

            logger.error(
                f"Error saving signal: {exc}",
                exc_info=True,
            )

            return ""

    def _model4_status(self, signal_record: Dict[str, Any]) -> str:
        m4 = signal_record.get("model4")
        if isinstance(m4, dict):
            bias = m4.get("bias") or m4.get("strategy_bias") or "ACTIVE"
            active = m4.get("active_count", 0)
            total = m4.get("total_count", 9)
            return f"{bias} ({active}/{total})"
        m4_strats = signal_record.get("model4_strategies")
        if isinstance(m4_strats, dict):
            active_strats = [k for k, v in m4_strats.items() if isinstance(v, dict) and v.get("active")]
            return f"{len(active_strats)} Active"
        return "N/A"

    # ============================================================
    # OUTCOME UPDATE
    # ============================================================

    def update_signal_outcome(
        self,
        signal_id: str,
        outcome: str,
        pnl: Optional[float] = None,
        pnl_percentage: Optional[
            float
        ] = None,
        exit_price: Optional[
            float
        ] = None,
        exit_time: Optional[Any] = None,
        position_id: Optional[str] = None,
    ) -> bool:
        """
        Update an existing signal after its position closes.

        Returns
        -------
        bool
            True if the storage update succeeds.
        """

        try:

            normalized_outcome = (
                str(outcome)
                .upper()
                .strip()
            )

            if not normalized_outcome:

                logger.error(
                    "Cannot update signal: "
                    "empty outcome"
                )

                return False

            # ----------------------------------------------------
            # Find signal.
            # ----------------------------------------------------

            signal = self.signals_cache.get(
                signal_id
            )

            if signal is None:

                signal = (
                    self.storage.get_signal(
                        signal_id
                    )
                )

            if not signal:

                logger.error(
                    f"Signal not found: {signal_id}"
                )

                return False

            previous_outcome = str(
                signal.get(
                    "outcome",
                    "OPEN",
                )
            ).upper()

            # ----------------------------------------------------
            # Update lifecycle fields.
            # ----------------------------------------------------

            signal = copy.deepcopy(
                signal
            )

            signal[
                "outcome"
            ] = normalized_outcome

            signal[
                "pnl"
            ] = pnl

            signal[
                "pnl_percentage"
            ] = pnl_percentage

            signal[
                "exit_price"
            ] = exit_price

            signal[
                "exit_time"
            ] = self._normalize_datetime(
                exit_time
            )

            if position_id is not None:

                signal[
                    "position_id"
                ] = position_id

            signal[
                "updated_at"
            ] = self._utc_iso()

            # ----------------------------------------------------
            # Persist.
            # ----------------------------------------------------

            updated = (
                self.storage.update_signal(
                    signal
                )
            )

            if not updated:

                logger.error(
                    "Storage failed updating signal: "
                    f"{signal_id}"
                )

                return False

            self.signals_cache[
                signal_id
            ] = signal

            # ----------------------------------------------------
            # Performance update.
            #
            # Only transition into a closed outcome should
            # create a new performance event.
            #
            # This prevents duplicate WIN/LOSS counting when
            # update_signal_outcome is called twice.
            # ----------------------------------------------------

            if (
                self._is_performance_outcome(
                    normalized_outcome
                )
                and not self._is_performance_outcome(
                    previous_outcome
                )
            ):

                self._update_performance(
                    signal
                )

            pnl_display = (
                f"{pnl_percentage:.4f}%"
                if pnl_percentage is not None
                else "N/A"
            )

            logger.info(
                "Signal outcome updated: "
                f"{signal.get('symbol')} "
                f"{signal.get('action')} -> "
                f"{normalized_outcome} | "
                f"PnL={pnl_display}"
            )

            return True

        except Exception as exc:

            logger.error(
                "Error updating signal outcome: "
                f"{exc}",
                exc_info=True,
            )

            return False

    # ============================================================
    # SIGNAL RETRIEVAL
    # ============================================================

    def get_recent_signals(
        self,
        symbol: Optional[str] = None,
        hours: int = 24,
        limit: int = 100,
        include_closed: bool = True,
    ) -> List[Dict]:
        """
        Retrieve recent signals from DataStorage.
        """

        try:

            return (
                self.storage.get_signals(
                    symbol,
                    hours,
                    limit,
                    include_closed,
                )
                or []
            )

        except Exception as exc:

            logger.error(
                f"Error retrieving signals: {exc}",
                exc_info=True,
            )

            return []

    def get_history(
        self,
        symbol: Optional[str] = None,
        hours: int = 720,
        limit: int = 100,
        include_closed: bool = True,
        **kwargs: Any,
    ) -> List[Dict]:
        """REST API adapter to get trading/signal history."""
        return self.get_recent_signals(
            symbol=symbol,
            hours=hours,
            limit=limit,
            include_closed=include_closed,
        )

    def get_trade_history(
        self,
        symbol: Optional[str] = None,
        hours: int = 720,
        limit: int = 100,
        include_closed: bool = True,
        **kwargs: Any,
    ) -> List[Dict]:
        """REST API adapter to get trade history."""
        return self.get_recent_signals(
            symbol=symbol,
            hours=hours,
            limit=limit,
            include_closed=include_closed,
        )

    def get_trades(
        self,
        symbol: Optional[str] = None,
        hours: int = 720,
        limit: int = 100,
        include_closed: bool = True,
        **kwargs: Any,
    ) -> List[Dict]:
        """REST API adapter to get trades."""
        return self.get_recent_signals(
            symbol=symbol,
            hours=hours,
            limit=limit,
            include_closed=include_closed,
        )

    def get_closed_trades(
        self,
        symbol: Optional[str] = None,
        hours: int = 720,
        limit: int = 100,
        **kwargs: Any,
    ) -> List[Dict]:
        """REST API adapter to get closed trades."""
        recent = self.get_recent_signals(
            symbol=symbol,
            hours=hours,
            limit=limit,
            include_closed=True,
        )
        return [
            s for s in recent
            if str(s.get("outcome", "")).upper() in {"WIN", "LOSS", "CLOSED"}
        ]

    def get_signals(
        self,
        symbol: Optional[str] = None,
        hours: int = 720,
        limit: int = 100,
        include_closed: bool = True,
        **kwargs: Any,
    ) -> List[Dict]:
        """REST API adapter to get signals."""
        return self.get_recent_signals(
            symbol=symbol,
            hours=hours,
            limit=limit,
            include_closed=include_closed,
        )

    def get_performance(
        self,
        symbol: Optional[str] = None,
        days: int = 30,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """REST API adapter to get performance metrics."""
        if symbol:
            return self.calculate_symbol_performance(symbol, days=days)
        return self.get_performance_summary()

    def get_signal(
        self,
        signal_id: str,
    ) -> Optional[Dict]:
        """
        Retrieve a single signal.
        """

        signal = self.signals_cache.get(
            signal_id
        )

        if signal is not None:
            return signal

        try:

            signal = (
                self.storage.get_signal(
                    signal_id
                )
            )

            if signal:

                self.signals_cache[
                    signal_id
                ] = signal

            return signal

        except Exception as exc:

            logger.error(
                f"Error retrieving signal "
                f"{signal_id}: {exc}"
            )

            return None

    # ============================================================
    # PERFORMANCE MANAGEMENT
    # ============================================================

    @classmethod
    def _is_performance_outcome(
        cls,
        outcome: Optional[str],
    ) -> bool:
        """
        Determine whether an outcome should contribute
        to win/loss performance.
        """

        return str(
            outcome or ""
        ).upper() in {
            "WIN",
            "LOSS",
        }

    def _update_performance(
        self,
        signal: Dict[str, Any],
    ) -> None:
        """
        Update symbol-level performance.

        NOTE:
            This preserves the existing DataStorage performance
            architecture rather than changing the storage schema.
        """

        symbol = signal.get(
            "symbol"
        )

        outcome = str(
            signal.get(
                "outcome",
                "",
            )
        ).upper()

        if (
            not symbol
            or not self._is_performance_outcome(
                outcome
            )
        ):
            return

        try:

            current = (
                self.storage.get_performance(
                    symbol
                )
                or {}
            )

            total_signals = int(
                current.get(
                    "total_signals",
                    0,
                )
            ) + 1

            total_wins = int(
                current.get(
                    "total_wins",
                    0,
                )
            ) + (
                1
                if outcome == "WIN"
                else 0
            )

            total_losses = int(
                current.get(
                    "total_losses",
                    0,
                )
            ) + (
                1
                if outcome == "LOSS"
                else 0
            )

            win_rate = (
                total_wins
                / total_signals
                if total_signals > 0
                else 0.0
            )

            signal_pnl = self._safe_float(
                signal.get(
                    "pnl_percentage"
                ),
                0.0,
            )

            total_pnl = (
                self._safe_float(
                    current.get(
                        "total_pnl",
                        0.0,
                    ),
                    0.0,
                )
                + signal_pnl
            )

            avg_pnl = (
                total_pnl
                / total_signals
                if total_signals > 0
                else 0.0
            )

            # ----------------------------------------------------
            # Historical closed trades.
            # ----------------------------------------------------

            recent = (
                self.storage.get_signals(
                    symbol,
                    hours=8760,
                    limit=10000,
                    include_closed=True,
                )
                or []
            )

            closed = [
                item
                for item in recent
                if str(
                    item.get(
                        "outcome",
                        "",
                    )
                ).upper()
                in {"WIN", "LOSS"}
                and item.get(
                    "pnl_percentage"
                )
                is not None
            ]

            pnls = [
                self._safe_float(
                    item.get(
                        "pnl_percentage"
                    ),
                    0.0,
                )
                for item in closed
            ]

            best_trade = (
                max(pnls)
                if pnls
                else 0.0
            )

            worst_trade = (
                min(pnls)
                if pnls
                else 0.0
            )

            # ----------------------------------------------------
            # Timeframe accuracy.
            # ----------------------------------------------------

            timeframe_accuracy = copy.deepcopy(
                current.get(
                    "timeframe_accuracy",
                    {},
                )
                or {}
            )

            for timeframe in (
                "1h",
                "4h",
                "1d",
            ):

                direction = signal.get(
                    f"direction_{timeframe}"
                )

                if not direction:
                    continue

                tf_data = (
                    timeframe_accuracy.setdefault(
                        timeframe,
                        {},
                    )
                )

                direction_data = (
                    tf_data.setdefault(
                        direction,
                        {
                            "correct": 0,
                            "total": 0,
                        },
                    )
                )

                direction_data[
                    "total"
                ] += 1

                if outcome == "WIN":

                    direction_data[
                        "correct"
                    ] += 1

            # ----------------------------------------------------
            # Model 4 historical statistics.
            # ----------------------------------------------------

            strategy_accuracy = self._build_strategy_accuracy(
                recent
            )

            metrics = {

                "symbol": symbol,

                "timestamp": self._utc_iso(),

                "total_signals": total_signals,

                "total_wins": total_wins,

                "total_losses": total_losses,

                "win_rate": win_rate,

                "total_pnl": total_pnl,

                "avg_pnl": avg_pnl,

                "best_trade": best_trade,

                "worst_trade": worst_trade,

                "sharpe_ratio": (
                    self._calculate_sharpe_ratio(
                        pnls
                    )
                ),

                "max_drawdown": (
                    self._calculate_max_drawdown(
                        pnls
                    )
                ),

                "timeframe_accuracy": (
                    timeframe_accuracy
                ),

                "signal_type_accuracy": (
                    current.get(
                        "signal_type_accuracy",
                        {},
                    )
                    or {}
                ),

                "strategy_accuracy": (
                    strategy_accuracy
                ),
            }

            self.storage.save_performance_metrics(
                metrics
            )

        except Exception as exc:

            logger.error(
                f"Error updating performance "
                f"for {symbol}: {exc}",
                exc_info=True,
            )

    # ============================================================
    # MODEL 4 PERFORMANCE
    # ============================================================

    def _build_strategy_accuracy(
        self,
        signals: List[Dict],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Calculate historical outcome performance for each
        Model 4 strategy.

        This is descriptive history only.

        It does NOT modify future signal decisions.
        """

        results: Dict[
            str,
            Dict[str, Any],
        ] = {}

        for strategy_name in (
            self.MODEL4_STRATEGIES
        ):

            results[
                strategy_name
            ] = {
                "detected": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
            }

        for signal in signals:

            outcome = str(
                signal.get(
                    "outcome",
                    "",
                )
            ).upper()

            if outcome not in {
                "WIN",
                "LOSS",
            }:
                continue

            strategy_detection = (
                signal.get(
                    "strategy_detection",
                    {},
                )
                or {}
            )

            bullish = set(
                strategy_detection.get(
                    "bullish_strategies",
                    [],
                )
                or []
            )

            bearish = set(
                strategy_detection.get(
                    "bearish_strategies",
                    [],
                )
                or []
            )

            detected = (
                bullish
                | bearish
            )

            for strategy_name in detected:

                if strategy_name not in results:

                    results[
                        strategy_name
                    ] = {
                        "detected": 0,
                        "wins": 0,
                        "losses": 0,
                        "win_rate": 0.0,
                    }

                item = results[
                    strategy_name
                ]

                item[
                    "detected"
                ] += 1

                if outcome == "WIN":

                    item[
                        "wins"
                    ] += 1

                elif outcome == "LOSS":

                    item[
                        "losses"
                    ] += 1

        for item in results.values():

            detected = item[
                "detected"
            ]

            item[
                "win_rate"
            ] = (
                item["wins"]
                / detected
                if detected > 0
                else 0.0
            )

        return results

    # ============================================================
    # STATISTICS
    # ============================================================

    @staticmethod
    def _safe_float(
        value: Any,
        default: float = 0.0,
    ) -> float:
        """
        Convert arbitrary numeric values safely.
        """

        try:

            result = float(
                value
            )

            if math.isfinite(
                result
            ):

                return result

        except (
            TypeError,
            ValueError,
        ):
            pass

        return default

    def _calculate_sharpe_ratio(
        self,
        returns: List[float],
        risk_free_rate: float = 0.0,
    ) -> float:
        """
        Calculate Sharpe ratio.

        Returns are expected to be percentages, e.g.:

            +2.0
            -1.5
            +0.8

        risk_free_rate is therefore also interpreted in
        percentage units.

        This is an unannualized trade-return Sharpe ratio.
        """

        if not returns or len(
            returns
        ) < 2:

            return 0.0

        returns_arr = np.asarray(
            returns,
            dtype=float,
        )

        returns_arr = (
            returns_arr[
                np.isfinite(
                    returns_arr
                )
            ]
        )

        if len(
            returns_arr
        ) < 2:

            return 0.0

        excess = (
            returns_arr
            - risk_free_rate
        )

        std = float(
            np.std(
                excess,
                ddof=1,
            )
        )

        if std <= 0:

            return 0.0

        return float(
            np.mean(
                excess
            )
            / std
        )

    def _calculate_max_drawdown(
        self,
        returns: List[float],
    ) -> float:
        """
        Calculate maximum drawdown from trade percentage
        returns.

        Example:

            +5%
            -2%
            -4%

        Returns drawdown as a positive percentage.
        """

        if not returns:
            return 0.0

        returns_arr = np.asarray(
            returns,
            dtype=float,
        )

        returns_arr = (
            returns_arr[
                np.isfinite(
                    returns_arr
                )
            ]
        )

        if len(
            returns_arr
        ) == 0:

            return 0.0

        # Convert percentage returns to multiplicative
        # equity factors.
        factors = (
            1.0
            + returns_arr / 100.0
        )

        # A return <= -100% is not a valid normal
        # trade-return input for this calculation.
        factors = np.maximum(
            factors,
            1e-12,
        )

        cumulative = np.cumprod(
            factors
        )

        running_max = np.maximum.accumulate(
            cumulative
        )

        drawdown = (
            cumulative
            - running_max
        ) / running_max

        return float(
            abs(
                np.min(
                    drawdown
                )
            )
            * 100.0
        )

    # ============================================================
    # SYMBOL PERFORMANCE
    # ============================================================

    def get_symbol_performance(
        self,
        symbol: str,
        days: int = 30,
    ) -> Dict:
        """
        Return performance for a specific symbol.
        """

        try:

            signals = (
                self.storage.get_signals(
                    symbol,
                    hours=days * 24,
                    limit=10000,
                    include_closed=True,
                )
                or []
            )

            if not signals:

                return {}

            closed = [
                signal
                for signal in signals
                if str(
                    signal.get(
                        "outcome",
                        "",
                    )
                ).upper()
                in {
                    "WIN",
                    "LOSS",
                }
            ]

            open_signals = [
                signal
                for signal in signals
                if str(
                    signal.get(
                        "outcome",
                        "",
                    )
                ).upper()
                == "OPEN"
            ]

            if not closed:

                return {

                    "symbol": symbol,

                    "total_signals": len(
                        signals
                    ),

                    "open_signals": len(
                        open_signals
                    ),

                    "closed_signals": 0,

                    "wins": 0,

                    "losses": 0,

                    "win_rate": 0.0,

                    "avg_pnl": 0.0,

                    "total_pnl": 0.0,

                }

            wins = sum(
                1
                for signal in closed
                if str(
                    signal.get(
                        "outcome",
                        "",
                    )
                ).upper()
                == "WIN"
            )

            total_closed = len(
                closed
            )

            pnls = [
                self._safe_float(
                    signal.get(
                        "pnl_percentage"
                    ),
                    0.0,
                )
                for signal in closed
                if signal.get(
                    "pnl_percentage"
                )
                is not None
            ]

            avg_pnl = (
                float(
                    np.mean(pnls)
                )
                if pnls
                else 0.0
            )

            total_pnl = (
                float(
                    np.sum(pnls)
                )
                if pnls
                else 0.0
            )

            model4_accuracy = (
                self._build_strategy_accuracy(
                    closed
                )
            )

            return {

                "symbol": symbol,

                "total_signals": len(
                    signals
                ),

                "open_signals": len(
                    open_signals
                ),

                "closed_signals": total_closed,

                "wins": wins,

                "losses": (
                    total_closed
                    - wins
                ),

                "win_rate": round(
                    wins
                    / total_closed,
                    4,
                ),

                "avg_pnl": round(
                    avg_pnl,
                    4,
                ),

                "total_pnl": round(
                    total_pnl,
                    4,
                ),

                "best_trade": round(
                    max(pnls)
                    if pnls
                    else 0.0,
                    4,
                ),

                "worst_trade": round(
                    min(pnls)
                    if pnls
                    else 0.0,
                    4,
                ),

                "sharpe_ratio": round(
                    self._calculate_sharpe_ratio(
                        pnls
                    ),
                    4,
                ),

                "max_drawdown": round(
                    self._calculate_max_drawdown(
                        pnls
                    ),
                    4,
                ),

                "model4_strategy_accuracy": (
                    model4_accuracy
                ),
            }

        except Exception as exc:

            logger.error(
                f"Error calculating performance "
                f"for {symbol}: {exc}",
                exc_info=True,
            )

            return {}

    # ============================================================
    # OVERALL PERFORMANCE
    # ============================================================

    def get_performance_summary(
        self,
    ) -> Dict:
        """
        Get complete performance summary.

        Includes Model 4 strategy history where available.
        """

        try:

            performance = (
                self.storage.get_performance()
                or {}
            )

        except Exception:

            performance = {}

        try:

            patterns = (
                self.storage.get_pattern_stats()
                or {}
            )

        except Exception:

            patterns = {}

        signals = list(
            self.signals_cache.values()
        )

        closed = [
            signal
            for signal in signals
            if str(
                signal.get(
                    "outcome",
                    "",
                )
            ).upper()
            in {
                "WIN",
                "LOSS",
            }
        ]

        model4_accuracy = (
            self._build_strategy_accuracy(
                closed
            )
        )

        return {

            "total_signals": len(
                signals
            ),

            "open_signals": sum(
                1
                for signal in signals
                if str(
                    signal.get(
                        "outcome",
                        "",
                    )
                ).upper()
                == "OPEN"
            ),

            "closed_signals": len(
                closed
            ),

            "symbol_count": len(
                {
                    signal.get("symbol")
                    for signal in signals
                    if signal.get("symbol")
                }
            ),

            "overall_win_rate": (
                performance.get(
                    "win_rate",
                    0.0,
                )
            ),

            "total_pnl": (
                performance.get(
                    "total_pnl",
                    0.0,
                )
            ),

            "avg_pnl": (
                performance.get(
                    "avg_pnl",
                    0.0,
                )
            ),

            "best_trade": (
                performance.get(
                    "best_trade",
                    0.0,
                )
            ),

            "worst_trade": (
                performance.get(
                    "worst_trade",
                    0.0,
                )
            ),

            "sharpe_ratio": (
                performance.get(
                    "sharpe_ratio",
                    0.0,
                )
            ),

            "max_drawdown": (
                performance.get(
                    "max_drawdown",
                    0.0,
                )
            ),

            "pattern_drawings": len(
                self.pattern_drawings_cache
            ),

            "model4_strategy_accuracy": (
                model4_accuracy
            ),

            "storage_pattern_stats": patterns,

            "last_updated": (
                self._utc_iso()
            ),
        }

    # ============================================================
    # PATTERN VISUALIZATION
    # ============================================================

    def save_pattern_drawing(
        self,
        signal: Dict[str, Any],
        candle_data: Optional[
            List[Dict]
        ] = None,
    ) -> Dict:
        """
        Generate and save a frontend-ready pattern record.

        Model 4 strategy detections are included so the
        frontend can visualize the actual strategy context
        behind the signal.
        """

        try:

            signal_id = signal.get(
                "signal_id"
            )

            timestamp = self._normalize_datetime(
                signal.get(
                    "timestamp"
                )
            )

            pattern_id = (
                f"pat_{signal_id or timestamp}"
            )

            analysis = (
                signal.get(
                    "analysis",
                    {},
                )
                or {}
            )

            strategy_detection = (
                signal.get(
                    "strategy_detection",
                    {},
                )
                or {}
            )

            drawing_record = {

                "pattern_id": pattern_id,

                "signal_id": signal_id,

                "symbol": signal.get(
                    "symbol"
                ),

                "pattern_type": (
                    analysis.get(
                        "detected_pattern",
                        "UNKNOWN",
                    )
                ),

                "action": signal.get(
                    "action",
                    "HOLD",
                ),

                "price": signal.get(
                    "price",
                    0,
                ),

                "confidence": signal.get(
                    "confidence",
                    0,
                ),

                "signal_strength": signal.get(
                    "signal_strength",
                    0,
                ),

                "timestamp": timestamp,

                "created_at": self._utc_iso(),

                # ------------------------------------------------
                # Model 4 visual context
                # ------------------------------------------------

                "model4": {

                    "bias": (
                        strategy_detection.get(
                            "bias",
                            "HOLD",
                        )
                    ),

                    "confirmation_score": (
                        strategy_detection.get(
                            "confirmation_score",
                            0.0,
                        )
                    ),

                    "agreement": (
                        strategy_detection.get(
                            "agreement",
                            0.0,
                        )
                    ),

                    "conflict": (
                        strategy_detection.get(
                            "conflict",
                            0.0,
                        )
                    ),

                    "bullish_strategies": (
                        strategy_detection.get(
                            "bullish_strategies",
                            [],
                        )
                    ),

                    "bearish_strategies": (
                        strategy_detection.get(
                            "bearish_strategies",
                            [],
                        )
                    ),

                    "neutral_strategies": (
                        strategy_detection.get(
                            "neutral_strategies",
                            [],
                        )
                    ),
                },

                # ------------------------------------------------
                # Optional candles
                # ------------------------------------------------

                "candle_data": (
                    candle_data
                    if candle_data is not None
                    else []
                ),
            }

            saved = (
                self.storage.save_pattern_drawing(
                    drawing_record
                )
            )

            if saved:

                self.pattern_drawings_cache[
                    pattern_id
                ] = drawing_record

                logger.info(
                    "Pattern drawing saved: "
                    f"{signal.get('symbol')} | "
                    f"{drawing_record['pattern_type']}"
                )

                return drawing_record

            return {}

        except Exception as exc:

            logger.error(
                f"Error saving pattern drawing: {exc}",
                exc_info=True,
            )

            return {}

    # ============================================================
    # SIMILARITY
    # ============================================================

    def get_similar_patterns(
        self,
        current_signal: Dict[str, Any],
        limit: int = 5,
    ) -> List[Dict]:
        """
        Find historically similar closed signals.

        Similarity considers:

            - same symbol
            - same action
            - confidence
            - signal strength
            - risk
            - market regime
            - timeframe directions
            - Model 4 strategy overlap
            - Model 4 directional bias
        """

        try:

            recent = (
                self.storage.get_signals(
                    hours=720,
                    limit=5000,
                    include_closed=True,
                )
                or []
            )

            similar: List[
                Dict[str, Any]
            ] = []

            for signal in recent:

                if signal.get(
                    "symbol"
                ) != current_signal.get(
                    "symbol"
                ):
                    continue

                if signal.get(
                    "action"
                ) != current_signal.get(
                    "action"
                ):
                    continue

                if str(
                    signal.get(
                        "outcome",
                        "",
                    )
                ).upper() not in {
                    "WIN",
                    "LOSS",
                }:
                    continue

                score = (
                    self._calculate_similarity(
                        current_signal,
                        signal,
                    )
                )

                if score <= 0.50:
                    continue

                timestamp = (
                    self._parse_datetime(
                        signal.get(
                            "timestamp"
                        )
                    )
                )

                days_ago = None

                if timestamp:

                    delta = (
                        self._utc_now()
                        - timestamp
                    )

                    days_ago = (
                        delta.total_seconds()
                        / 86400.0
                    )

                item = copy.deepcopy(
                    signal
                )

                item[
                    "similarity_score"
                ] = round(
                    score,
                    4,
                )

                item[
                    "days_ago"
                ] = round(
                    days_ago,
                    3,
                ) if days_ago is not None else None

                similar.append(
                    item
                )

            similar.sort(
                key=lambda item: item.get(
                    "similarity_score",
                    0.0,
                ),
                reverse=True,
            )

            return similar[:limit]

        except Exception as exc:

            logger.error(
                "Error finding similar patterns: "
                f"{exc}",
                exc_info=True,
            )

            return []

    def _calculate_similarity(
        self,
        signal1: Dict[str, Any],
        signal2: Dict[str, Any],
    ) -> float:
        """
        Calculate weighted similarity between two signals.

        Model 4 strategy overlap receives meaningful weight
        because it represents the market-structure/strategy
        context that accompanied the signal.
        """

        score = 0.0

        total_weight = 0.0

        # --------------------------------------------------------
        # Confidence
        # --------------------------------------------------------

        conf1 = self._safe_float(
            signal1.get(
                "confidence"
            ),
            0.0,
        )

        conf2 = self._safe_float(
            signal2.get(
                "confidence"
            ),
            0.0,
        )

        confidence_similarity = 1.0 - min(
            abs(
                conf1 - conf2
            ),
            1.0,
        )

        score += (
            confidence_similarity
            * 0.12
        )

        total_weight += 0.12

        # --------------------------------------------------------
        # Signal strength
        # --------------------------------------------------------

        strength1 = self._safe_float(
            signal1.get(
                "signal_strength"
            ),
            0.0,
        )

        strength2 = self._safe_float(
            signal2.get(
                "signal_strength"
            ),
            0.0,
        )

        strength_similarity = 1.0 - min(
            abs(
                strength1
                - strength2
            ),
            1.0,
        )

        score += (
            strength_similarity
            * 0.12
        )

        total_weight += 0.12

        # --------------------------------------------------------
        # Risk level
        # --------------------------------------------------------

        risk1 = signal1.get(
            "risk_level"
        )

        risk2 = signal2.get(
            "risk_level"
        )

        risk_similarity = (
            1.0
            if (
                risk1
                and risk2
                and risk1 == risk2
            )
            else 0.0
        )

        score += (
            risk_similarity
            * 0.08
        )

        total_weight += 0.08

        # --------------------------------------------------------
        # Market regime
        # --------------------------------------------------------

        regime1 = signal1.get(
            "market_regime"
        )

        regime2 = signal2.get(
            "market_regime"
        )

        regime_similarity = (
            1.0
            if (
                regime1
                and regime2
                and regime1 == regime2
            )
            else 0.0
        )

        score += (
            regime_similarity
            * 0.12
        )

        total_weight += 0.12

        # --------------------------------------------------------
        # Timeframe alignment
        # --------------------------------------------------------

        timeframe_matches = 0
        timeframe_count = 0

        for timeframe in (
            "1h",
            "4h",
            "1d",
        ):

            key = (
                f"direction_{timeframe}"
            )

            d1 = signal1.get(
                key
            )

            d2 = signal2.get(
                key
            )

            if d1 is None or d2 is None:
                continue

            timeframe_count += 1

            if d1 == d2:

                timeframe_matches += 1

        timeframe_similarity = (
            timeframe_matches
            / timeframe_count
            if timeframe_count > 0
            else 0.0
        )

        score += (
            timeframe_similarity
            * 0.16
        )

        total_weight += 0.16

        # --------------------------------------------------------
        # Model 4 strategy overlap
        # --------------------------------------------------------

        strategies1 = (
            self._extract_model4_strategies(
                signal1
            )
        )

        strategies2 = (
            self._extract_model4_strategies(
                signal2
            )
        )

        if (
            strategies1
            or strategies2
        ):

            union = (
                strategies1
                | strategies2
            )

            intersection = (
                strategies1
                & strategies2
            )

            strategy_similarity = (
                len(intersection)
                / len(union)
                if union
                else 0.0
            )

        else:

            strategy_similarity = 0.0

        score += (
            strategy_similarity
            * 0.20
        )

        total_weight += 0.20

        # --------------------------------------------------------
        # Model 4 directional bias
        # --------------------------------------------------------

        bias1 = (
            signal1.get(
                "strategy_detection",
                {},
            )
            or {}
        ).get(
            "bias",
            "HOLD",
        )

        bias2 = (
            signal2.get(
                "strategy_detection",
                {},
            )
            or {}
        ).get(
            "bias",
            "HOLD",
        )

        bias_similarity = (
            1.0
            if bias1 == bias2
            else 0.0
        )

        score += (
            bias_similarity
            * 0.10
        )

        total_weight += 0.10

        if total_weight <= 0:

            return 0.0

        return float(
            np.clip(
                score
                / total_weight,
                0.0,
                1.0,
            )
        )

    # ============================================================
    # MODEL 4 EXTRACTION
    # ============================================================

    @staticmethod
    def _extract_model4_strategies(
        signal: Dict[str, Any],
    ) -> set:
        """
        Extract all directional Model 4 strategies from
        a historical signal.

        Supports both:

            strategy_detection

        and:

            ai_model_breakdown.model_4_strategy_detector
        """

        strategies = set()

        detection = (
            signal.get(
                "strategy_detection",
                {},
            )
            or {}
        )

        for key in (
            "bullish_strategies",
            "bearish_strategies",
            "detected_strategies",
        ):

            value = detection.get(
                key
            )

            if isinstance(
                value,
                list,
            ):

                strategies.update(
                    value
                )

            elif isinstance(
                value,
                dict,
            ):

                strategies.update(
                    value.keys()
                )

        # --------------------------------------------------------
        # Fallback to AI breakdown.
        # --------------------------------------------------------

        breakdown = (
            signal.get(
                "ai_model_breakdown",
                {},
            )
            or {}
        )

        model4 = breakdown.get(
            "model_4_strategy_detector",
            {},
        ) or {}

        for key in (
            "bullish_strategies",
            "bearish_strategies",
        ):

            value = model4.get(
                key
            )

            if isinstance(
                value,
                list,
            ):

                strategies.update(
                    value
                )

        return strategies

    # ============================================================
    # MODEL 4 STATUS
    # ============================================================

    @staticmethod
    def _model4_status(
        signal: Dict[str, Any],
    ) -> str:
        """
        Return compact Model 4 status for logging.
        """

        detection = (
            signal.get(
                "strategy_detection",
                {},
            )
            or {}
        )

        if not detection:

            return "UNAVAILABLE"

        bias = detection.get(
            "bias",
            "HOLD",
        )

        confirmation = (
            detection.get(
                "confirmation_score",
                0.0,
            )
        )

        try:

            confirmation = float(
                confirmation
            )

        except (
            TypeError,
            ValueError,
        ):

            confirmation = 0.0

        return (
            f"{bias}:{confirmation:.2f}"
        )

    # ============================================================
    # HEALTH CHECK
    # ============================================================

    def health_check(
        self,
    ) -> Dict:
        """
        Return storage and cache health.
        """

        try:

            storage_health = (
                self.storage.health_check()
            )

        except Exception as exc:

            logger.error(
                f"Storage health check failed: {exc}"
            )

            storage_health = {
                "status": "unhealthy",
                "error": str(exc),
            }

        return {

            **storage_health,

            "cache_size": {

                "signals": len(
                    self.signals_cache
                ),

                "drawings": len(
                    self.pattern_drawings_cache
                ),
            },

            "storage_mode": (
                "database"
                if self.storage.use_db
                else "files"
            ),

            "model4_history_supported": True,

            "model4_strategy_count": len(
                self.MODEL4_STRATEGIES
            ),

            "status": (
                "healthy"
                if storage_health.get(
                    "status"
                )
                not in {
                    "unhealthy",
                    "error",
                }
                else "degraded"
            ),
        }