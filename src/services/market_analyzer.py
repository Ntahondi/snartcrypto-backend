"""
src/services/market_analyzer.py

SmartCrypto Market Analyzer v3.1.0

Responsibilities
----------------
- Maintain real-time 1H market data.
- Maintain a processed-feature DataFrame per symbol.
- Receive Binance Futures WebSocket kline updates.
- Update the working candle without generating signals.
- Generate signals only after a candle closes.
- Provide an hourly REST recovery/reconciliation guard.
- Prevent duplicate candle processing.
- Pass signals through PortfolioManager before opening positions.
- Isolate symbol failures so one broken symbol cannot crash the engine.
- Gracefully reconnect failed WebSocket connections.
- Gracefully shut down all background tasks.

Architecture
------------
Binance WebSocket
        |
        v
MarketAnalyzer
        |
        +--> DataCollector
        |
        +--> DataProcessor
        |
        +--> SignalGenerator
        |
        +--> PortfolioManager
        |
        +--> HistoryManager
        |
        +--> TelegramService

Important
---------
The WebSocket and hourly REST checker are NOT independent signal engines.

The WebSocket is the primary real-time source.

The hourly checker is a recovery/reconciliation mechanism used to
recover from missed WebSocket candle-close events.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import random
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
import websockets
from websockets.exceptions import ConnectionClosed, ConnectionClosedError

from src.core.config import Settings
from src.data.collectors import MultiExchangeCollector
from src.data.processors import DataProcessor
from src.services.history_manager import HistoryManager
from src.services.portfolio_manager import PortfolioManager
from src.services.signal_generator import SignalGenerator
from src.services.telegram_service import TelegramService
from src.utils.safe_logger import SafeLogger


logger = SafeLogger.get_logger(__name__)


class MarketAnalyzer:
    """
    Production-oriented real-time market analyzer.

    Core guarantees
    ---------------
    1. A failure in one symbol does not stop other symbols.
    2. A WebSocket failure does not terminate the analyzer.
    3. Only closed 1H candles trigger committee signal generation.
    4. The same candle cannot be processed twice.
    5. PortfolioManager approves a signal before opening a position.
    6. Hourly REST recovery does not duplicate WebSocket signals.
    7. Shutdown cancels all internally-created tasks cleanly.
    """

    VERSION = "3.1.0"
    TIMEFRAME = "1h"

    # WebSocket behaviour
    WS_PING_INTERVAL = 30
    WS_PING_TIMEOUT = 60
    WS_CLOSE_TIMEOUT = 10
    WS_MAX_SIZE = 2 ** 21

    # Reconnection
    WS_INITIAL_BACKOFF = 2.0
    WS_MAX_BACKOFF = 120.0

    # Historical data
    MIN_REQUIRED_ROWS = 50

    # REST hourly recovery
    HOURLY_RECOVERY_SECOND = 8

    # Deduplication memory
    PROCESSED_CANDLE_CACHE_SIZE = 256

    def __init__(self, settings: Settings):
        self.settings = settings

        self.logger = logger
        self.logger.setLevel(logging.INFO)

        self.exchange_type = str(
            getattr(settings, "EXCHANGE_TYPE", "future")
        ).lower()

        self.symbols: List[str] = list(
            getattr(settings, "SYMBOLS", [])
        )

        self.max_historical_data = int(
            getattr(settings, "MAX_HISTORICAL_DATA", 1000)
        )

        # ---------------------------------------------------------
        # Core services
        # ---------------------------------------------------------

        self.data_collector = MultiExchangeCollector(settings)
        self.data_processor = DataProcessor()

        self.signal_generator: Optional[SignalGenerator] = None
        self.portfolio_manager: Optional[PortfolioManager] = None
        self.telegram_service: Optional[TelegramService] = None
        self.history_manager: Optional[HistoryManager] = None
        self.orderbook_monitor: Any = None

        # Optional future component
        self.model_trainer: Any = None

        # ---------------------------------------------------------
        # Runtime state
        # ---------------------------------------------------------

        self.is_running = False
        self.initialized = False

        self.market_data: Dict[str, pd.DataFrame] = {}
        self.latest_signals: Dict[str, Dict] = {}

        self.ws_connections: Dict[str, Any] = {}

        self.last_candle_times: Dict[str, pd.Timestamp] = {}

        # Primary deduplication mechanism.
        self.processed_candles: Dict[str, List[pd.Timestamp]] = {}

        # Prevent two concurrent signal evaluations for the same symbol.
        self._symbol_locks: Dict[str, asyncio.Lock] = {}

        # Global data lock for market_data modifications.
        self._data_lock = asyncio.Lock()

        # Tasks owned by this analyzer.
        self._tasks: Dict[str, asyncio.Task] = {}

        # ---------------------------------------------------------
        # Derivatives cache
        # ---------------------------------------------------------

        self.derivatives_cache: Dict[str, Dict] = {}
        self.last_derivatives_fetch: Dict[str, datetime] = {}

        # ---------------------------------------------------------
        # Health/performance
        # ---------------------------------------------------------

        self.performance_metrics = {
            "accuracy_1h": 0.56,
            "accuracy_4h": 0.60,
            "accuracy_1d": 0.58,

            "total_signals": 0,

            # IMPORTANT:
            # This is not incremented merely because a signal opened.
            # It should only be updated when actual outcomes are known.
            "successful_signals": 0,

            "features_count": 0,
            "has_derivatives": False,
            "has_orderbook": False,

            "websocket_connections": 0,
            "websocket_reconnections": 0,

            "last_signal_time": None,
            "last_data_update": None,

            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        self._last_health_error: Optional[str] = None

    # =============================================================
    # INITIALIZATION
    # =============================================================

    async def initialize(self) -> bool:
        """
        Initialize all services and preload market data.

        Returns
        -------
        bool
            True if the core analyzer initialized successfully.
        """

        if self.initialized:
            self.logger.warning("MarketAnalyzer is already initialized.")
            return True

        self.logger.info(
            f"📊 MarketAnalyzer v{self.VERSION} initializing..."
        )

        # ---------------------------------------------------------
        # 1. Telegram
        # ---------------------------------------------------------

        await self._initialize_telegram()

        # ---------------------------------------------------------
        # 2. History manager
        # ---------------------------------------------------------

        try:
            self.history_manager = HistoryManager()
            self.logger.info("✅ History Manager initialized")
        except Exception as exc:
            self.history_manager = None
            self.logger.warning(
                f"⚠️ History Manager unavailable: {exc}"
            )

        # ---------------------------------------------------------
        # 3. Signal generator
        # ---------------------------------------------------------

        if not await self._initialize_signal_generator():
            self.logger.error(
                "❌ Core initialization failed: SignalGenerator unavailable."
            )
            return False

        # ---------------------------------------------------------
        # 4. Portfolio manager
        # ---------------------------------------------------------

        await self._initialize_portfolio_manager()

        # ---------------------------------------------------------
        # 5. Historical data
        # ---------------------------------------------------------

        await self._load_initial_market_data()

        # ---------------------------------------------------------
        # 6. Validate usable market data
        # ---------------------------------------------------------

        usable_symbols = [
            symbol
            for symbol in self.symbols
            if symbol in self.market_data
            and not self.market_data[symbol].empty
        ]

        if not usable_symbols:
            self.logger.error(
                "❌ No usable market data was loaded."
            )
            return False

        # ---------------------------------------------------------
        # 7. Initial signal evaluation
        # ---------------------------------------------------------

        self.logger.info(
            "🎯 Running initial signal evaluation..."
        )

        for symbol in usable_symbols:
            try:
                signal = await self.generate_signal(symbol)

                if signal:
                    self.logger.info(
                        f"🚀 Initial Signal [{symbol}]: "
                        f"{signal.get('action')} "
                        f"(Confidence: "
                        f"{signal.get('confidence', 0):.2f})"
                    )
                else:
                    self.logger.info(
                        f"⏭️ No initial signal for {symbol}"
                    )

            except Exception as exc:
                self.logger.error(
                    f"❌ Initial signal failure for {symbol}: {exc}",
                    exc_info=True,
                )

        self.initialized = True

        self.logger.info(
            f"✅ MarketAnalyzer v{self.VERSION} initialized successfully "
            f"({len(usable_symbols)}/{len(self.symbols)} symbols ready)"
        )

        return True

    async def _initialize_telegram(self) -> None:
        """Initialize Telegram without making it a hard dependency."""

        try:
            self.telegram_service = TelegramService(self.settings)

            self.logger.info(
                "✅ Telegram service initialized"
            )

        except Exception as exc:
            self.telegram_service = None

            self.logger.warning(
                f"⚠️ Telegram service unavailable: {exc}"
            )

    async def _initialize_signal_generator(self) -> bool:
        """
        Initialize SignalGenerator.

        The method supports both:
            SignalGenerator(settings)
        and future implementations accepting:
            SignalGenerator(settings, trading_profile=...)
        """

        try:
            profile_name = getattr(
                self.settings,
                "TRADING_PROFILE",
                "day_trader",
            )

            # -----------------------------------------------------
            # First try the new profile-aware interface.
            # -----------------------------------------------------

            try:
                self.signal_generator = SignalGenerator(
                    self.settings,
                    trading_profile=profile_name,
                )
            except TypeError:
                # Backward compatibility with current implementation.
                self.signal_generator = SignalGenerator(
                    self.settings
                )

            await self.signal_generator.load_model()

            if not getattr(
                self.signal_generator,
                "model_loaded",
                False,
            ):
                self.logger.error(
                    "❌ SignalGenerator exists but models are not loaded."
                )
                self.signal_generator = None
                return False

            features = getattr(
                self.signal_generator,
                "feature_columns",
                None,
            )

            if features:
                self.performance_metrics["features_count"] = len(
                    features
                )

                self.performance_metrics["has_derivatives"] = any(
                    (
                        "funding" in str(feature).lower()
                        or "oi_" in str(feature).lower()
                    )
                    for feature in features
                )

                self.performance_metrics["has_orderbook"] = any(
                    (
                        "buy_pressure" in str(feature).lower()
                        or "order_imbalance" in str(feature).lower()
                    )
                    for feature in features
                )

            self.logger.info(
                "✅ Signal Generator initialized successfully"
            )

            self.logger.info(
                f"📊 Signal features: "
                f"{self.performance_metrics['features_count']}"
            )

            return True

        except Exception as exc:
            self.signal_generator = None

            self.logger.error(
                f"❌ Failed to initialize SignalGenerator: {exc}",
                exc_info=True,
            )

            return False

    async def _initialize_portfolio_manager(self) -> None:
        """
        Initialize PortfolioManager.

        Portfolio failure does NOT crash market-data processing.
        The analyzer can continue operating in signal-observation mode.
        """

        try:
            profile_name = getattr(
                self.settings,
                "TRADING_PROFILE",
                "day_trader",
            )

            self.portfolio_manager = PortfolioManager(
                initial_capital=self.settings.INITIAL_CAPITAL,
                profile_name=profile_name,
                history_manager=self.history_manager,
                telegram_service=self.telegram_service,
            )

            self.logger.info(
                f"💰 Portfolio Manager initialized "
                f"with profile: {profile_name}"
            )

        except Exception as exc:
            self.portfolio_manager = None

            self.logger.error(
                f"⚠️ Portfolio Manager unavailable: {exc}"
            )

            self.logger.warning(
                "⚠️ MarketAnalyzer will continue without "
                "automatic portfolio execution."
            )

    # =============================================================
    # INITIAL HISTORICAL DATA
    # =============================================================

    async def _load_initial_market_data(self) -> None:
        """Load historical data independently for every symbol."""

        self.logger.info(
            f"📥 Loading historical data for {len(self.symbols)} symbols..."
        )

        tasks = [
            self._load_symbol_history(symbol)
            for symbol in self.symbols
        ]

        if not tasks:
            return

        results = await asyncio.gather(
            *tasks,
            return_exceptions=True,
        )

        successful = 0

        for symbol, result in zip(self.symbols, results):
            if isinstance(result, Exception):
                self.logger.error(
                    f"❌ Historical loading failed for {symbol}: {result}"
                )
            elif result:
                successful += 1

        self.logger.info(
            f"📊 Historical loading complete: "
            f"{successful}/{len(self.symbols)} symbols"
        )

    async def _load_symbol_history(self, symbol: str) -> bool:
        """Load and process historical data for one symbol."""

        try:
            df = await self.data_collector.fetch_data(
                symbol,
                limit=self.max_historical_data,
            )

            if df is None or df.empty:
                self.logger.warning(
                    f"⚠️ No historical data for {symbol}"
                )
                return False

            df = self._prepare_data_dataframe(df)

            featured_df = self.data_processor.engineer_features(df)

            if featured_df is None or featured_df.empty:
                self.logger.warning(
                    f"⚠️ Feature engineering produced no data for {symbol}"
                )
                return False

            featured_df = self._prepare_data_dataframe(
                featured_df
            )

            # Keep only the most recent configured history.
            featured_df = featured_df.tail(
                self.max_historical_data
            ).reset_index(drop=True)

            async with self._data_lock:
                self.market_data[symbol] = featured_df

            self._symbol_locks.setdefault(
                symbol,
                asyncio.Lock(),
            )

            self.performance_metrics[
                "last_data_update"
            ] = datetime.now(timezone.utc).isoformat()

            self.logger.info(
                f"✅ Loaded {len(featured_df)} processed rows "
                f"for {symbol}"
            )

            return True

        except Exception as exc:
            self.logger.error(
                f"❌ Failed loading historical data "
                f"for {symbol}: {exc}",
                exc_info=True,
            )

            return False

    # =============================================================
    # DATA FETCHING
    # =============================================================

    async def fetch_historical_data(
        self,
        symbol: str,
    ) -> pd.DataFrame:
        """Fetch and process historical market data."""

        try:
            df = await self.data_collector.fetch_data(
                symbol,
                limit=self.max_historical_data,
            )

            if df is None or df.empty:
                return pd.DataFrame()

            df = self._prepare_data_dataframe(df)

            featured = self.data_processor.engineer_features(df)

            if featured is None or featured.empty:
                return pd.DataFrame()

            return self._prepare_data_dataframe(
                featured
            ).tail(
                self.max_historical_data
            ).reset_index(drop=True)

        except Exception as exc:
            self.logger.error(
                f"❌ Error fetching data for {symbol}: {exc}"
            )

            return pd.DataFrame()

    async def fetch_current_derivatives(
        self,
        symbol: str,
    ) -> Dict:
        """Fetch current funding/open-interest data with caching."""

        try:
            symbol_ccxt = self._to_ccxt_symbol(symbol)

            now = datetime.now(timezone.utc)

            last_fetch = self.last_derivatives_fetch.get(symbol)

            if last_fetch:
                age = (
                    now - last_fetch
                ).total_seconds()

                if age < 60:
                    return self.derivatives_cache.get(
                        symbol,
                        {},
                    )

            collector = self.data_collector.collectors.get(
                "binance"
            )

            if not collector:
                return {}

            if not hasattr(
                collector,
                "fetch_current_derivatives",
            ):
                return {}

            data = await collector.fetch_current_derivatives(
                symbol_ccxt
            )

            if not isinstance(data, dict):
                return {}

            self.derivatives_cache[symbol] = data
            self.last_derivatives_fetch[symbol] = now

            return data

        except Exception as exc:
            self.logger.warning(
                f"⚠️ Derivatives fetch failed for {symbol}: {exc}"
            )

            return {}

    # =============================================================
    # DATAFRAME MANAGEMENT
    # =============================================================

    @staticmethod
    def _prepare_data_dataframe(
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Normalize DataFrame index/timestamp representation."""

        if df is None or df.empty:
            return pd.DataFrame()

        result = df.copy()

        result = result.loc[
            :,
            ~result.columns.duplicated(),
        ]

        if "timestamp" not in result.columns:
            result = result.reset_index()

        if "timestamp" in result.columns:
            result["timestamp"] = pd.to_datetime(
                result["timestamp"],
                errors="coerce",
            )

            result = result.dropna(
                subset=["timestamp"]
            )

        result = result.reset_index(
            drop=True
        )

        return result

    # =============================================================
    # LIVE CANDLE UPDATE
    # =============================================================

    async def update_market_data_only(
        self,
        symbol: str,
        kline_data: Dict,
    ) -> None:
        """
        Update the currently forming candle.

        IMPORTANT:
        This method NEVER generates a signal.

        Signal generation occurs only after kline_data['x'] == True.
        """

        try:
            async with self._data_lock:

                current_df = self.market_data.get(symbol)

                if current_df is None or current_df.empty:
                    return

                required_keys = (
                    "c",
                    "h",
                    "l",
                    "v",
                )

                if not all(
                    key in kline_data
                    for key in required_keys
                ):
                    return

                # Work on a copy to avoid mutating a DataFrame
                # while another coroutine is reading it.
                updated = current_df.copy()

                idx = updated.index[-1]

                new_close = float(
                    kline_data["c"]
                )

                new_high = float(
                    kline_data["h"]
                )

                new_low = float(
                    kline_data["l"]
                )

                new_volume = float(
                    kline_data["v"]
                )

                updated.at[idx, "close"] = new_close

                updated.at[idx, "high"] = max(
                    float(updated.at[idx, "high"]),
                    new_high,
                )

                updated.at[idx, "low"] = min(
                    float(updated.at[idx, "low"]),
                    new_low,
                )

                updated.at[idx, "volume"] = new_volume

                self.market_data[symbol] = updated

        except Exception as exc:
            self.logger.error(
                f"❌ Tick update failed for {symbol}: {exc}"
            )

    # =============================================================
    # CLOSED CANDLE PROCESSING
    # =============================================================

    async def process_market_data(
        self,
        symbol: str,
        kline_data: Dict,
    ) -> Optional[Dict]:
        """
        Process a Binance kline event.

        Only a CLOSED candle can reach signal generation.
        """

        try:
            if not kline_data.get("x", False):
                await self.update_market_data_only(
                    symbol,
                    kline_data,
                )
                return None

            candle_timestamp = self._extract_candle_timestamp(
                kline_data
            )

            if candle_timestamp is None:
                self.logger.warning(
                    f"⚠️ Invalid candle timestamp for {symbol}"
                )
                return None

            # -----------------------------------------------------
            # Duplicate protection
            # -----------------------------------------------------

            if self._candle_already_processed(
                symbol,
                candle_timestamp,
            ):
                self.logger.debug(
                    f"⏭️ Candle already processed: "
                    f"{symbol} {candle_timestamp}"
                )
                return None

            self.logger.info(
                f"🕒 {symbol} 1H candle closed at "
                f"{candle_timestamp.isoformat()}"
            )

            # -----------------------------------------------------
            # Convert Binance kline to normalized row
            # -----------------------------------------------------

            new_point = {
                "timestamp": candle_timestamp,
                "open": float(kline_data["o"]),
                "high": float(kline_data["h"]),
                "low": float(kline_data["l"]),
                "close": float(kline_data["c"]),
                "volume": float(kline_data["v"]),
            }

            # -----------------------------------------------------
            # Rebuild features from the newly closed candle.
            # -----------------------------------------------------

            await self.update_market_data(
                symbol,
                new_point,
            )

            # Mark ONLY after successful market-data update.
            self._mark_candle_processed(
                symbol,
                candle_timestamp,
            )

            # -----------------------------------------------------
            # Generate one signal for this candle.
            # -----------------------------------------------------

            signal = await self.generate_signal(
                symbol
            )

            if signal:
                await self._handle_signal(
                    signal
                )

            return signal

        except Exception as exc:
            self.logger.error(
                f"❌ Error processing closed candle "
                f"for {symbol}: {exc}",
                exc_info=True,
            )

            return None

    async def update_market_data(
        self,
        symbol: str,
        new_point: Dict,
    ) -> None:
        """
        Append a closed candle and regenerate engineered features.
        """

        try:
            async with self._data_lock:

                current_df = self._prepare_data_dataframe(
                    self.market_data.get(
                        symbol,
                        pd.DataFrame(),
                    )
                )

                new_row = pd.DataFrame(
                    [new_point]
                )

                combined = pd.concat(
                    [
                        current_df,
                        new_row,
                    ],
                    ignore_index=True,
                )

                if "timestamp" in combined.columns:

                    combined["timestamp"] = pd.to_datetime(
                        combined["timestamp"],
                        errors="coerce",
                    )

                    combined = combined.dropna(
                        subset=["timestamp"]
                    )

                    combined = combined.drop_duplicates(
                        subset=["timestamp"],
                        keep="last",
                    )

                    combined = combined.sort_values(
                        "timestamp"
                    ).reset_index(
                        drop=True
                    )

                featured = self.data_processor.engineer_features(
                    combined
                )

                if featured is None or featured.empty:
                    raise ValueError(
                        f"Feature engineering returned no data for {symbol}"
                    )

                featured = self._prepare_data_dataframe(
                    featured
                )

                featured = featured.tail(
                    self.max_historical_data
                ).reset_index(
                    drop=True
                )

                self.market_data[symbol] = featured

                self.last_candle_times[symbol] = (
                    pd.Timestamp(
                        new_point["timestamp"]
                    )
                )

                self.performance_metrics[
                    "last_data_update"
                ] = datetime.now(
                    timezone.utc
                ).isoformat()

        except Exception as exc:
            self.logger.error(
                f"❌ Error updating market data "
                f"for {symbol}: {exc}",
                exc_info=True,
            )

            raise

    # =============================================================
    # SIGNAL GENERATION
    # =============================================================

    async def generate_signal(
        self,
        symbol: str,
    ) -> Optional[Dict]:
        """
        Generate exactly one committee decision for the current
        closed-candle market state.
        """

        lock = self._symbol_locks.setdefault(
            symbol,
            asyncio.Lock(),
        )

        async with lock:

            try:
                current_data = self.market_data.get(
                    symbol
                )

                if (
                    current_data is None
                    or current_data.empty
                    or len(current_data)
                    < self.MIN_REQUIRED_ROWS
                ):
                    self.logger.warning(
                        f"⚠️ Insufficient data for {symbol}: "
                        f"{0 if current_data is None else len(current_data)} rows"
                    )
                    return None

                current_data_clean = (
                    current_data.copy()
                )

                current_data_clean = (
                    current_data_clean.loc[
                        :,
                        ~current_data_clean.columns.duplicated(),
                    ]
                )

                if "close" not in current_data_clean.columns:
                    self.logger.error(
                        f"❌ Close column missing for {symbol}"
                    )
                    return None

                current_price = float(
                    current_data_clean[
                        "close"
                    ].iloc[-1]
                )

                if not (
                    current_price > 0
                ):
                    self.logger.warning(
                        f"⚠️ Invalid current price for {symbol}: "
                        f"{current_price}"
                    )
                    return None

                if not self.signal_generator:
                    self.logger.warning(
                        "⚠️ SignalGenerator unavailable."
                    )
                    return None

                signal = await self.signal_generator.generate_signal(
                    symbol,
                    current_data_clean,
                    current_price,
                )

                if not signal:
                    return None

                # -------------------------------------------------
                # Validate signal before accepting it.
                # -------------------------------------------------

                if signal.get("action") not in {
                    "BUY",
                    "SELL",
                }:
                    self.logger.warning(
                        f"⚠️ Invalid signal action for {symbol}: "
                        f"{signal.get('action')}"
                    )
                    return None

                signal["analyzer_version"] = self.VERSION

                signal["generated_at"] = (
                    datetime.now(
                        timezone.utc
                    ).isoformat()
                )

                self.performance_metrics[
                    "total_signals"
                ] += 1

                self.performance_metrics[
                    "last_signal_time"
                ] = datetime.now(
                    timezone.utc
                ).isoformat()

                # -------------------------------------------------
                # Attach derivatives.
                # -------------------------------------------------

                try:
                    derivatives = (
                        await self.fetch_current_derivatives(
                            symbol
                        )
                    )

                    if derivatives:
                        signal["derivatives"] = {
                            "funding_rate": derivatives.get(
                                "funding_rate",
                                0.0,
                            ),
                            "open_interest": derivatives.get(
                                "open_interest",
                                0.0,
                            ),
                            "open_interest_usd": derivatives.get(
                                "open_interest_usd",
                                0.0,
                            ),
                        }

                except Exception as exc:
                    self.logger.warning(
                        f"⚠️ Could not attach derivatives "
                        f"for {symbol}: {exc}"
                    )

                # -------------------------------------------------
                # Attach order-book information.
                # -------------------------------------------------

                try:
                    if self.orderbook_monitor:
                        ob_data = (
                            self.orderbook_monitor.get_imbalance(
                                symbol
                            )
                        )

                        if ob_data:
                            signal["orderbook"] = ob_data

                except Exception as exc:
                    self.logger.warning(
                        f"⚠️ Order-book data unavailable "
                        f"for {symbol}: {exc}"
                    )

                self.latest_signals[
                    symbol
                ] = signal

                self.logger.info(
                    f"🎯 SIGNAL [{symbol}] "
                    f"{signal.get('action')} | "
                    f"Confidence: "
                    f"{signal.get('confidence', 0):.1%} | "
                    f"Strength: "
                    f"{signal.get('signal_strength', 0):.1%}"
                )

                # -------------------------------------------------
                # Persist signal.
                # -------------------------------------------------

                await self._save_signal_safely(
                    signal
                )

                return signal

            except Exception as exc:
                self.logger.error(
                    f"❌ Signal generation failed "
                    f"for {symbol}: {exc}",
                    exc_info=True,
                )

                return None

    async def _handle_signal(
        self,
        signal: Dict,
    ) -> None:
        """
        Process an accepted signal.

        Order of operations:

            Signal
               ↓
            Portfolio risk check
               ↓
            Approved?
             /   \
           NO     YES
           ↓       ↓
         Skip    Open
        """

        symbol = signal.get(
            "symbol",
            "UNKNOWN",
        )

        try:
            # -----------------------------------------------------
            # 1. Always broadcast valid trading signals to Telegram VIP Channel
            # -----------------------------------------------------
            if signal.get("action") in ("BUY", "SELL"):
                await self._broadcast_signal_safely(signal)

            # -----------------------------------------------------
            # 2. Portfolio risk validation & execution
            # -----------------------------------------------------
            if not self.portfolio_manager:
                self.logger.warning(
                    f"⚠️ Signal generated for {symbol}, "
                    "but PortfolioManager is unavailable. "
                    "No position opened."
                )
                return

            # -----------------------------------------------------
            # Ask portfolio manager BEFORE opening.
            # -----------------------------------------------------

            should_trade = False
            reason = "Unknown"

            if hasattr(
                self.portfolio_manager,
                "should_open_position",
            ):
                decision = (
                    self.portfolio_manager.should_open_position(
                        signal
                    )
                )

                if (
                    isinstance(decision, tuple)
                    and len(decision) >= 2
                ):
                    should_trade = bool(
                        decision[0]
                    )
                    reason = str(
                        decision[1]
                    )
                else:
                    should_trade = bool(
                        decision
                    )
                    reason = (
                        "Portfolio approved"
                        if should_trade
                        else "Portfolio rejected"
                    )

            else:
                # Do NOT silently bypass risk controls.
                self.logger.error(
                    "❌ PortfolioManager does not expose "
                    "should_open_position(). Position will NOT open."
                )
                return

            if not should_trade:
                self.logger.info(
                    f"⏭️ Portfolio rejected {symbol}: "
                    f"{reason}"
                )
                return

            # -----------------------------------------------------
            # Open only after approval.
            # -----------------------------------------------------

            position = (
                self.portfolio_manager.open_position(
                    signal
                )
            )

            if position:
                action = getattr(
                    position,
                    "action",
                    signal.get("action"),
                )

                entry_price = getattr(
                    position,
                    "entry_price",
                    signal.get("price", 0.0),
                )

                self.logger.info(
                    f"💰 PORTFOLIO POSITION OPENED | "
                    f"{symbol} | "
                    f"{action} | "
                    f"Entry: ${float(entry_price):.6f}"
                )

                await self._broadcast_signal_safely(
                    signal
                )

            else:
                self.logger.info(
                    f"⏭️ Portfolio did not open position "
                    f"for {symbol} after approval."
                )

        except Exception as exc:
            self.logger.error(
                f"❌ Signal handling failed "
                f"for {symbol}: {exc}",
                exc_info=True,
            )

    # =============================================================
    # TELEGRAM / HISTORY SAFE OPERATIONS
    # =============================================================

    async def _broadcast_signal_safely(
        self,
        signal: Dict,
    ) -> None:
        """Broadcast Telegram signal without affecting trading."""

        if not self.telegram_service:
            return

        try:
            enabled = getattr(
                self.telegram_service,
                "enable_telegram",
                False,
            )

            if not enabled:
                return

            result = (
                self.telegram_service.broadcast_signal(
                    signal
                )
            )

            if asyncio.iscoroutine(result):
                await result

        except Exception as exc:
            self.logger.warning(
                f"⚠️ Telegram broadcast failed: {exc}"
            )

    async def _save_signal_safely(
        self,
        signal: Dict,
    ) -> None:
        """Persist signal without making storage a hard dependency."""

        if not self.history_manager:
            return

        try:
            result = (
                self.history_manager.save_signal(
                    signal,
                    outcome="OPEN",
                )
            )

            if asyncio.iscoroutine(result):
                await result

        except Exception as exc:
            self.logger.warning(
                f"⚠️ Could not save signal history: {exc}"
            )

    # =============================================================
    # WEBSOCKET
    # =============================================================

    async def _start_symbol_websocket(
        self,
        symbol: str,
    ) -> None:
        """
        Maintain a persistent WebSocket connection for one symbol.

        Failure of this task never propagates to other symbols.
        """

        stream_name = (
            f"{symbol.lower()}@kline_{self.TIMEFRAME}"
        )

        if self.exchange_type == "future":
            base_url = "wss://fstream.binance.com/ws"
        else:
            base_url = "wss://stream.binance.com:9443/ws"

        url = (
            f"{base_url}/{stream_name}"
        )

        backoff = self.WS_INITIAL_BACKOFF

        self.logger.info(
            f"🔗 Starting WebSocket worker for {symbol}"
        )

        await asyncio.sleep(
            random.uniform(
                0.5,
                2.5,
            )
        )

        while self.is_running:

            try:
                self.logger.info(
                    f"🔌 Connecting WebSocket: {symbol}"
                )

                async with websockets.connect(
                    url,
                    ping_interval=self.WS_PING_INTERVAL,
                    ping_timeout=self.WS_PING_TIMEOUT,
                    close_timeout=self.WS_CLOSE_TIMEOUT,
                    max_size=self.WS_MAX_SIZE,
                ) as websocket:

                    self.ws_connections[
                        symbol
                    ] = websocket

                    self.performance_metrics[
                        "websocket_connections"
                    ] = sum(
                        1
                        for ws in self.ws_connections.values()
                        if ws is not None
                    )

                    self.logger.info(
                        f"✅ WebSocket connected: {symbol}"
                    )

                    # Successful connection resets backoff.
                    backoff = self.WS_INITIAL_BACKOFF

                    async for raw_message in websocket:

                        if not self.is_running:
                            break

                        try:
                            data = json.loads(
                                raw_message
                            )

                            kline = data.get("k")

                            if not kline:
                                continue

                            if kline.get("x", False):
                                await self.process_market_data(
                                    symbol,
                                    kline,
                                )
                            else:
                                await self.update_market_data_only(
                                    symbol,
                                    kline,
                                )

                        except json.JSONDecodeError:
                            self.logger.warning(
                                f"⚠️ Invalid WebSocket JSON "
                                f"for {symbol}"
                            )

                        except asyncio.CancelledError:
                            raise

                        except Exception as exc:
                            self.logger.error(
                                f"❌ Message processing error "
                                f"for {symbol}: {exc}",
                                exc_info=True,
                            )

            except asyncio.CancelledError:
                self.logger.info(
                    f"🛑 WebSocket worker cancelled: {symbol}"
                )
                raise

            except (
                asyncio.TimeoutError,
                ConnectionClosed,
                ConnectionClosedError,
            ) as exc:

                self.logger.warning(
                    f"🔄 WebSocket disconnected "
                    f"for {symbol}: {exc}"
                )

                self.performance_metrics[
                    "websocket_reconnections"
                ] += 1

            except Exception as exc:

                self.logger.error(
                    f"❌ WebSocket error for {symbol}: {exc}"
                )

                self.performance_metrics[
                    "websocket_reconnections"
                ] += 1

            finally:

                current_ws = (
                    self.ws_connections.get(
                        symbol
                    )
                )

                if current_ws is not None:
                    self.ws_connections[
                        symbol
                    ] = None

                self.performance_metrics[
                    "websocket_connections"
                ] = sum(
                    1
                    for ws in self.ws_connections.values()
                    if ws is not None
                )

            if self.is_running:

                jitter = random.uniform(
                    0.0,
                    2.0,
                )

                delay = min(
                    backoff + jitter,
                    self.WS_MAX_BACKOFF,
                )

                self.logger.info(
                    f"🔄 Reconnecting {symbol} "
                    f"in {delay:.1f}s..."
                )

                await asyncio.sleep(
                    delay
                )

                backoff = min(
                    backoff * 2,
                    self.WS_MAX_BACKOFF,
                )

    # =============================================================
    # HOURLY RECOVERY GUARD
    # =============================================================

    async def start_hourly_clock_checker(
        self,
    ) -> None:
        """
        Hourly REST reconciliation guard.

        This does NOT blindly generate another signal.

        It refreshes the latest historical data and checks whether
        the latest closed candle has already been processed.

        If the WebSocket missed that candle, it processes it once.
        """

        self.logger.info(
            "⏰ Hourly recovery guard started"
        )

        while self.is_running:

            try:
                now = datetime.now()

                next_hour = (
                    now + timedelta(hours=1)
                ).replace(
                    minute=0,
                    second=self.HOURLY_RECOVERY_SECOND,
                    microsecond=0,
                )

                wait_seconds = max(
                    (
                        next_hour - now
                    ).total_seconds(),
                    1.0,
                )

                self.logger.info(
                    f"⏰ Next recovery check in "
                    f"{wait_seconds / 60:.1f} minutes"
                )

                await asyncio.sleep(
                    wait_seconds
                )

                if not self.is_running:
                    break

                self.logger.info(
                    "🛡️ Hourly REST reconciliation started"
                )

                tasks = [
                    self._recover_symbol(symbol)
                    for symbol in self.symbols
                ]

                await asyncio.gather(
                    *tasks,
                    return_exceptions=True,
                )

            except asyncio.CancelledError:
                raise

            except Exception as exc:
                self.logger.error(
                    f"❌ Hourly recovery loop error: {exc}",
                    exc_info=True,
                )

                # Prevent a tight failure loop.
                await asyncio.sleep(
                    30
                )

    async def _recover_symbol(
        self,
        symbol: str,
    ) -> None:
        """
        Recover one symbol from REST if WebSocket missed data.
        """

        try:
            fresh_df = await self.fetch_historical_data(
                symbol
            )

            if fresh_df is None or fresh_df.empty:
                self.logger.warning(
                    f"⚠️ Recovery returned no data for {symbol}"
                )
                return

            # -----------------------------------------------------
            # The REST collector should already return closed
            # historical candles. We identify the latest candle.
            # -----------------------------------------------------

            if "timestamp" not in fresh_df.columns:
                return

            latest_timestamp = pd.Timestamp(
                fresh_df["timestamp"].iloc[-1]
            )

            if self._candle_already_processed(
                symbol,
                latest_timestamp,
            ):
                self.logger.debug(
                    f"✅ {symbol} latest candle already processed"
                )

                # Refresh data anyway, because this also repairs
                # any missed WebSocket updates.
                async with self._data_lock:
                    self.market_data[
                        symbol
                    ] = fresh_df.tail(
                        self.max_historical_data
                    ).reset_index(
                        drop=True
                    )

                return

            # -----------------------------------------------------
            # Replace the local dataset with the fresh REST dataset.
            # -----------------------------------------------------

            async with self._data_lock:
                self.market_data[
                    symbol
                ] = fresh_df.tail(
                    self.max_historical_data
                ).reset_index(
                    drop=True
                )

            self.logger.warning(
                f"🛡️ REST recovery detected unprocessed candle "
                f"for {symbol}: {latest_timestamp}"
            )

            # -----------------------------------------------------
            # Generate only if this candle was genuinely missed.
            # -----------------------------------------------------

            if len(fresh_df) < self.MIN_REQUIRED_ROWS:
                return

            await self._process_recovered_candle(
                symbol,
                latest_timestamp,
            )

        except Exception as exc:
            self.logger.error(
                f"❌ Recovery failed for {symbol}: {exc}",
                exc_info=True,
            )

    async def _process_recovered_candle(
        self,
        symbol: str,
        candle_timestamp: pd.Timestamp,
    ) -> None:
        """
        Generate a signal for a candle missed by WebSocket.

        Uses the already-refreshed market_data.
        """

        if self._candle_already_processed(
            symbol,
            candle_timestamp,
        ):
            return

        self._mark_candle_processed(
            symbol,
            candle_timestamp,
        )

        signal = await self.generate_signal(
            symbol
        )

        if signal:
            await self._handle_signal(
                signal
            )

            self.logger.info(
                f"🛡️ RECOVERED SIGNAL [{symbol}] "
                f"{signal.get('action')}"
            )

    # =============================================================
    # ORDER BOOK MONITOR
    # =============================================================

    async def start_orderbook_monitor(
        self,
    ) -> None:
        """Start optional order-book monitoring."""

        try:
            from src.services.orderbook_monitor import (
                OrderBookMonitor,
            )

            self.orderbook_monitor = (
                OrderBookMonitor(
                    self.settings
                )
            )

            task = asyncio.create_task(
                self.orderbook_monitor.start_monitoring(),
                name="orderbook-monitor",
            )

            self._tasks[
                "orderbook"
            ] = task

            self.logger.info(
                "📊 Order Book Monitor started"
            )

        except ImportError:
            self.logger.warning(
                "⚠️ Order Book Monitor module unavailable. "
                "Continuing without it."
            )

        except Exception as exc:
            self.orderbook_monitor = None

            self.logger.warning(
                f"⚠️ Order Book Monitor unavailable: {exc}"
            )

    # =============================================================
    # REAL-TIME ENGINE
    # =============================================================

    async def start_real_time_analysis(
        self,
    ) -> None:
        """
        Start the complete real-time analysis engine.

        Each symbol receives an independent WebSocket task.
        """

        if not self.initialized:
            initialized = await self.initialize()

            if not initialized:
                raise RuntimeError(
                    "MarketAnalyzer initialization failed."
                )

        if self.is_running:
            self.logger.warning(
                "⚠️ Real-time analysis is already running."
            )
            return

        self.is_running = True

        self.logger.info(
            f"🚀 Starting MarketAnalyzer v{self.VERSION} "
            f"real-time engine..."
        )

        # ---------------------------------------------------------
        # Optional order-book monitor
        # ---------------------------------------------------------

        await self.start_orderbook_monitor()

        # ---------------------------------------------------------
        # Hourly recovery guard
        # ---------------------------------------------------------

        hourly_task = asyncio.create_task(
            self.start_hourly_clock_checker(),
            name="hourly-recovery-guard",
        )

        self._tasks[
            "hourly_guard"
        ] = hourly_task

        # ---------------------------------------------------------
        # Symbol WebSocket workers
        # ---------------------------------------------------------

        for symbol in self.symbols:

            task = asyncio.create_task(
                self._start_symbol_websocket(
                    symbol
                ),
                name=f"websocket-{symbol}",
            )

            self._tasks[
                f"websocket:{symbol}"
            ] = task

        self.logger.info(
            f"🟢 Real-time engine online: "
            f"{len(self.symbols)} symbol workers"
        )

        # ---------------------------------------------------------
        # Supervisor loop.
        #
        # IMPORTANT:
        # We do NOT await gather() forever without supervision.
        # Failed symbol tasks are detected and restarted.
        # ---------------------------------------------------------

        try:

            while self.is_running:

                await asyncio.sleep(
                    10
                )

                await self._supervise_tasks()

        except asyncio.CancelledError:
            raise

        finally:
            await self.cleanup()

    async def _supervise_tasks(self) -> None:
        """
        Detect unexpectedly terminated internal tasks.

        A failed symbol worker is restarted independently.
        """

        if not self.is_running:
            return

        # ---------------------------------------------------------
        # Symbol workers
        # ---------------------------------------------------------

        for symbol in self.symbols:

            task_name = (
                f"websocket:{symbol}"
            )

            task = self._tasks.get(
                task_name
            )

            if task is None:
                self.logger.warning(
                    f"🛡️ Missing WebSocket task "
                    f"for {symbol}; restarting."
                )

                self._restart_symbol_task(
                    symbol
                )

                continue

            if task.done():

                if task.cancelled():
                    self.logger.warning(
                        f"🛡️ WebSocket task cancelled "
                        f"unexpectedly: {symbol}"
                    )
                else:
                    exception = task.exception()

                    if exception:
                        self.logger.error(
                            f"❌ WebSocket worker crashed "
                            f"for {symbol}: {exception}"
                        )
                    else:
                        self.logger.warning(
                            f"⚠️ WebSocket worker stopped "
                            f"for {symbol}"
                        )

                self._restart_symbol_task(
                    symbol
                )

        # ---------------------------------------------------------
        # Hourly guard
        # ---------------------------------------------------------

        hourly = self._tasks.get(
            "hourly_guard"
        )

        if (
            hourly is not None
            and hourly.done()
            and self.is_running
        ):
            if not hourly.cancelled():

                with contextlib.suppress(
                    Exception
                ):
                    exc = hourly.exception()

                    if exc:
                        self.logger.error(
                            f"❌ Hourly guard stopped: {exc}"
                        )

            self.logger.warning(
                "🛡️ Restarting hourly recovery guard."
            )

            self._tasks[
                "hourly_guard"
            ] = asyncio.create_task(
                self.start_hourly_clock_checker(),
                name="hourly-recovery-guard",
            )

    def _restart_symbol_task(
        self,
        symbol: str,
    ) -> None:
        """Restart a single WebSocket worker."""

        if not self.is_running:
            return

        task_name = (
            f"websocket:{symbol}"
        )

        old_task = self._tasks.get(
            task_name
        )

        if (
            old_task is not None
            and not old_task.done()
        ):
            return

        self._tasks[
            task_name
        ] = asyncio.create_task(
            self._start_symbol_websocket(
                symbol
            ),
            name=task_name,
        )

    # =============================================================
    # DEDUPLICATION
    # =============================================================

    def _candle_already_processed(
        self,
        symbol: str,
        timestamp: pd.Timestamp,
    ) -> bool:
        """Return True when this candle was already processed."""

        timestamp = pd.Timestamp(
            timestamp
        )

        processed = self.processed_candles.get(
            symbol,
            [],
        )

        return timestamp in processed

    def _mark_candle_processed(
        self,
        symbol: str,
        timestamp: pd.Timestamp,
    ) -> None:
        """Record a processed candle with bounded memory."""

        timestamp = pd.Timestamp(
            timestamp
        )

        processed = self.processed_candles.setdefault(
            symbol,
            [],
        )

        if timestamp in processed:
            return

        processed.append(
            timestamp
        )

        if len(processed) > self.PROCESSED_CANDLE_CACHE_SIZE:
            del processed[
                : len(processed)
                - self.PROCESSED_CANDLE_CACHE_SIZE
            ]

    # =============================================================
    # HELPERS
    # =============================================================

    @staticmethod
    def _extract_candle_timestamp(
        kline_data: Dict,
    ) -> Optional[pd.Timestamp]:
        """Extract Binance candle open timestamp."""

        try:
            timestamp_ms = kline_data.get(
                "t"
            )

            if timestamp_ms is None:
                return None

            return pd.Timestamp(
                pd.to_datetime(
                    int(timestamp_ms),
                    unit="ms",
                    utc=True,
                )
            )

        except Exception:
            return None

    @staticmethod
    def _to_ccxt_symbol(
        symbol: str,
    ) -> str:
        """
        Convert Binance symbol such as BTCUSDT to CCXT futures
        symbol BTC/USDT:USDT.
        """

        normalized = (
            str(symbol)
            .upper()
            .strip()
        )

        if normalized.endswith(
            ":USDT"
        ):
            return normalized

        if normalized.endswith(
            "USDT"
        ):
            base = normalized[
                :-4
            ]

            return (
                f"{base}/USDT:USDT"
            )

        if "/" in normalized:
            return normalized

        return normalized

    # =============================================================
    # PUBLIC GETTERS
    # =============================================================

    def get_signal(
        self,
        symbol: str,
    ) -> Optional[Dict]:
        return self.latest_signals.get(
            symbol
        )

    def get_latest_signals(
        self,
    ) -> Dict:
        return self.latest_signals.copy()

    def get_market_data(
        self,
        symbol: str,
    ) -> Optional[pd.DataFrame]:
        df = self.market_data.get(
            symbol
        )

        if df is None:
            return None

        return df.copy()

    def is_healthy(self) -> bool:
        """
        Basic health state.

        The analyzer can be healthy even when an individual
        WebSocket is reconnecting, provided there is usable market
        data and the engine is running.
        """

        if not self.is_running:
            return False

        if not self.market_data:
            return False

        return any(
            df is not None
            and not df.empty
            for df in self.market_data.values()
        )

    def health_check(self) -> Dict:
        """Return detailed analyzer health information."""

        websocket_status = {
            symbol: (
                "connected"
                if self.ws_connections.get(symbol)
                else "disconnected"
            )
            for symbol in self.symbols
        }

        task_status = {
            name: (
                "cancelled"
                if task.cancelled()
                else "done"
                if task.done()
                else "running"
            )
            for name, task in self._tasks.items()
        }

        return {
            "service": "MarketAnalyzer",
            "version": self.VERSION,
            "running": self.is_running,
            "initialized": self.initialized,
            "healthy": self.is_healthy(),

            "symbols": self.symbols,

            "market_data_symbols": list(
                self.market_data.keys()
            ),

            "websocket_status": websocket_status,

            "task_status": task_status,

            "signal_generator_loaded": bool(
                self.signal_generator
                and getattr(
                    self.signal_generator,
                    "model_loaded",
                    False,
                )
            ),

            "portfolio_manager_loaded": (
                self.portfolio_manager is not None
            ),

            "telegram_available": (
                self.telegram_service is not None
            ),

            "orderbook_monitor_available": (
                self.orderbook_monitor is not None
            ),

            "performance": self.performance_metrics.copy(),

            "last_error": self._last_health_error,
        }

    # =============================================================
    # CLEANUP
    # =============================================================

    async def cleanup(self) -> None:
        """
        Gracefully shut down all WebSockets, background tasks,
        order-book monitors and exchange clients.
        """

        if not self.is_running and not self._tasks:
            return

        self.logger.info(
            "🛑 Shutting down MarketAnalyzer..."
        )

        self.is_running = False

        # ---------------------------------------------------------
        # Close WebSocket connections first.
        # ---------------------------------------------------------

        websocket_values = list(
            self.ws_connections.items()
        )

        for symbol, websocket in websocket_values:

            if websocket is None:
                continue

            try:
                await websocket.close()

            except Exception as exc:
                self.logger.debug(
                    f"WebSocket close error "
                    f"for {symbol}: {exc}"
                )

        self.ws_connections.clear()

        # ---------------------------------------------------------
        # Cancel internal tasks.
        # ---------------------------------------------------------

        current_task = asyncio.current_task()

        tasks_to_cancel = []

        for name, task in list(
            self._tasks.items()
        ):

            if task is None:
                continue

            if task is current_task:
                continue

            if task.done():
                continue

            task.cancel()

            tasks_to_cancel.append(
                task
            )

        if tasks_to_cancel:

            results = await asyncio.gather(
                *tasks_to_cancel,
                return_exceptions=True,
            )

            for result in results:

                if isinstance(
                    result,
                    Exception,
                ) and not isinstance(
                    result,
                    asyncio.CancelledError,
                ):
                    self.logger.debug(
                        f"Background task shutdown result: "
                        f"{result}"
                    )

        self._tasks.clear()

        # ---------------------------------------------------------
        # Optional order-book monitor.
        # ---------------------------------------------------------

        if self.orderbook_monitor:

            try:

                stop_method = getattr(
                    self.orderbook_monitor,
                    "stop_monitoring",
                    None,
                )

                if stop_method:

                    result = stop_method()

                    if asyncio.iscoroutine(
                        result
                    ):
                        await result

            except Exception as exc:
                self.logger.debug(
                    f"OrderBookMonitor shutdown error: {exc}"
                )

            finally:
                self.orderbook_monitor = None

        # ---------------------------------------------------------
        # Close CCXT exchange.
        # ---------------------------------------------------------

        try:

            collector = (
                self.data_collector.collectors.get(
                    "binance"
                )
            )

            exchange = getattr(
                collector,
                "exchange",
                None,
            )

            if exchange:

                close_method = getattr(
                    exchange,
                    "close",
                    None,
                )

                if close_method:

                    result = close_method()

                    if asyncio.iscoroutine(
                        result
                    ):
                        await result

        except Exception as exc:
            self.logger.debug(
                f"Exchange shutdown error: {exc}"
            )

        # ---------------------------------------------------------
        # Close collector aiohttp session if present.
        # ---------------------------------------------------------

        try:

            collector = (
                self.data_collector.collectors.get(
                    "binance"
                )
            )

            session = getattr(
                collector,
                "session",
                None,
            )

            if session and not session.closed:
                await session.close()

        except Exception as exc:
            self.logger.debug(
                f"HTTP session shutdown error: {exc}"
            )

        self.logger.info(
            "✅ MarketAnalyzer shutdown complete."
        )

    # =============================================================
    # PERFORMANCE
    # =============================================================

    def get_performance_stats(
        self,
    ) -> Dict:
        """Return analyzer performance statistics."""

        total = int(
            self.performance_metrics.get(
                "total_signals",
                0,
            )
        )

        successful = int(
            self.performance_metrics.get(
                "successful_signals",
                0,
            )
        )

        return {
            "version": self.VERSION,

            "total_signals": total,

            "successful_signals": successful,

            "success_rate": (
                successful / total
                if total > 0
                else 0.0
            ),

            "accuracy_1h": self.performance_metrics[
                "accuracy_1h"
            ],

            "accuracy_4h": self.performance_metrics[
                "accuracy_4h"
            ],

            "accuracy_1d": self.performance_metrics[
                "accuracy_1d"
            ],

            "features_count": self.performance_metrics[
                "features_count"
            ],

            "has_derivatives": self.performance_metrics[
                "has_derivatives"
            ],

            "has_orderbook": self.performance_metrics[
                "has_orderbook"
            ],

            "websocket_connections": self.performance_metrics[
                "websocket_connections"
            ],

            "websocket_reconnections": self.performance_metrics[
                "websocket_reconnections"
            ],

            "last_signal_time": self.performance_metrics[
                "last_signal_time"
            ],

            "last_data_update": self.performance_metrics[
                "last_data_update"
            ],
        }

    def update_performance_metrics(
        self,
        success: bool,
    ) -> None:
        """
        Update actual signal outcome.

        This should be called by the history/outcome system after
        the trade's outcome is known.

        It is intentionally NOT called when a signal is generated.
        """

        if success:
            self.performance_metrics[
                "successful_signals"
            ] += 1

        self.performance_metrics[
            "timestamp"
        ] = datetime.now(
            timezone.utc
        ).isoformat()

    def get_symbol_performance(
        self,
        symbol: str,
        days: int = 30,
    ) -> Dict:

        if not self.history_manager:
            return {}

        try:
            return (
                self.history_manager.get_symbol_performance(
                    symbol,
                    days,
                )
            )

        except Exception as exc:
            self.logger.warning(
                f"⚠️ Symbol performance unavailable "
                f"for {symbol}: {exc}"
            )

            return {}

    def get_recent_signals(
        self,
        symbol: Optional[str] = None,
        hours: int = 24,
        limit: int = 50,
    ) -> List[Dict]:

        if not self.history_manager:
            return []

        try:
            return (
                self.history_manager.get_recent_signals(
                    symbol,
                    hours,
                    limit,
                )
            )

        except Exception as exc:
            self.logger.warning(
                f"⚠️ Recent signals unavailable: {exc}"
            )

            return []