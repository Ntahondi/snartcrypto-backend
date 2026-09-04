# src/services/real_trade_executor.py

"""
SmartCrypto AI - Real Trade Executor
Version: 3.2.0

Responsibilities
----------------
- Execute real futures positions on Binance USD-M Futures.
- Execute real futures positions on Bybit USDT Linear Futures.
- Optionally execute on both exchanges simultaneously.
- Configure isolated/cross margin and leverage.
- Validate exchange markets and order quantities.
- Execute market entries and exits.
- Verify actual fills.
- Install independent protective Stop Loss / Take Profit orders.
- Track exchange-side orders.
- Prevent duplicate real entries.
- Handle partial fills and exchange failures.
- Expose balances and exchange positions.
- Provide health checks.
- Support paper execution helpers.
- Close all CCXT connections safely.

Important
---------
This executor is intentionally an execution layer.

PortfolioManager remains the FINAL risk gate.

The flow is:

    MarketAnalyzer
          |
          v
    PortfolioManager
          |
          v
    RealTradeExecutor
          |
          +------ Binance Futures
          |
          +------ Bybit Linear Futures

When dual-exchange execution is enabled, the same portfolio
quantity is executed independently on each enabled exchange.
That means a quantity of 0.01 BTC results in:

    Binance -> 0.01 BTC
    Bybit   -> 0.01 BTC

This is intentional and must be accounted for in capital/risk
planning.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import ccxt.async_support as ccxt

from src.core.config import Settings, get_settings
from src.utils.safe_logger import SafeLogger


logger = SafeLogger.get_logger(__name__)


class RealTradeExecutor:
    """
    Production-oriented asynchronous futures executor.

    Supported exchanges:
        - Binance USD-M Futures
        - Bybit USDT Linear Futures

    The executor is intentionally defensive because it sits
    directly between PortfolioManager and real money exchanges.
    """

    VERSION = "3.2.0"

    SUPPORTED_EXCHANGES = (
        "BINANCE",
        "BYBIT",
    )

    DEFAULT_LEVERAGE = 3
    DEFAULT_MARGIN_TYPE = "ISOLATED"

    DEFAULT_ORDER_TIMEOUT = 30
    DEFAULT_MAX_RETRIES = 3

    FILL_POLL_INTERVAL = 1.0
    FILL_CONFIRM_TIMEOUT = 15.0

    BALANCE_SAFETY_BUFFER = 0.95
    NOTIONAL_SLIPPAGE_BUFFER = 1.01

    def __init__(
        self,
        settings: Optional[Settings] = None,
    ):
        self.settings = (
            settings
            or get_settings()
        )

        # =====================================================
        # GLOBAL EXECUTION FLAGS
        # =====================================================

        self.use_testnet = bool(
            getattr(
                self.settings,
                "USE_TESTNET",
                False,
            )
        )

        self.enable_real_trading = bool(
            getattr(
                self.settings,
                "ENABLE_REAL_TRADING",
                False,
            )
        )

        self.enable_binance = bool(
            getattr(
                self.settings,
                "ENABLE_BINANCE",
                True,
            )
        )

        self.enable_bybit = bool(
            getattr(
                self.settings,
                "ENABLE_BYBIT",
                True,
            )
        )

        # =====================================================
        # RISK / ORDER SETTINGS
        # =====================================================

        self.leverage = max(
            1,
            int(
                getattr(
                    self.settings,
                    "DEFAULT_LEVERAGE",
                    self.DEFAULT_LEVERAGE,
                )
            ),
        )

        self.margin_type = str(
            getattr(
                self.settings,
                "MARGIN_TYPE",
                self.DEFAULT_MARGIN_TYPE,
            )
        ).lower()

        if self.margin_type not in {
            "isolated",
            "cross",
        }:
            self.margin_type = "isolated"

        self.order_timeout = max(
            5,
            int(
                getattr(
                    self.settings,
                    "ORDER_TIMEOUT",
                    self.DEFAULT_ORDER_TIMEOUT,
                )
            ),
        )

        self.max_retries = max(
            1,
            int(
                getattr(
                    self.settings,
                    "MAX_ORDER_RETRIES",
                    self.DEFAULT_MAX_RETRIES,
                )
            ),
        )

        # =====================================================
        # EXCHANGE INSTANCES
        # =====================================================

        self.binance_exchange = None
        self.bybit_exchange = None

        self.is_initialized = False

        self._initialization_lock = asyncio.Lock()

        # =====================================================
        # ORDER STATE
        # =====================================================

        # symbol -> exchange -> order record
        self.active_orders: Dict[
            str,
            Dict[str, Dict[str, Any]],
        ] = {}

        # order_id -> order record
        self._order_cache: Dict[
            str,
            Dict[str, Any],
        ] = {}

        # portfolio position ID -> execution state
        self._position_execution: Dict[
            str,
            Dict[str, Any],
        ] = {}

        # position ID -> asyncio lock
        self._position_locks: Dict[
            str,
            asyncio.Lock,
        ] = {}

        # =====================================================
        # SHUTDOWN STATE
        # =====================================================

        self._closed = False

    # =========================================================
    # INITIALIZATION
    # =========================================================

    async def initialize(self) -> bool:
        """
        Initialize enabled exchanges.

        Returns:
            True if at least one exchange is connected.
        """

        if not self.enable_real_trading:
            logger.info(
                "📝 RealTradeExecutor: "
                "ENABLE_REAL_TRADING=False. "
                "Running in paper mode."
            )
            return False

        async with self._initialization_lock:

            if self.is_initialized:
                return True

            self._closed = False

            logger.warning(
                "⚠️ INITIALIZING REAL-MONEY "
                "FUTURES EXECUTION."
            )

            successful = []

            # -------------------------------------------------
            # Binance
            # -------------------------------------------------

            if self.enable_binance:

                try:

                    exchange = (
                        await self._create_binance()
                    )

                    if exchange:

                        self.binance_exchange = (
                            exchange
                        )

                        successful.append(
                            "BINANCE"
                        )

                except Exception as exc:

                    logger.error(
                        "❌ Binance initialization "
                        f"failed: {exc}",
                        exc_info=True,
                    )

                    await self._safe_close_exchange(
                        self.binance_exchange
                    )

                    self.binance_exchange = None

            # -------------------------------------------------
            # Bybit
            # -------------------------------------------------

            if self.enable_bybit:

                try:

                    exchange = (
                        await self._create_bybit()
                    )

                    if exchange:

                        self.bybit_exchange = (
                            exchange
                        )

                        successful.append(
                            "BYBIT"
                        )

                except Exception as exc:

                    logger.error(
                        "❌ Bybit initialization "
                        f"failed: {exc}",
                        exc_info=True,
                    )

                    await self._safe_close_exchange(
                        self.bybit_exchange
                    )

                    self.bybit_exchange = None

            self.is_initialized = bool(
                successful
            )

            if self.is_initialized:

                logger.warning(
                    "🚀 RealTradeExecutor "
                    f"v{self.VERSION} initialized. "
                    f"Exchanges: "
                    f"{', '.join(successful)}"
                )

                if self.use_testnet:

                    logger.warning(
                        "🧪 TESTNET / SANDBOX MODE "
                        "IS ENABLED."
                    )

                else:

                    logger.warning(
                        "🔴 LIVE MAINNET REAL-MONEY "
                        "EXECUTION IS ENABLED."
                    )

            else:

                logger.error(
                    "❌ No futures exchanges "
                    "were initialized."
                )

            return self.is_initialized

    async def _create_binance(self):
        """Create and initialize Binance USD-M Futures."""

        api_key = str(
            getattr(
                self.settings,
                "BINANCE_API_KEY",
                "",
            )
            or ""
        ).strip()

        api_secret = str(
            getattr(
                self.settings,
                "BINANCE_API_SECRET",
                "",
            )
            or ""
        ).strip()

        if not api_key or not api_secret:

            logger.warning(
                "⚠️ Binance API credentials "
                "are missing."
            )

            return None

        exchange_config = {
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
            "options": {
                "defaultType": "future",
                "adjustForTimeDifference": True,
                "recvWindow": 20000,
                "fetchCurrencies": False,
            },
        }

        proxy_url = getattr(self.settings, "EXCHANGE_PROXY_URL", None)
        if proxy_url:
            exchange_config["aiohttp_proxy"] = proxy_url

        exchange = ccxt.binanceusdm(exchange_config)
        exchange.has['fetchCurrencies'] = False

        try:
            binance_testnet = getattr(self.settings, "BINANCE_USE_TESTNET", None)
            use_testnet = self.use_testnet if binance_testnet is None else binance_testnet

            if use_testnet:
                if hasattr(exchange, "enable_demo_trading"):
                    exchange.enable_demo_trading(True)
                else:
                    exchange.options["disableFuturesSandboxWarning"] = True
                    exchange.set_sandbox_mode(True)

            try:
                await exchange.load_time_difference()
            except Exception as e:
                logger.debug(f"Binance time sync fallback: {e}")

            await exchange.load_markets()
            await exchange.fetch_balance()

            logger.info(
                "✅ Binance USD-M Futures connection established."
            )

            return exchange

        except Exception:
            await self._safe_close_exchange(exchange)
            raise

    async def _create_bybit(self):
        """Create and initialize Bybit USDT Linear Futures."""

        api_key = str(
            getattr(
                self.settings,
                "BYBIT_API_KEY",
                "",
            )
            or ""
        ).strip()

        api_secret = str(
            getattr(
                self.settings,
                "BYBIT_API_SECRET",
                "",
            )
            or ""
        ).strip()

        if not api_key or not api_secret:
            logger.warning(
                "⚠️ Bybit API credentials are missing."
            )
            return None

        bybit_config = {
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
            "options": {
                "defaultType": "linear",
                "adjustForTimeDifference": True,
                "recvWindow": 20000,
                "fetchCurrencies": False,
            },
        }

        proxy_url = getattr(self.settings, "EXCHANGE_PROXY_URL", None)
        if proxy_url:
            bybit_config["aiohttp_proxy"] = proxy_url

        exchange = ccxt.bybit(bybit_config)
        exchange.has['fetchCurrencies'] = False

        try:
            bybit_testnet = getattr(self.settings, "BYBIT_USE_TESTNET", None)
            use_testnet = self.use_testnet if bybit_testnet is None else bybit_testnet

            if use_testnet:
                exchange.set_sandbox_mode(True)

            try:
                await exchange.load_time_difference()
            except Exception as e:
                logger.debug(f"Bybit time sync fallback: {e}")

            await exchange.load_markets()
            await exchange.fetch_balance()

            logger.info(
                "✅ Bybit USDT Linear Futures connection established."
            )

            return exchange

        except Exception:
            await self._safe_close_exchange(exchange)
            raise

    # =========================================================
    # SYMBOL NORMALIZATION
    # =========================================================

    def _normalize_base_symbol(
        self,
        symbol: str,
    ) -> str:
        """
        Convert common SmartCrypto symbols into a base symbol.

        Examples:

            BTCUSDT
            BTC/USDT
            BTC/USDT:USDT

        -> BTC
        """

        value = str(
            symbol
            or ""
        ).strip().upper()

        value = value.replace(
            ":USDT",
            "",
        )

        value = value.replace(
            "/USDT",
            "",
        )

        if value.endswith(
            "USDT"
        ):
            value = value[:-4]

        return value

    def _get_ccxt_symbol(
        self,
        exchange_name: str,
        symbol: str,
    ) -> str:
        """
        Return a verified CCXT futures symbol.

        We do not construct the symbol blindly. We first
        attempt to find the actual loaded market.
        """

        exchange = (
            self._get_exchange_by_name(
                exchange_name
            )
        )

        base = self._normalize_base_symbol(
            symbol
        )

        candidates = [
            f"{base}/USDT:USDT",
            f"{base}/USDT",
        ]

        if exchange is None:
            return candidates[0]

        for candidate in candidates:

            market = exchange.markets.get(
                candidate
            )

            if market and (
                market.get("swap")
                or market.get("future")
                or market.get("contract")
            ):
                return candidate

        # Search by base/quote if exchange uses
        # another unified representation.
        for market_symbol, market in (
            exchange.markets.items()
        ):

            if (
                str(
                    market.get("base", "")
                ).upper()
                == base
                and str(
                    market.get("quote", "")
                ).upper()
                == "USDT"
                and (
                    market.get("swap")
                    or market.get("future")
                    or market.get("contract")
                )
            ):

                return market_symbol

        raise ValueError(
            f"{exchange_name}: "
            f"No USDT futures market found "
            f"for {symbol}"
        )

    # =========================================================
    # ACTION / SIDE
    # =========================================================

    @staticmethod
    def _get_action_side(
        action: str,
    ) -> str:

        action = str(
            action
            or ""
        ).upper()

        if action == "BUY":
            return "buy"

        if action == "SELL":
            return "sell"

        raise ValueError(
            f"Invalid trading action: {action}"
        )

    @staticmethod
    def _get_close_side(
        action: str,
    ) -> str:

        action = str(
            action
            or ""
        ).upper()

        if action == "BUY":
            return "sell"

        if action == "SELL":
            return "buy"

        raise ValueError(
            f"Invalid trading action: {action}"
        )

    # =========================================================
    # POSITION LOCKING
    # =========================================================

    def _get_position_lock(
        self,
        position: Any,
    ) -> asyncio.Lock:

        position_id = str(
            getattr(
                position,
                "id",
                ""),
        )

        if not position_id:

            position_id = (
                f"{getattr(position, 'symbol', 'UNKNOWN')}:"
                f"{getattr(position, 'action', 'UNKNOWN')}"
            )

        lock = self._position_locks.get(
            position_id
        )

        if lock is None:

            lock = asyncio.Lock()

            self._position_locks[
                position_id
            ] = lock

        return lock

    # =========================================================
    # PRECISION / MARKET VALIDATION
    # =========================================================

    def _get_market(
        self,
        exchange,
        symbol_ccxt: str,
    ) -> Dict[str, Any]:

        if not exchange.markets:
            raise RuntimeError(
                "Exchange markets have not been loaded."
            )

        market = exchange.markets.get(
            symbol_ccxt
        )

        if not market:

            raise ValueError(
                f"Market not found: "
                f"{symbol_ccxt}"
            )

        if not (
            market.get("contract")
            or market.get("swap")
            or market.get("future")
        ):

            raise ValueError(
                f"Market is not a futures "
                f"contract: {symbol_ccxt}"
            )

        return market

    def _format_amount(
        self,
        exchange,
        symbol: str,
        amount: float,
    ) -> float:

        if not math.isfinite(
            amount
        ) or amount <= 0:

            raise ValueError(
                f"Invalid order amount: "
                f"{amount}"
            )

        formatted = float(
            exchange.amount_to_precision(
                symbol,
                amount,
            )
        )

        if formatted <= 0:
            raise ValueError(
                f"Amount became zero after "
                f"precision formatting: "
                f"{amount}"
            )

        market = self._get_market(
            exchange,
            symbol,
        )

        limits = market.get(
            "limits",
            {},
        )

        amount_limits = limits.get(
            "amount",
            {},
        )

        minimum = amount_limits.get(
            "min"
        )

        maximum = amount_limits.get(
            "max"
        )

        if (
            minimum is not None
            and formatted < float(minimum)
        ):

            raise ValueError(
                f"Amount {formatted} is below "
                f"minimum {minimum} for "
                f"{symbol}"
            )

        if (
            maximum is not None
            and formatted > float(maximum)
        ):

            raise ValueError(
                f"Amount {formatted} exceeds "
                f"maximum {maximum} for "
                f"{symbol}"
            )

        return formatted

    def _format_price(
        self,
        exchange,
        symbol: str,
        price: float,
    ) -> float:

        if not math.isfinite(
            price
        ) or price <= 0:

            raise ValueError(
                f"Invalid price: {price}"
            )

        return float(
            exchange.price_to_precision(
                symbol,
                price,
            )
        )

    # =========================================================
    # BALANCE
    # =========================================================

    async def _get_free_usdt(
        self,
        exchange,
    ) -> float:

        balance = await exchange.fetch_balance()

        free = (
            balance.get(
                "free",
                {}
            ).get(
                "USDT",
                0,
            )
        )

        if free is None:

            free = (
                balance.get(
                    "USDT",
                    {}
                ).get(
                    "free",
                    0,
                )
            )

        try:

            value = float(
                free
            )

        except (
            TypeError,
            ValueError,
        ):

            value = 0.0

        return max(
            value,
            0.0,
        )

    async def _check_balance(
        self,
        exchange,
        exchange_name: str,
        symbol: str,
        amount: float,
        reference_price: float,
    ) -> Tuple[bool, float, float]:
        """
        Validate approximate initial margin.

        IMPORTANT:
        Futures balance must NOT be compared directly with
        full notional when leverage is used.

        Approximate margin:

            notional / leverage

        A small safety buffer is then applied.
        """

        try:

            free_usdt = (
                await self._get_free_usdt(
                    exchange
                )
            )

            notional = (
                amount
                * reference_price
            )

            estimated_margin = (
                notional
                / max(
                    self.leverage,
                    1,
                )
            )

            required = (
                estimated_margin
                * self.NOTIONAL_SLIPPAGE_BUFFER
            )

            available = (
                free_usdt
                * self.BALANCE_SAFETY_BUFFER
            )

            if required > available:

                logger.error(
                    f"❌ {exchange_name}: "
                    f"Insufficient futures margin. "
                    f"Required≈${required:.2f}, "
                    f"Available=${available:.2f}, "
                    f"Notional≈${notional:.2f}, "
                    f"Leverage={self.leverage}x"
                )

                return (
                    False,
                    free_usdt,
                    required,
                )

            return (
                True,
                free_usdt,
                required,
            )

        except Exception as exc:

            logger.error(
                f"❌ {exchange_name}: "
                f"Balance check failed: {exc}"
            )

            return (
                False,
                0.0,
                0.0,
            )

    # =========================================================
    # MARGIN / LEVERAGE
    # =========================================================

    async def _configure_margin_and_leverage(
        self,
        exchange,
        exchange_name: str,
        symbol_ccxt: str,
    ) -> bool:
        """
        Configure margin mode and leverage.

        Some exchanges may return a harmless "already set"
        error. Those are logged without treating them as fatal.
        """

        try:

            try:

                await exchange.set_margin_mode(
                    self.margin_type,
                    symbol_ccxt,
                )

            except Exception as exc:

                logger.debug(
                    f"{exchange_name}: "
                    f"Margin mode notice for "
                    f"{symbol_ccxt}: {exc}"
                )

            try:

                await exchange.set_leverage(
                    self.leverage,
                    symbol_ccxt,
                )

            except Exception as exc:

                logger.debug(
                    f"{exchange_name}: "
                    f"Leverage notice for "
                    f"{symbol_ccxt}: {exc}"
                )

            logger.info(
                f"⚙️ {exchange_name}: "
                f"{symbol_ccxt} configured "
                f"with {self.leverage}x "
                f"{self.margin_type} margin."
            )

            return True

        except Exception as exc:

            logger.error(
                f"❌ {exchange_name}: "
                f"Failed margin/leverage setup: "
                f"{exc}"
            )

            return False

    # =========================================================
    # ORDER FETCH / FILL VERIFICATION
    # =========================================================

    async def _fetch_order_safe(
        self,
        exchange,
        order_id: str,
        symbol_ccxt: str,
    ) -> Optional[Dict[str, Any]]:

        try:

            return await asyncio.wait_for(
                exchange.fetch_order(
                    order_id,
                    symbol_ccxt,
                ),
                timeout=self.order_timeout,
            )

        except Exception as exc:

            logger.warning(
                f"⚠️ Failed fetching order "
                f"{order_id}: {exc}"
            )

            return None

    async def _wait_for_fill(
        self,
        exchange,
        exchange_name: str,
        order_id: str,
        symbol_ccxt: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Poll an order until filled or terminal failure.

        Returns the final CCXT order structure.
        """

        deadline = (
            time.monotonic()
            + self.FILL_CONFIRM_TIMEOUT
        )

        last_order = None

        while (
            time.monotonic()
            < deadline
        ):

            order = (
                await self._fetch_order_safe(
                    exchange,
                    order_id,
                    symbol_ccxt,
                )
            )

            if order:

                last_order = order

                status = str(
                    order.get(
                        "status",
                        "",
                    )
                ).lower()

                filled = float(
                    order.get(
                        "filled",
                        0,
                    )
                    or 0
                )

                remaining = float(
                    order.get(
                        "remaining",
                        0,
                    )
                    or 0
                )

                if status == "closed":

                    return order

                if (
                    filled > 0
                    and remaining <= 0
                ):

                    return order

                if status in {
                    "canceled",
                    "cancelled",
                    "expired",
                    "rejected",
                }:

                    logger.error(
                        f"❌ {exchange_name}: "
                        f"Order {order_id} "
                        f"terminal status={status}"
                    )

                    return order

            await asyncio.sleep(
                self.FILL_POLL_INTERVAL
            )

        # -----------------------------------------------------
        # Final fetch after timeout.
        # -----------------------------------------------------

        final_order = (
            await self._fetch_order_safe(
                exchange,
                order_id,
                symbol_ccxt,
            )
        )

        return (
            final_order
            or last_order
        )

    @staticmethod
    def _order_is_fully_filled(
        order: Optional[Dict[str, Any]],
    ) -> bool:

        if not order:
            return False

        status = str(
            order.get(
                "status",
                "",
            )
        ).lower()

        filled = float(
            order.get(
                "filled",
                0,
            )
            or 0
        )

        remaining = float(
            order.get(
                "remaining",
                0,
            )
            or 0
        )

        if status == "closed":
            return True

        return (
            filled > 0
            and remaining <= 0
        )

    @staticmethod
    def _get_filled_amount(
        order: Dict[str, Any],
        fallback: float,
    ) -> float:

        try:

            filled = float(
                order.get(
                    "filled",
                    0,
                )
                or 0
            )

            if filled > 0:
                return filled

        except (
            TypeError,
            ValueError,
        ):
            pass

        return float(
            fallback
        )

    @staticmethod
    def _get_average_price(
        order: Dict[str, Any],
        fallback: float,
    ) -> float:

        for key in (
            "average",
            "price",
        ):

            try:

                value = float(
                    order.get(
                        key,
                        0,
                    )
                    or 0
                )

                if value > 0:
                    return value

            except (
                TypeError,
                ValueError,
            ):
                continue

        return float(
            fallback
        )

    # =========================================================
    # PROTECTION ORDERS
    # =========================================================

    def _trigger_params(
        self,
        action: str,
        trigger_price: float,
        order_purpose: str,
    ) -> Dict[str, Any]:
        """
        Build conservative unified trigger parameters.

        Both SL and TP close the existing position, therefore
        reduceOnly=True is essential.
        """

        close_side = (
            self._get_close_side(
                action
            )
        )

        params: Dict[str, Any] = {
            "triggerPrice": trigger_price,
            "reduceOnly": True,
        }

        # Some exchanges understand closePosition, but it is
        # deliberately NOT used generically here because CCXT
        # does not guarantee identical semantics across all
        # futures exchanges.

        # Trigger direction can be supplied when useful.
        #
        # BUY / long:
        #   SL -> descending
        #   TP -> ascending
        #
        # SELL / short:
        #   SL -> ascending
        #   TP -> descending

        if action.upper() == "BUY":

            if order_purpose == "STOP_LOSS":
                params["triggerDirection"] = (
                    "descending"
                )
            else:
                params["triggerDirection"] = (
                    "ascending"
                )

        else:

            if order_purpose == "STOP_LOSS":
                params["triggerDirection"] = (
                    "ascending"
                )
            else:
                params["triggerDirection"] = (
                    "descending"
                )

        params["_close_side"] = (
            close_side
        )

        return params

    async def _create_protection_order(
        self,
        exchange,
        exchange_name: str,
        position: Any,
        symbol_ccxt: str,
        amount: float,
        purpose: str,
        trigger_price: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Create one protective trigger order.

        Uses CCXT unified triggerPrice first.

        If the exchange rejects the unified trigger form,
        exchange-specific stop-market forms are attempted.
        """

        side = self._get_close_side(
            position.action
        )

        trigger = self._format_price(
            exchange,
            symbol_ccxt,
            trigger_price,
        )

        primary_params = (
            self._trigger_params(
                position.action,
                trigger,
                purpose,
            )
        )

        # Internal helper value must not be sent.
        primary_params.pop(
            "_close_side",
            None,
        )

        # -----------------------------------------------------
        # First attempt: unified CCXT trigger order.
        # -----------------------------------------------------

        try:

            order = await asyncio.wait_for(
                exchange.create_order(
                    symbol_ccxt,
                    "market",
                    side,
                    amount,
                    None,
                    primary_params,
                ),
                timeout=self.order_timeout,
            )

            return order

        except Exception as primary_exc:

            logger.warning(
                f"⚠️ {exchange_name}: "
                f"Unified {purpose} order failed "
                f"for {position.symbol}: "
                f"{primary_exc}"
            )

        # -----------------------------------------------------
        # Exchange-specific fallback.
        # -----------------------------------------------------

        try:

            if purpose == "STOP_LOSS":

                order_type = (
                    "STOP_MARKET"
                )

            else:

                order_type = (
                    "TAKE_PROFIT_MARKET"
                )

            fallback_params = {
                "stopPrice": trigger,
                "reduceOnly": True,
            }

            order = await asyncio.wait_for(
                exchange.create_order(
                    symbol_ccxt,
                    order_type,
                    side,
                    amount,
                    None,
                    fallback_params,
                ),
                timeout=self.order_timeout,
            )

            return order

        except Exception as fallback_exc:

            logger.error(
                f"❌ {exchange_name}: "
                f"{purpose} protection order "
                f"failed for "
                f"{position.symbol}. "
                f"Fallback error: "
                f"{fallback_exc}"
            )

            return None

    async def _install_protection_orders(
        self,
        exchange,
        exchange_name: str,
        position: Any,
        symbol_ccxt: str,
        filled_amount: float,
    ) -> Tuple[
        Optional[Dict[str, Any]],
        Optional[Dict[str, Any]],
    ]:
        """
        Install SL and TP independently.

        If one protection order fails, the failure is explicit.
        """

        sl_order = None
        tp_order = None

        # -----------------------------------------------------
        # Stop Loss
        # -----------------------------------------------------

        try:

            sl_order = (
                await self._create_protection_order(
                    exchange,
                    exchange_name,
                    position,
                    symbol_ccxt,
                    filled_amount,
                    "STOP_LOSS",
                    float(
                        position.stop_loss
                    ),
                )
            )

        except Exception as exc:

            logger.error(
                f"❌ {exchange_name}: "
                f"SL installation failed: {exc}"
            )

        # -----------------------------------------------------
        # Take Profit
        # -----------------------------------------------------

        try:

            tp_order = (
                await self._create_protection_order(
                    exchange,
                    exchange_name,
                    position,
                    symbol_ccxt,
                    filled_amount,
                    "TAKE_PROFIT",
                    float(
                        position.take_profit
                    ),
                )
            )

        except Exception as exc:

            logger.error(
                f"❌ {exchange_name}: "
                f"TP installation failed: {exc}"
            )

        return (
            sl_order,
            tp_order,
        )

    # =========================================================
    # ORDER TRACKING
    # =========================================================

    def _record_active_order(
        self,
        position: Any,
        exchange_name: str,
        record: Dict[str, Any],
    ) -> None:

        symbol = str(
            position.symbol
        )

        if symbol not in self.active_orders:

            self.active_orders[
                symbol
            ] = {}

        self.active_orders[
            symbol
        ][exchange_name] = record

        for order_key in (
            "entry_order_id",
            "sl_order_id",
            "tp_order_id",
        ):

            order_id = record.get(
                order_key
            )

            if order_id:

                self._order_cache[
                    str(order_id)
                ] = record

    def _remove_active_exchange_order(
        self,
        symbol: str,
        exchange_name: str,
    ) -> None:

        exchange_orders = (
            self.active_orders.get(
                symbol
            )
        )

        if not exchange_orders:
            return

        exchange_orders.pop(
            exchange_name,
            None,
        )

        if not exchange_orders:

            self.active_orders.pop(
                symbol,
                None,
            )

    # =========================================================
    # DUPLICATE EXECUTION PROTECTION
    # =========================================================

    def _has_execution_for_position(
        self,
        position: Any,
    ) -> bool:

        position_id = str(
            getattr(
                position,
                "id",
                "",
            )
        )

        if not position_id:
            return False

        state = self._position_execution.get(
            position_id
        )

        if not state:
            return False

        return bool(
            state.get(
                "entry_submitted"
            )
            or state.get(
                "entry_filled"
            )
        )

    # =========================================================
    # SINGLE EXCHANGE OPEN
    # =========================================================

    async def _execute_single_exchange_open(
        self,
        exchange,
        exchange_name: str,
        position: Any,
    ) -> Optional[Dict[str, Any]]:
        """
        Execute one real position on one exchange.
        """

        symbol_ccxt = (
            self._get_ccxt_symbol(
                exchange_name,
                position.symbol,
            )
        )

        side = self._get_action_side(
            position.action
        )

        reference_price = float(
            position.entry_price
        )

        amount = float(
            position.quantity
        )

        # -----------------------------------------------------
        # Validate position data.
        # -----------------------------------------------------

        if reference_price <= 0:

            raise ValueError(
                f"{exchange_name}: "
                f"Invalid entry price "
                f"{reference_price}"
            )

        if amount <= 0:

            raise ValueError(
                f"{exchange_name}: "
                f"Invalid position quantity "
                f"{amount}"
            )

        self._get_market(
            exchange,
            symbol_ccxt,
        )

        # -----------------------------------------------------
        # Configure margin/leverage.
        # -----------------------------------------------------

        await self._configure_margin_and_leverage(
            exchange,
            exchange_name,
            symbol_ccxt,
        )

        # -----------------------------------------------------
        # Format quantity.
        # -----------------------------------------------------

        amount_float = (
            self._format_amount(
                exchange,
                symbol_ccxt,
                amount,
            )
        )

        # -----------------------------------------------------
        # Balance.
        # -----------------------------------------------------

        (
            has_funds,
            free_usdt,
            estimated_margin,
        ) = await self._check_balance(
            exchange,
            exchange_name,
            symbol_ccxt,
            amount_float,
            reference_price,
        )

        if not has_funds:
            return None

        logger.info(
            f"📊 {exchange_name}: "
            f"OPEN {position.action} "
            f"{amount_float} "
            f"{position.symbol} | "
            f"Estimated margin="
            f"${estimated_margin:.2f} | "
            f"Free USDT="
            f"${free_usdt:.2f}"
        )

        # -----------------------------------------------------
        # Entry order.
        # -----------------------------------------------------

        client_order_id = (
            f"SC-{str(position.id)[:12]}-"
            f"{exchange_name[:3]}-"
            f"{uuid.uuid4().hex[:8]}"
        )

        entry_params: Dict[str, Any] = {
            "marginMode": self.margin_type,
            "leverage": self.leverage,
        }

        # CCXT/exchange support varies for client IDs.
        if exchange_name == "BINANCE":

            entry_params[
                "newClientOrderId"
            ] = client_order_id

        elif exchange_name == "BYBIT":

            entry_params[
                "orderLinkId"
            ] = client_order_id

        entry_order = None

        for attempt in range(
            1,
            self.max_retries + 1,
        ):

            try:

                execution_state = (
                    self._position_execution.setdefault(
                        str(position.id),
                        {},
                    )
                )

                execution_state[
                    "entry_submitted"
                ] = True

                entry_order = (
                    await asyncio.wait_for(
                        exchange.create_order(
                            symbol_ccxt,
                            "market",
                            side,
                            amount_float,
                            None,
                            entry_params,
                        ),
                        timeout=self.order_timeout,
                    )
                )

                break

            except (
                ccxt.NetworkError,
                ccxt.RequestTimeout,
                ccxt.ExchangeNotAvailable,
            ) as exc:

                logger.warning(
                    f"⚠️ {exchange_name}: "
                    f"Entry attempt "
                    f"{attempt}/"
                    f"{self.max_retries} "
                    f"network failure: "
                    f"{exc}"
                )

                if attempt >= self.max_retries:

                    raise

                await asyncio.sleep(
                    min(
                        2 ** (attempt - 1),
                        5,
                    )
                )

            except Exception:

                raise

        if not entry_order:

            return None

        entry_id = (
            entry_order.get(
                "id"
            )
        )

        if not entry_id:

            raise RuntimeError(
                f"{exchange_name}: "
                f"Exchange returned entry "
                f"order without ID."
            )

        # -----------------------------------------------------
        # Verify actual fill.
        # -----------------------------------------------------

        final_entry = (
            await self._wait_for_fill(
                exchange,
                exchange_name,
                str(entry_id),
                symbol_ccxt,
            )
        )

        if not final_entry:

            logger.error(
                f"❌ {exchange_name}: "
                f"Could not verify entry "
                f"{entry_id}"
            )

            return None

        if not self._order_is_fully_filled(
            final_entry
        ):

            filled = float(
                final_entry.get(
                    "filled",
                    0,
                )
                or 0
            )

            status = final_entry.get(
                "status"
            )

            logger.error(
                f"❌ {exchange_name}: "
                f"Entry {entry_id} "
                f"not fully filled. "
                f"status={status}, "
                f"filled={filled}, "
                f"requested={amount_float}"
            )

            # If there was a partial fill, it must not be
            # silently treated as a successful full position.
            if filled > 0:

                logger.critical(
                    f"🚨 {exchange_name}: "
                    f"PARTIAL REAL FILL detected "
                    f"for {position.symbol}. "
                    f"Filled={filled}. "
                    f"Manual/reconciliation action "
                    f"may be required."
                )

            return None

        filled_amount = (
            self._get_filled_amount(
                final_entry,
                amount_float,
            )
        )

        filled_price = (
            self._get_average_price(
                final_entry,
                reference_price,
            )
        )

        execution_state = (
            self._position_execution.setdefault(
                str(position.id),
                {},
            )
        )

        execution_state[
            "entry_filled"
        ] = True

        execution_state[
            "entry_order_id"
        ] = str(entry_id)

        # -----------------------------------------------------
        # Protective orders.
        # -----------------------------------------------------

        (
            sl_order,
            tp_order,
        ) = await self._install_protection_orders(
            exchange,
            exchange_name,
            position,
            symbol_ccxt,
            filled_amount,
        )

        sl_id = (
            sl_order.get("id")
            if sl_order
            else None
        )

        tp_id = (
            tp_order.get("id")
            if tp_order
            else None
        )

        protection_complete = bool(
            sl_id
            and tp_id
        )

        if not protection_complete:

            logger.critical(
                f"🚨 {exchange_name}: "
                f"POSITION {position.symbol} "
                f"IS OPEN BUT FULL PROTECTION "
                f"WAS NOT INSTALLED. "
                f"SL={sl_id}, TP={tp_id}"
            )

        # -----------------------------------------------------
        # Record.
        # -----------------------------------------------------

        record = {
            "exchange": exchange_name,
            "symbol": position.symbol,
            "ccxt_symbol": symbol_ccxt,
            "position_id": str(
                position.id
            ),
            "entry_order_id": str(
                entry_id
            ),
            "sl_order_id": (
                str(sl_id)
                if sl_id
                else None
            ),
            "tp_order_id": (
                str(tp_id)
                if tp_id
                else None
            ),
            "requested_amount": (
                float(amount_float)
            ),
            "filled_amount": (
                float(filled_amount)
            ),
            "filled_price": (
                float(filled_price)
            ),
            "side": side,
            "leverage": self.leverage,
            "margin_type": self.margin_type,
            "stop_loss": float(
                position.stop_loss
            ),
            "take_profit": float(
                position.take_profit
            ),
            "protection_complete": (
                protection_complete
            ),
            "status": "OPEN",
            "timestamp": (
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),
        }

        self._record_active_order(
            position,
            exchange_name,
            record,
        )

        logger.info(
            f"✅ {exchange_name}: "
            f"REAL POSITION OPENED | "
            f"{position.symbol} | "
            f"{position.action} | "
            f"Qty={filled_amount} | "
            f"Entry=${filled_price:.6f} | "
            f"SL={position.stop_loss:.6f} | "
            f"TP={position.take_profit:.6f} | "
            f"Protected={protection_complete}"
        )

        return record

    # =========================================================
    # SINGLE EXCHANGE CLOSE
    # =========================================================

    async def _execute_single_exchange_close(
        self,
        exchange,
        exchange_name: str,
        position: Any,
        reason: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Close an existing real position on one exchange.

        Before closing:
            - cancel protection orders
            - determine actual exchange position size
            - submit reduce-only market order
            - verify fill
        """

        symbol_ccxt = (
            self._get_ccxt_symbol(
                exchange_name,
                position.symbol,
            )
        )

        close_side = (
            self._get_close_side(
                position.action
            )
        )

        self._get_market(
            exchange,
            symbol_ccxt,
        )

        # -----------------------------------------------------
        # Retrieve active execution record.
        # -----------------------------------------------------

        execution_record = (
            self.active_orders
            .get(
                position.symbol,
                {}
            )
            .get(
                exchange_name,
            )
        )

        # -----------------------------------------------------
        # Cancel known protection orders first.
        # -----------------------------------------------------

        known_order_ids = []

        if execution_record:

            for key in (
                "sl_order_id",
                "tp_order_id",
            ):

                value = execution_record.get(
                    key
                )

                if value:
                    known_order_ids.append(
                        str(value)
                    )

        for order_id in known_order_ids:

            try:

                await exchange.cancel_order(
                    order_id,
                    symbol_ccxt,
                )

                logger.info(
                    f"🧹 {exchange_name}: "
                    f"Cancelled protection "
                    f"order {order_id}"
                )

            except Exception as exc:

                logger.debug(
                    f"{exchange_name}: "
                    f"Protection cancellation "
                    f"notice for {order_id}: "
                    f"{exc}"
                )

        # -----------------------------------------------------
        # Do NOT blindly cancel every order on the symbol.
        #
        # This is safer in multi-strategy environments.
        # -----------------------------------------------------

        # -----------------------------------------------------
        # Determine actual position amount.
        # -----------------------------------------------------

        actual_amount = (
            await self._get_actual_position_amount(
                exchange,
                exchange_name,
                symbol_ccxt,
                position.action,
            )
        )

        requested_amount = float(
            getattr(
                position,
                "quantity",
                0,
            )
            or 0
        )

        amount = (
            actual_amount
            if actual_amount > 0
            else requested_amount
        )

        if amount <= 0:

            logger.warning(
                f"⚠️ {exchange_name}: "
                f"No open exchange position "
                f"found for "
                f"{position.symbol}."
            )

            self._remove_active_exchange_order(
                position.symbol,
                exchange_name,
            )

            return {
                "exchange": exchange_name,
                "symbol": position.symbol,
                "status": "ALREADY_CLOSED",
                "reason": reason,
                "timestamp": (
                    datetime.now(
                        timezone.utc
                    ).isoformat()
                ),
            }

        amount = self._format_amount(
            exchange,
            symbol_ccxt,
            amount,
        )

        # -----------------------------------------------------
        # Reduce-only close.
        # -----------------------------------------------------

        close_params: Dict[str, Any] = {
            "reduceOnly": True,
            "marginMode": self.margin_type,
        }

        close_client_id = (
            f"SC-CLOSE-"
            f"{str(position.id)[:12]}-"
            f"{exchange_name[:3]}-"
            f"{uuid.uuid4().hex[:8]}"
        )

        if exchange_name == "BINANCE":

            close_params[
                "newClientOrderId"
            ] = close_client_id

        elif exchange_name == "BYBIT":

            close_params[
                "orderLinkId"
            ] = close_client_id

        logger.info(
            f"📤 {exchange_name}: "
            f"CLOSING {position.symbol} "
            f"qty={amount} "
            f"reason={reason}"
        )

        close_order = await asyncio.wait_for(
            exchange.create_order(
                symbol_ccxt,
                "market",
                close_side,
                amount,
                None,
                close_params,
            ),
            timeout=self.order_timeout,
        )

        close_id = (
            close_order.get(
                "id"
            )
        )

        if not close_id:

            raise RuntimeError(
                f"{exchange_name}: "
                f"Close order returned "
                f"without ID."
            )

        final_close = (
            await self._wait_for_fill(
                exchange,
                exchange_name,
                str(close_id),
                symbol_ccxt,
            )
        )

        if not final_close:

            raise RuntimeError(
                f"{exchange_name}: "
                f"Could not verify close "
                f"order {close_id}"
            )

        if not self._order_is_fully_filled(
            final_close
        ):

            logger.critical(
                f"🚨 {exchange_name}: "
                f"CLOSE ORDER NOT FULLY "
                f"FILLED | "
                f"Position may remain open | "
                f"Order={close_id}"
            )

            return {
                "exchange": exchange_name,
                "symbol": position.symbol,
                "close_order_id": str(
                    close_id
                ),
                "status": "PARTIAL_OR_UNCONFIRMED",
                "reason": reason,
                "filled": float(
                    final_close.get(
                        "filled",
                        0,
                    )
                    or 0
                ),
                "timestamp": (
                    datetime.now(
                        timezone.utc
                    ).isoformat()
                ),
            }

        exit_price = (
            self._get_average_price(
                final_close,
                float(
                    getattr(
                        position,
                        "current_price",
                        position.entry_price,
                    )
                ),
            )
        )

        filled_amount = (
            self._get_filled_amount(
                final_close,
                amount,
            )
        )

        close_record = {
            "exchange": exchange_name,
            "symbol": position.symbol,
            "ccxt_symbol": symbol_ccxt,
            "position_id": str(
                position.id
            ),
            "close_order_id": str(
                close_id
            ),
            "exit_price": float(
                exit_price
            ),
            "filled_amount": float(
                filled_amount
            ),
            "reason": reason,
            "status": "CLOSED",
            "timestamp": (
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),
        }

        self._order_cache[
            str(close_id)
        ] = close_record

        self._remove_active_exchange_order(
            position.symbol,
            exchange_name,
        )

        logger.info(
            f"✅ {exchange_name}: "
            f"REAL POSITION CLOSED | "
            f"{position.symbol} | "
            f"Exit=${exit_price:.6f} | "
            f"Reason={reason}"
        )

        return close_record

    # =========================================================
    # ACTUAL EXCHANGE POSITION
    # =========================================================

    async def _get_actual_position_amount(
        self,
        exchange,
        exchange_name: str,
        symbol_ccxt: str,
        expected_action: str,
    ) -> float:

        try:

            positions = (
                await exchange.fetch_positions(
                    [symbol_ccxt]
                )
            )

            expected_side = (
                "long"
                if str(
                    expected_action
                ).upper()
                == "BUY"
                else "short"
            )

            for position in positions:

                if (
                    position.get(
                        "symbol"
                    )
                    != symbol_ccxt
                ):
                    continue

                side = str(
                    position.get(
                        "side",
                        "",
                    )
                ).lower()

                contracts = float(
                    position.get(
                        "contracts",
                        0,
                    )
                    or 0
                )

                if (
                    side == expected_side
                    and contracts > 0
                ):

                    return contracts

            return 0.0

        except Exception as exc:

            logger.warning(
                f"⚠️ {exchange_name}: "
                f"Could not retrieve actual "
                f"position size for "
                f"{symbol_ccxt}: {exc}"
            )

            return 0.0

    # =========================================================
    # OPEN POSITION - PUBLIC
    # =========================================================

    async def execute_open_position(
        self,
        position: Any,
    ) -> Dict[str, Any]:
        """
        Execute a PortfolioManager position.

        The same quantity is sent independently to each enabled
        exchange.
        """

        if not self.enable_real_trading:

            logger.info(
                "📝 Real trading disabled. "
                "Skipping exchange execution."
            )

            return {
                "success": False,
                "paper": True,
                "results": [],
            }

        if not self.is_initialized:

            initialized = (
                await self.initialize()
            )

            if not initialized:

                return {
                    "success": False,
                    "paper": False,
                    "results": [],
                    "error": (
                        "No exchange initialized"
                    ),
                }

        lock = (
            self._get_position_lock(
                position
            )
        )

        async with lock:

            # -------------------------------------------------
            # Duplicate protection.
            # -------------------------------------------------

            if self._has_execution_for_position(
                position
            ):

                logger.warning(
                    f"🛡️ Duplicate real "
                    f"entry blocked for "
                    f"{position.symbol} "
                    f"position={position.id}"
                )

                return {
                    "success": False,
                    "duplicate": True,
                    "results": [],
                }

            tasks = []

            if self.binance_exchange:

                tasks.append(
                    self._execute_single_exchange_open(
                        self.binance_exchange,
                        "BINANCE",
                        position,
                    )
                )

            if self.bybit_exchange:

                tasks.append(
                    self._execute_single_exchange_open(
                        self.bybit_exchange,
                        "BYBIT",
                        position,
                    )
                )

            if not tasks:

                return {
                    "success": False,
                    "results": [],
                    "error": (
                        "No exchanges available"
                    ),
                }

            results = await asyncio.gather(
                *tasks,
                return_exceptions=True,
            )

            normalized = []

            for result in results:

                if isinstance(
                    result,
                    Exception,
                ):

                    logger.error(
                        "❌ Exchange open "
                        f"exception: {result}",
                        exc_info=True,
                    )

                    normalized.append(
                        {
                            "status": "FAILED",
                            "error": str(
                                result
                            ),
                        }
                    )

                elif result:

                    normalized.append(
                        result
                    )

                else:

                    normalized.append(
                        {
                            "status": "FAILED"
                        }
                    )

            successful = [
                item
                for item in normalized
                if item.get(
                    "status"
                )
                not in {
                    "FAILED",
                }
                and item.get(
                    "entry_order_id"
                )
            ]

            failed = (
                len(normalized)
                - len(successful)
            )

            # -------------------------------------------------
            # A real position may exist on one exchange while
            # failing on another. Do not pretend the execution
            # was globally successful.
            # -------------------------------------------------

            overall_status = (
                "SUCCESS"
                if successful
                and failed == 0
                else (
                    "PARTIAL_SUCCESS"
                    if successful
                    else "FAILED"
                )
            )

            result_payload = {
                "success": bool(
                    successful
                ),
                "status": overall_status,
                "results": normalized,
                "successful_exchanges": len(
                    successful
                ),
                "failed_exchanges": failed,
                "exchange_order_ids": [
                    item.get(
                        "entry_order_id"
                    )
                    for item in successful
                    if item.get(
                        "entry_order_id"
                    )
                ],
            }

            logger.info(
                f"📊 OPEN EXECUTION SUMMARY | "
                f"{position.symbol} | "
                f"Status={overall_status} | "
                f"Success={len(successful)} | "
                f"Failed={failed}"
            )

            return result_payload

    # =========================================================
    # CLOSE POSITION - PUBLIC
    # =========================================================

    async def execute_close_position(
        self,
        position: Any,
        reason: str = "MANUAL",
    ) -> Dict[str, Any]:
        """
        Close a PortfolioManager position across all
        initialized exchanges.
        """

        if not self.enable_real_trading:

            return {
                "success": False,
                "paper": True,
                "results": [],
            }

        if not self.is_initialized:

            return {
                "success": False,
                "results": [],
                "error": (
                    "Executor is not initialized"
                ),
            }

        lock = (
            self._get_position_lock(
                position
            )
        )

        async with lock:

            tasks = []

            if self.binance_exchange:

                tasks.append(
                    self._execute_single_exchange_close(
                        self.binance_exchange,
                        "BINANCE",
                        position,
                        reason,
                    )
                )

            if self.bybit_exchange:

                tasks.append(
                    self._execute_single_exchange_close(
                        self.bybit_exchange,
                        "BYBIT",
                        position,
                        reason,
                    )
                )

            if not tasks:

                return {
                    "success": False,
                    "results": [],
                }

            results = await asyncio.gather(
                *tasks,
                return_exceptions=True,
            )

            normalized = []

            for result in results:

                if isinstance(
                    result,
                    Exception,
                ):

                    logger.error(
                        "❌ Exchange close "
                        f"exception: {result}",
                        exc_info=True,
                    )

                    normalized.append(
                        {
                            "status": "FAILED",
                            "error": str(
                                result
                            ),
                        }
                    )

                elif result:

                    normalized.append(
                        result
                    )

                else:

                    normalized.append(
                        {
                            "status": "FAILED"
                        }
                    )

            successful = [
                item
                for item in normalized
                if item.get(
                    "status"
                ) in {
                    "CLOSED",
                    "ALREADY_CLOSED",
                }
            ]

            failed = (
                len(normalized)
                - len(successful)
            )

            status = (
                "SUCCESS"
                if successful
                and failed == 0
                else (
                    "PARTIAL_SUCCESS"
                    if successful
                    else "FAILED"
                )
            )

            logger.info(
                f"📊 CLOSE EXECUTION SUMMARY | "
                f"{position.symbol} | "
                f"Status={status} | "
                f"Success={len(successful)} | "
                f"Failed={failed}"
            )

            return {
                "success": bool(
                    successful
                ),
                "status": status,
                "results": normalized,
                "successful_exchanges": len(
                    successful
                ),
                "failed_exchanges": failed,
                "exchange_order_ids": [
                    item.get(
                        "close_order_id"
                    )
                    for item in successful
                    if item.get(
                        "close_order_id"
                    )
                ],
            }

    # =========================================================
    # ORDER MANAGEMENT
    # =========================================================

    async def cancel_order(
        self,
        exchange_name: str,
        order_id: str,
        symbol: str,
    ) -> bool:

        exchange = (
            self._get_exchange_by_name(
                exchange_name
            )
        )

        if not exchange:
            return False

        try:

            symbol_ccxt = (
                self._get_ccxt_symbol(
                    exchange_name,
                    symbol,
                )
            )

            await exchange.cancel_order(
                order_id,
                symbol_ccxt,
            )

            logger.info(
                f"✅ {exchange_name}: "
                f"Order {order_id} cancelled."
            )

            return True

        except Exception as exc:

            logger.error(
                f"❌ {exchange_name}: "
                f"Failed cancelling "
                f"{order_id}: {exc}"
            )

            return False

    async def cancel_all_orders(
        self,
        exchange_name: str,
        symbol: str,
    ) -> bool:

        exchange = (
            self._get_exchange_by_name(
                exchange_name
            )
        )

        if not exchange:
            return False

        try:

            symbol_ccxt = (
                self._get_ccxt_symbol(
                    exchange_name,
                    symbol,
                )
            )

            await exchange.cancel_all_orders(
                symbol_ccxt
            )

            logger.info(
                f"🧹 {exchange_name}: "
                f"All orders cancelled for "
                f"{symbol}"
            )

            return True

        except Exception as exc:

            logger.error(
                f"❌ {exchange_name}: "
                f"Failed cancelling orders "
                f"for {symbol}: {exc}"
            )

            return False

    async def get_order_status(
        self,
        exchange_name: str,
        order_id: str,
        symbol: str,
    ) -> Optional[Dict[str, Any]]:

        exchange = (
            self._get_exchange_by_name(
                exchange_name
            )
        )

        if not exchange:
            return None

        try:

            symbol_ccxt = (
                self._get_ccxt_symbol(
                    exchange_name,
                    symbol,
                )
            )

            order = (
                await exchange.fetch_order(
                    order_id,
                    symbol_ccxt,
                )
            )

            return {
                "id": order.get(
                    "id"
                ),
                "symbol": order.get(
                    "symbol"
                ),
                "side": order.get(
                    "side"
                ),
                "type": order.get(
                    "type"
                ),
                "status": order.get(
                    "status"
                ),
                "price": float(
                    order.get(
                        "price",
                        0,
                    )
                    or 0
                ),
                "average": float(
                    order.get(
                        "average",
                        0,
                    )
                    or 0
                ),
                "filled": float(
                    order.get(
                        "filled",
                        0,
                    )
                    or 0
                ),
                "remaining": float(
                    order.get(
                        "remaining",
                        0,
                    )
                    or 0
                ),
                "cost": float(
                    order.get(
                        "cost",
                        0,
                    )
                    or 0
                ),
                "fee": order.get(
                    "fee"
                ),
                "timestamp": order.get(
                    "timestamp"
                ),
                "exchange": exchange_name,
            }

        except Exception as exc:

            logger.error(
                f"❌ {exchange_name}: "
                f"Failed getting order status "
                f"{order_id}: {exc}"
            )

            return None

    # =========================================================
    # EXCHANGE ACCESS
    # =========================================================

    def _get_exchange_by_name(
        self,
        exchange_name: str,
    ):

        name = str(
            exchange_name
            or ""
        ).upper()

        if name == "BINANCE":
            return self.binance_exchange

        if name == "BYBIT":
            return self.bybit_exchange

        return None

    # =========================================================
    # BALANCE QUERIES
    # =========================================================

    async def get_balance(
        self,
        exchange_name: str,
        currency: str = "USDT",
    ) -> Dict[str, Any]:

        exchange = (
            self._get_exchange_by_name(
                exchange_name
            )
        )

        if not exchange:

            return {
                "free": 0.0,
                "used": 0.0,
                "total": 0.0,
                "is_live": False,
                "exchange": exchange_name,
            }

        try:

            balance = (
                await exchange.fetch_balance()
            )

            currency_data = balance.get(
                currency,
                {}
            )

            return {
                "free": float(
                    currency_data.get(
                        "free",
                        0,
                    )
                    or 0
                ),
                "used": float(
                    currency_data.get(
                        "used",
                        0,
                    )
                    or 0
                ),
                "total": float(
                    currency_data.get(
                        "total",
                        0,
                    )
                    or 0
                ),
                "is_live": True,
                "exchange": exchange_name,
            }

        except Exception as exc:

            logger.error(
                f"❌ {exchange_name}: "
                f"Balance fetch failed: {exc}"
            )

            return {
                "free": 0.0,
                "used": 0.0,
                "total": 0.0,
                "is_live": False,
                "exchange": exchange_name,
                "error": str(exc),
            }

    async def get_balances(
        self,
        currency: str = "USDT",
    ) -> Dict[str, Dict[str, Any]]:

        tasks = {}

        if self.binance_exchange:

            tasks["BINANCE"] = (
                self.get_balance(
                    "BINANCE",
                    currency,
                )
            )

        if self.bybit_exchange:

            tasks["BYBIT"] = (
                self.get_balance(
                    "BYBIT",
                    currency,
                )
            )

        if not tasks:
            return {}

        names = list(
            tasks.keys()
        )

        results = await asyncio.gather(
            *tasks.values(),
            return_exceptions=True,
        )

        output = {}

        for name, result in zip(
            names,
            results,
        ):

            if isinstance(
                result,
                Exception,
            ):

                output[name] = {
                    "free": 0.0,
                    "used": 0.0,
                    "total": 0.0,
                    "is_live": False,
                    "exchange": name,
                    "error": str(
                        result
                    ),
                }

            else:

                output[name] = result

        return output

    # =========================================================
    # POSITION QUERIES
    # =========================================================

    async def get_position(
        self,
        exchange_name: str,
        symbol: str,
    ) -> Optional[Dict[str, Any]]:

        exchange = (
            self._get_exchange_by_name(
                exchange_name
            )
        )

        if not exchange:
            return None

        try:

            symbol_ccxt = (
                self._get_ccxt_symbol(
                    exchange_name,
                    symbol,
                )
            )

            positions = (
                await exchange.fetch_positions(
                    [symbol_ccxt]
                )
            )

            for position in positions:

                if (
                    position.get(
                        "symbol"
                    )
                    != symbol_ccxt
                ):
                    continue

                contracts = float(
                    position.get(
                        "contracts",
                        0,
                    )
                    or 0
                )

                if contracts <= 0:
                    continue

                return {
                    "symbol": position.get(
                        "symbol"
                    ),
                    "side": position.get(
                        "side"
                    ),
                    "contracts": contracts,
                    "entry_price": float(
                        position.get(
                            "entryPrice",
                            0,
                        )
                        or 0
                    ),
                    "mark_price": float(
                        position.get(
                            "markPrice",
                            0,
                        )
                        or 0
                    ),
                    "pnl": float(
                        position.get(
                            "unrealizedPnl",
                            0,
                        )
                        or 0
                    ),
                    "percentage": float(
                        position.get(
                            "percentage",
                            0,
                        )
                        or 0
                    ),
                    "leverage": float(
                        position.get(
                            "leverage",
                            1,
                        )
                        or 1
                    ),
                    "margin_mode": position.get(
                        "marginMode"
                    ),
                    "exchange": exchange_name,
                }

            return None

        except Exception as exc:

            logger.error(
                f"❌ {exchange_name}: "
                f"Position query failed for "
                f"{symbol}: {exc}"
            )

            return None

    async def get_positions(
        self,
        symbols: Optional[List[str]] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Query real positions.

        Unlike the previous implementation, this is not
        hard-coded to BTC.
        """

        output = {}

        if symbols is None:

            symbols = []

            for exchange_name in (
                "BINANCE",
                "BYBIT",
            ):

                exchange = (
                    self._get_exchange_by_name(
                        exchange_name
                    )
                )

                if not exchange:
                    continue

                try:

                    positions = (
                        await exchange.fetch_positions()
                    )

                    output[
                        exchange_name
                    ] = [
                        self._normalize_exchange_position(
                            item,
                            exchange_name,
                        )
                        for item in positions
                        if float(
                            item.get(
                                "contracts",
                                0,
                            )
                            or 0
                        ) > 0
                    ]

                except Exception as exc:

                    output[
                        exchange_name
                    ] = [
                        {
                            "error": str(
                                exc
                            )
                        }
                    ]

            return output

        for exchange_name in (
            "BINANCE",
            "BYBIT",
        ):

            if not self._get_exchange_by_name(
                exchange_name
            ):
                continue

            results = []

            for symbol in symbols:

                position = (
                    await self.get_position(
                        exchange_name,
                        symbol,
                    )
                )

                if position:
                    results.append(
                        position
                    )

            output[
                exchange_name
            ] = results

        return output

    @staticmethod
    def _normalize_exchange_position(
        position: Dict[str, Any],
        exchange_name: str,
    ) -> Dict[str, Any]:

        return {
            "exchange": exchange_name,
            "symbol": position.get(
                "symbol"
            ),
            "side": position.get(
                "side"
            ),
            "contracts": float(
                position.get(
                    "contracts",
                    0,
                )
                or 0
            ),
            "entry_price": float(
                position.get(
                    "entryPrice",
                    0,
                )
                or 0
            ),
            "mark_price": float(
                position.get(
                    "markPrice",
                    0,
                )
                or 0
            ),
            "unrealized_pnl": float(
                position.get(
                    "unrealizedPnl",
                    0,
                )
                or 0
            ),
            "leverage": float(
                position.get(
                    "leverage",
                    1,
                )
                or 1
            ),
            "margin_mode": position.get(
                "marginMode"
            ),
        }

    # =========================================================
    # HEALTH CHECK
    # =========================================================

    async def health_check(
        self,
        exchange_name: Optional[str] = None,
    ) -> Dict[str, Any]:

        if exchange_name:

            result = (
                await self._health_check_single(
                    exchange_name
                )
            )

            return {
                "status": (
                    "healthy"
                    if result.get(
                        "status"
                    )
                    == "healthy"
                    else "degraded"
                ),
                "exchanges": {
                    exchange_name.upper():
                        result
                },
                "is_live": (
                    self.enable_real_trading
                ),
                "timestamp": (
                    datetime.now(
                        timezone.utc
                    ).isoformat()
                ),
            }

        names = []

        if self.binance_exchange:
            names.append("BINANCE")

        if self.bybit_exchange:
            names.append("BYBIT")

        if not names:

            return {
                "status": "unhealthy",
                "exchanges": {},
                "is_live": (
                    self.enable_real_trading
                ),
                "timestamp": (
                    datetime.now(
                        timezone.utc
                    ).isoformat()
                ),
            }

        results = await asyncio.gather(
            *[
                self._health_check_single(
                    name
                )
                for name in names
            ],
            return_exceptions=True,
        )

        exchange_results = {}

        for name, result in zip(
            names,
            results,
        ):

            if isinstance(
                result,
                Exception,
            ):

                exchange_results[
                    name
                ] = {
                    "status": "unhealthy",
                    "error": str(
                        result
                    ),
                    "exchange": name,
                }

            else:

                exchange_results[
                    name
                ] = result

        healthy = [
            item
            for item in exchange_results.values()
            if item.get(
                "status"
            )
            == "healthy"
        ]

        overall = (
            "healthy"
            if healthy
            and len(healthy)
            == len(exchange_results)
            else (
                "degraded"
                if healthy
                else "unhealthy"
            )
        )

        return {
            "status": overall,
            "exchanges": exchange_results,
            "is_live": (
                self.enable_real_trading
            ),
            "timestamp": (
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),
        }

    async def _health_check_single(
        self,
        exchange_name: str,
    ) -> Dict[str, Any]:

        exchange = (
            self._get_exchange_by_name(
                exchange_name
            )
        )

        if not exchange:

            return {
                "status": "disabled",
                "exchange": exchange_name,
                "message": (
                    "Exchange is not initialized."
                ),
            }

        try:

            ticker = await exchange.fetch_ticker(
                "BTC/USDT:USDT"
            )

            return {
                "status": "healthy",
                "exchange": exchange_name,
                "btc_price": ticker.get(
                    "last"
                ),
                "exchange_type": "futures",
                "leverage": self.leverage,
                "margin_type": self.margin_type,
                "testnet": self.use_testnet,
                "is_live": (
                    self.enable_real_trading
                ),
                "timestamp": (
                    datetime.now(
                        timezone.utc
                    ).isoformat()
                ),
            }

        except Exception as exc:

            return {
                "status": "unhealthy",
                "exchange": exchange_name,
                "error": str(exc),
                "timestamp": (
                    datetime.now(
                        timezone.utc
                    ).isoformat()
                ),
            }

    # =========================================================
    # PAPER TRADING
    # =========================================================

    async def paper_execute_open_position(
        self,
        position: Any,
    ) -> Dict[str, Any]:

        timestamp = (
            datetime.now(
                timezone.utc
            ).isoformat()
        )

        order_id = (
            f"PAPER-OPEN-"
            f"{uuid.uuid4().hex[:12]}"
        )

        logger.info(
            f"📝 PAPER TRADE OPEN | "
            f"{position.action} "
            f"{position.quantity} "
            f"{position.symbol} @ "
            f"${position.entry_price:.6f}"
        )

        return {
            "success": True,
            "paper_trade": True,
            "exchange": "PAPER",
            "entry_order_id": order_id,
            "sl_order_id": (
                f"{order_id}-SL"
            ),
            "tp_order_id": (
                f"{order_id}-TP"
            ),
            "filled_price": float(
                position.entry_price
            ),
            "amount": float(
                position.quantity
            ),
            "symbol": position.symbol,
            "timestamp": timestamp,
        }

    async def paper_execute_close_position(
        self,
        position: Any,
        reason: str = "MANUAL",
    ) -> Dict[str, Any]:

        timestamp = (
            datetime.now(
                timezone.utc
            ).isoformat()
        )

        order_id = (
            f"PAPER-CLOSE-"
            f"{uuid.uuid4().hex[:12]}"
        )

        logger.info(
            f"📝 PAPER TRADE CLOSE | "
            f"{position.symbol} @ "
            f"${position.current_price:.6f} | "
            f"Reason={reason}"
        )

        return {
            "success": True,
            "paper_trade": True,
            "exchange": "PAPER",
            "close_order_id": order_id,
            "exit_price": float(
                position.current_price
            ),
            "reason": reason,
            "symbol": position.symbol,
            "timestamp": timestamp,
        }

    # =========================================================
    # STATUS
    # =========================================================

    def get_status(self) -> Dict[str, Any]:

        return {
            "version": self.VERSION,
            "real_trading_enabled": (
                self.enable_real_trading
            ),
            "testnet": self.use_testnet,
            "initialized": self.is_initialized,
            "closed": self._closed,
            "binance_connected": bool(
                self.binance_exchange
            ),
            "bybit_connected": bool(
                self.bybit_exchange
            ),
            "leverage": self.leverage,
            "margin_type": self.margin_type,
            "active_symbols": list(
                self.active_orders.keys()
            ),
            "tracked_positions": len(
                self._position_execution
            ),
        }

    # =========================================================
    # CLEANUP
    # =========================================================

    async def _safe_close_exchange(
        self,
        exchange,
    ) -> None:

        if not exchange:
            return

        try:

            await exchange.close()

        except Exception as exc:

            logger.debug(
                f"Exchange close notice: "
                f"{exc}"
            )

    async def close(self):
        """
        Gracefully close all exchange connections.
        """

        if self._closed:

            return

        logger.info(
            "🛑 Closing RealTradeExecutor..."
        )

        exchanges = [
            (
                "BINANCE",
                self.binance_exchange,
            ),
            (
                "BYBIT",
                self.bybit_exchange,
            ),
        ]

        for name, exchange in exchanges:

            if not exchange:
                continue

            try:

                await exchange.close()

                logger.info(
                    f"✅ {name} exchange "
                    f"connection closed."
                )

            except Exception as exc:

                logger.error(
                    f"❌ Error closing "
                    f"{name}: {exc}"
                )

        self.binance_exchange = None
        self.bybit_exchange = None

        self.is_initialized = False
        self._closed = True

        logger.info(
            "✅ RealTradeExecutor cleanup "
            "complete."
        )

    # =========================================================
    # ASYNC CONTEXT MANAGER
    # =========================================================

    async def __aenter__(self):

        await self.initialize()

        return self

    async def __aexit__(
        self,
        exc_type,
        exc_val,
        exc_tb,
    ):

        await self.close()