# src/services/real_trade_executor.py

"""
SmartCrypto AI - Real Trade Executor
Version: 3.3.0

Responsibilities
----------------
- Execute real futures positions on Binance USD-M Futures.
- Execute real futures positions on Bybit USDT Linear Futures.
- Execute real futures positions on Bitget USDT Futures.
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

IMPORTANT BITGET RULE
---------------------
Bitget one-way position mode MUST NOT receive tradeSide.

Correct one-way entry:

    side=buy/sell
    oneWayMode=True
    NO tradeSide

Correct one-way close:

    side=opposite side
    reduceOnly=True
    NO tradeSide

tradeSide belongs to Bitget hedge-mode semantics.

PortfolioManager remains the FINAL risk gate.
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


class InsufficientBalanceError(Exception):
    """Raised when exchange futures balance is below minimum viable threshold."""

    def __init__(
        self,
        exchange_name: str,
        balance: float,
        message: Optional[str] = None,
    ) -> None:
        self.exchange_name = exchange_name
        self.balance = balance
        super().__init__(
            message
            or f"{exchange_name}: Available balance is ${balance:.2f} (< $5.00 min threshold)."
        )


class RealTradeExecutor:
    """
    Production-oriented asynchronous futures executor.

    Supported exchanges:

        - Binance USD-M Futures
        - Bybit USDT Linear Futures
        - Bitget USDT Futures

    This class is an execution layer only.

    PortfolioManager remains responsible for:
        - signal acceptance
        - risk gating
        - portfolio allocation
        - position lifecycle
    """

    VERSION = "3.3.0"

    SUPPORTED_EXCHANGES = (
        "BINANCE",
        "BYBIT",
        "BITGET",
    )

    DEFAULT_LEVERAGE = 3
    DEFAULT_MARGIN_TYPE = "ISOLATED"

    DEFAULT_ORDER_TIMEOUT = 30
    DEFAULT_MAX_RETRIES = 3

    FILL_POLL_INTERVAL = 1.0
    FILL_CONFIRM_TIMEOUT = 15.0

    BALANCE_SAFETY_BUFFER = 0.95
    NOTIONAL_SLIPPAGE_BUFFER = 1.01

    # Existing live sizing behavior:
    # allocate 25% of free USDT as margin.
    DEFAULT_MARGIN_ALLOCATION_PCT = 0.25

    def __init__(
        self,
        settings: Optional[Settings] = None,
    ):
        self.settings = settings or get_settings()

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

        self.enable_bitget = bool(
            getattr(
                self.settings,
                "ENABLE_BITGET",
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

        self.margin_allocation_pct = float(
            getattr(
                self.settings,
                "REAL_TRADE_MARGIN_ALLOCATION_PCT",
                self.DEFAULT_MARGIN_ALLOCATION_PCT,
            )
        )

        self.margin_allocation_pct = min(
            max(
                self.margin_allocation_pct,
                0.01,
            ),
            0.95,
        )

        # =====================================================
        # EXCHANGE INSTANCES
        # =====================================================

        self.binance_exchange = None
        self.bybit_exchange = None
        self.bitget_exchange = None

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

        self._closed = False

    # =========================================================
    # INITIALIZATION
    # =========================================================

    async def initialize(self) -> bool:
        """
        Initialize all enabled exchanges.

        Returns True when at least one exchange is connected.
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

            successful: List[str] = []

            # -------------------------------------------------
            # Binance
            # -------------------------------------------------

            if self.enable_binance:

                try:
                    exchange = await self._create_binance()

                    if exchange:
                        self.binance_exchange = exchange
                        successful.append("BINANCE")

                except Exception as exc:

                    logger.error(
                        f"❌ Binance initialization failed: {exc}",
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
                    exchange = await self._create_bybit()

                    if exchange:
                        self.bybit_exchange = exchange
                        successful.append("BYBIT")

                except Exception as exc:

                    logger.error(
                        f"❌ Bybit initialization failed: {exc}",
                        exc_info=True,
                    )

                    await self._safe_close_exchange(
                        self.bybit_exchange
                    )

                    self.bybit_exchange = None

            # -------------------------------------------------
            # Bitget
            # -------------------------------------------------

            if self.enable_bitget:

                try:
                    exchange = await self._create_bitget()

                    if exchange:
                        self.bitget_exchange = exchange
                        successful.append("BITGET")

                except Exception as exc:

                    logger.error(
                        f"❌ Bitget initialization failed: {exc}",
                        exc_info=True,
                    )

                    await self._safe_close_exchange(
                        self.bitget_exchange
                    )

                    self.bitget_exchange = None

            self.is_initialized = bool(successful)

            if self.is_initialized:

                logger.warning(
                    f"🚀 RealTradeExecutor v{self.VERSION} "
                    f"initialized. Exchanges: "
                    f"{', '.join(successful)}"
                )

                if self.use_testnet:
                    logger.warning(
                        "🧪 TESTNET / SANDBOX MODE IS ENABLED."
                    )
                else:
                    logger.warning(
                        "🔴 LIVE MAINNET REAL-MONEY "
                        "EXECUTION IS ENABLED."
                    )

            else:

                logger.error(
                    "❌ No futures exchanges were initialized."
                )

            return self.is_initialized

    # =========================================================
    # EXCHANGE CREATION
    # =========================================================

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
                "⚠️ Binance API credentials are missing."
            )

            return None

        config = {
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

        proxy_url = getattr(
            self.settings,
            "EXCHANGE_PROXY_URL",
            None,
        )

        if proxy_url:
            config["aiohttp_proxy"] = proxy_url

        exchange = ccxt.binanceusdm(config)

        exchange.has["fetchCurrencies"] = False

        try:

            exchange_testnet = getattr(
                self.settings,
                "BINANCE_USE_TESTNET",
                None,
            )

            use_testnet = (
                self.use_testnet
                if exchange_testnet is None
                else bool(exchange_testnet)
            )

            if use_testnet:

                if hasattr(
                    exchange,
                    "enable_demo_trading",
                ):
                    exchange.enable_demo_trading(True)
                else:
                    exchange.set_sandbox_mode(True)

            try:
                await exchange.load_time_difference()
            except Exception as exc:
                logger.debug(
                    f"Binance time sync fallback: {exc}"
                )

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

        config = {
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

        proxy_url = getattr(
            self.settings,
            "EXCHANGE_PROXY_URL",
            None,
        )

        if proxy_url:
            config["aiohttp_proxy"] = proxy_url

        exchange = ccxt.bybit(config)

        exchange.has["fetchCurrencies"] = False

        try:

            exchange_testnet = getattr(
                self.settings,
                "BYBIT_USE_TESTNET",
                None,
            )

            use_testnet = (
                self.use_testnet
                if exchange_testnet is None
                else bool(exchange_testnet)
            )

            if use_testnet:
                exchange.set_sandbox_mode(True)

            try:
                await exchange.load_time_difference()
            except Exception as exc:
                logger.debug(
                    f"Bybit time sync fallback: {exc}"
                )

            await exchange.load_markets()
            await exchange.fetch_balance()

            logger.info(
                "✅ Bybit USDT Linear Futures connection established."
            )

            return exchange

        except Exception:

            await self._safe_close_exchange(exchange)

            raise

    async def _create_bitget(self):
        """Create and initialize Bitget USDT Futures."""

        api_key = str(
            getattr(
                self.settings,
                "BITGET_API_KEY",
                "",
            )
            or ""
        ).strip()

        api_secret = str(
            getattr(
                self.settings,
                "BITGET_API_SECRET",
                "",
            )
            or ""
        ).strip()

        api_passphrase = str(
            getattr(
                self.settings,
                "BITGET_API_PASSPHRASE",
                "",
            )
            or ""
        ).strip()

        if (
            not api_key
            or not api_secret
            or not api_passphrase
        ):

            logger.warning(
                "⚠️ Bitget API credentials "
                "(apiKey, secret, passphrase) are incomplete."
            )

            return None

        config = {
            "apiKey": api_key,
            "secret": api_secret,
            "password": api_passphrase,
            "enableRateLimit": True,
            "options": {
                "defaultType": "swap",
                "adjustForTimeDifference": True,
                "recvWindow": 20000,
                "fetchCurrencies": False,
            },
        }

        proxy_url = getattr(
            self.settings,
            "EXCHANGE_PROXY_URL",
            None,
        )

        if proxy_url:
            config["aiohttp_proxy"] = proxy_url

        exchange = ccxt.bitget(config)

        exchange.has["fetchCurrencies"] = False

        try:

            exchange_testnet = getattr(
                self.settings,
                "BITGET_USE_TESTNET",
                None,
            )

            use_testnet = (
                self.use_testnet
                if exchange_testnet is None
                else bool(exchange_testnet)
            )

            if use_testnet:
                exchange.set_sandbox_mode(True)

            try:
                await exchange.load_time_difference()
            except Exception as exc:
                logger.debug(
                    f"Bitget time sync fallback: {exc}"
                )

            await exchange.load_markets()
            await exchange.fetch_balance()

            logger.info(
                "✅ Bitget USDT Futures connection established."
            )

            return exchange

        except Exception:

            await self._safe_close_exchange(exchange)

            raise

    # =========================================================
    # SYMBOL HELPERS
    # =========================================================

    def _normalize_base_symbol(
        self,
        symbol: str,
    ) -> str:

        value = str(
            symbol or ""
        ).strip().upper()

        value = value.replace(
            ":USDT",
            "",
        )

        value = value.replace(
            "/USDT",
            "",
        )

        if value.endswith("USDT"):
            value = value[:-4]

        return value

    def _get_ccxt_symbol(
        self,
        exchange,
        symbol: str,
    ) -> str:

        value = str(
            symbol or ""
        ).strip().upper()

        if value in exchange.markets:
            return value

        base = self._normalize_base_symbol(
            value
        )

        candidates = [
            f"{base}/USDT:USDT",
            f"{base}/USDT",
        ]

        for candidate in candidates:

            if candidate in exchange.markets:
                return candidate

        # Final market-ID lookup.
        for market_symbol, market in exchange.markets.items():

            market_id = str(
                market.get("id", "")
            ).upper()

            if market_id == value:
                return market_symbol

            if (
                market_id == f"{base}USDT"
                and market.get("swap", False)
            ):
                return market_symbol

        raise ValueError(
            f"Unsupported futures symbol {symbol!r} "
            f"for {exchange.id}."
        )

    # =========================================================
    # EXCHANGE SELECTION
    # =========================================================

    def _get_exchange(
        self,
        exchange_name: str,
    ):
        name = str(
            exchange_name
        ).upper()

        if name == "BINANCE":
            return self.binance_exchange

        if name == "BYBIT":
            return self.bybit_exchange

        if name == "BITGET":
            return self.bitget_exchange

        raise ValueError(
            f"Unsupported exchange: {exchange_name}"
        )

    # =========================================================
    # MARGIN / LEVERAGE
    # =========================================================

    async def _configure_margin_and_leverage(
        self,
        exchange,
        exchange_name: str,
        symbol_ccxt: str,
    ) -> None:

        exchange_name = exchange_name.upper()

        # -----------------------------------------------------
        # Margin mode
        # -----------------------------------------------------

        try:
            margin_params: Dict[str, Any] = {}
            if exchange_name == "BITGET":
                margin_params["productType"] = "USDT-FUTURES"
                margin_params["marginCoin"] = "USDT"

            await exchange.set_margin_mode(
                self.margin_type,
                symbol_ccxt,
                margin_params,
            )

        except Exception as exc:

            message = str(exc).lower()

            # Exchanges often report "already isolated".
            already_set = (
                "already" in message
                or "same" in message
                or "not modified" in message
                or "margin mode is the same" in message
            )

            if not already_set:
                logger.warning(
                    f"⚠️ {exchange_name}: "
                    f"Could not set {self.margin_type} margin "
                    f"for {symbol_ccxt}: {exc}"
                )

        # -----------------------------------------------------
        # Leverage
        # -----------------------------------------------------

        try:

            params: Dict[str, Any] = {}

            if exchange_name == "BITGET":
                params["productType"] = "USDT-FUTURES"
                params["marginCoin"] = "USDT"
                if self.margin_type == "isolated":
                    # For Bitget isolated margin, configure leverage for long side
                    params["holdSide"] = "long"

            await exchange.set_leverage(
                self.leverage,
                symbol_ccxt,
                params,
            )

            # If isolated on Bitget, also ensure short side is configured
            if exchange_name == "BITGET" and self.margin_type == "isolated":
                try:
                    params_short = dict(params)
                    params_short["holdSide"] = "short"
                    await exchange.set_leverage(
                        self.leverage,
                        symbol_ccxt,
                        params_short,
                    )
                except Exception:
                    pass

        except Exception as exc:

            logger.warning(
                f"⚠️ {exchange_name}: "
                f"Could not set leverage {self.leverage}x "
                f"for {symbol_ccxt}: {exc}"
            )

        logger.info(
            f"⚙️ {exchange_name}: "
            f"{symbol_ccxt} configured with "
            f"{self.leverage}x "
            f"{self.margin_type} margin."
        )

    # =========================================================
    # BITGET POSITION MODE
    # =========================================================

    async def _ensure_bitget_one_way_mode(
        self,
        exchange,
        symbol_ccxt: str,
    ) -> None:
        """
        Ensure Bitget uses one-way position mode.

        IMPORTANT:
        We do not blindly switch position mode while positions/orders
        exist. Bitget position mode is account/product scoped and
        may reject a mode change when positions/orders are active.

        The normal path simply attempts to set one-way mode.
        """

        if exchange is None:
            raise RuntimeError(
                "Bitget exchange is not initialized."
            )

        try:

            await exchange.set_position_mode(
                False,
                symbol_ccxt,
                {
                    "productType": "USDT-FUTURES",
                },
            )

            logger.info(
                f"BITGET: {symbol_ccxt} configured for one-way position mode."
            )

        except Exception as exc:

            message = str(exc).lower()

            # Already in one-way mode or active positions exist
            already_one_way = (
                "already" in message
                or "same" in message
                or "no change" in message
                or "not modified" in message
                or "consistent" in message
                or "repeat" in message
                or "40775" in message
                or "40017" in message
                or "has position" in message
                or "cannot be changed" in message
            )

            if already_one_way:

                logger.info(
                    f"⚙️ BITGET: {symbol_ccxt} "
                    "already in one-way position mode."
                )

                return

            # Do not hide a real mode conflict.
            raise RuntimeError(
                f"BITGET position-mode configuration failed "
                f"for {symbol_ccxt}: {exc}"
            ) from exc

    # =========================================================
    # BALANCE HELPERS
    # =========================================================

    async def _fetch_free_usdt(
        self,
        exchange,
        exchange_name: str,
    ) -> float:

        balance = await exchange.fetch_balance()

        free = None

        try:
            free = balance.get("free", {}).get("USDT")
        except Exception:
            free = None

        if free is None:

            try:
                free = balance.get("USDT", {}).get("free")
            except Exception:
                free = None

        if free is None:

            try:
                free = balance.get("total", {}).get("USDT")
            except Exception:
                free = None

        try:
            free_float = float(
                free or 0.0
            )
        except (
            TypeError,
            ValueError,
        ):
            free_float = 0.0

        if not math.isfinite(free_float):
            free_float = 0.0

        logger.info(
            f"💰 {exchange_name}: "
            f"Free USDT=${free_float:.2f}"
        )

        return max(
            0.0,
            free_float,
        )

    # =========================================================
    # POSITION QUANTITY SIZING
    # =========================================================

    async def _calculate_exchange_quantity(
        self,
        exchange,
        exchange_name: str,
        symbol_ccxt: str,
        reference_price: float,
    ) -> Tuple[float, float, float]:
        """
        Size the live exchange position from actual exchange equity.

        Returns:

            quantity
            target_margin
            notional
        """

        if (
            reference_price <= 0
            or not math.isfinite(reference_price)
        ):
            raise ValueError(
                f"Invalid reference price: {reference_price}"
            )

        free_usdt = await self._fetch_free_usdt(
            exchange,
            exchange_name,
        )

        if free_usdt < 5.0:
            raise InsufficientBalanceError(
                exchange_name,
                free_usdt,
            )

        target_margin = (
            free_usdt
            * self.margin_allocation_pct
            * self.BALANCE_SAFETY_BUFFER
        )

        if target_margin <= 0:
            raise RuntimeError(
                f"{exchange_name}: "
                "Calculated margin allocation is zero."
            )

        notional = (
            target_margin
            * self.leverage
        )

        notional *= self.NOTIONAL_SLIPPAGE_BUFFER

        raw_quantity = (
            notional
            / reference_price
        )

        market = exchange.market(
            symbol_ccxt
        )

        min_amount = (
            market.get("limits", {})
            .get("amount", {})
            .get("min")
        )

        if min_amount is not None:

            try:
                raw_quantity = max(
                    raw_quantity,
                    float(min_amount),
                )
            except (
                TypeError,
                ValueError,
            ):
                pass

        quantity = float(
            exchange.amount_to_precision(
                symbol_ccxt,
                raw_quantity,
            )
        )

        if quantity <= 0:
            raise RuntimeError(
                f"{exchange_name}: "
                "Exchange precision reduced quantity to zero."
            )

        logger.info(
            f"⚖️ {exchange_name}: "
            f"Sized trade to real exchange equity. "
            f"Free=${free_usdt:.2f}, "
            f"Margin Allocation=${target_margin:.2f} "
            f"({self.margin_allocation_pct * 100:.0f}%), "
            f"Leverage={self.leverage}x, "
            f"Notional=${notional:.2f}, "
            f"Qty={quantity}"
        )

        return (
            quantity,
            target_margin,
            notional,
        )

    # =========================================================
    # ENTRY PARAMETER CONSTRUCTION
    # =========================================================

    def _build_entry_params(
        self,
        exchange_name: str,
        client_order_id: str,
    ) -> Dict[str, Any]:

        exchange_name = exchange_name.upper()

        params: Dict[str, Any] = {}

        if exchange_name == "BINANCE":

            params["newClientOrderId"] = client_order_id

        elif exchange_name == "BYBIT":

            params["orderLinkId"] = client_order_id

        elif exchange_name == "BITGET":

            params["clientOid"] = client_order_id

            # =================================================
            # CRITICAL BITGET FIX
            # =================================================
            #
            # Bitget one-way mode:
            #
            #     oneWayMode=True
            #     tradeSide MUST NOT be sent.
            #
            # The previous implementation sent:
            #
            #     oneWayMode=True
            #     tradeSide="open"
            #
            # which caused:
            #
            #     40774
            #
            # "The order type for unilateral position must also
            # be the unilateral position type."
            #
            # Do NOT add tradeSide here.
            # =================================================

            params["oneWayMode"] = True

            params.pop(
                "tradeSide",
                None,
            )

            params.pop(
                "hedged",
                None,
            )

        return params

    # =========================================================
    # CLOSE PARAMETER CONSTRUCTION
    # =========================================================

    def _build_close_params(
        self,
        exchange_name: str,
        client_order_id: str,
    ) -> Dict[str, Any]:

        exchange_name = exchange_name.upper()

        params: Dict[str, Any] = {}

        if exchange_name == "BINANCE":

            params["newClientOrderId"] = client_order_id
            params["reduceOnly"] = True

        elif exchange_name == "BYBIT":

            params["orderLinkId"] = client_order_id
            params["reduceOnly"] = True

        elif exchange_name == "BITGET":

            params["clientOid"] = client_order_id

            # One-way Bitget close:
            #
            # opposite side
            # reduceOnly=True
            # NO tradeSide

            params["reduceOnly"] = True
            params["oneWayMode"] = True

            params.pop(
                "tradeSide",
                None,
            )

            params.pop(
                "hedged",
                None,
            )

        return params

    # =========================================================
    # CREATE MARKET ORDER
    # =========================================================

    async def _create_market_order(
        self,
        exchange,
        exchange_name: str,
        symbol_ccxt: str,
        side: str,
        quantity: float,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:

        side = str(
            side
        ).lower()

        if side not in {
            "buy",
            "sell",
        }:
            raise ValueError(
                f"Invalid order side: {side}"
            )

        params = dict(
            params or {}
        )

        exchange_name = exchange_name.upper()

        logger.info(
            f"📤 {exchange_name}: "
            f"Submitting MARKET {side.upper()} "
            f"{quantity} {symbol_ccxt}"
        )

        try:

            order = await exchange.create_order(
                symbol_ccxt,
                "market",
                side,
                quantity,
                None,
                params,
            )

            return order

        except Exception as exc:

            error_text = str(exc)

            # -------------------------------------------------
            # BITGET 40774
            # -------------------------------------------------

            if (
                exchange_name == "BITGET"
                and "40774" in error_text
            ):

                logger.error(
                    "❌ BITGET 40774: "
                    "one-way position order contains an "
                    "incompatible tradeSide/position-mode parameter. "
                    "The order will NOT be retried with another "
                    "invalid parameter combination."
                )

                raise

            raise

    # =========================================================
    # FILL VERIFICATION
    # =========================================================

    async def _verify_fill(
        self,
        exchange,
        exchange_name: str,
        symbol_ccxt: str,
        order: Dict[str, Any],
    ) -> Dict[str, Any]:

        order_id = order.get("id")

        if not order_id:
            return order

        status = str(
            order.get(
                "status",
                "",
            )
        ).lower()

        filled = order.get(
            "filled"
        )

        average = order.get(
            "average"
        )

        if status in {
            "closed",
            "filled",
        }:

            return order

        deadline = (
            time.monotonic()
            + self.FILL_CONFIRM_TIMEOUT
        )

        latest = order

        while time.monotonic() < deadline:

            try:

                latest = await exchange.fetch_order(
                    order_id,
                    symbol_ccxt,
                )

                latest_status = str(
                    latest.get(
                        "status",
                        "",
                    )
                ).lower()

                if latest_status in {
                    "closed",
                    "filled",
                }:
                    return latest

                if latest_status in {
                    "canceled",
                    "cancelled",
                    "rejected",
                    "expired",
                }:
                    return latest

            except Exception as exc:

                logger.debug(
                    f"{exchange_name}: "
                    f"fill polling failed for "
                    f"{order_id}: {exc}"
                )

            await asyncio.sleep(
                self.FILL_POLL_INTERVAL
            )

        logger.warning(
            f"⚠️ {exchange_name}: "
            f"Fill confirmation timeout for "
            f"order {order_id}."
        )

        return latest

    # =========================================================
    # PROTECTIVE ORDERS
    # =========================================================

    async def _create_protective_orders(
        self,
        exchange,
        exchange_name: str,
        symbol_ccxt: str,
        action: str,
        quantity: float,
        stop_loss: Optional[float],
        take_profit: Optional[float],
    ) -> Dict[str, Any]:

        result = {
            "stop_loss": None,
            "take_profit": None,
            "errors": [],
        }

        action = str(
            action
        ).upper()

        close_side = (
            "sell"
            if action == "BUY"
            else "buy"
        )

        # -----------------------------------------------------
        # Stop Loss
        # -----------------------------------------------------

        if (
            stop_loss is not None
            and float(stop_loss) > 0
        ):

            try:

                order_type = "stop_market"
                stop_params: Dict[str, Any] = {
                    "stopPrice": float(stop_loss),
                    "reduceOnly": True,
                }

                if exchange_name == "BITGET":

                    order_type = "market"
                    stop_params["stopLossPrice"] = float(stop_loss)
                    stop_params.pop("stopPrice", None)
                    stop_params["clientOid"] = (
                        f"sl-{uuid.uuid4().hex[:20]}"
                    )
                    stop_params["oneWayMode"] = True
                    stop_params.pop(
                        "tradeSide",
                        None,
                    )

                elif exchange_name == "BYBIT":

                    stop_params["triggerPrice"] = (
                        float(stop_loss)
                    )

                    stop_params.pop(
                        "stopPrice",
                        None,
                    )

                    stop_params["orderLinkId"] = (
                        f"sl-{uuid.uuid4().hex[:20]}"
                    )

                elif exchange_name == "BINANCE":

                    stop_params["stopPrice"] = (
                        float(stop_loss)
                    )

                order = await exchange.create_order(
                    symbol_ccxt,
                    order_type,
                    close_side,
                    quantity,
                    None,
                    stop_params,
                )

                result["stop_loss"] = order

                logger.info(
                    f"🛡️ {exchange_name}: "
                    f"Stop Loss installed at "
                    f"{float(stop_loss):.8f}"
                )

            except Exception as exc:

                result["errors"].append(
                    f"stop_loss: {exc}"
                )

                logger.error(
                    f"❌ {exchange_name}: "
                    f"Failed to install Stop Loss: {exc}"
                )

        # -----------------------------------------------------
        # Take Profit
        # -----------------------------------------------------

        if (
            take_profit is not None
            and float(take_profit) > 0
        ):

            try:

                order_type = "take_profit_market"
                tp_params: Dict[str, Any] = {
                    "stopPrice": float(take_profit),
                    "reduceOnly": True,
                }

                if exchange_name == "BITGET":

                    order_type = "market"
                    tp_params["takeProfitPrice"] = float(take_profit)
                    tp_params.pop("stopPrice", None)
                    tp_params["clientOid"] = (
                        f"tp-{uuid.uuid4().hex[:20]}"
                    )
                    tp_params["oneWayMode"] = True
                    tp_params.pop(
                        "tradeSide",
                        None,
                    )

                elif exchange_name == "BYBIT":

                    tp_params["triggerPrice"] = (
                        float(take_profit)
                    )

                    tp_params.pop(
                        "stopPrice",
                        None,
                    )

                    tp_params["orderLinkId"] = (
                        f"tp-{uuid.uuid4().hex[:20]}"
                    )

                elif exchange_name == "BINANCE":

                    tp_params["stopPrice"] = (
                        float(take_profit)
                    )

                order = await exchange.create_order(
                    symbol_ccxt,
                    order_type,
                    close_side,
                    quantity,
                    None,
                    tp_params,
                )

                result["take_profit"] = order

                logger.info(
                    f"🎯 {exchange_name}: "
                    f"Take Profit installed at "
                    f"{float(take_profit):.8f}"
                )

            except Exception as exc:

                result["errors"].append(
                    f"take_profit: {exc}"
                )

                logger.error(
                    f"❌ {exchange_name}: "
                    f"Failed to install Take Profit: {exc}"
                )

        return result

    # =========================================================
    # OPEN POSITION - PUBLIC
    # =========================================================

    async def execute_open_position(
        self,
        position: Any,
    ) -> Dict[str, Any]:
        """
        Execute an opening position on every enabled exchange.

        Expected position attributes:

            symbol
            action
            current_price / entry_price
            quantity
            stop_loss
            take_profit
            id

        Returns an execution summary.
        """

        if not self.enable_real_trading:

            return {
                "status": "PAPER",
                "success": True,
                "executed": False,
                "exchange_results": {},
            }

        if not self.is_initialized:

            await self.initialize()

        if not self.is_initialized:

            return {
                "status": "FAILED",
                "success": False,
                "executed": False,
                "error": "No exchange initialized.",
                "exchange_results": {},
            }

        position_id = str(
            getattr(
                position,
                "id",
                uuid.uuid4().hex,
            )
        )

        lock = self._position_locks.setdefault(
            position_id,
            asyncio.Lock(),
        )

        async with lock:

            existing = self._position_execution.get(
                position_id
            )

            if existing and existing.get("opened"):

                logger.warning(
                    f"⚠️ Duplicate execution prevented "
                    f"for position {position_id}."
                )

                return existing

            symbol = str(
                getattr(
                    position,
                    "symbol",
                    "",
                )
            ).strip()

            action = str(
                getattr(
                    position,
                    "action",
                    "",
                )
            ).upper()

            if action not in {
                "BUY",
                "SELL",
            }:
                return {
                    "status": "FAILED",
                    "success": False,
                    "executed": False,
                    "error": f"Invalid action: {action}",
                    "exchange_results": {},
                }

            reference_price = self._safe_float(
                getattr(
                    position,
                    "entry_price",
                    getattr(
                        position,
                        "current_price",
                        0.0,
                    ),
                ),
                0.0,
            )

            if reference_price <= 0:

                reference_price = self._safe_float(
                    getattr(
                        position,
                        "current_price",
                        0.0,
                    ),
                    0.0,
                )

            if reference_price <= 0:

                return {
                    "status": "FAILED",
                    "success": False,
                    "executed": False,
                    "error": "Invalid position price.",
                    "exchange_results": {},
                }

            exchange_results: Dict[
                str,
                Dict[str, Any],
            ] = {}

            successful: List[str] = []
            failed: List[str] = []

            for exchange_name in self.SUPPORTED_EXCHANGES:

                exchange = self._get_exchange(
                    exchange_name
                )

                if exchange is None:
                    continue

                try:

                    symbol_ccxt = self._get_ccxt_symbol(
                        exchange,
                        symbol,
                    )

                    # -------------------------------------------------
                    # Margin / leverage
                    # -------------------------------------------------

                    await self._configure_margin_and_leverage(
                        exchange,
                        exchange_name,
                        symbol_ccxt,
                    )

                    # -------------------------------------------------
                    # Bitget position mode
                    # -------------------------------------------------

                    if exchange_name == "BITGET":

                        await self._ensure_bitget_one_way_mode(
                            exchange,
                            symbol_ccxt,
                        )

                    # -------------------------------------------------
                    # Actual exchange equity sizing
                    # -------------------------------------------------

                    (
                        quantity,
                        target_margin,
                        notional,
                    ) = await self._calculate_exchange_quantity(
                        exchange,
                        exchange_name,
                        symbol_ccxt,
                        reference_price,
                    )

                    side = (
                        "buy"
                        if action == "BUY"
                        else "sell"
                    )

                    client_order_id = (
                        f"st-{uuid.uuid4().hex[:20]}"
                    )

                    entry_params = (
                        self._build_entry_params(
                            exchange_name,
                            client_order_id,
                        )
                    )

                    logger.info(
                        f"📤 {exchange_name}: "
                        f"OPEN {side.upper()} "
                        f"{quantity} "
                        f"{exchange.market(symbol_ccxt)['id']}"
                    )

                    order = await self._create_market_order(
                        exchange,
                        exchange_name,
                        symbol_ccxt,
                        side,
                        quantity,
                        entry_params,
                    )

                    order = await self._verify_fill(
                        exchange,
                        exchange_name,
                        symbol_ccxt,
                        order,
                    )

                    status = str(
                        order.get(
                            "status",
                            "",
                        )
                    ).lower()

                    filled = self._safe_float(
                        order.get(
                            "filled",
                            quantity,
                        ),
                        quantity,
                    )

                    average_price = self._safe_float(
                        order.get(
                            "average",
                            order.get(
                                "price",
                                reference_price,
                            ),
                        ),
                        reference_price,
                    )

                    success = (
                        status in {
                            "",
                            "open",
                            "closed",
                            "filled",
                        }
                        and filled > 0
                    )

                    if not success:

                        raise RuntimeError(
                            f"Order was not confirmed filled. "
                            f"status={status!r}, "
                            f"filled={filled}"
                        )

                    protective = (
                        await self._create_protective_orders(
                            exchange,
                            exchange_name,
                            symbol_ccxt,
                            action,
                            filled,
                            getattr(
                                position,
                                "stop_loss",
                                None,
                            ),
                            getattr(
                                position,
                                "take_profit",
                                None,
                            ),
                        )
                    )

                    exchange_results[
                        exchange_name
                    ] = {
                        "success": True,
                        "status": "FILLED",
                        "symbol": symbol_ccxt,
                        "exchange_symbol": exchange.market(
                            symbol_ccxt
                        ).get("id"),
                        "side": side,
                        "quantity": filled,
                        "requested_quantity": quantity,
                        "average_price": average_price,
                        "notional": notional,
                        "margin": target_margin,
                        "leverage": self.leverage,
                        "order_id": order.get("id"),
                        "client_order_id": client_order_id,
                        "order": order,
                        "protective_orders": protective,
                    }

                    successful.append(
                        exchange_name
                    )

                    self._order_cache[
                        str(order.get("id"))
                    ] = exchange_results[
                        exchange_name
                    ]

                    self.active_orders.setdefault(
                        symbol,
                        {},
                    )[exchange_name] = (
                        exchange_results[
                            exchange_name
                        ]
                    )

                except InsufficientBalanceError as exc:

                    logger.info(
                        f"ℹ️ {exchange_name}: Available balance is "
                        f"${exc.balance:.2f} (< $5.00 min). Skipping order execution on this exchange."
                    )

                    exchange_results[
                        exchange_name
                    ] = {
                        "success": True,
                        "status": "SKIPPED_INSUFFICIENT_BALANCE",
                        "symbol": symbol_ccxt,
                        "free_usdt": exc.balance,
                        "note": "Skipped due to insufficient balance without failing trade.",
                    }

                except Exception as exc:

                    failed.append(
                        exchange_name
                    )

                    exchange_results[
                        exchange_name
                    ] = {
                        "success": False,
                        "status": "FAILED",
                        "error": str(exc),
                    }

                    logger.error(
                        f"❌ {exchange_name}: "
                        f"Open execution failed for "
                        f"{symbol}: {exc}",
                        exc_info=True,
                    )

            overall_status = (
                "SUCCESS"
                if successful and not failed
                else (
                    "PARTIAL"
                    if successful
                    else "FAILED"
                )
            )

            result = {
                "status": overall_status,
                "success": bool(successful),
                "executed": bool(successful),
                "position_id": position_id,
                "symbol": symbol,
                "action": action,
                "exchange_results": exchange_results,
                "successful_exchanges": successful,
                "failed_exchanges": failed,
                "timestamp": datetime.now(
                    timezone.utc
                ).isoformat(),
            }

            self._position_execution[
                position_id
            ] = {
                **result,
                "opened": bool(successful),
            }

            logger.info(
                f"📊 OPEN EXECUTION SUMMARY | "
                f"{symbol} | "
                f"Status={overall_status} | "
                f"Success={len(successful)} | "
                f"Failed={len(failed)}"
            )

            return result

    # =========================================================
    # CLOSE POSITION - PUBLIC
    # =========================================================

    async def execute_close_position(
        self,
        position: Any,
        reason: str = "MANUAL",
    ) -> Dict[str, Any]:
        """
        Close a live futures position.

        For Bitget one-way mode:

            long  -> SELL + reduceOnly
            short -> BUY  + reduceOnly

        tradeSide is never sent.
        """

        if not self.enable_real_trading:

            return {
                "status": "PAPER",
                "success": True,
                "executed": False,
                "reason": reason,
                "exchange_results": {},
            }

        if not self.is_initialized:

            await self.initialize()

        if not self.is_initialized:

            return {
                "status": "FAILED",
                "success": False,
                "executed": False,
                "error": "No exchange initialized.",
                "exchange_results": {},
            }

        position_id = str(
            getattr(
                position,
                "id",
                uuid.uuid4().hex,
            )
        )

        lock = self._position_locks.setdefault(
            position_id,
            asyncio.Lock(),
        )

        async with lock:

            symbol = str(
                getattr(
                    position,
                    "symbol",
                    "",
                )
            ).strip()

            action = str(
                getattr(
                    position,
                    "action",
                    "",
                )
            ).upper()

            quantity = self._safe_float(
                getattr(
                    position,
                    "quantity",
                    0.0,
                ),
                0.0,
            )

            if quantity <= 0:

                return {
                    "status": "FAILED",
                    "success": False,
                    "executed": False,
                    "error": "Invalid position quantity.",
                    "exchange_results": {},
                }

            close_side = (
                "sell"
                if action == "BUY"
                else "buy"
            )

            exchange_results: Dict[
                str,
                Dict[str, Any],
            ] = {}

            successful: List[str] = []
            failed: List[str] = []

            for exchange_name in self.SUPPORTED_EXCHANGES:

                exchange = self._get_exchange(
                    exchange_name
                )

                if exchange is None:
                    continue

                try:

                    symbol_ccxt = self._get_ccxt_symbol(
                        exchange,
                        symbol,
                    )

                    # -------------------------------------------------
                    # Use actual live exchange position size where
                    # possible instead of blindly trusting the
                    # portfolio quantity.
                    # -------------------------------------------------

                    live_quantity = await self._get_live_position_quantity(
                        exchange,
                        exchange_name,
                        symbol_ccxt,
                        action,
                    )

                    if live_quantity <= 0:
                        logger.info(
                            f"ℹ️ {exchange_name}: No active live position found on exchange for "
                            f"{symbol_ccxt} (live qty=0.0). Treating as already closed."
                        )

                        await self._cancel_symbol_open_orders(
                            exchange,
                            exchange_name,
                            symbol_ccxt,
                        )

                        exchange_results[
                            exchange_name
                        ] = {
                            "success": True,
                            "status": "ALREADY_CLOSED",
                            "symbol": symbol_ccxt,
                            "quantity": 0.0,
                            "reason": reason,
                            "note": "Position already closed or was never opened on exchange.",
                        }

                        successful.append(
                            exchange_name
                        )

                        continue

                    close_quantity = live_quantity

                    close_quantity = float(
                        exchange.amount_to_precision(
                            symbol_ccxt,
                            close_quantity,
                        )
                    )

                    if close_quantity <= 0:
                        raise RuntimeError(
                            "Exchange precision reduced close "
                            "quantity to zero."
                        )

                    client_order_id = (
                        f"cl-{uuid.uuid4().hex[:20]}"
                    )

                    close_params = (
                        self._build_close_params(
                            exchange_name,
                            client_order_id,
                        )
                    )

                    logger.info(
                        f"📤 {exchange_name}: "
                        f"CLOSE {close_side.upper()} "
                        f"{close_quantity} "
                        f"{exchange.market(symbol_ccxt)['id']} "
                        f"| Reason={reason}"
                    )

                    order = await self._create_market_order(
                        exchange,
                        exchange_name,
                        symbol_ccxt,
                        close_side,
                        close_quantity,
                        close_params,
                    )

                    order = await self._verify_fill(
                        exchange,
                        exchange_name,
                        symbol_ccxt,
                        order,
                    )

                    status = str(
                        order.get(
                            "status",
                            "",
                        )
                    ).lower()

                    filled = self._safe_float(
                        order.get(
                            "filled",
                            close_quantity,
                        ),
                        close_quantity,
                    )

                    average_price = self._safe_float(
                        order.get(
                            "average",
                            order.get(
                                "price",
                                getattr(
                                    position,
                                    "current_price",
                                    0.0,
                                ),
                            ),
                        ),
                        0.0,
                    )

                    success = (
                        status in {
                            "",
                            "open",
                            "closed",
                            "filled",
                        }
                        and filled > 0
                    )

                    if not success:

                        raise RuntimeError(
                            f"Close order was not confirmed filled. "
                            f"status={status!r}, "
                            f"filled={filled}"
                        )

                    # -------------------------------------------------
                    # Cancel remaining protective orders.
                    # -------------------------------------------------

                    await self._cancel_symbol_open_orders(
                        exchange,
                        exchange_name,
                        symbol_ccxt,
                    )

                    exchange_results[
                        exchange_name
                    ] = {
                        "success": True,
                        "status": "CLOSED",
                        "symbol": symbol_ccxt,
                        "exchange_symbol": exchange.market(
                            symbol_ccxt
                        ).get("id"),
                        "side": close_side,
                        "quantity": filled,
                        "average_price": average_price,
                        "order_id": order.get("id"),
                        "client_order_id": client_order_id,
                        "reason": reason,
                        "order": order,
                    }

                    successful.append(
                        exchange_name
                    )

                    logger.info(
                        f"✅ {exchange_name}: "
                        f"CLOSE {close_side.upper()} "
                        f"{filled} {symbol_ccxt} "
                        f"@ {average_price:.8f}"
                    )

                except Exception as exc:

                    failed.append(
                        exchange_name
                    )

                    exchange_results[
                        exchange_name
                    ] = {
                        "success": False,
                        "status": "FAILED",
                        "error": str(exc),
                    }

                    logger.error(
                        f"❌ {exchange_name}: "
                        f"Close execution failed for "
                        f"{symbol}: {exc}",
                        exc_info=True,
                    )

            overall_status = (
                "SUCCESS"
                if successful and not failed
                else (
                    "PARTIAL"
                    if successful
                    else "FAILED"
                )
            )

            result = {
                "status": overall_status,
                "success": bool(successful),
                "executed": bool(successful),
                "position_id": position_id,
                "symbol": symbol,
                "action": action,
                "reason": reason,
                "exchange_results": exchange_results,
                "successful_exchanges": successful,
                "failed_exchanges": failed,
                "timestamp": datetime.now(
                    timezone.utc
                ).isoformat(),
            }

            logger.info(
                f"📊 CLOSE EXECUTION SUMMARY | "
                f"{symbol} | "
                f"Status={overall_status} | "
                f"Success={len(successful)} | "
                f"Failed={len(failed)}"
            )

            return result

    # =========================================================
    # LIVE POSITION QUANTITY
    # =========================================================

    async def _get_live_position_quantity(
        self,
        exchange,
        exchange_name: str,
        symbol_ccxt: str,
        action: str,
    ) -> float:
        """
        Attempt to read actual live exchange position size.

        Returns zero when unavailable.
        """

        try:

            if not hasattr(
                exchange,
                "fetch_positions",
            ):
                return 0.0

            params: Dict[str, Any] = {}

            if exchange_name == "BITGET":
                params = {
                    "productType": "USDT-FUTURES",
                    "marginCoin": "USDT",
                }

            positions = await exchange.fetch_positions(
                [symbol_ccxt],
                params,
            )

            wanted_side = (
                "long"
                if action == "BUY"
                else "short"
            )

            best_quantity = 0.0

            for position in positions:

                side = str(
                    position.get(
                        "side",
                        "",
                    )
                ).lower()

                contracts = self._safe_float(
                    position.get(
                        "contracts",
                        position.get(
                            "contractSize",
                            0.0,
                        ),
                    ),
                    0.0,
                )

                if side == wanted_side:

                    best_quantity = max(
                        best_quantity,
                        abs(contracts),
                    )

            return best_quantity

        except Exception as exc:

            logger.debug(
                f"{exchange_name}: "
                f"Could not fetch live position size: {exc}"
            )

            return 0.0

    # =========================================================
    # CANCEL PROTECTIVE / OPEN ORDERS
    # =========================================================

    async def _cancel_symbol_open_orders(
        self,
        exchange,
        exchange_name: str,
        symbol_ccxt: str,
    ) -> None:

        try:

            if not hasattr(
                exchange,
                "fetch_open_orders",
            ):
                return

            orders = await exchange.fetch_open_orders(
                symbol_ccxt
            )

            for order in orders:

                order_id = order.get("id")

                if not order_id:
                    continue

                try:

                    await exchange.cancel_order(
                        order_id,
                        symbol_ccxt,
                    )

                    logger.info(
                        f"🧹 {exchange_name}: "
                        f"Cancelled protective/open order "
                        f"{order_id}"
                    )

                except Exception as exc:

                    logger.debug(
                        f"{exchange_name}: "
                        f"Could not cancel order "
                        f"{order_id}: {exc}"
                    )

        except Exception as exc:

            logger.debug(
                f"{exchange_name}: "
                f"Could not fetch open orders for "
                f"{symbol_ccxt}: {exc}"
            )

    # =========================================================
    # BALANCE QUERIES
    # =========================================================

    async def get_balance(
        self,
        exchange_name: str,
        currency: str = "USDT",
    ) -> Optional[Dict[str, Any]]:

        try:

            exchange = self._get_exchange(
                exchange_name
            )

            if exchange is None:
                return None

            balance = await exchange.fetch_balance()

            currency = currency.upper()

            free = self._safe_float(
                balance.get(
                    "free",
                    {},
                ).get(
                    currency,
                    0.0,
                ),
                0.0,
            )

            used = self._safe_float(
                balance.get(
                    "used",
                    {},
                ).get(
                    currency,
                    0.0,
                ),
                0.0,
            )

            total = self._safe_float(
                balance.get(
                    "total",
                    {},
                ).get(
                    currency,
                    free + used,
                ),
                free + used,
            )

            return {
                "exchange": exchange_name.upper(),
                "currency": currency,
                "free": free,
                "used": used,
                "total": total,
                "timestamp": datetime.now(
                    timezone.utc
                ).isoformat(),
            }

        except Exception as exc:

            logger.error(
                f"❌ {exchange_name}: "
                f"Balance query failed: {exc}"
            )

            return None

    # =========================================================
    # POSITION QUERIES
    # =========================================================

    async def get_positions(
        self,
        symbols: Optional[List[str]] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:

        result: Dict[
            str,
            List[Dict[str, Any]],
        ] = {}

        for exchange_name in self.SUPPORTED_EXCHANGES:

            exchange = self._get_exchange(
                exchange_name
            )

            if exchange is None:
                continue

            try:

                ccxt_symbols = None

                if symbols:

                    ccxt_symbols = []

                    for symbol in symbols:

                        try:

                            ccxt_symbols.append(
                                self._get_ccxt_symbol(
                                    exchange,
                                    symbol,
                                )
                            )

                        except Exception:
                            continue

                params: Dict[str, Any] = {}

                if exchange_name == "BITGET":

                    params = {
                        "productType": "USDT-FUTURES",
                        "marginCoin": "USDT",
                    }

                positions = await exchange.fetch_positions(
                    ccxt_symbols,
                    params,
                )

                cleaned: List[
                    Dict[str, Any]
                ] = []

                for position in positions:

                    contracts = self._safe_float(
                        position.get(
                            "contracts",
                            0.0,
                        ),
                        0.0,
                    )

                    if abs(contracts) <= 0:
                        continue

                    cleaned.append(
                        {
                            "symbol": position.get(
                                "symbol"
                            ),
                            "side": position.get(
                                "side"
                            ),
                            "contracts": contracts,
                            "entry_price": position.get(
                                "entryPrice"
                            ),
                            "mark_price": position.get(
                                "markPrice"
                            ),
                            "unrealized_pnl": position.get(
                                "unrealizedPnl"
                            ),
                            "leverage": position.get(
                                "leverage"
                            ),
                            "margin_mode": position.get(
                                "marginMode"
                            ),
                            "raw": position,
                        }
                    )

                result[
                    exchange_name
                ] = cleaned

            except Exception as exc:

                logger.error(
                    f"❌ {exchange_name}: "
                    f"Position query failed: {exc}"
                )

                result[
                    exchange_name
                ] = []

        return result

    # =========================================================
    # OPEN ORDERS
    # =========================================================

    async def get_open_orders(
        self,
        exchange_name: str,
        symbol: Optional[str] = None,
    ) -> List[Dict[str, Any]]:

        exchange = self._get_exchange(
            exchange_name
        )

        if exchange is None:
            return []

        try:

            ccxt_symbol = None

            if symbol:
                ccxt_symbol = self._get_ccxt_symbol(
                    exchange,
                    symbol,
                )

            return await exchange.fetch_open_orders(
                ccxt_symbol
            )

        except Exception as exc:

            logger.error(
                f"❌ {exchange_name}: "
                f"Open-order query failed: {exc}"
            )

            return []

    # =========================================================
    # HEALTH CHECK
    # =========================================================

    async def health_check(
        self,
        exchange_name: Optional[str] = None,
    ) -> Dict[str, Any]:

        names = (
            [exchange_name.upper()]
            if exchange_name
            else list(self.SUPPORTED_EXCHANGES)
        )

        result: Dict[str, Any] = {
            "version": self.VERSION,
            "initialized": self.is_initialized,
            "real_trading_enabled": self.enable_real_trading,
            "testnet": self.use_testnet,
            "exchanges": {},
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
        }

        for name in names:

            try:

                exchange = self._get_exchange(
                    name
                )

                if exchange is None:

                    result["exchanges"][name] = {
                        "connected": False,
                        "status": "NOT_INITIALIZED",
                    }

                    continue

                started = time.monotonic()

                await exchange.fetch_balance()

                latency_ms = (
                    time.monotonic()
                    - started
                ) * 1000.0

                result["exchanges"][name] = {
                    "connected": True,
                    "status": "HEALTHY",
                    "latency_ms": round(
                        latency_ms,
                        2,
                    ),
                    "exchange_id": exchange.id,
                }

            except Exception as exc:

                result["exchanges"][name] = {
                    "connected": False,
                    "status": "ERROR",
                    "error": str(exc),
                }

        return result

    # =========================================================
    # PAPER EXECUTION HELPER
    # =========================================================

    async def execute_paper_order(
        self,
        symbol: str,
        action: str,
        quantity: float,
        price: float,
    ) -> Dict[str, Any]:

        action = str(
            action
        ).upper()

        if action not in {
            "BUY",
            "SELL",
        }:
            raise ValueError(
                f"Invalid action: {action}"
            )

        if quantity <= 0:
            raise ValueError(
                "Quantity must be greater than zero."
            )

        if price <= 0:
            raise ValueError(
                "Price must be greater than zero."
            )

        return {
            "status": "PAPER_FILLED",
            "success": True,
            "executed": False,
            "paper": True,
            "symbol": symbol,
            "action": action,
            "quantity": quantity,
            "price": price,
            "notional": quantity * price,
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
        }

    # =========================================================
    # ORDER LOOKUP
    # =========================================================

    async def get_order(
        self,
        exchange_name: str,
        order_id: str,
        symbol: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:

        exchange = self._get_exchange(
            exchange_name
        )

        if exchange is None:
            return None

        try:

            ccxt_symbol = None

            if symbol:
                ccxt_symbol = self._get_ccxt_symbol(
                    exchange,
                    symbol,
                )

            return await exchange.fetch_order(
                order_id,
                ccxt_symbol,
            )

        except Exception as exc:

            logger.error(
                f"❌ {exchange_name}: "
                f"Order lookup failed: {exc}"
            )

            return None

    # =========================================================
    # CANCEL ORDER
    # =========================================================

    async def cancel_order(
        self,
        exchange_name: str,
        order_id: str,
        symbol: Optional[str] = None,
    ) -> bool:

        exchange = self._get_exchange(
            exchange_name
        )

        if exchange is None:
            return False

        try:

            ccxt_symbol = None

            if symbol:
                ccxt_symbol = self._get_ccxt_symbol(
                    exchange,
                    symbol,
                )

            await exchange.cancel_order(
                order_id,
                ccxt_symbol,
            )

            return True

        except Exception as exc:

            logger.error(
                f"❌ {exchange_name}: "
                f"Cancel order failed: {exc}"
            )

            return False

    # =========================================================
    # SAFE FLOAT
    # =========================================================

    @staticmethod
    def _safe_float(
        value: Any,
        default: float = 0.0,
    ) -> float:

        try:

            result = float(
                value
            )

            if not math.isfinite(
                result
            ):
                return float(default)

            return result

        except (
            TypeError,
            ValueError,
        ):

            return float(default)

    # =========================================================
    # SAFE CLOSE EXCHANGE
    # =========================================================

    async def _safe_close_exchange(
        self,
        exchange,
    ) -> None:

        if exchange is None:
            return

        try:

            close_method = getattr(
                exchange,
                "close",
                None,
            )

            if close_method:

                result = close_method()

                if asyncio.iscoroutine(result):
                    await result

        except Exception as exc:

            logger.debug(
                f"Exchange close error: {exc}"
            )

    # =========================================================
    # CLEANUP
    # =========================================================

    async def cleanup(
        self,
    ) -> None:
        """
        Gracefully close all CCXT connections.
        """

        if self._closed:
            return

        self._closed = True

        logger.info(
            "🧹 RealTradeExecutor cleanup started."
        )

        exchanges = [
            self.binance_exchange,
            self.bybit_exchange,
            self.bitget_exchange,
        ]

        for exchange in exchanges:

            await self._safe_close_exchange(
                exchange
            )

        self.binance_exchange = None
        self.bybit_exchange = None
        self.bitget_exchange = None

        self.is_initialized = False

        logger.info(
            "✅ RealTradeExecutor cleanup complete."
        )

    async def shutdown(
        self,
    ) -> None:
        """Alias for cleanup()."""

        await self.cleanup()

    async def close(
        self,
    ) -> None:
        """CCXT-style close alias."""

        await self.cleanup()