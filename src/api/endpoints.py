"""
api_endpoints.py
================

REST API layer for SmartCrypto AI v3.

Responsibilities
----------------
- Expose market analysis
- Expose order-book intelligence
- Expose AI trading signals
- Expose portfolio/position information
- Expose trade execution controls
- Expose trade history/performance
- Expose Telegram status
- Expose system health/readiness

Architecture
------------
FastAPI
    |
    +-- MarketAnalyzer
    +-- OrderBookMonitor
    +-- SignalGenerator
    +-- PortfolioManager
    +-- RealTradeExecutor
    +-- HistoryManager
    +-- TelegramService

Important
---------
This module should NOT contain trading logic.
Trading logic belongs inside the service layer.

The API is only an interface to those services.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field


# ============================================================================
# LOGGING
# ============================================================================

logger = logging.getLogger(__name__)


# ============================================================================
# ROUTER
# ============================================================================

router = APIRouter(
    prefix="/api/v1",
    tags=["SmartCrypto AI"],
)


# ============================================================================
# SERVICE REGISTRY
# ============================================================================
#
# The services are injected here rather than recreated inside every endpoint.
# This is important because several of these services maintain state.
#
# In server.py / main.py:
#
#     from api_endpoints import router, configure_services
#
#     configure_services(
#         market_analyzer=market_analyzer,
#         orderbook_monitor=orderbook_monitor,
#         portfolio_manager=portfolio_manager,
#         signal_generator=signal_generator,
#         trade_executor=trade_executor,
#         history_manager=history_manager,
#         telegram_service=telegram_service,
#     )
#
#     app.include_router(router)
#
# ============================================================================

class ServiceRegistry:
    def __init__(self) -> None:
        self.market_analyzer: Any = None
        self.orderbook_monitor: Any = None
        self.portfolio_manager: Any = None
        self.signal_generator: Any = None
        self.trade_executor: Any = None
        self.history_manager: Any = None
        self.telegram_service: Any = None


services = ServiceRegistry()


def configure_services(
    *,
    market_analyzer: Any = None,
    orderbook_monitor: Any = None,
    portfolio_manager: Any = None,
    signal_generator: Any = None,
    trade_executor: Any = None,
    history_manager: Any = None,
    telegram_service: Any = None,
) -> None:
    """
    Register application services.

    This should be called once during application startup.
    """

    services.market_analyzer = market_analyzer
    services.orderbook_monitor = orderbook_monitor
    services.portfolio_manager = portfolio_manager
    services.signal_generator = signal_generator
    services.trade_executor = trade_executor
    services.history_manager = history_manager
    services.telegram_service = telegram_service


def set_market_analyzer(analyzer: Any) -> None:
    """
    Register the MarketAnalyzer and its associated services.
    
    Compatible with main.py startup workflow.
    """
    services.market_analyzer = analyzer
    if analyzer is not None:
        if hasattr(analyzer, 'orderbook_monitor') and analyzer.orderbook_monitor:
            services.orderbook_monitor = analyzer.orderbook_monitor
        if hasattr(analyzer, 'portfolio_manager') and analyzer.portfolio_manager:
            services.portfolio_manager = analyzer.portfolio_manager
        if hasattr(analyzer, 'signal_generator') and analyzer.signal_generator:
            services.signal_generator = analyzer.signal_generator
        if hasattr(analyzer, 'history_manager') and analyzer.history_manager:
            services.history_manager = analyzer.history_manager
        if hasattr(analyzer, 'telegram_service') and analyzer.telegram_service:
            services.telegram_service = analyzer.telegram_service
        if hasattr(analyzer, 'trade_executor') and analyzer.trade_executor:
            services.trade_executor = analyzer.trade_executor


def get_market_analyzer() -> Any:
    """Return the registered MarketAnalyzer instance."""
    if services.market_analyzer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Market analyzer service is not initialized.",
        )
    return services.market_analyzer


# ============================================================================
# HELPERS
# ============================================================================

START_TIME = time.time()


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


def require_service(service: Any, name: str) -> Any:
    """
    Ensure a service has been configured.
    """

    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"{name} service is not initialized.",
        )

    return service


async def maybe_await(value: Any) -> Any:
    """
    Allow endpoints to work with both synchronous and asynchronous
    service methods.
    """

    if asyncio.iscoroutine(value):
        return await value

    return value


async def call_service(
    service: Any,
    method_names: List[str],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """
    Call the first available method from a list.

    This provides a small compatibility layer while the service
    interfaces are being consolidated.
    """

    for method_name in method_names:

        method = getattr(service, method_name, None)

        if callable(method):

            try:
                return await maybe_await(
                    method(*args, **kwargs)
                )

            except TypeError:
                # Retry without kwargs for services whose signatures
                # are more restrictive.
                if kwargs:
                    try:
                        return await maybe_await(
                            method(*args)
                        )
                    except TypeError:
                        continue

                raise

    raise AttributeError(
        f"{type(service).__name__} does not expose any of: "
        f"{', '.join(method_names)}"
    )


def service_error(
    service_name: str,
    exc: Exception,
) -> HTTPException:

    logger.exception(
        "%s service error: %s",
        service_name,
        exc,
    )

    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail={
            "service": service_name,
            "error": str(exc),
        },
    )


# ============================================================================
# RESPONSE MODELS
# ============================================================================

class APIResponse(BaseModel):
    success: bool = True
    timestamp: str
    data: Any = None


class ErrorResponse(BaseModel):
    success: bool = False
    timestamp: str
    error: str


class TradeRequest(BaseModel):
    symbol: str = Field(..., min_length=1)
    side: str = Field(..., description="BUY or SELL")
    quantity: float = Field(..., gt=0)
    leverage: Optional[int] = Field(default=None, ge=1)
    stop_loss: Optional[float] = Field(default=None, gt=0)
    take_profit: Optional[float] = Field(default=None, gt=0)
    reduce_only: bool = False


class ClosePositionRequest(BaseModel):
    symbol: str = Field(..., min_length=1)
    quantity: Optional[float] = Field(default=None, gt=0)


class SignalRequest(BaseModel):
    symbol: str = Field(..., min_length=1)
    timeframe: Optional[str] = None


# ============================================================================
# ROOT / SYSTEM
# ============================================================================

@router.get(
    "/",
    response_model=APIResponse,
    summary="API information",
)
async def api_root() -> APIResponse:

    return APIResponse(
        timestamp=utc_now(),
        data={
            "name": "SmartCrypto AI",
            "version": "3.0.0",
            "api_version": "v1",
            "status": "online",
        },
    )


@router.get(
    "/health",
    response_model=APIResponse,
    summary="Application health",
)
async def health() -> APIResponse:

    return APIResponse(
        timestamp=utc_now(),
        data={
            "status": "healthy",
            "uptime_seconds": round(
                time.time() - START_TIME,
                2,
            ),
            "services": {
                "market_analyzer": services.market_analyzer is not None,
                "orderbook_monitor": services.orderbook_monitor is not None,
                "portfolio_manager": services.portfolio_manager is not None,
                "signal_generator": services.signal_generator is not None,
                "trade_executor": services.trade_executor is not None,
                "history_manager": services.history_manager is not None,
                "telegram_service": services.telegram_service is not None,
            },
        },
    )


@router.get(
    "/ready",
    response_model=APIResponse,
    summary="Readiness status",
)
async def readiness() -> APIResponse:

    required_services = {
        "market_analyzer": services.market_analyzer,
        "orderbook_monitor": services.orderbook_monitor,
        "signal_generator": services.signal_generator,
        "portfolio_manager": services.portfolio_manager,
        "history_manager": services.history_manager,
    }

    missing = [
        name
        for name, service in required_services.items()
        if service is None
    ]

    ready = len(missing) == 0

    if not ready:
        return APIResponse(
            success=False,
            timestamp=utc_now(),
            data={
                "ready": False,
                "missing_services": missing,
            },
        )

    return APIResponse(
        timestamp=utc_now(),
        data={
            "ready": True,
            "missing_services": [],
        },
    )


@router.get(
    "/status",
    response_model=APIResponse,
    summary="Full system status",
)
async def system_status() -> APIResponse:

    return APIResponse(
        timestamp=utc_now(),
        data={
            "application": "SmartCrypto AI",
            "version": "3.0.0",
            "environment": os.getenv(
                "ENVIRONMENT",
                "development",
            ),
            "uptime_seconds": round(
                time.time() - START_TIME,
                2,
            ),
        },
    )


# ============================================================================
# MARKET ANALYSIS
# ============================================================================

@router.get(
    "/market/{symbol}",
    response_model=APIResponse,
    summary="Get current market analysis",
)
async def market_analysis(
    symbol: str,
) -> APIResponse:

    analyzer = require_service(
        services.market_analyzer,
        "MarketAnalyzer",
    )

    try:

        result = await call_service(
            analyzer,
            [
                "analyze_market",
                "get_market_analysis",
                "analyze",
                "get_analysis",
            ],
            symbol,
        )

        return APIResponse(
            timestamp=utc_now(),
            data=result,
        )

    except Exception as exc:
        raise service_error(
            "MarketAnalyzer",
            exc,
        )


@router.get(
    "/market/{symbol}/regime",
    response_model=APIResponse,
    summary="Get current market regime",
)
async def market_regime(
    symbol: str,
) -> APIResponse:

    analyzer = require_service(
        services.market_analyzer,
        "MarketAnalyzer",
    )

    try:

        result = await call_service(
            analyzer,
            [
                "get_market_regime",
                "detect_regime",
                "market_regime",
                "get_regime",
            ],
            symbol,
        )

        return APIResponse(
            timestamp=utc_now(),
            data=result,
        )

    except Exception as exc:
        raise service_error(
            "MarketAnalyzer",
            exc,
        )


# ============================================================================
# ORDER BOOK
# ============================================================================

@router.get(
    "/orderbook/{symbol}",
    response_model=APIResponse,
    summary="Get current order-book intelligence",
)
async def orderbook(
    symbol: str,
) -> APIResponse:

    monitor = require_service(
        services.orderbook_monitor,
        "OrderBookMonitor",
    )

    try:

        result = await call_service(
            monitor,
            [
                "get_full_features",
                "get_orderbook_features",
                "get_features",
                "get_snapshot",
            ],
            symbol,
        )

        return APIResponse(
            timestamp=utc_now(),
            data=result,
        )

    except Exception as exc:
        raise service_error(
            "OrderBookMonitor",
            exc,
        )


@router.get(
    "/orderbook/{symbol}/imbalance",
    response_model=APIResponse,
    summary="Get order-book imbalance",
)
async def orderbook_imbalance(
    symbol: str,
) -> APIResponse:

    monitor = require_service(
        services.orderbook_monitor,
        "OrderBookMonitor",
    )

    try:

        result = await call_service(
            monitor,
            [
                "get_imbalance",
                "calculate_imbalance",
                "get_order_imbalance",
            ],
            symbol,
        )

        return APIResponse(
            timestamp=utc_now(),
            data=result,
        )

    except Exception as exc:
        raise service_error(
            "OrderBookMonitor",
            exc,
        )


@router.get(
    "/orderbook/{symbol}/pressure",
    response_model=APIResponse,
    summary="Get order-book pressure",
)
async def orderbook_pressure(
    symbol: str,
) -> APIResponse:

    monitor = require_service(
        services.orderbook_monitor,
        "OrderBookMonitor",
    )

    try:

        result = await call_service(
            monitor,
            [
                "get_pressure",
                "get_pressure_signal",
                "calculate_pressure",
                "get_buy_sell_pressure",
            ],
            symbol,
        )

        return APIResponse(
            timestamp=utc_now(),
            data=result,
        )

    except Exception as exc:
        raise service_error(
            "OrderBookMonitor",
            exc,
        )


# ============================================================================
# AI SIGNALS
# ============================================================================

@router.get(
    "/signals/latest",
    response_model=APIResponse,
    summary="Get latest AI signals",
)
async def latest_signals(
    symbol: Optional[str] = Query(
        default=None,
    ),
) -> APIResponse:

    generator = require_service(
        services.signal_generator,
        "SignalGenerator",
    )

    try:

        if symbol:

            result = await call_service(
                generator,
                [
                    "generate_signal",
                    "get_latest_signal",
                    "get_signal",
                ],
                symbol,
            )

        else:

            result = await call_service(
                generator,
                [
                    "get_latest_signals",
                    "get_all_signals",
                    "latest_signals",
                    "get_signals",
                ],
            )

        return APIResponse(
            timestamp=utc_now(),
            data=result,
        )

    except Exception as exc:
        raise service_error(
            "SignalGenerator",
            exc,
        )


@router.post(
    "/signals/generate",
    response_model=APIResponse,
    summary="Generate an AI trading signal",
)
async def generate_signal(
    request: SignalRequest,
) -> APIResponse:

    generator = require_service(
        services.signal_generator,
        "SignalGenerator",
    )

    try:

        result = await call_service(
            generator,
            [
                "generate_signal",
                "create_signal",
                "predict",
            ],
            request.symbol,
            timeframe=request.timeframe,
        )

        return APIResponse(
            timestamp=utc_now(),
            data=result,
        )

    except Exception as exc:
        raise service_error(
            "SignalGenerator",
            exc,
        )


# ============================================================================
# MODEL 4 STRATEGY DETECTORS
# ============================================================================

@router.get(
    "/strategies",
    response_model=APIResponse,
    summary="Get all 9 Model 4 strategy detector definitions and statuses",
)
async def list_strategies() -> APIResponse:
    generator = getattr(services, "signal_generator", None)
    engine = getattr(generator, "model4_engine", None) if generator else None

    detectors_info = {}
    strategy_descriptions = {
        "momentum_reversal": "Detects mean-reversion turning points following momentum exhaustion.",
        "ma_crossover": "Identifies moving average convergence/divergence and trend cross triggers.",
        "heikin_ashi": "Evaluates directional trend momentum and wick balance from smoothed HA candles.",
        "swing_trading": "Detects higher-high / higher-low swing continuation and pullback setups.",
        "candlestick": "Recognizes price action formations (hammers, shooting stars, inside/outside bars).",
        "role_reversal": "Detects broken resistance acting as support or broken support acting as resistance.",
        "bollinger_squeeze": "Identifies extreme volatility compression and breakout expansion.",
        "narrow_range": "Detects NR4/NR7/NR14 volatility contractions preceding explosive price moves.",
        "rsi_2": "Identifies extreme short-term overbought/oversold mean-reversion setups.",
    }

    if engine and hasattr(engine, "detectors"):
        for name in engine.STRATEGY_NAMES:
            det = engine.detectors.get(name, {})
            detectors_info[name] = {
                "name": name,
                "description": strategy_descriptions.get(name, ""),
                "loaded": bool(det),
                "threshold": float(det.get("threshold", 0.50)) if isinstance(det, dict) else 0.50,
                "features_count": len(det.get("feature_columns", [])) if isinstance(det, dict) else 94,
            }
    else:
        for name, desc in strategy_descriptions.items():
            detectors_info[name] = {
                "name": name,
                "description": desc,
                "loaded": False,
                "threshold": 0.50,
                "features_count": 94,
            }

    return APIResponse(
        timestamp=utc_now(),
        data={
            "total_detectors": len(detectors_info),
            "loaded_count": sum(1 for d in detectors_info.values() if d["loaded"]),
            "role": "Strategy Intelligence / Confirmation Layer",
            "detectors": detectors_info,
        },
    )


@router.get(
    "/strategies/{symbol}",
    response_model=APIResponse,
    summary="Evaluate all 9 Model 4 strategy detectors on a specific symbol",
)
async def evaluate_symbol_strategies(
    symbol: str,
) -> APIResponse:
    analyzer = require_service(
        services.market_analyzer,
        "MarketAnalyzer",
    )
    generator = getattr(services, "signal_generator", None)
    engine = getattr(generator, "model4_engine", None) if generator else None

    if not engine or not engine.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model 4 strategy engine is not initialized or loaded.",
        )

    # Get market data for symbol (case-insensitive & formatted)
    symbol_clean = symbol.upper().replace("/", "").replace("-", "").replace("_", "")
    market_data = None

    if hasattr(analyzer, "market_data") and analyzer.market_data:
        for k, v in analyzer.market_data.items():
            if k.upper().replace("/", "").replace("-", "").replace("_", "") == symbol_clean:
                market_data = v
                break

    if market_data is None and hasattr(analyzer, "historical_data") and analyzer.historical_data:
        for k, v in analyzer.historical_data.items():
            if k.upper().replace("/", "").replace("-", "").replace("_", "") == symbol_clean:
                market_data = v
                break

    if market_data is None or len(market_data) < 30:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Insufficient market data for symbol {symbol} (minimum 30 candles required).",
        )

    evaluation = engine.evaluate(market_data)
    return APIResponse(
        timestamp=utc_now(),
        data={
            "symbol": symbol.upper(),
            "price": float(market_data["close"].iloc[-1]),
            "evaluation": evaluation,
        },
    )


# ============================================================================
# PORTFOLIO
# ============================================================================

@router.get(
    "/portfolio",
    response_model=APIResponse,
    summary="Get portfolio overview",
)
async def portfolio() -> APIResponse:

    manager = require_service(
        services.portfolio_manager,
        "PortfolioManager",
    )

    try:

        result = await call_service(
            manager,
            [
                "get_portfolio",
                "get_portfolio_summary",
                "get_account_summary",
                "get_summary",
            ],
        )

        return APIResponse(
            timestamp=utc_now(),
            data=result,
        )

    except Exception as exc:
        raise service_error(
            "PortfolioManager",
            exc,
        )


@router.get(
    "/portfolio/positions",
    response_model=APIResponse,
    summary="Get open positions",
)
async def positions() -> APIResponse:

    manager = require_service(
        services.portfolio_manager,
        "PortfolioManager",
    )

    try:

        result = await call_service(
            manager,
            [
                "get_positions",
                "get_open_positions",
                "positions",
            ],
        )

        return APIResponse(
            timestamp=utc_now(),
            data=result,
        )

    except Exception as exc:
        raise service_error(
            "PortfolioManager",
            exc,
        )


@router.get(
    "/portfolio/positions/{symbol}",
    response_model=APIResponse,
    summary="Get position for a symbol",
)
async def position(
    symbol: str,
) -> APIResponse:

    manager = require_service(
        services.portfolio_manager,
        "PortfolioManager",
    )

    try:

        result = await call_service(
            manager,
            [
                "get_position",
                "get_symbol_position",
            ],
            symbol,
        )

        return APIResponse(
            timestamp=utc_now(),
            data=result,
        )

    except Exception as exc:
        raise service_error(
            "PortfolioManager",
            exc,
        )


# ============================================================================
# TRADING
# ============================================================================

@router.get(
    "/trading/orders",
    response_model=APIResponse,
    summary="Get active orders",
)
async def active_orders() -> APIResponse:

    executor = require_service(
        services.trade_executor,
        "RealTradeExecutor",
    )

    try:

        result = await call_service(
            executor,
            [
                "get_open_orders",
                "fetch_open_orders",
                "get_orders",
            ],
        )

        return APIResponse(
            timestamp=utc_now(),
            data=result,
        )

    except Exception as exc:
        raise service_error(
            "RealTradeExecutor",
            exc,
        )


@router.post(
    "/trading/order",
    response_model=APIResponse,
    summary="Execute a real trade",
)
async def execute_trade(
    request: TradeRequest,
) -> APIResponse:

    executor = require_service(
        services.trade_executor,
        "RealTradeExecutor",
    )

    side = request.side.upper()

    if side not in {"BUY", "SELL"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="side must be BUY or SELL",
        )

    try:

        result = await call_service(
            executor,
            [
                "execute_trade",
                "execute_order",
                "place_order",
                "create_order",
            ],
            symbol=request.symbol,
            side=side,
            quantity=request.quantity,
            leverage=request.leverage,
            stop_loss=request.stop_loss,
            take_profit=request.take_profit,
            reduce_only=request.reduce_only,
        )

        return APIResponse(
            timestamp=utc_now(),
            data=result,
        )

    except Exception as exc:
        raise service_error(
            "RealTradeExecutor",
            exc,
        )


@router.post(
    "/trading/close",
    response_model=APIResponse,
    summary="Close an open position",
)
async def close_position(
    request: ClosePositionRequest,
) -> APIResponse:

    executor = require_service(
        services.trade_executor,
        "RealTradeExecutor",
    )

    try:

        result = await call_service(
            executor,
            [
                "close_position",
                "close_trade",
                "close_symbol_position",
            ],
            symbol=request.symbol,
            quantity=request.quantity,
        )

        return APIResponse(
            timestamp=utc_now(),
            data=result,
        )

    except Exception as exc:
        raise service_error(
            "RealTradeExecutor",
            exc,
        )


# ============================================================================
# TRADE HISTORY
# ============================================================================

@router.get(
    "/history",
    response_model=APIResponse,
    summary="Get trading history",
)
async def history(
    symbol: Optional[str] = None,
    limit: int = Query(
        default=100,
        ge=1,
        le=1000,
    ),
) -> APIResponse:

    manager = require_service(
        services.history_manager,
        "HistoryManager",
    )

    try:

        result = await call_service(
            manager,
            [
                "get_recent_signals",
                "get_signals",
                "get_history",
                "get_trade_history",
                "get_trades",
                "get_closed_trades",
            ],
            symbol=symbol,
            limit=limit,
        )

        return APIResponse(
            timestamp=utc_now(),
            data=result,
        )

    except Exception as exc:
        raise service_error(
            "HistoryManager",
            exc,
        )


@router.get(
    "/history/performance",
    response_model=APIResponse,
    summary="Get trading performance",
)
async def performance() -> APIResponse:

    manager = require_service(
        services.history_manager,
        "HistoryManager",
    )

    try:

        result = await call_service(
            manager,
            [
                "get_performance",
                "calculate_performance",
                "get_performance_summary",
                "performance",
            ],
        )

        return APIResponse(
            timestamp=utc_now(),
            data=result,
        )

    except Exception as exc:
        raise service_error(
            "HistoryManager",
            exc,
        )


# ============================================================================
# TELEGRAM
# ============================================================================

@router.get(
    "/notifications/telegram",
    response_model=APIResponse,
    summary="Get Telegram service status",
)
async def telegram_status() -> APIResponse:

    telegram = require_service(
        services.telegram_service,
        "TelegramService",
    )

    try:

        result = await call_service(
            telegram,
            [
                "get_status",
                "status",
                "get_telegram_status",
            ],
        )

        return APIResponse(
            timestamp=utc_now(),
            data=result,
        )

    except Exception as exc:
        raise service_error(
            "TelegramService",
            exc,
        )


# ============================================================================
# COMBINED AI MARKET SNAPSHOT
# ============================================================================

@router.get(
    "/snapshot/{symbol}",
    response_model=APIResponse,
    summary="Get complete AI market snapshot",
)
async def market_snapshot(
    symbol: str,
) -> APIResponse:

    """
    Unified endpoint for dashboards.

    Instead of the frontend making five or six requests:

        /market/BTCUSDT
        /orderbook/BTCUSDT
        /orderbook/BTCUSDT/imbalance
        /signals/latest?symbol=BTCUSDT
        /portfolio/positions/BTCUSDT

    it can request:

        /snapshot/BTCUSDT
    """

    result: Dict[str, Any] = {
        "symbol": symbol,
        "timestamp": utc_now(),
    }

    # ------------------------------------------------------------------------
    # Market
    # ------------------------------------------------------------------------

    if services.market_analyzer is not None:

        try:

            result["market"] = await call_service(
                services.market_analyzer,
                [
                    "analyze_market",
                    "get_market_analysis",
                    "analyze",
                    "get_analysis",
                ],
                symbol,
            )

        except Exception as exc:

            logger.warning(
                "Snapshot market analysis failed: %s",
                exc,
            )

            result["market"] = {
                "error": str(exc),
            }

    # ------------------------------------------------------------------------
    # Order book
    # ------------------------------------------------------------------------

    if services.orderbook_monitor is not None:

        try:

            result["orderbook"] = await call_service(
                services.orderbook_monitor,
                [
                    "get_full_features",
                    "get_orderbook_features",
                    "get_features",
                    "get_snapshot",
                ],
                symbol,
            )

        except Exception as exc:

            logger.warning(
                "Snapshot orderbook failed: %s",
                exc,
            )

            result["orderbook"] = {
                "error": str(exc),
            }

    # ------------------------------------------------------------------------
    # AI signal
    # ------------------------------------------------------------------------

    if services.signal_generator is not None:

        try:

            result["signal"] = await call_service(
                services.signal_generator,
                [
                    "generate_signal",
                    "get_latest_signal",
                    "get_signal",
                ],
                symbol,
            )

        except Exception as exc:

            logger.warning(
                "Snapshot signal failed: %s",
                exc,
            )

            result["signal"] = {
                "error": str(exc),
            }

    # ------------------------------------------------------------------------
    # Position
    # ------------------------------------------------------------------------

    if services.portfolio_manager is not None:

        try:

            result["position"] = await call_service(
                services.portfolio_manager,
                [
                    "get_position",
                    "get_symbol_position",
                ],
                symbol,
            )

        except Exception as exc:

            logger.warning(
                "Snapshot position failed: %s",
                exc,
            )

            result["position"] = {
                "error": str(exc),
            }

    return APIResponse(
        timestamp=utc_now(),
        data=result,
    )


# ============================================================================
# EXPORT
# ============================================================================

__all__ = [
    "router",
    "configure_services",
]