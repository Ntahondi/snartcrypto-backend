"""
src/services/orderbook_monitor.py

SnartCrypto AI v3.0.0+
Real-Time Binance Futures L2 Order Book Monitor

Purpose
-------
Provides market-microstructure information to the signal-generation
pipeline.

The order book is NOT an independent AI voting model.

It provides:
    - bid/ask spread
    - multi-depth liquidity
    - order-book imbalance
    - top-of-book pressure
    - weighted pressure
    - wall detection
    - liquidity density
    - VWAP information
    - short-term pressure changes
    - data freshness

Architecture
------------

Binance Futures L2
        │
        ▼
OrderBookMonitor
        │
        ├── liquidity
        ├── imbalance
        ├── walls
        ├── spread
        ├── pressure
        └── temporal change
                │
                ▼
        Signal Generator
                │
        ┌───────┴────────┐
        │                │
    AI Committee      Model 4
        │                │
        └───────┬────────┘
                ▼
             SIGNAL
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional

import numpy as np
import websockets

from src.core.config import Settings, get_settings
from src.utils.safe_logger import SafeLogger


logger = SafeLogger.get_logger(__name__)


class OrderBookMonitor:
    """
    Real-time L2 order-book monitor for Binance Futures.

    Important:
        This class provides market microstructure features.
        It does not make final BUY/SELL decisions.
    """

    # Binance stream provides 20 levels here.
    STREAM_DEPTH = 20

    # Snapshot history retained per symbol.
    HISTORY_SIZE = 120

    # Data older than this is considered stale.
    DEFAULT_STALE_SECONDS = 5.0

    # Depth windows expressed as distance from mid-price.
    DEPTH_LEVELS = (
        0.002,   # 0.2%
        0.005,   # 0.5%
        0.010,   # 1%
        0.020,   # 2%
    )

    def __init__(
        self,
        settings: Optional[Settings] = None,
        stale_seconds: float = DEFAULT_STALE_SECONDS,
    ):
        self.settings = settings or get_settings()

        self.stale_seconds = max(
            float(stale_seconds),
            0.5,
        )

        self.is_running = False

        # Latest processed features.
        self.orderbook_data: Dict[str, Dict[str, Any]] = {}

        # Short-term history for temporal analysis.
        self.orderbook_history: Dict[
            str,
            Deque[Dict[str, Any]]
        ] = {}

        # Active websocket connections.
        self.ws_connections: Dict[
            str,
            Any
        ] = {}

        # Statistics.
        self.update_count = 0
        self.error_count = 0
        self.connection_count = 0

        # Last update timestamp in Unix seconds.
        self.last_update: Dict[str, float] = {}

        logger.info(
            "📊 OrderBookMonitor initialized | "
            "depth=%s | stale_limit=%.1fs",
            self.STREAM_DEPTH,
            self.stale_seconds,
        )

    # ==================================================================
    # SYMBOL / URL HELPERS
    # ==================================================================

    @staticmethod
    def normalize_symbol(symbol: str) -> str:
        """
        Convert symbols into Binance stream-compatible form.

        Examples:
            BTCUSDT -> btcusdt
            BTC/USDT -> btcusdt
            BTC/USDT:USDT -> btcusdt
        """

        symbol = str(symbol or "").upper().strip()

        symbol = symbol.replace(
            "/",
            "",
        )

        symbol = symbol.replace(
            ":USDT",
            "",
        )

        return symbol.lower()

    def _get_ws_url(
        self,
        symbol: str,
    ) -> str:
        """
        Build Binance websocket URL according to exchange type.
        """

        exchange_type = str(
            getattr(
                self.settings,
                "EXCHANGE_TYPE",
                "future",
            )
            or "future"
        ).lower()

        stream = (
            f"{self.normalize_symbol(symbol)}"
            f"@depth{self.STREAM_DEPTH}@100ms"
        )

        if exchange_type in {
            "future",
            "futures",
            "usdm",
        }:
            host = "fstream.binance.com"

        else:
            host = "stream.binance.com:9443"

        return f"wss://{host}/ws/{stream}"

    # ==================================================================
    # MAIN MONITOR
    # ==================================================================

    async def start_monitoring(
        self,
        symbols: Optional[List[str]] = None,
    ) -> None:
        """
        Start concurrent order-book monitoring.
        """

        if symbols is None:
            symbols = list(
                getattr(
                    self.settings,
                    "SYMBOLS",
                    [],
                )
                or []
            )

        if not symbols:
            logger.warning(
                "⚠️ No symbols configured for OrderBookMonitor."
            )
            return

        self.is_running = True

        logger.info(
            "📊 Starting OrderBookMonitor for %d symbols",
            len(symbols),
        )

        tasks = [
            asyncio.create_task(
                self._monitor_symbol(symbol)
            )
            for symbol in symbols
        ]

        try:
            await asyncio.gather(
                *tasks,
                return_exceptions=True,
            )

        finally:
            self.is_running = False

    async def _monitor_symbol(
        self,
        symbol: str,
    ) -> None:
        """
        Maintain a resilient websocket connection for one symbol.
        """

        normalized_symbol = (
            self.normalize_symbol(symbol)
        )

        url = self._get_ws_url(
            normalized_symbol
        )

        logger.info(
            "🔗 Connecting OrderBook WebSocket: %s",
            normalized_symbol.upper(),
        )

        reconnect_delay = 1.0

        while self.is_running:

            websocket = None

            try:

                async with websockets.connect(
                    url,
                    ping_interval=20,
                    ping_timeout=20,
                    close_timeout=5,
                    max_size=2 ** 20,
                ) as websocket:

                    self.ws_connections[
                        normalized_symbol.upper()
                    ] = websocket

                    self.connection_count += 1
                    reconnect_delay = 1.0

                    logger.info(
                        "✅ OrderBook connected: %s",
                        normalized_symbol.upper(),
                    )

                    async for raw_message in websocket:

                        if not self.is_running:
                            break

                        try:
                            data = json.loads(
                                raw_message
                            )

                            if not (
                                isinstance(
                                    data,
                                    dict,
                                )
                            ):
                                continue

                            bids = data.get(
                                "bids"
                            )

                            asks = data.get(
                                "asks"
                            )

                            if not bids or not asks:
                                continue

                            features = (
                                self._process_orderbook(
                                    data,
                                    normalized_symbol.upper(),
                                )
                            )

                            self._store_features(
                                normalized_symbol.upper(),
                                features,
                            )

                        except json.JSONDecodeError:

                            self.error_count += 1

                            logger.warning(
                                "⚠️ Invalid order-book JSON: %s",
                                normalized_symbol.upper(),
                            )

                        except Exception as exc:

                            self.error_count += 1

                            logger.error(
                                "❌ Order-book processing error "
                                "%s: %s",
                                normalized_symbol.upper(),
                                exc,
                            )

            except asyncio.CancelledError:

                raise

            except Exception as exc:

                self.error_count += 1

                logger.warning(
                    "🔄 OrderBook connection lost "
                    "%s: %s",
                    normalized_symbol.upper(),
                    exc,
                )

            finally:

                self.ws_connections.pop(
                    normalized_symbol.upper(),
                    None,
                )

            if self.is_running:

                await asyncio.sleep(
                    reconnect_delay
                )

                reconnect_delay = min(
                    reconnect_delay * 2,
                    30.0,
                )

    # ==================================================================
    # PROCESS ORDER BOOK
    # ==================================================================

    def _process_orderbook(
        self,
        data: Dict[str, Any],
        symbol: str,
    ) -> Dict[str, Any]:
        """
        Convert raw L2 data into model-ready features.
        """

        try:

            bids_raw = data.get(
                "bids",
                [],
            )

            asks_raw = data.get(
                "asks",
                [],
            )

            bids = self._convert_levels(
                bids_raw
            )

            asks = self._convert_levels(
                asks_raw
            )

            if not bids or not asks:

                return self._get_default_features(
                    symbol
                )

            # Binance depth stream is normally sorted.
            # Still sort defensively.
            bids.sort(
                key=lambda x: x[0],
                reverse=True,
            )

            asks.sort(
                key=lambda x: x[0]
            )

            bids_array = np.asarray(
                bids,
                dtype=np.float64,
            )

            asks_array = np.asarray(
                asks,
                dtype=np.float64,
            )

            best_bid = float(
                bids_array[0, 0]
            )

            best_ask = float(
                asks_array[0, 0]
            )

            best_bid_size = float(
                bids_array[0, 1]
            )

            best_ask_size = float(
                asks_array[0, 1]
            )

            if best_bid <= 0 or best_ask <= 0:
                return self._get_default_features(
                    symbol
                )

            mid_price = (
                best_bid + best_ask
            ) / 2.0

            spread = (
                best_ask - best_bid
            )

            spread_ratio = (
                spread / mid_price
                if mid_price > 0
                else 0.0
            )

            spread_bps = (
                spread_ratio * 10_000
            )

            features: Dict[str, Any] = {
                "symbol": symbol,
                "timestamp": datetime.now(
                    timezone.utc
                ).isoformat(),

                "timestamp_unix": time.time(),

                "best_bid": best_bid,
                "best_ask": best_ask,
                "mid_price": mid_price,

                "spread": spread,
                "spread_ratio": spread_ratio,
                "spread_bps": spread_bps,

                "best_bid_size": best_bid_size,
                "best_ask_size": best_ask_size,

                "levels_available": min(
                    len(bids),
                    len(asks),
                ),
            }

            # ==========================================================
            # TOP OF BOOK IMBALANCE
            # ==========================================================

            top_total = (
                best_bid_size
                + best_ask_size
            )

            top_imbalance = (
                (
                    best_bid_size
                    - best_ask_size
                )
                / top_total
                if top_total > 0
                else 0.0
            )

            features[
                "top_imbalance"
            ] = float(
                np.clip(
                    top_imbalance,
                    -1,
                    1,
                )
            )

            # ==========================================================
            # MULTI-DEPTH LIQUIDITY
            # ==========================================================

            for depth in self.DEPTH_LEVELS:

                suffix = int(
                    depth * 1000
                )

                bid_limit = (
                    mid_price
                    * (1 - depth)
                )

                ask_limit = (
                    mid_price
                    * (1 + depth)
                )

                bid_mask = (
                    bids_array[:, 0]
                    >= bid_limit
                )

                ask_mask = (
                    asks_array[:, 0]
                    <= ask_limit
                )

                bid_volume = float(
                    np.sum(
                        bids_array[
                            bid_mask,
                            1,
                        ]
                    )
                )

                ask_volume = float(
                    np.sum(
                        asks_array[
                            ask_mask,
                            1,
                        ]
                    )
                )

                total_volume = (
                    bid_volume
                    + ask_volume
                )

                imbalance = (
                    (
                        bid_volume
                        - ask_volume
                    )
                    / total_volume
                    if total_volume > 0
                    else 0.0
                )

                features[
                    f"bid_volume_{suffix}bp"
                ] = bid_volume

                features[
                    f"ask_volume_{suffix}bp"
                ] = ask_volume

                features[
                    f"imbalance_{suffix}bp"
                ] = float(
                    np.clip(
                        imbalance,
                        -1,
                        1,
                    )
                )

            # ==========================================================
            # WEIGHTED DEPTH IMBALANCE
            # ==========================================================

            weighted_imbalance = (
                self._weighted_imbalance(
                    bids_array,
                    asks_array,
                    mid_price,
                )
            )

            features[
                "weighted_imbalance"
            ] = weighted_imbalance

            # ==========================================================
            # VWAP
            # ==========================================================

            bid_vwap = self._weighted_average_price(
                bids_array
            )

            ask_vwap = self._weighted_average_price(
                asks_array
            )

            features[
                "bid_vwap"
            ] = bid_vwap

            features[
                "ask_vwap"
            ] = ask_vwap

            # This measures the width between the two
            # liquidity-weighted prices.
            features[
                "vwap_spread_ratio"
            ] = (
                (ask_vwap - bid_vwap)
                / mid_price
                if mid_price > 0
                else 0.0
            )

            # Directional VWAP displacement.
            features[
                "bid_vwap_distance"
            ] = (
                (mid_price - bid_vwap)
                / mid_price
                if mid_price > 0
                else 0.0
            )

            features[
                "ask_vwap_distance"
            ] = (
                (ask_vwap - mid_price)
                / mid_price
                if mid_price > 0
                else 0.0
            )

            # ==========================================================
            # WALL DETECTION
            # ==========================================================

            bid_sizes = bids_array[:, 1]
            ask_sizes = asks_array[:, 1]

            avg_bid = float(
                np.mean(bid_sizes)
            )

            avg_ask = float(
                np.mean(ask_sizes)
            )

            median_bid = float(
                np.median(bid_sizes)
            )

            median_ask = float(
                np.median(ask_sizes)
            )

            max_bid = float(
                np.max(bid_sizes)
            )

            max_ask = float(
                np.max(ask_sizes)
            )

            features[
                "bid_wall_ratio"
            ] = (
                max_bid
                / max(
                    median_bid,
                    1e-12,
                )
            )

            features[
                "ask_wall_ratio"
            ] = (
                max_ask
                / max(
                    median_ask,
                    1e-12,
                )
            )

            max_bid_idx = int(
                np.argmax(bid_sizes)
            )

            max_ask_idx = int(
                np.argmax(ask_sizes)
            )

            features[
                "wall_bid_price"
            ] = float(
                bids_array[
                    max_bid_idx,
                    0,
                ]
            )

            features[
                "wall_ask_price"
            ] = float(
                asks_array[
                    max_ask_idx,
                    0,
                ]
            )

            features[
                "wall_imbalance"
            ] = float(
                np.clip(
                    (
                        max_bid
                        - max_ask
                    )
                    / (
                        max_bid
                        + max_ask
                        + 1e-12
                    ),
                    -1,
                    1,
                )
            )

            # ==========================================================
            # LIQUIDITY DENSITY
            # ==========================================================

            bid_reference = max(
                median_bid,
                1e-12,
            )

            ask_reference = max(
                median_ask,
                1e-12,
            )

            bid_density = float(
                np.mean(
                    bid_sizes
                    >= bid_reference
                )
            )

            ask_density = float(
                np.mean(
                    ask_sizes
                    >= ask_reference
                )
            )

            features[
                "bid_density"
            ] = bid_density

            features[
                "ask_density"
            ] = ask_density

            features[
                "liquidity_density_imbalance"
            ] = float(
                bid_density
                - ask_density
            )

            # ==========================================================
            # PRESSURE
            # ==========================================================

            # Combine independent pieces rather than counting
            # buy_pressure twice.
            pressure = (
                0.50
                * features[
                    "imbalance_10bp"
                ]
                + 0.25
                * features[
                    "top_imbalance"
                ]
                + 0.15
                * weighted_imbalance
                + 0.10
                * features[
                    "wall_imbalance"
                ]
            )

            pressure = float(
                np.clip(
                    pressure,
                    -1,
                    1,
                )
            )

            features[
                "order_pressure"
            ] = pressure

            features[
                "buy_pressure"
            ] = float(
                (pressure + 1)
                / 2
            )

            if pressure >= 0.30:

                pressure_direction = "BUY"
                pressure_strength = "STRONG"

            elif pressure >= 0.10:

                pressure_direction = "BUY"
                pressure_strength = "MODERATE"

            elif pressure <= -0.30:

                pressure_direction = "SELL"
                pressure_strength = "STRONG"

            elif pressure <= -0.10:

                pressure_direction = "SELL"
                pressure_strength = "MODERATE"

            else:

                pressure_direction = "NEUTRAL"
                pressure_strength = "LOW"

            features[
                "pressure_direction"
            ] = pressure_direction

            features[
                "pressure_strength"
            ] = pressure_strength

            # ==========================================================
            # QUALITY
            # ==========================================================

            features[
                "data_quality"
            ] = self._calculate_data_quality(
                bids_array,
                asks_array,
                spread_bps,
            )

            return features

        except Exception as exc:

            logger.error(
                "❌ Failed to process order book "
                "%s: %s",
                symbol,
                exc,
                exc_info=True,
            )

            return self._get_default_features(
                symbol
            )

    # ==================================================================
    # LEVEL PROCESSING
    # ==================================================================

    @staticmethod
    def _convert_levels(
        levels: List[Any],
    ) -> List[List[float]]:
        """
        Convert raw Binance [price, quantity] levels.
        """

        result = []

        for level in levels:

            try:

                if (
                    not isinstance(
                        level,
                        (list, tuple),
                    )
                    or len(level) < 2
                ):
                    continue

                price = float(
                    level[0]
                )

                quantity = float(
                    level[1]
                )

                if (
                    price <= 0
                    or quantity < 0
                ):
                    continue

                result.append(
                    [
                        price,
                        quantity,
                    ]
                )

            except (
                TypeError,
                ValueError,
            ):
                continue

        return result

    # ==================================================================
    # WEIGHTED METRICS
    # ==================================================================

    @staticmethod
    def _weighted_average_price(
        levels: np.ndarray,
    ) -> float:

        if (
            levels.size == 0
            or np.sum(levels[:, 1]) <= 0
        ):
            return 0.0

        return float(
            np.average(
                levels[:, 0],
                weights=levels[:, 1],
            )
        )

    @staticmethod
    def _weighted_imbalance(
        bids: np.ndarray,
        asks: np.ndarray,
        mid_price: float,
    ) -> float:
        """
        Give more importance to liquidity closer to the market.

        Distance weighting:
            weight = 1 / (1 + distance)

        where distance is relative distance from mid-price.
        """

        def weighted_volume(
            levels: np.ndarray,
        ) -> float:

            if levels.size == 0:
                return 0.0

            prices = levels[:, 0]
            volumes = levels[:, 1]

            distance = (
                np.abs(
                    prices - mid_price
                )
                / mid_price
            )

            weights = 1.0 / (
                1.0 + distance * 100
            )

            return float(
                np.sum(
                    volumes * weights
                )
            )

        bid_weighted = weighted_volume(
            bids
        )

        ask_weighted = weighted_volume(
            asks
        )

        total = (
            bid_weighted
            + ask_weighted
        )

        if total <= 0:
            return 0.0

        return float(
            np.clip(
                (
                    bid_weighted
                    - ask_weighted
                )
                / total,
                -1,
                1,
            )
        )

    # ==================================================================
    # TEMPORAL HISTORY
    # ==================================================================

    def _store_features(
        self,
        symbol: str,
        features: Dict[str, Any],
    ) -> None:
        """
        Store current snapshot and retain a short history.
        """

        self.orderbook_data[
            symbol
        ] = features

        if symbol not in self.orderbook_history:

            self.orderbook_history[
                symbol
            ] = deque(
                maxlen=self.HISTORY_SIZE
            )

        history = self.orderbook_history[
            symbol
        ]

        # Calculate changes before adding current snapshot.
        if history:

            previous = history[-1]

            features.update(
                self._calculate_changes(
                    previous,
                    features,
                )
            )

        else:

            features[
                "pressure_change"
            ] = 0.0

            features[
                "imbalance_change"
            ] = 0.0

            features[
                "spread_change_bps"
            ] = 0.0

        history.append(
            dict(features)
        )

        self.last_update[
            symbol
        ] = time.time()

        self.update_count += 1

    @staticmethod
    def _calculate_changes(
        previous: Dict[str, Any],
        current: Dict[str, Any],
    ) -> Dict[str, float]:

        return {
            "pressure_change": (
                float(
                    current.get(
                        "order_pressure",
                        0.0,
                    )
                )
                - float(
                    previous.get(
                        "order_pressure",
                        0.0,
                    )
                )
            ),

            "imbalance_change": (
                float(
                    current.get(
                        "imbalance_10bp",
                        0.0,
                    )
                )
                - float(
                    previous.get(
                        "imbalance_10bp",
                        0.0,
                    )
                )
            ),

            "spread_change_bps": (
                float(
                    current.get(
                        "spread_bps",
                        0.0,
                    )
                )
                - float(
                    previous.get(
                        "spread_bps",
                        0.0,
                    )
                )
            ),
        }

    # ==================================================================
    # DATA QUALITY
    # ==================================================================

    @staticmethod
    def _calculate_data_quality(
        bids: np.ndarray,
        asks: np.ndarray,
        spread_bps: float,
    ) -> float:
        """
        Basic quality score.

        This is NOT a trading confidence score.
        """

        score = 0.0

        if len(bids) >= 10:
            score += 0.35
        elif len(bids) >= 5:
            score += 0.20

        if len(asks) >= 10:
            score += 0.35
        elif len(asks) >= 5:
            score += 0.20

        if spread_bps > 0:
            score += 0.20

        if np.all(
            np.diff(
                bids[:, 0]
            ) <= 0
        ):
            score += 0.05

        if np.all(
            np.diff(
                asks[:, 0]
            ) >= 0
        ):
            score += 0.05

        return float(
            np.clip(
                score,
                0,
                1,
            )
        )

    # ==================================================================
    # DEFAULT
    # ==================================================================

    def _get_default_features(
        self,
        symbol: str = "UNKNOWN",
    ) -> Dict[str, Any]:

        features: Dict[str, Any] = {
            "symbol": symbol,

            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),

            "timestamp_unix": time.time(),

            "best_bid": 0.0,
            "best_ask": 0.0,
            "mid_price": 0.0,

            "spread": 0.0,
            "spread_ratio": 0.0,
            "spread_bps": 0.0,

            "best_bid_size": 0.0,
            "best_ask_size": 0.0,

            "top_imbalance": 0.0,
            "weighted_imbalance": 0.0,

            "bid_vwap": 0.0,
            "ask_vwap": 0.0,

            "vwap_spread_ratio": 0.0,
            "bid_vwap_distance": 0.0,
            "ask_vwap_distance": 0.0,

            "bid_wall_ratio": 1.0,
            "ask_wall_ratio": 1.0,

            "wall_bid_price": 0.0,
            "wall_ask_price": 0.0,

            "wall_imbalance": 0.0,

            "bid_density": 0.0,
            "ask_density": 0.0,

            "liquidity_density_imbalance": 0.0,

            "order_pressure": 0.0,
            "buy_pressure": 0.5,

            "pressure_direction": "NEUTRAL",
            "pressure_strength": "LOW",

            "pressure_change": 0.0,
            "imbalance_change": 0.0,
            "spread_change_bps": 0.0,

            "levels_available": 0,

            "data_quality": 0.0,
        }

        for depth in self.DEPTH_LEVELS:

            suffix = int(
                depth * 1000
            )

            features[
                f"bid_volume_{suffix}bp"
            ] = 0.0

            features[
                f"ask_volume_{suffix}bp"
            ] = 0.0

            features[
                f"imbalance_{suffix}bp"
            ] = 0.0

        return features

    # ==================================================================
    # PUBLIC API
    # ==================================================================

    def get_full_features(
        self,
        symbol: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Return the latest complete feature snapshot.

        Returns None when no data exists or data is stale.
        """

        normalized = (
            self.normalize_symbol(symbol)
            .upper()
        )

        features = self.orderbook_data.get(
            normalized
        )

        if not features:
            return None

        if not self.is_data_fresh(
            normalized
        ):
            return None

        return dict(features)

    def get_imbalance(
        self,
        symbol: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Return compact imbalance information.
        """

        features = self.get_full_features(
            symbol
        )

        if features is None:
            return None

        return {
            "symbol": features[
                "symbol"
            ],

            "imbalance": features.get(
                "imbalance_10bp",
                0.0,
            ),

            "imbalance_50bp": features.get(
                "imbalance_20bp",
                0.0,
            ),

            "top_imbalance": features.get(
                "top_imbalance",
                0.0,
            ),

            "weighted_imbalance": features.get(
                "weighted_imbalance",
                0.0,
            ),

            "spread_bps": features.get(
                "spread_bps",
                0.0,
            ),

            "bid_volume": features.get(
                "bid_volume_10bp",
                0.0,
            ),

            "ask_volume": features.get(
                "ask_volume_10bp",
                0.0,
            ),

            "buy_pressure": features.get(
                "buy_pressure",
                0.5,
            ),

            "order_pressure": features.get(
                "order_pressure",
                0.0,
            ),

            "pressure_direction": features.get(
                "pressure_direction",
                "NEUTRAL",
            ),

            "pressure_strength": features.get(
                "pressure_strength",
                "LOW",
            ),

            "wall_imbalance": features.get(
                "wall_imbalance",
                0.0,
            ),

            "pressure_change": features.get(
                "pressure_change",
                0.0,
            ),

            "data_quality": features.get(
                "data_quality",
                0.0,
            ),

            "timestamp": features.get(
                "timestamp"
            ),
        }

    def get_pressure_signal(
        self,
        symbol: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Return the order-book pressure signal.

        This is contextual information for SignalGenerator.
        It is NOT a standalone trading signal.
        """

        features = self.get_full_features(
            symbol
        )

        if features is None:
            return None

        pressure = float(
            features.get(
                "order_pressure",
                0.0,
            )
        )

        return {
            "direction": features.get(
                "pressure_direction",
                "NEUTRAL",
            ),

            "strength": float(
                abs(pressure)
            ),

            "confidence": float(
                min(
                    1.0,
                    abs(pressure) * 2.0,
                )
            ),

            "pressure": pressure,

            "imbalance": features.get(
                "imbalance_10bp",
                0.0,
            ),

            "top_imbalance": features.get(
                "top_imbalance",
                0.0,
            ),

            "weighted_imbalance": features.get(
                "weighted_imbalance",
                0.0,
            ),

            "wall_imbalance": features.get(
                "wall_imbalance",
                0.0,
            ),

            "pressure_change": features.get(
                "pressure_change",
                0.0,
            ),

            "spread_bps": features.get(
                "spread_bps",
                0.0,
            ),

            "data_quality": features.get(
                "data_quality",
                0.0,
            ),

            "timestamp": features.get(
                "timestamp"
            ),
        }

    def get_all_pressure_signals(
        self,
    ) -> Dict[str, Dict[str, Any]]:

        result = {}

        for symbol in list(
            self.orderbook_data.keys()
        ):

            pressure = (
                self.get_pressure_signal(
                    symbol
                )
            )

            if pressure:
                result[symbol] = pressure

        return result

    def is_data_fresh(
        self,
        symbol: str,
    ) -> bool:

        normalized = (
            self.normalize_symbol(symbol)
            .upper()
        )

        timestamp = self.last_update.get(
            normalized
        )

        if timestamp is None:
            return False

        age = (
            time.time()
            - timestamp
        )

        return age <= self.stale_seconds

    def get_history(
        self,
        symbol: str,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Return recent order-book feature history.
        """

        normalized = (
            self.normalize_symbol(symbol)
            .upper()
        )

        history = self.orderbook_history.get(
            normalized
        )

        if not history:
            return []

        limit = max(
            1,
            min(
                int(limit),
                self.HISTORY_SIZE,
            ),
        )

        return list(
            history
        )[-limit:]

    def get_stats(self) -> Dict[str, Any]:

        now = time.time()

        ages = {}

        for symbol, timestamp in (
            self.last_update.items()
        ):
            ages[symbol] = round(
                max(
                    0,
                    now - timestamp,
                ),
                3,
            )

        return {
            "is_running": self.is_running,

            "symbols_monitored": len(
                self.orderbook_data
            ),

            "update_count": self.update_count,

            "error_count": self.error_count,

            "connection_count": self.connection_count,

            "active_connections": len(
                self.ws_connections
            ),

            "fresh_symbols": sum(
                1
                for symbol in self.orderbook_data
                if self.is_data_fresh(symbol)
            ),

            "last_update_age_seconds": ages,
        }

    # ==================================================================
    # CLEANUP
    # ==================================================================

    async def cleanup(self) -> None:
        """
        Stop monitoring and close all websocket connections.
        """

        self.is_running = False

        connections = list(
            self.ws_connections.items()
        )

        self.ws_connections.clear()

        for symbol, websocket in connections:

            try:

                await websocket.close()

            except Exception as exc:

                logger.debug(
                    "Error closing OrderBook websocket "
                    "%s: %s",
                    symbol,
                    exc,
                )

        logger.info(
            "🧹 OrderBookMonitor cleaned up | "
            "updates=%d | errors=%d",
            self.update_count,
            self.error_count,
        )