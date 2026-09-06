"""
src/data/collectors.py

SmartCrypto AI v3.1.0
Market data collection layer.

Responsibilities:
    - Binance Futures OHLCV
    - Funding-rate history
    - Open-interest history
    - Current derivatives state
    - Current order-book microstructure
    - Clean alignment of heterogeneous data
    - Multi-exchange fallback architecture

This module only collects and aligns raw/current market data.
Feature engineering remains inside DataProcessor.
Signal generation remains inside signal_generator.py.
"""

import asyncio
import logging
from typing import Dict, List, Optional

import aiohttp
import ccxt.async_support as ccxt
import pandas as pd

from src.core.config import Settings
from src.utils.safe_logger import SafeLogger


logger = SafeLogger.get_logger(__name__)


class BinanceDataCollector:
    """
    Binance Futures data collector.

    Uses:
        - Binance Spot REST for OHLCV
        - Binance Futures through CCXT for derivatives/order book

    The returned dataframe is standardized for DataProcessor.
    """

    def __init__(self, settings: Settings):
        self.settings = settings

        self.base_url = getattr(
            settings,
            "BINANCE_API_BASE",
            "https://api.binance.com",
        )

        self.futures_base_url = getattr(
            settings,
            "BINANCE_FUTURES_API_BASE",
            "https://fapi.binance.com",
        )

        self.session: Optional[aiohttp.ClientSession] = None

        self.exchange = ccxt.binanceusdm(
            {
                "enableRateLimit": True,
                "options": {
                    "defaultType": "future",
                },
            }
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # SESSION MANAGEMENT
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def get_session(self) -> aiohttp.ClientSession:
        """Create or reuse the asynchronous HTTP session."""

        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(
                total=30,
                connect=10,
            )

            self.session = aiohttp.ClientSession(
                timeout=timeout,
            )

        return self.session

    async def close(self) -> None:
        """Close HTTP and CCXT resources."""

        try:
            if self.session and not self.session.closed:
                await self.session.close()

            self.session = None

        except Exception as exc:
            logger.warning(
                f"Error closing HTTP session: {exc}"
            )

        try:
            await self.exchange.close()

        except Exception as exc:
            logger.warning(
                f"Error closing exchange connection: {exc}"
            )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # SYMBOL HELPERS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @staticmethod
    def to_ccxt_symbol(symbol: str) -> str:
        """
        Convert Binance symbol to CCXT Futures symbol.

        BTCUSDT -> BTC/USDT:USDT
        ETHUSDT -> ETH/USDT:USDT
        """

        symbol = str(symbol).upper().strip()

        if symbol.endswith("USDT"):
            base = symbol[:-4]
            return f"{base}/USDT:USDT"

        return symbol

    @staticmethod
    def to_binance_symbol(symbol: str) -> str:
        """
        Normalize symbol to Binance format.

        BTC/USDT:USDT -> BTCUSDT
        """

        symbol = str(symbol).upper().strip()

        return (
            symbol
            .replace("/", "")
            .replace(":USDT", "")
            .replace(":USDC", "")
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # OHLCV
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def fetch_historical_data(
        self,
        symbol: str,
        interval: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch Binance OHLCV candles.

        The interval and limit default to configuration values.
        """

        interval = interval or getattr(
            self.settings,
            "DEFAULT_INTERVAL",
            "1h",
        )

        limit = limit or getattr(
            self.settings,
            "DEFAULT_LIMIT",
            500,
        )

        binance_symbol = self.to_binance_symbol(
            symbol
        )

        candidate_bases = [
            self.base_url,
            "https://data-api.binance.vision",
            "https://api1.binance.com",
            "https://api2.binance.com",
            "https://api3.binance.com",
            self.futures_base_url,
        ]
        # Deduplicate while preserving order
        unique_bases = []
        for b in candidate_bases:
            if b and b not in unique_bases:
                unique_bases.append(b)

        params = {
            "symbol": binance_symbol,
            "interval": interval,
            "limit": min(int(limit), 1000),
        }

        proxy_url = getattr(self.settings, "EXCHANGE_PROXY_URL", None)

        for attempt, base in enumerate(unique_bases[:4]):
            endpoint = "/fapi/v1/klines" if "fapi.binance.com" in base else "/api/v3/klines"
            url = f"{base.rstrip('/')}{endpoint}"

            try:
                session = await self.get_session()

                async with session.get(
                    url,
                    params=params,
                    proxy=proxy_url,
                ) as response:

                    if response.status == 200:
                        data = await response.json()

                        if not data:
                            logger.warning(
                                f"Empty OHLCV response for {binance_symbol}"
                            )
                            return None

                        return self.parse_kline_data(
                            data,
                            binance_symbol,
                            interval,
                        )

                    body = await response.text()
                    logger.warning(
                        f"OHLCV attempt {attempt + 1} ({base}) failed "
                        f"for {binance_symbol}: HTTP {response.status} - {body[:150]}"
                    )

            except asyncio.CancelledError:
                raise

            except Exception as exc:
                logger.warning(
                    f"OHLCV attempt {attempt + 1} ({base}) failed "
                    f"for {binance_symbol}: {exc}"
                )

            if attempt < 2:
                await asyncio.sleep(
                    2 ** attempt
                )

        return None

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # KLINE PARSER
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def parse_kline_data(
        self,
        data: List,
        symbol: str,
        interval: str = "1h",
    ) -> pd.DataFrame:
        """
        Convert Binance kline response to standardized dataframe.
        """

        columns = [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
            "ignore",
        ]

        df = pd.DataFrame(
            data,
            columns=columns,
        )

        df = df.rename(
            columns={
                "number_of_trades": "trades_count",
                "taker_buy_base_asset_volume":
                    "taker_buy_base_volume",
                "taker_buy_quote_asset_volume":
                    "taker_buy_quote_volume",
            }
        )

        numeric_columns = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_asset_volume",
            "trades_count",
            "taker_buy_base_volume",
            "taker_buy_quote_volume",
        ]

        for column in numeric_columns:

            if column in df.columns:

                df[column] = pd.to_numeric(
                    df[column],
                    errors="coerce",
                )

        df["timestamp"] = pd.to_datetime(
            df["open_time"],
            unit="ms",
            utc=True,
        )

        df["close_timestamp"] = pd.to_datetime(
            df["close_time"],
            unit="ms",
            utc=True,
        )

        df["symbol"] = symbol
        df["interval"] = interval

        df = (
            df.sort_values("timestamp")
            .drop_duplicates(
                subset=["timestamp"],
                keep="last",
            )
            .reset_index(drop=True)
        )

        return df

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # FUNDING RATE HISTORY
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def fetch_funding_rate_history(
        self,
        symbol_ccxt: str,
        limit: int = 1000,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical Binance Futures funding rates.
        """

        try:

            funding_raw = (
                await self.exchange.fetch_funding_rate_history(
                    symbol=symbol_ccxt,
                    limit=min(int(limit), 1000),
                )
            )

            if not funding_raw:
                logger.warning(
                    f"No funding data for {symbol_ccxt}"
                )
                return None

            df = pd.DataFrame(
                funding_raw
            )

            if "timestamp" not in df.columns:
                return None

            funding_column = None

            if "fundingRate" in df.columns:
                funding_column = "fundingRate"

            elif "info" in df.columns:
                values = []

                for info in df["info"]:
                    try:
                        values.append(
                            float(
                                info.get(
                                    "fundingRate",
                                    0.0,
                                )
                            )
                        )
                    except Exception:
                        values.append(0.0)

                df["funding_rate"] = values
                funding_column = "funding_rate"

            if funding_column is None:
                return None

            if funding_column != "funding_rate":
                df["funding_rate"] = pd.to_numeric(
                    df[funding_column],
                    errors="coerce",
                )

            df["timestamp"] = pd.to_datetime(
                df["timestamp"],
                unit="ms",
                utc=True,
            )

            df = df[
                [
                    "timestamp",
                    "funding_rate",
                ]
            ]

            return (
                df.sort_values("timestamp")
                .drop_duplicates(
                    subset=["timestamp"],
                    keep="last",
                )
                .reset_index(drop=True)
            )

        except asyncio.CancelledError:
            raise

        except Exception as exc:

            logger.warning(
                f"Funding history failed for "
                f"{symbol_ccxt}: {exc}"
            )

            return None

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # OPEN INTEREST HISTORY
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def fetch_open_interest_history(
        self,
        symbol_ccxt: str,
        period: str = "1h",
        limit: int = 1000,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical Binance Futures open interest.

        Uses asynchronous aiohttp rather than blocking requests.
        """

        try:

            session = await self.get_session()

            symbol = self.to_binance_symbol(
                symbol_ccxt
            )

            url = (
                f"{self.futures_base_url}"
                "/futures/data/openInterestHist"
            )

            params = {
                "symbol": symbol,
                "period": period,
                "limit": min(int(limit), 500),
            }

            async with session.get(
                url,
                params=params,
            ) as response:

                if response.status != 200:

                    body = await response.text()

                    logger.warning(
                        f"Open interest request failed "
                        f"for {symbol}: "
                        f"HTTP {response.status} "
                        f"{body[:200]}"
                    )

                    return None

                data = await response.json()

            if not data:
                return None

            df = pd.DataFrame(data)

            if "timestamp" not in df.columns:
                return None

            df["timestamp"] = pd.to_datetime(
                pd.to_numeric(
                    df["timestamp"],
                    errors="coerce",
                ),
                unit="ms",
                utc=True,
            )

            if "sumOpenInterest" in df.columns:

                df["open_interest"] = pd.to_numeric(
                    df["sumOpenInterest"],
                    errors="coerce",
                )

            if "sumOpenInterestValue" in df.columns:

                df["open_interest_usd"] = pd.to_numeric(
                    df["sumOpenInterestValue"],
                    errors="coerce",
                )

            required = [
                "timestamp",
                "open_interest",
                "open_interest_usd",
            ]

            for column in required:

                if column not in df.columns:
                    df[column] = 0.0

            return (
                df[required]
                .sort_values("timestamp")
                .drop_duplicates(
                    subset=["timestamp"],
                    keep="last",
                )
                .reset_index(drop=True)
            )

        except asyncio.CancelledError:
            raise

        except Exception as exc:

            logger.warning(
                f"Open interest failed for "
                f"{symbol_ccxt}: {exc}"
            )

            return None

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # CURRENT DERIVATIVES
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def fetch_current_derivatives(
        self,
        symbol_ccxt: str,
    ) -> Dict:
        """
        Fetch current funding rate and open interest.
        """

        now = pd.Timestamp.now(
            tz="UTC"
        )

        default = {
            "funding_rate": 0.0,
            "open_interest": 0.0,
            "open_interest_usd": 0.0,
            "timestamp": now,
        }

        try:

            funding, oi = await asyncio.gather(
                self.exchange.fetch_funding_rate(
                    symbol_ccxt
                ),
                self.exchange.fetch_open_interest(
                    symbol_ccxt
                ),
                return_exceptions=True,
            )

            if not isinstance(
                funding,
                Exception,
            ):

                default["funding_rate"] = float(
                    funding.get(
                        "fundingRate",
                        0.0,
                    )
                    or 0.0
                )

            if not isinstance(
                oi,
                Exception,
            ):

                default["open_interest"] = float(
                    oi.get(
                        "openInterest",
                        0.0,
                    )
                    or 0.0
                )

                default["open_interest_usd"] = float(
                    oi.get(
                        "openInterestValue",
                        0.0,
                    )
                    or 0.0
                )

            return default

        except asyncio.CancelledError:
            raise

        except Exception as exc:

            logger.warning(
                f"Current derivatives failed for "
                f"{symbol_ccxt}: {exc}"
            )

            return default

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # ORDER BOOK
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def fetch_order_book_imbalance(
        self,
        symbol_ccxt: str,
        depth_pct: float = 0.01,
        limit: int = 100,
    ) -> Dict:
        """
        Fetch current order-book microstructure.

        imbalance:
            -1 = ask dominated
             0 = balanced
            +1 = bid dominated
        """

        default = {
            "imbalance": 0.0,
            "spread_pct": 0.0,
            "bid_volume": 0.0,
            "ask_volume": 0.0,
            "mid_price": 0.0,
            "best_bid": 0.0,
            "best_ask": 0.0,
        }

        try:

            orderbook = (
                await self.exchange.fetch_order_book(
                    symbol_ccxt,
                    limit=limit,
                )
            )

            bids = orderbook.get(
                "bids",
                [],
            )

            asks = orderbook.get(
                "asks",
                [],
            )

            if not bids or not asks:
                return default

            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])

            mid_price = (
                best_bid + best_ask
            ) / 2.0

            if mid_price <= 0:
                return default

            lower_bound = (
                mid_price
                * (1.0 - depth_pct)
            )

            upper_bound = (
                mid_price
                * (1.0 + depth_pct)
            )

            bid_volume = sum(
                float(level[1])
                for level in bids
                if float(level[0])
                >= lower_bound
            )

            ask_volume = sum(
                float(level[1])
                for level in asks
                if float(level[0])
                <= upper_bound
            )

            total_volume = (
                bid_volume
                + ask_volume
            )

            imbalance = (
                bid_volume
                - ask_volume
            ) / (
                total_volume
                + 1e-8
            )

            spread_pct = (
                best_ask
                - best_bid
            ) / mid_price

            return {
                "imbalance": float(
                    imbalance
                ),
                "spread_pct": float(
                    spread_pct
                ),
                "bid_volume": float(
                    bid_volume
                ),
                "ask_volume": float(
                    ask_volume
                ),
                "mid_price": float(
                    mid_price
                ),
                "best_bid": float(
                    best_bid
                ),
                "best_ask": float(
                    best_ask
                ),
            }

        except asyncio.CancelledError:
            raise

        except Exception as exc:

            logger.warning(
                f"Order book failed for "
                f"{symbol_ccxt}: {exc}"
            )

            return default

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # COMPLETE DATA
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def fetch_complete_data(
        self,
        symbol: str,
        interval: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch and align:

            OHLCV
            + funding rate
            + open interest
            + current order-book state

        Historical derivatives are aligned using backward
        time matching rather than exact timestamp matching.
        """

        interval = interval or getattr(
            self.settings,
            "DEFAULT_INTERVAL",
            "1h",
        )

        limit = limit or getattr(
            self.settings,
            "DEFAULT_LIMIT",
            500,
        )

        try:

            df_ohlcv = (
                await self.fetch_historical_data(
                    symbol=symbol,
                    interval=interval,
                    limit=limit,
                )
            )

            if (
                df_ohlcv is None
                or df_ohlcv.empty
            ):

                logger.error(
                    f"No OHLCV data for {symbol}"
                )

                return None

            symbol_ccxt = self.to_ccxt_symbol(
                symbol
            )

            # Fetch independent sources concurrently.
            funding_task = (
                self.fetch_funding_rate_history(
                    symbol_ccxt,
                    limit=limit,
                )
            )

            oi_task = (
                self.fetch_open_interest_history(
                    symbol_ccxt,
                    period=interval,
                    limit=limit,
                )
            )

            orderbook_task = (
                self.fetch_order_book_imbalance(
                    symbol_ccxt
                )
            )

            funding_df, oi_df, orderbook = (
                await asyncio.gather(
                    funding_task,
                    oi_task,
                    orderbook_task,
                    return_exceptions=True,
                )
            )

            df = df_ohlcv.copy()

            df["timestamp"] = pd.to_datetime(
                df["timestamp"],
                utc=True,
            )

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # FUNDING
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            if (
                isinstance(
                    funding_df,
                    pd.DataFrame,
                )
                and not funding_df.empty
            ):

                funding_df = funding_df.copy()

                funding_df["timestamp"] = (
                    pd.to_datetime(
                        funding_df["timestamp"],
                        utc=True,
                    )
                )

                funding_df = (
                    funding_df
                    .sort_values("timestamp")
                    .drop_duplicates(
                        "timestamp",
                        keep="last",
                    )
                )

                df = pd.merge_asof(
                    df.sort_values("timestamp"),
                    funding_df.sort_values(
                        "timestamp"
                    ),
                    on="timestamp",
                    direction="backward",
                )

            if (
                "funding_rate" not in df.columns
            ):
                df["funding_rate"] = 0.0

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # OPEN INTEREST
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            if (
                isinstance(
                    oi_df,
                    pd.DataFrame,
                )
                and not oi_df.empty
            ):

                oi_df = oi_df.copy()

                oi_df["timestamp"] = (
                    pd.to_datetime(
                        oi_df["timestamp"],
                        utc=True,
                    )
                )

                oi_df = (
                    oi_df
                    .sort_values("timestamp")
                    .drop_duplicates(
                        "timestamp",
                        keep="last",
                    )
                )

                df = pd.merge_asof(
                    df.sort_values("timestamp"),
                    oi_df.sort_values(
                        "timestamp"
                    ),
                    on="timestamp",
                    direction="backward",
                )

            if (
                "open_interest"
                not in df.columns
            ):
                df["open_interest"] = 0.0

            if (
                "open_interest_usd"
                not in df.columns
            ):
                df["open_interest_usd"] = 0.0

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # FILL DERIVATIVE GAPS
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            derivative_columns = [
                "funding_rate",
                "open_interest",
                "open_interest_usd",
            ]

            for column in derivative_columns:

                df[column] = pd.to_numeric(
                    df[column],
                    errors="coerce",
                )

                df[column] = (
                    df[column]
                    .ffill()
                    .fillna(0.0)
                )

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # CURRENT ORDER BOOK
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            if isinstance(
                orderbook,
                dict,
            ):

                imbalance = float(
                    orderbook.get(
                        "imbalance",
                        0.0,
                    )
                    or 0.0
                )

                buy_pressure = (
                    imbalance * 0.5
                    + 0.5
                )

                if len(df) > 0:

                    latest = df.index[-1]

                    df.loc[
                        latest,
                        "order_imbalance"
                    ] = imbalance

                    df.loc[
                        latest,
                        "buy_pressure"
                    ] = buy_pressure

                    df.loc[
                        latest,
                        "spread_pct"
                    ] = float(
                        orderbook.get(
                            "spread_pct",
                            0.0,
                        )
                        or 0.0
                    )

                    df.loc[
                        latest,
                        "best_bid"
                    ] = float(
                        orderbook.get(
                            "best_bid",
                            0.0,
                        )
                        or 0.0
                    )

                    df.loc[
                        latest,
                        "best_ask"
                    ] = float(
                        orderbook.get(
                            "best_ask",
                            0.0,
                        )
                        or 0.0
                    )

                    df.loc[
                        latest,
                        "orderbook_mid_price"
                    ] = float(
                        orderbook.get(
                            "mid_price",
                            0.0,
                        )
                        or 0.0
                    )

            # Historical rows do not have a historical
            # order-book snapshot, so keep neutral values.
            if (
                "order_imbalance"
                not in df.columns
            ):
                df["order_imbalance"] = 0.0

            if (
                "buy_pressure"
                not in df.columns
            ):
                df["buy_pressure"] = 0.5

            if (
                "spread_pct"
                not in df.columns
            ):
                df["spread_pct"] = 0.0

            if (
                "best_bid"
                not in df.columns
            ):
                df["best_bid"] = 0.0

            if (
                "best_ask"
                not in df.columns
            ):
                df["best_ask"] = 0.0

            if (
                "orderbook_mid_price"
                not in df.columns
            ):
                df["orderbook_mid_price"] = 0.0

            df["order_imbalance"] = (
                df["order_imbalance"]
                .fillna(0.0)
            )

            df["buy_pressure"] = (
                df["buy_pressure"]
                .fillna(0.5)
            )

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # FINAL NORMALIZATION
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            df = (
                df
                .sort_values("timestamp")
                .drop_duplicates(
                    subset=["timestamp"],
                    keep="last",
                )
                .reset_index(drop=True)
            )

            numeric_columns = (
                df.select_dtypes(
                    include=["number"]
                ).columns
            )

            df[numeric_columns] = (
                df[numeric_columns]
                .replace(
                    [float("inf"), float("-inf")],
                    pd.NA,
                )
                .ffill()
                .fillna(0.0)
            )

            logger.info(
                f"Complete market data fetched "
                f"for {symbol}: {len(df)} rows"
            )

            return df

        except asyncio.CancelledError:
            raise

        except Exception as exc:

            logger.error(
                f"Complete data fetch failed "
                f"for {symbol}: {exc}",
                exc_info=True,
            )

            return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BYBIT FALLBACK DATA COLLECTOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class BybitDataCollector:
    """
    Bybit UTA public market data collector.
    Serves as an automatic secondary fallback when Binance endpoints encounter DNS, rate limit, or regional issues.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.base_url = "https://api.bybit.com"
        self.session: Optional[aiohttp.ClientSession] = None

    async def get_session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=20, connect=8)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session

    async def close(self) -> None:
        try:
            if self.session and not self.session.closed:
                await self.session.close()
            self.session = None
        except Exception:
            pass

    @staticmethod
    def to_bybit_symbol(symbol: str) -> str:
        return symbol.upper().replace("/", "").replace(":USDT", "").replace(":USDC", "").strip()

    async def fetch_historical_data(
        self,
        symbol: str,
        interval: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Optional[pd.DataFrame]:
        interval = interval or getattr(self.settings, "DEFAULT_INTERVAL", "1h")
        limit = limit or getattr(self.settings, "DEFAULT_LIMIT", 500)
        bybit_symbol = self.to_bybit_symbol(symbol)

        bybit_intervals = {
            "1m": "1", "3m": "3", "5m": "5", "15m": "15", "30m": "30",
            "1h": "60", "2h": "120", "4h": "240", "6h": "360", "12h": "720",
            "1d": "D", "1w": "W", "1M": "M"
        }
        b_interval = bybit_intervals.get(interval, "60")
        url = f"{self.base_url}/v5/market/kline"
        params = {
            "category": "linear",
            "symbol": bybit_symbol,
            "interval": b_interval,
            "limit": min(int(limit), 1000),
        }

        try:
            session = await self.get_session()
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    payload = await resp.json()
                    raw_list = payload.get("result", {}).get("list", [])
                    if raw_list:
                        # Bybit list format: [startTime, openPrice, highPrice, lowPrice, closePrice, volume, turnover]
                        raw_list = list(reversed(raw_list))
                        records = []
                        for row in raw_list:
                            open_t = int(row[0])
                            vol = float(row[5])
                            turnover = float(row[6])
                            records.append({
                                "open_time": open_t,
                                "open": float(row[1]),
                                "high": float(row[2]),
                                "low": float(row[3]),
                                "close": float(row[4]),
                                "volume": vol,
                                "close_time": open_t + 3599999,
                                "quote_asset_volume": turnover,
                                "trades_count": 1000,
                                "taker_buy_base_volume": vol * 0.5,
                                "taker_buy_quote_volume": turnover * 0.5,
                                "ignore": 0,
                            })
                        df = pd.DataFrame(records)
                        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
                        df["symbol"] = bybit_symbol
                        df = df.set_index("timestamp").sort_index()
                        logger.info(f"✅ Bybit fallback klines fetched for {bybit_symbol} ({len(df)} candles)")
                        return df
        except Exception as exc:
            logger.warning(f"Bybit klines failed for {bybit_symbol}: {exc}")
        return None

    async def fetch_ticker(self, symbol: str) -> Optional[Dict[str, float]]:
        """Fetch genuine live Bybit ticker with funding rate, open interest, and bid/ask spread."""
        bybit_symbol = self.to_bybit_symbol(symbol)
        url = f"{self.base_url}/v5/market/tickers"
        params = {"category": "linear", "symbol": bybit_symbol}
        try:
            session = await self.get_session()
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    payload = await resp.json()
                    item = (payload.get("result", {}).get("list", []) or [{}])[0]
                    if item:
                        last_price = float(item.get("lastPrice", 0.0) or 0.0)
                        funding_rate = float(item.get("fundingRate", 0.0) or 0.0)
                        open_interest = float(item.get("openInterest", 0.0) or 0.0)
                        open_interest_usd = float(item.get("openInterestValue", 0.0) or (open_interest * last_price if last_price > 0 else 0.0))
                        best_bid = float(item.get("bid1Price", 0.0) or 0.0)
                        best_ask = float(item.get("ask1Price", 0.0) or 0.0)
                        spread_pct = ((best_ask - best_bid) / last_price) if (last_price > 0 and best_ask >= best_bid) else 0.0
                        return {
                            "last_price": last_price,
                            "funding_rate": funding_rate,
                            "open_interest": open_interest,
                            "open_interest_usd": open_interest_usd,
                            "best_bid": best_bid,
                            "best_ask": best_ask,
                            "spread_pct": spread_pct,
                        }
        except Exception as exc:
            logger.warning(f"Bybit ticker fetch failed for {bybit_symbol}: {exc}")
        return None

    async def fetch_complete_data(
        self,
        symbol: str,
        interval: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Optional[pd.DataFrame]:
        df = await self.fetch_historical_data(symbol, interval, limit)
        if df is None or df.empty:
            return None
            
        ticker = await self.fetch_ticker(symbol)
        if ticker:
            df["funding_rate"] = ticker["funding_rate"]
            df["funding_rate_annualized"] = ticker["funding_rate"] * 3 * 365
            df["open_interest"] = ticker["open_interest"]
            df["open_interest_usd"] = ticker["open_interest_usd"]
            df["spread_pct"] = ticker["spread_pct"]
            df["best_bid"] = ticker["best_bid"]
            df["best_ask"] = ticker["best_ask"]
            df["order_imbalance"] = 0.0
            df["buy_pressure"] = 0.5
        else:
            df["funding_rate"] = 0.0
            df["funding_rate_annualized"] = 0.0
            df["open_interest"] = 0.0
            df["open_interest_usd"] = 0.0
            df["spread_pct"] = 0.0
            df["order_imbalance"] = 0.0
            df["buy_pressure"] = 0.5
        return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BITGET FALLBACK DATA COLLECTOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class BitgetDataCollector:
    """
    Bitget public market data collector.
    Serves as an automatic tertiary fallback when Binance and Bybit endpoints are unreachable.
    """
    def __init__(self, settings: Settings):
        self.settings = settings
        self.base_url = "https://api.bitget.com"
        self.session: Optional[aiohttp.ClientSession] = None

    async def get_session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=20, connect=8)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session

    async def close(self) -> None:
        try:
            if self.session and not self.session.closed:
                await self.session.close()
            self.session = None
        except Exception:
            pass

    @staticmethod
    def to_bitget_symbol(symbol: str) -> str:
        return symbol.upper().replace("/", "").replace(":USDT", "").replace(":USDC", "").strip()

    async def fetch_historical_data(
        self,
        symbol: str,
        interval: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Optional[pd.DataFrame]:
        interval = interval or getattr(self.settings, "DEFAULT_INTERVAL", "1h")
        limit = limit or getattr(self.settings, "DEFAULT_LIMIT", 500)
        bitget_symbol = self.to_bitget_symbol(symbol)

        bitget_intervals = {
            "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
            "1h": "1H", "2h": "2H", "4h": "4H", "6h": "6H", "12h": "12H",
            "1d": "1D", "1w": "1W",
        }
        granularity = bitget_intervals.get(interval.lower(), "1H")

        url = f"{self.base_url}/api/v2/mix/market/candles"
        params = {
            "symbol": bitget_symbol,
            "granularity": granularity,
            "limit": str(min(limit, 1000)),
            "productType": "USDT-FUTURES",
        }
        try:
            session = await self.get_session()
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    payload = await resp.json()
                    candles = payload.get("data", [])
                    if candles:
                        df = pd.DataFrame(
                            candles,
                            columns=["timestamp", "open", "high", "low", "close", "volume", "quote_volume"]
                        )
                        df["timestamp"] = pd.to_datetime(pd.to_numeric(df["timestamp"]), unit="ms")
                        for col in ["open", "high", "low", "close", "volume"]:
                            df[col] = df[col].astype(float)
                        df.sort_values("timestamp", inplace=True)
                        df.reset_index(drop=True, inplace=True)
                        return df
        except Exception as exc:
            logger.warning(f"Bitget klines failed for {bitget_symbol}: {exc}")
        return None

    async def fetch_ticker(self, symbol: str) -> Optional[Dict[str, float]]:
        bitget_symbol = self.to_bitget_symbol(symbol)
        url = f"{self.base_url}/api/v2/mix/market/ticker"
        params = {"symbol": bitget_symbol, "productType": "USDT-FUTURES"}
        try:
            session = await self.get_session()
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    payload = await resp.json()
                    items = payload.get("data", [])
                    if items:
                        item = items[0]
                        last_price = float(item.get("lastPr", 0.0) or 0.0)
                        funding_rate = float(item.get("fundingRate", 0.0) or 0.0)
                        open_interest = float(item.get("openInterest", 0.0) or 0.0)
                        best_bid = float(item.get("bidPr", 0.0) or 0.0)
                        best_ask = float(item.get("askPr", 0.0) or 0.0)
                        spread_pct = ((best_ask - best_bid) / last_price) if (last_price > 0 and best_ask >= best_bid) else 0.0
                        return {
                            "last_price": last_price,
                            "funding_rate": funding_rate,
                            "open_interest": open_interest,
                            "open_interest_usd": open_interest * last_price,
                            "best_bid": best_bid,
                            "best_ask": best_ask,
                            "spread_pct": spread_pct,
                        }
        except Exception as exc:
            logger.warning(f"Bitget ticker fetch failed for {bitget_symbol}: {exc}")
        return None

    async def fetch_complete_data(
        self,
        symbol: str,
        interval: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Optional[pd.DataFrame]:
        df = await self.fetch_historical_data(symbol, interval, limit)
        if df is None or df.empty:
            return None
        ticker = await self.fetch_ticker(symbol)
        if ticker:
            df["funding_rate"] = ticker["funding_rate"]
            df["funding_rate_annualized"] = ticker["funding_rate"] * 3 * 365
            df["open_interest"] = ticker["open_interest"]
            df["open_interest_usd"] = ticker["open_interest_usd"]
            df["spread_pct"] = ticker["spread_pct"]
            df["best_bid"] = ticker["best_bid"]
            df["best_ask"] = ticker["best_ask"]
        else:
            df["funding_rate"] = 0.0
            df["funding_rate_annualized"] = 0.0
            df["open_interest"] = 0.0
            df["open_interest_usd"] = 0.0
            df["spread_pct"] = 0.0
        df["order_imbalance"] = 0.0
        df["buy_pressure"] = 0.5
        return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MULTI-EXCHANGE COLLECTOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class MultiExchangeCollector:
    """
    Exchange abstraction with multi-tier fallback support.

    Binance is primary, Bybit is secondary, and Bitget is tertiary fallback.
    """

    def __init__(
        self,
        settings: Settings,
    ):
        self.settings = settings

        self.collectors = {
            "binance": BinanceDataCollector(
                settings
            ),
            "bybit": BybitDataCollector(
                settings
            ),
            "bitget": BitgetDataCollector(
                settings
            ),
        }

    async def fetch_data(
        self,
        symbol: str,
        **kwargs,
    ) -> Optional[pd.DataFrame]:
        """Fetch complete market data with exchange fallback."""

        for (
            exchange_name,
            collector,
        ) in self.collectors.items():

            try:

                data = await collector.fetch_complete_data(
                    symbol,
                    **kwargs,
                )

                if (
                    data is not None
                    and not data.empty
                ):

                    logger.info(
                        f"Data fetched from "
                        f"{exchange_name} "
                        f"for {symbol}"
                    )

                    return data

            except asyncio.CancelledError:
                raise

            except Exception as exc:

                logger.warning(
                    f"{exchange_name} complete "
                    f"data failed for {symbol}: "
                    f"{exc}"
                )

        logger.error(
            f"All data sources failed for {symbol}"
        )

        return None

    async def fetch_historical_data(
        self,
        symbol: str,
        **kwargs,
    ) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data with exchange fallback."""

        for (
            exchange_name,
            collector,
        ) in self.collectors.items():

            try:

                data = await collector.fetch_historical_data(
                    symbol,
                    **kwargs,
                )

                if (
                    data is not None
                    and not data.empty
                ):

                    logger.info(
                        f"OHLCV fetched from "
                        f"{exchange_name} "
                        f"for {symbol}"
                    )

                    return data

            except asyncio.CancelledError:
                raise

            except Exception as exc:

                logger.warning(
                    f"{exchange_name} OHLCV "
                    f"failed for {symbol}: {exc}"
                )

        logger.error(
            f"All OHLCV sources failed for {symbol}"
        )

        return None

    async def close(self) -> None:
        """Close all exchange collectors."""

        for collector in self.collectors.values():

            try:
                await collector.close()

            except Exception as exc:

                logger.warning(
                    f"Error closing collector: {exc}"
                )