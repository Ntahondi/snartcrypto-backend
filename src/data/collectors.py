"""
Enhanced data collectors with multiple exchange support
Fetches OHLCV, derivatives data (funding rates, open interest) for AI models
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
import ccxt.async_support as ccxt

from src.core.config import Settings
from src.utils.safe_logger import SafeLogger

logger = SafeLogger.get_logger(__name__)


class BinanceDataCollector:
    """Enhanced Binance data collector with derivatives support"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.base_url = getattr(settings, 'BINANCE_API_BASE', 'https://api.binance.com')
        self.session: Optional[aiohttp.ClientSession] = None
        
        self.exchange = ccxt.binanceusdm({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        
    async def get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # OHLCV DATA
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    async def fetch_historical_data(self, symbol: str, interval: str = '1h', 
                                  limit: int = 1000) -> Optional[pd.DataFrame]:
        """Fetch historical kline data with error handling"""
        for attempt in range(3):
            try:
                session = await self.get_session()
                url = f"{self.base_url}/api/v3/klines"
                params = {
                    'symbol': symbol,
                    'interval': interval,
                    'limit': limit
                }
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self.parse_kline_data(data, symbol)
                    else:
                        logger.warning(f"Attempt {attempt + 1} failed for {symbol}")
                        await asyncio.sleep(2 ** attempt)
                        
            except Exception as e:
                logger.error(f"Error fetching OHLCV data for {symbol}: {e}")
                await asyncio.sleep(1)
                
        return None

    def parse_kline_data(self, data: List, symbol: str) -> pd.DataFrame:
        """Parse Binance kline data into DataFrame with standardized column names"""
        df = pd.DataFrame(data, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # FIX: Rename raw Binance columns to match AI Feature Engineer expectations
        df = df.rename(columns={
            'number_of_trades': 'trades_count',
            'taker_buy_base_asset_volume': 'taker_buy_base_volume',
            'taker_buy_quote_asset_volume': 'taker_buy_quote_volume'
        })

        # Convert numeric types
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 
                          'quote_asset_volume', 'trades_count',
                          'taker_buy_base_volume', 'taker_buy_quote_volume']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].astype(float)
        
        df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
        df['symbol'] = symbol
        
        return df

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # DERIVATIVES DATA
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    async def fetch_funding_rate_history(self, symbol_ccxt: str, limit: int = 1000) -> Optional[pd.DataFrame]:
        """Fetch historical funding rates from Binance Futures via CCXT"""
        try:
            funding_raw = await self.exchange.fetch_funding_rate_history(
                symbol=symbol_ccxt, 
                limit=limit
            )
            
            if funding_raw:
                df_funding = pd.DataFrame(funding_raw)
                df_funding['timestamp'] = pd.to_datetime(df_funding['timestamp'], unit='ms')
                df_funding = df_funding[['timestamp', 'fundingRate']].rename(
                    columns={'fundingRate': 'funding_rate'}
                )
                return df_funding
            else:
                logger.warning(f"No funding rate data for {symbol_ccxt}")
                return None
                
        except Exception as e:
            logger.warning(f"Error fetching funding rates for {symbol_ccxt}: {e}")
            return None

    async def fetch_open_interest_history(self, symbol_ccxt: str, period: str = '1h', 
                                        limit: int = 1000) -> Optional[pd.DataFrame]:
        """Fetch historical open interest from Binance Futures"""
        try:
            symbol_raw = symbol_ccxt.replace('/', '').replace(':USDT', '')
            import requests
            url = "https://fapi.binance.com/futures/data/openInterestHist"
            params = {
                'symbol': symbol_raw,
                'period': period,
                'limit': limit
            }
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: requests.get(url, params=params, timeout=15)
            )
            
            if response.status_code == 200:
                data = response.json()
                if data:
                    df_oi = pd.DataFrame(data)
                    df_oi['timestamp'] = pd.to_datetime(df_oi['timestamp'].astype(int), unit='ms')
                    df_oi['open_interest'] = df_oi['sumOpenInterest'].astype(float)
                    df_oi['open_interest_usd'] = df_oi['sumOpenInterestValue'].astype(float)
                    return df_oi[['timestamp', 'open_interest', 'open_interest_usd']]
            
            return None
            
        except Exception as e:
            logger.warning(f"Error fetching open interest for {symbol_ccxt}: {e}")
            return None

    async def fetch_current_derivatives(self, symbol_ccxt: str) -> Dict:
        """Fetch current derivatives data (funding rate, open interest)"""
        try:
            funding = await self.exchange.fetch_funding_rate(symbol_ccxt)
            funding_rate = funding.get('fundingRate', 0)
            
            oi = await self.exchange.fetch_open_interest(symbol_ccxt)
            open_interest = oi.get('openInterest', 0)
            open_interest_usd = oi.get('openInterestValue', 0)
            
            return {
                'funding_rate': funding_rate,
                'open_interest': open_interest,
                'open_interest_usd': open_interest_usd,
                'timestamp': pd.Timestamp.now()
            }
        except Exception as e:
            logger.warning(f"Error fetching current derivatives for {symbol_ccxt}: {e}")
            return {
                'funding_rate': 0,
                'open_interest': 0,
                'open_interest_usd': 0,
                'timestamp': pd.Timestamp.now()
            }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # ORDER BOOK DATA
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    async def fetch_order_book_imbalance(self, symbol_ccxt: str, depth_pct: float = 0.01) -> Dict:
        """Fetch order book imbalance ratio (-1 to +1)"""
        try:
            orderbook = await self.exchange.fetch_order_book(symbol_ccxt, limit=100)
            
            best_bid = orderbook['bids'][0][0]
            best_ask = orderbook['asks'][0][0]
            mid_price = (best_bid + best_ask) / 2.0
            
            bids_in_depth = sum([
                b[1] for b in orderbook['bids'] 
                if b[0] >= mid_price * (1.0 - depth_pct)
            ])
            asks_in_depth = sum([
                a[1] for a in orderbook['asks'] 
                if a[0] <= mid_price * (1.0 + depth_pct)
            ])
            
            total = bids_in_depth + asks_in_depth
            imbalance = (bids_in_depth - asks_in_depth) / (total + 1e-8)
            
            return {
                'imbalance': imbalance,
                'spread_pct': (best_ask - best_bid) / mid_price,
                'bid_volume': bids_in_depth,
                'ask_volume': asks_in_depth,
                'mid_price': mid_price,
                'best_bid': best_bid,
                'best_ask': best_ask,
            }
        except Exception as e:
            logger.warning(f"Error fetching order book for {symbol_ccxt}: {e}")
            return {
                'imbalance': 0,
                'spread_pct': 0.0001,
                'bid_volume': 0,
                'ask_volume': 0,
                'mid_price': 0,
                'best_bid': 0,
                'best_ask': 0,
            }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # COMPLETE DATA FETCH
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    async def fetch_complete_data(self, symbol: str, limit: int = 1000) -> Optional[pd.DataFrame]:
        """
        Fetch complete data: OHLCV + derivatives + order book.
        Merges all data sources cleanly into a single DataFrame.
        """
        try:
            df_ohlcv = await self.fetch_historical_data(symbol, limit=limit)
            if df_ohlcv is None or df_ohlcv.empty:
                logger.error(f"No OHLCV data for {symbol}")
                return None
            
            symbol_ccxt = symbol.replace('USDT', '/USDT')
            
            df_funding = await self.fetch_funding_rate_history(symbol_ccxt, limit=limit)
            df_oi = await self.fetch_open_interest_history(symbol_ccxt, limit=limit)
            
            df = df_ohlcv.copy()
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            if df_funding is not None and not df_funding.empty:
                df_funding['timestamp'] = pd.to_datetime(df_funding['timestamp'])
                df = pd.merge(df, df_funding, on='timestamp', how='left')
            else:
                df['funding_rate'] = 0.0
            
            if df_oi is not None and not df_oi.empty:
                df_oi['timestamp'] = pd.to_datetime(df_oi['timestamp'])
                df = pd.merge(df, df_oi, on='timestamp', how='left')
            else:
                df['open_interest'] = 0.0
                df['open_interest_usd'] = 0.0
            
            df['funding_rate'] = df['funding_rate'].ffill().fillna(0.0)
            df['open_interest'] = df['open_interest'].ffill().fillna(0.0)
            df['open_interest_usd'] = df['open_interest_usd'].ffill().fillna(0.0)
            
            ob_data = await self.fetch_order_book_imbalance(symbol_ccxt)
            
            if len(df) > 0:
                df.loc[df.index[-1], 'order_imbalance'] = ob_data.get('imbalance', 0.0)
                df.loc[df.index[-1], 'buy_pressure'] = ob_data.get('imbalance', 0.0) * 0.5 + 0.5
                
                df['order_imbalance'] = df['order_imbalance'].fillna(0.0)
                df['buy_pressure'] = df['buy_pressure'].fillna(0.5)
            
            logger.info(f"✅ Complete data fetched for {symbol}: {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching complete data for {symbol}: {e}")
            return None


class MultiExchangeCollector:
    """Collect data from multiple exchanges with fallbacks"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.collectors = {
            'binance': BinanceDataCollector(settings),
        }
        
    async def fetch_data(self, symbol: str, **kwargs) -> Optional[pd.DataFrame]:
        for exchange_name, collector in self.collectors.items():
            try:
                data = await collector.fetch_complete_data(symbol, **kwargs)
                if data is not None and len(data) > 0:
                    logger.info(f"✅ Data fetched from {exchange_name} for {symbol}")
                    return data
            except Exception as e:
                logger.warning(f"Failed to fetch from {exchange_name}: {e}")
                
        logger.error(f"❌ All data sources failed for {symbol}")
        return None
    
    async def fetch_historical_data(self, symbol: str, **kwargs) -> Optional[pd.DataFrame]:
        for exchange_name, collector in self.collectors.items():
            try:
                data = await collector.fetch_historical_data(symbol, **kwargs)
                if data is not None and len(data) > 0:
                    return data
            except Exception as e:
                logger.warning(f"Failed to fetch OHLCV from {exchange_name}: {e}")
        
        logger.error(f"❌ All OHLCV sources failed for {symbol}")
        return None