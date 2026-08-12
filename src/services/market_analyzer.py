"""
Market Analyzer - Real-time market data processing and signal generation
Compatible with new AI model v3.0.0 (derivatives, order book, stationary features)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
import json
import logging
import random
from typing import Dict, Optional, List, Any
import websockets
import requests
import warnings

from src.core.config import Settings, get_settings
from src.utils.safe_logger import SafeLogger
from src.data.collectors import MultiExchangeCollector
from src.data.processors import DataProcessor
from src.services.signal_generator import SignalGenerator
from src.services.portfolio_manager import PortfolioManager
from src.services.history_manager import HistoryManager
from src.services.telegram_service import TelegramService

warnings.filterwarnings('ignore', category=RuntimeWarning)

logger = SafeLogger.get_logger(__name__)


class MarketAnalyzer:
    """
    Market Analyzer - Real-time data analysis and signal generation.
    Supports derivatives data (funding rates, open interest) and order book imbalance.
    Guarantees hourly signal evaluations at the top of every hour.
    """
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.exchange_type = getattr(settings, 'EXCHANGE_TYPE', 'future')
        
        self.data_collector = MultiExchangeCollector(settings)
        self.data_processor = DataProcessor()
        
        self.is_running = False
        self.signal_generator = None
        self.portfolio_manager = None
        self.telegram_service = None  # Declare cleanly here
        self.history_manager = None
        self.model_trainer = None
        self.orderbook_monitor = None
        
        self.market_data = {}  # symbol -> DataFrame with features
        self.latest_signals = {}
        self.ws_connections = {}
        self.last_candle_times = {}
        
        self._data_lock = asyncio.Lock()
        
        self.performance_metrics = {
            'accuracy_1h': 0.56,
            'accuracy_4h': 0.60,
            'accuracy_1d': 0.58,
            'total_signals': 0,
            'successful_signals': 0,
            'features_count': 0,
            'has_derivatives': False,
            'has_orderbook': False,
            'timestamp': datetime.now().isoformat()
        }
        
        self.derivatives_cache = {}
        self.last_derivatives_fetch = {}
        self.logger = logger

    async def initialize(self):
        """Initialize the market analyzer with all components"""
        self.logger.info("📊 MarketAnalyzer v3.0.0 initializing...")

        try:
            from src.services.telegram_service import TelegramService
            self.telegram_service = TelegramService(self.settings)
        except Exception as e:
            self.logger.warning(f"Telegram service disabled or unavailable: {e}")
            self.telegram_service = None
        # 1. Initialize Signal Generator
        try:
            self.signal_generator = SignalGenerator(self.settings)
            await self.signal_generator.load_model()
            
            if self.signal_generator.model_loaded:
                self.logger.info("✅ Signal generator loaded successfully")
            else:
                self.logger.error("❌ Signal generator failed to load")
                raise Exception("Signal generator not loaded")
            
            if hasattr(self.signal_generator, 'feature_columns'):
                features = self.signal_generator.feature_columns
                self.performance_metrics['features_count'] = len(features)
                self.performance_metrics['has_derivatives'] = any(
                    'funding' in f or 'oi_' in f for f in features
                )
                self.performance_metrics['has_orderbook'] = any(
                    'buy_pressure' in f or 'order_imbalance' in f for f in features
                )
                self.logger.info(f"📊 Model loaded with {len(features)} stationary features")
            
            self.history_manager = HistoryManager()
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize signal generator: {e}")
            raise

# 2. Initialize Portfolio Manager
        try:
            from src.services.portfolio_manager import PortfolioManager
            
            profile_name = getattr(self.settings, 'TRADING_PROFILE', 'day_trader')
            
            self.portfolio_manager = PortfolioManager(
                initial_capital=self.settings.INITIAL_CAPITAL,
                profile_name=profile_name,  # Pass profile_name string cleanly
                history_manager=self.history_manager,
                telegram_service=self.telegram_service
            )
            self.logger.info(f"💰 Portfolio Manager initialized with profile: {profile_name}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize portfolio manager: {e}")

        # 3. Fetch initial historical data
        self.logger.info("📥 Fetching initial historical data...")
        for symbol in self.settings.SYMBOLS:
            try:
                df = await self.data_collector.fetch_data(symbol, limit=self.settings.MAX_HISTORICAL_DATA)
                
                if df is not None and not df.empty:
                    if 'timestamp' not in df.columns:
                        df = df.reset_index()
                    df = df.reset_index(drop=True)

                    df_featured = self.data_processor.engineer_features(df)
                    
                    if 'timestamp' not in df_featured.columns:
                        df_featured = df_featured.reset_index()
                    df_featured = df_featured.reset_index(drop=True)

                    self.market_data[symbol] = df_featured.fillna(0)
                    
                    if not self.data_processor.feature_columns:
                        self.data_processor.feature_columns = self.data_processor.get_stationary_features(df_featured)
                    
                    self.logger.info(f"✅ Loaded {len(df_featured)} records for {symbol} with features")
                else:
                    self.logger.warning(f"⚠️ No data for {symbol}")
                    
            except Exception as e:
                self.logger.error(f"❌ Failed to fetch data for {symbol}: {e}")

        # 4. Immediate startup signal check
        self.logger.info("🎯 Running initial signal evaluation on startup...")
        for symbol in self.settings.SYMBOLS:
            try:
                signal = await self.generate_signal(symbol)
                if signal:
                    self.logger.info(f"🚀 Initial Signal [{symbol}]: {signal['action']} (Conf: {signal['confidence']:.2f})")
                else:
                    self.logger.info(f"⏭️ No initial signal for {symbol}")
            except Exception as e:
                self.logger.error(f"❌ Error generating initial signal for {symbol}: {e}")

        self.logger.info("✅ MarketAnalyzer v3.0.0 initialized successfully")

    async def fetch_historical_data(self, symbol: str) -> pd.DataFrame:
        """Fetch historical data with derivatives support"""
        try:
            df = await self.data_collector.fetch_data(symbol, limit=self.settings.MAX_HISTORICAL_DATA)
            if df is not None and not df.empty:
                if 'timestamp' not in df.columns:
                    df = df.reset_index()
                return self.data_processor.engineer_features(df.reset_index(drop=True)).fillna(0)
            return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

    async def fetch_current_derivatives(self, symbol: str) -> Dict:
        """Fetch current derivatives data (funding rate, open interest)"""
        try:
            symbol_ccxt = symbol.replace('USDT', '/USDT')
            now = datetime.now()
            if symbol in self.last_derivatives_fetch:
                if (now - self.last_derivatives_fetch[symbol]).total_seconds() < 60:
                    return self.derivatives_cache.get(symbol, {})
            
            collector = self.data_collector.collectors.get('binance')
            if collector and hasattr(collector, 'fetch_current_derivatives'):
                data = await collector.fetch_current_derivatives(symbol_ccxt)
                self.derivatives_cache[symbol] = data
                self.last_derivatives_fetch[symbol] = now
                return data
            
            return {}
        except Exception as e:
            self.logger.error(f"Error fetching derivatives for {symbol}: {e}")
            return {}

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Helper to ensure timestamp is a column and index is clean"""
        if df.empty:
            return df
        df_copy = df.copy()
        if 'timestamp' not in df_copy.columns:
            df_copy = df_copy.reset_index()
        return df_copy.reset_index(drop=True)

    async def update_market_data(self, symbol: str, new_point: Dict):
        """Update market data with new closed point and re-engineer features"""
        async with self._data_lock:
            try:
                current_df = self._prepare_dataframe(self.market_data.get(symbol, pd.DataFrame()))
                new_row_df = pd.DataFrame([new_point])
                
                combined_df = pd.concat([current_df, new_row_df], ignore_index=True)
                
                if 'timestamp' in combined_df.columns:
                    combined_df = combined_df.drop_duplicates(subset=['timestamp'], keep='last')
                    combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
                
                featured_df = self.data_processor.engineer_features(combined_df)
                featured_df = self._prepare_dataframe(featured_df).fillna(0)
                
                self.market_data[symbol] = featured_df.tail(
                    self.settings.MAX_HISTORICAL_DATA
                ).reset_index(drop=True)
                
                self.last_candle_times[symbol] = datetime.now()
                
            except Exception as e:
                self.logger.error(f"❌ Error updating market data for {symbol}: {e}")

    async def update_market_data_only(self, symbol: str, kline_data: Dict):
        """Update last candle price for unclosed ticks efficiently"""
        async with self._data_lock:
            try:
                current_df = self.market_data.get(symbol)
                if current_df is None or current_df.empty:
                    return

                idx = current_df.index[-1]
                current_df.at[idx, 'close'] = float(kline_data['c'])
                current_df.at[idx, 'high'] = max(current_df.at[idx, 'high'], float(kline_data['h']))
                current_df.at[idx, 'low'] = min(current_df.at[idx, 'low'], float(kline_data['l']))
                current_df.at[idx, 'volume'] = float(kline_data['v'])
                
            except Exception as e:
                self.logger.error(f"❌ Error in tick update for {symbol}: {e}")

    async def generate_signal(self, symbol: str) -> Optional[Dict]:
            """Generate trading signal for a symbol using SignalGenerator"""
            try:
                current_data = self.market_data.get(symbol)
                if current_data is None or len(current_data) < 50:
                    self.logger.warning(f"⚠️ Insufficient data for {symbol}")
                    return None
                    
                current_data_clean = current_data.fillna(0).copy()
                current_price = float(current_data_clean['close'].iloc[-1])
                
                if self.signal_generator:
                    signal = await self.signal_generator.generate_signal(
                        symbol, current_data_clean, current_price
                    )
                    
                    if signal:
                        self.performance_metrics['total_signals'] += 1
                        self.performance_metrics['successful_signals'] += 1
                        
                        deriv_data = await self.fetch_current_derivatives(symbol)
                        if deriv_data:
                            signal['derivatives'] = {
                                'funding_rate': deriv_data.get('funding_rate', 0),
                                'open_interest': deriv_data.get('open_interest', 0),
                                'open_interest_usd': deriv_data.get('open_interest_usd', 0),
                            }

                        # Broadcast accepted VIP signal to Telegram Channel
                        if getattr(self.telegram_service, 'enable_telegram', False):
                            asyncio.create_task(self.telegram_service.broadcast_signal(signal))

                        if hasattr(self, 'orderbook_monitor') and self.orderbook_monitor:
                            ob_data = self.orderbook_monitor.get_imbalance(symbol)
                            if ob_data:
                                signal['orderbook'] = ob_data
                        
                        self.latest_signals[symbol] = signal
                        return signal
                        
                return None
                
            except Exception as e:
                self.logger.error(f"❌ Error generating signal for {symbol}: {e}", exc_info=True)
                return None

    async def process_market_data(self, symbol: str, kline_data: Dict):
        """Process incoming CLOSED kline data and generate signal"""
        try:
            if kline_data.get('x'):  # Only on closed candles
                self.logger.info(f"🕒 {symbol} 1h candle closed - Processing signal...")
                
                new_point = {
                    'timestamp': pd.to_datetime(kline_data['t'], unit='ms'),
                    'open': float(kline_data['o']),
                    'high': float(kline_data['h']),
                    'low': float(kline_data['l']),
                    'close': float(kline_data['c']),
                    'volume': float(kline_data['v']),
                }

                await self.update_market_data(symbol, new_point)
                
                signal = await self.generate_signal(symbol)
                if signal:
                    self.latest_signals[symbol] = signal
                    
                    if self.portfolio_manager:
                        position = self.portfolio_manager.open_position(signal)
                        if position:
                            self.logger.info(f"💰 PORTFOLIO: Opened {position.action} position for {symbol} "
                                            f"at ${position.entry_price:.4f}")
                        else:
                            should_trade, reason = self.portfolio_manager.should_open_position(signal)
                            if not should_trade:
                                self.logger.debug(f"⏭️ Portfolio skipped {symbol}: {reason}")
                    
                    self.logger.info(f"🎯 HOURLY SIGNAL: {symbol} {signal.get('action', 'HOLD')} "
                                    f"(Confidence: {signal.get('confidence', 0):.1%})")
                            
        except Exception as e:
            self.logger.error(f"❌ Error processing market data for {symbol}: {e}")

    async def _start_symbol_websocket(self, symbol: str):
        """Start WebSocket connection for a symbol"""
        base_url = "fstream.binance.com" if self.settings.EXCHANGE_TYPE.lower() == "future" else "stream.binance.com"
        stream_name = f"{symbol.lower()}@kline_1h"
        url = f"wss://{base_url}/ws/{stream_name}"
        
        self.logger.info(f"🔗 Connecting WebSocket for {symbol}...")
        await asyncio.sleep(random.uniform(0.5, 2.5))

        while self.is_running:
            try:
                async with websockets.connect(
                    url, ping_interval=30, ping_timeout=60, close_timeout=60, max_size=2**21
                ) as websocket:
                    self.ws_connections[symbol] = websocket
                    self.logger.info(f"✅ WebSocket connected for {symbol}")

                    async for message in websocket:
                        if not self.is_running:
                            break
                        try:
                            data = json.loads(message)
                            if "k" in data:
                                kline = data["k"]
                                if kline['x']:
                                    await self.process_market_data(symbol, kline)
                                else:
                                    await self.update_market_data_only(symbol, kline)
                        except Exception as e:
                            self.logger.error(f"❌ Error processing {symbol} message: {e}")

            except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
                self.logger.warning(f"🔄 Reconnecting {symbol} WebSocket...")
            except Exception as e:
                self.logger.error(f"❌ WebSocket error for {symbol}: {e}")

            if self.is_running:
                await asyncio.sleep(10)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # HOURLY CLOCK CHECKER (GUARANTEES 1H SIGNALS)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def start_hourly_clock_checker(self):
            """
            Background guard timer running at top of every hour (xx:00:05).
            Fetches fresh 1H candles and evaluates signals across all symbols,
            guaranteeing hourly execution even if WebSocket drops a frame.
            """
            self.logger.info("⏰ Top-of-hour clock timer started")
            while self.is_running:
                try:
                    now = datetime.now()
                    next_hour = (now + timedelta(hours=1)).replace(minute=0, second=5, microsecond=0)
                    wait_seconds = (next_hour - now).total_seconds()
                    
                    self.logger.info(f"⏰ Next hourly candle evaluation scheduled in {wait_seconds / 60:.1f} minutes")
                    await asyncio.sleep(wait_seconds)

                    if not self.is_running:
                        break

                    self.logger.info("🕒 TOP OF THE HOUR REACHED - Evaluating 1H Candle Signals...")
                    
                    for symbol in self.settings.SYMBOLS:
                        try:
                            df = await self.fetch_historical_data(symbol)
                            if df is not None and not df.empty:
                                async with self._data_lock:
                                    self.market_data[symbol] = df.fillna(0)
                                
                                signal = await self.generate_signal(symbol)
                                if signal:
                                    # ✅ FIX: Use signal['action'] for accurate log output
                                    trade_action = signal.get('action', 'HOLD')
                                    
                                    self.logger.info(
                                        f"🎯 HOURLY SIGNAL [{symbol}]: {trade_action} "
                                        f"(Conf: {signal['confidence']:.1%}, Strength: {signal['signal_strength']:.1%})"
                                    )
                                    if self.portfolio_manager:
                                        self.portfolio_manager.open_position(signal)
                                else:
                                    self.logger.info(f"⏭️ Hourly check: No accepted signal for {symbol}")
                        except Exception as e:
                            self.logger.error(f"❌ Error during hourly clock check for {symbol}: {e}")

                except Exception as e:
                    self.logger.error(f"❌ Error in hourly clock loop: {e}")
                    await asyncio.sleep(60)

    async def start_orderbook_monitor(self):
        """Start order book monitoring if module exists"""
        try:
            from src.services.orderbook_monitor import OrderBookMonitor
            self.orderbook_monitor = OrderBookMonitor(self.settings)
            asyncio.create_task(self.orderbook_monitor.start_monitoring())
            self.logger.info("📊 Order Book Monitor started")
        except ImportError:
            self.logger.warning("⚠️ Order Book Monitor module 'src.services.orderbook_monitor' not found. Skipping.")
        except Exception as e:
            self.logger.warning(f"⚠️ Order Book Monitor error: {e}")

    async def start_real_time_analysis(self):
        """Start real-time market data analysis with top-of-hour clock guard"""
        self.is_running = True
        self.logger.info("🚀 Starting real-time market analysis v3.0.0...")
        
        await self.start_orderbook_monitor()
        
        # Launch guaranteed top-of-hour clock timer
        asyncio.create_task(self.start_hourly_clock_checker())
        
        tasks = []
        for symbol in self.settings.SYMBOLS:
            tasks.append(asyncio.create_task(self._start_symbol_websocket(symbol)))
        
        await asyncio.gather(*tasks, return_exceptions=True)

    def get_signal(self, symbol: str) -> Optional[Dict]:
        return self.latest_signals.get(symbol)

    def get_latest_signals(self) -> Dict:
        return self.latest_signals

    def get_market_data(self, symbol: str) -> Optional[pd.DataFrame]:
        return self.market_data.get(symbol)

    def is_healthy(self) -> bool:
        return self.is_running and bool(self.market_data)

    async def cleanup(self):
        self.is_running = False
        for symbol, ws in self.ws_connections.items():
            if ws:
                try:
                    await ws.close()
                except Exception:
                    pass
        
        try:
            collector = self.data_collector.collectors.get('binance')
            if collector and hasattr(collector, 'exchange'):
                await collector.exchange.close()
        except Exception:
            pass

        self.logger.info("✅ All connections closed")