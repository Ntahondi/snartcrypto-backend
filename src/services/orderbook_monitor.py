"""
Order Book Monitor - Real-time L2 order book analysis
Provides imbalance, depth, and pressure signals for the AI model
Compatible with SmartCrypto AI v3.0.0
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Optional, List, Any
import websockets
import ccxt.async_support as ccxt
import numpy as np

from src.core.config import Settings, get_settings
from src.utils.logger import get_logger

from src.utils.safe_logger import SafeLogger
logger = SafeLogger.get_logger(__name__)


class OrderBookMonitor:
    """
    Real-time order book monitor for cryptocurrency trading.
    Provides:
    - Order book imbalance (-1 to +1)
    - Bid/Ask depth at multiple levels
    - Price pressure signals
    - Wall detection (large orders)
    - Spread analysis
    """
    
    def __init__(self, settings: Settings = None):
        self.settings = settings or get_settings()
        self.exchange = ccxt.binanceusdm({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        
        self.is_running = False
        self.orderbook_data = {}  # symbol -> latest snapshot
        self.orderbook_history = {}  # symbol -> historical snapshots
        self.ws_connections = {}
        
        # Depth levels to track (as percentage of mid price)
        self.depth_levels = [0.002, 0.005, 0.01, 0.02, 0.05]  # 0.2%, 0.5%, 1%, 2%, 5%
        
        # Feature cache for fast access
        self.feature_cache = {}
        self.last_update = {}
        
        # Performance tracking
        self.update_count = 0
        self.error_count = 0
        
        logger.info("📊 Order Book Monitor initialized")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # MAIN MONITORING LOOP
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def start_monitoring(self, symbols: List[str] = None):
        """Start order book monitoring for all symbols"""
        if symbols is None:
            symbols = self.settings.SYMBOLS
        
        self.is_running = True
        logger.info(f"📊 Starting Order Book Monitor for {len(symbols)} symbols...")
        
        tasks = []
        for symbol in symbols:
            task = asyncio.create_task(self._monitor_symbol(symbol))
            tasks.append(task)
        
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _monitor_symbol(self, symbol: str):
        """Monitor a single symbol's order book"""
        symbol_ccxt = symbol.replace('USDT', '/USDT')
        stream_name = f"{symbol.lower()}@depth20@100ms"
        
        # Determine WebSocket URL
        if self.settings.EXCHANGE_TYPE.lower() == "future":
            base_url = "fstream.binance.com"
        else:
            base_url = "stream.binance.com"
        
        url = f"wss://{base_url}/ws/{stream_name}"
        
        logger.info(f"🔗 Connecting OrderBook WebSocket for {symbol}...")
        
        while self.is_running:
            try:
                async with websockets.connect(
                    url,
                    ping_interval=30,
                    ping_timeout=60,
                    close_timeout=60,
                    max_size=2**21
                ) as websocket:
                    self.ws_connections[symbol] = websocket
                    logger.info(f"✅ OrderBook WebSocket connected for {symbol}")
                    
                    async for message in websocket:
                        if not self.is_running:
                            break
                        
                        try:
                            data = json.loads(message)
                            if 'bids' in data and 'asks' in data:
                                # Process order book update
                                features = self._process_orderbook(data, symbol)
                                self.orderbook_data[symbol] = features
                                self.feature_cache[symbol] = features
                                self.last_update[symbol] = datetime.now()
                                self.update_count += 1
                                
                        except json.JSONDecodeError:
                            logger.warning(f"⚠️ Invalid JSON from {symbol}")
                        except Exception as e:
                            logger.error(f"❌ Error processing {symbol} orderbook: {e}")
                            self.error_count += 1
                            
            except websockets.exceptions.ConnectionClosed:
                logger.warning(f"🔄 OrderBook reconnecting {symbol}...")
            except Exception as e:
                logger.error(f"❌ OrderBook error for {symbol}: {e}")
            
            if self.is_running:
                await asyncio.sleep(5)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # ORDER BOOK PROCESSING
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _process_orderbook(self, data: Dict, symbol: str) -> Dict:
        """
        Process raw order book data into meaningful features.
        Returns features compatible with the AI model.
        """
        try:
            bids = data.get('bids', [])
            asks = data.get('asks', [])
            
            if not bids or not asks:
                return self._get_default_features()
            
            # Convert to numpy for faster processing
            bids_array = np.array([[float(b[0]), float(b[1])] for b in bids[:50]])
            asks_array = np.array([[float(a[0]), float(a[1])] for a in asks[:50]])
            
            best_bid = bids_array[0][0]
            best_ask = asks_array[0][0]
            mid_price = (best_bid + best_ask) / 2.0
            spread = best_ask - best_bid
            spread_pct = spread / mid_price if mid_price > 0 else 0.0001
            
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # 1. DEPTH IMBALANCE AT MULTIPLE LEVELS
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            features = {
                'best_bid': best_bid,
                'best_ask': best_ask,
                'mid_price': mid_price,
                'spread': spread,
                'spread_pct': spread_pct,
                'best_bid_size': bids_array[0][1],
                'best_ask_size': asks_array[0][1],
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
            }
            
            # Imbalance at each depth level
            for pct in self.depth_levels:
                bid_limit = mid_price * (1 - pct)
                ask_limit = mid_price * (1 + pct)
                
                bid_vol = np.sum(bids_array[bids_array[:, 0] >= bid_limit][:, 1])
                ask_vol = np.sum(asks_array[asks_array[:, 0] <= ask_limit][:, 1])
                total_vol = bid_vol + ask_vol
                
                imbalance = (bid_vol - ask_vol) / (total_vol + 1e-8)
                
                features[f'bid_volume_{int(pct*1000)}bp'] = float(bid_vol)
                features[f'ask_volume_{int(pct*1000)}bp'] = float(ask_vol)
                features[f'imbalance_{int(pct*1000)}bp'] = float(imbalance)
            
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # 2. ORDER BOOK SLOPE (Depth Drop-off)
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            if len(bids_array) >= 20:
                # Average size at first 10 vs next 10 levels
                bid_slope = (np.mean(bids_array[:10, 1]) - np.mean(bids_array[10:20, 1])) / 10
                ask_slope = (np.mean(asks_array[:10, 1]) - np.mean(asks_array[10:20, 1])) / 10
                features['bid_slope'] = float(bid_slope)
                features['ask_slope'] = float(ask_slope)
            else:
                features['bid_slope'] = 0.0
                features['ask_slope'] = 0.0
            
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # 3. VOLUME-WEIGHTED AVERAGE PRICE (VWAP)
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            if len(bids_array) > 0:
                bid_vwap = np.average(bids_array[:, 0], weights=bids_array[:, 1] + 1e-8)
                ask_vwap = np.average(asks_array[:, 0], weights=asks_array[:, 1] + 1e-8)
                features['bid_vwap'] = float(bid_vwap)
                features['ask_vwap'] = float(ask_vwap)
                features['vwap_pressure'] = float((ask_vwap - bid_vwap) / mid_price)
            else:
                features['bid_vwap'] = mid_price
                features['ask_vwap'] = mid_price
                features['vwap_pressure'] = 0.0
            
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # 4. WALL DETECTION (Large Orders)
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            max_bid = np.max(bids_array[:, 1])
            max_ask = np.max(asks_array[:, 1])
            avg_bid = np.mean(bids_array[:, 1])
            avg_ask = np.mean(asks_array[:, 1])
            
            features['bid_wall_ratio'] = float(max_bid / (avg_bid + 1e-8))
            features['ask_wall_ratio'] = float(max_ask / (avg_ask + 1e-8))
            features['wall_imbalance'] = float((max_bid - max_ask) / (max_bid + max_ask + 1e-8))
            
            # Wall price levels (where the largest orders are)
            if len(bids_array) > 0:
                max_bid_idx = np.argmax(bids_array[:, 1])
                max_ask_idx = np.argmax(asks_array[:, 1])
                features['wall_bid_price'] = float(bids_array[max_bid_idx][0])
                features['wall_ask_price'] = float(asks_array[max_ask_idx][0])
            else:
                features['wall_bid_price'] = best_bid
                features['wall_ask_price'] = best_ask
            
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # 5. ORDER BOOK DENSITY
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            bid_density = np.sum(bids_array[:, 1] > np.mean(bids_array[:, 1]) * 0.1)
            ask_density = np.sum(asks_array[:, 1] > np.mean(asks_array[:, 1]) * 0.1)
            features['bid_density'] = int(bid_density)
            features['ask_density'] = int(ask_density)
            
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # 6. PRESSURE INDICATORS (for AI model)
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # Buy pressure = normalized imbalance
            features['buy_pressure'] = float(max(0, min(1, (features['imbalance_10bp'] + 1) / 2)))
            
            # Order imbalance (compatible with model features)
            features['order_imbalance'] = float(features['imbalance_10bp'])
            
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # 7. PRESSURE DIRECTION
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            if features['imbalance_10bp'] > 0.3:
                features['pressure_direction'] = 'BUY'
                features['pressure_strength'] = 'STRONG'
            elif features['imbalance_10bp'] > 0.1:
                features['pressure_direction'] = 'BUY'
                features['pressure_strength'] = 'MODERATE'
            elif features['imbalance_10bp'] < -0.3:
                features['pressure_direction'] = 'SELL'
                features['pressure_strength'] = 'STRONG'
            elif features['imbalance_10bp'] < -0.1:
                features['pressure_direction'] = 'SELL'
                features['pressure_strength'] = 'MODERATE'
            else:
                features['pressure_direction'] = 'NEUTRAL'
                features['pressure_strength'] = 'LOW'
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Error processing orderbook: {e}")
            return self._get_default_features()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # DEFAULT FEATURES (Fallback)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _get_default_features(self) -> Dict:
        """Return default features when order book fetch fails"""
        return {
            'best_bid': 0,
            'best_ask': 0,
            'mid_price': 0,
            'spread': 0,
            'spread_pct': 0.0001,
            'best_bid_size': 0,
            'best_ask_size': 0,
            'bid_volume_2bp': 0,
            'ask_volume_2bp': 0,
            'imbalance_2bp': 0,
            'bid_volume_5bp': 0,
            'ask_volume_5bp': 0,
            'imbalance_5bp': 0,
            'bid_volume_10bp': 0,
            'ask_volume_10bp': 0,
            'imbalance_10bp': 0,
            'bid_volume_20bp': 0,
            'ask_volume_20bp': 0,
            'imbalance_20bp': 0,
            'bid_volume_50bp': 0,
            'ask_volume_50bp': 0,
            'imbalance_50bp': 0,
            'bid_slope': 0,
            'ask_slope': 0,
            'bid_vwap': 0,
            'ask_vwap': 0,
            'vwap_pressure': 0,
            'bid_wall_ratio': 1,
            'ask_wall_ratio': 1,
            'wall_imbalance': 0,
            'wall_bid_price': 0,
            'wall_ask_price': 0,
            'bid_density': 0,
            'ask_density': 0,
            'buy_pressure': 0.5,
            'order_imbalance': 0,
            'pressure_direction': 'NEUTRAL',
            'pressure_strength': 'LOW',
            'timestamp': datetime.now().isoformat(),
            'symbol': 'UNKNOWN',
        }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PUBLIC METHODS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def get_imbalance(self, symbol: str) -> Optional[Dict]:
        """Get latest order book imbalance for a symbol"""
        features = self.feature_cache.get(symbol)
        if features:
            return {
                'imbalance': features.get('imbalance_10bp', 0),
                'spread_pct': features.get('spread_pct', 0.0001),
                'bid_volume': features.get('bid_volume_10bp', 0),
                'ask_volume': features.get('ask_volume_10bp', 0),
                'mid_price': features.get('mid_price', 0),
                'buy_pressure': features.get('buy_pressure', 0.5),
                'order_imbalance': features.get('order_imbalance', 0),
                'pressure_direction': features.get('pressure_direction', 'NEUTRAL'),
                'pressure_strength': features.get('pressure_strength', 'LOW'),
                'wall_imbalance': features.get('wall_imbalance', 0),
                'timestamp': features.get('timestamp', datetime.now().isoformat()),
            }
        return None

    def get_full_features(self, symbol: str) -> Optional[Dict]:
        """Get all order book features for a symbol"""
        return self.feature_cache.get(symbol)

    def get_pressure_signal(self, symbol: str) -> Optional[Dict]:
        """
        Get a simple pressure signal for trading decisions.
        Returns: {'direction': 'BUY'|'SELL'|'NEUTRAL', 'strength': 0-1, 'confidence': 0-1}
        """
        features = self.feature_cache.get(symbol)
        if not features:
            return None
        
        imbalance = features.get('imbalance_10bp', 0)
        wall_imbalance = features.get('wall_imbalance', 0)
        buy_pressure = features.get('buy_pressure', 0.5)
        
        # Combine signals
        combined = (imbalance * 0.6 + wall_imbalance * 0.2 + (buy_pressure - 0.5) * 2 * 0.2)
        confidence = min(1, abs(combined) * 2)
        
        if combined > 0.15:
            direction = 'BUY'
        elif combined < -0.15:
            direction = 'SELL'
        else:
            direction = 'NEUTRAL'
        
        return {
            'direction': direction,
            'strength': min(1, abs(combined) * 2),
            'confidence': confidence,
            'imbalance': imbalance,
            'buy_pressure': buy_pressure,
            'wall_imbalance': wall_imbalance,
            'timestamp': datetime.now().isoformat(),
        }

    def get_all_pressure_signals(self) -> Dict[str, Dict]:
        """Get pressure signals for all symbols"""
        signals = {}
        for symbol in self.feature_cache:
            signal = self.get_pressure_signal(symbol)
            if signal:
                signals[symbol] = signal
        return signals

    def get_stats(self) -> Dict:
        """Get monitor statistics"""
        return {
            'is_running': self.is_running,
            'symbols_monitored': len(self.feature_cache),
            'update_count': self.update_count,
            'error_count': self.error_count,
            'last_update': max(self.last_update.values()).isoformat() if self.last_update else None,
            'active_connections': len(self.ws_connections),
        }

    async def cleanup(self):
        """Clean up connections"""
        self.is_running = False
        for symbol, ws in self.ws_connections.items():
            try:
                await ws.close()
            except:
                pass
        
        try:
            await self.exchange.close()
        except:
            pass
        
        logger.info("🧹 Order Book Monitor cleaned up")