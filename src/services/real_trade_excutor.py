# src/services/real_trade_executor.py

"""
Real Trade Executor for SmartCrypto AI v3.0.0
Handles live order execution on Binance (Spot / Futures) using CCXT Async
Supports Leverage setting, Isolated Margin, Market Orders, and attached SL/TP
"""

import asyncio
import ccxt.async_support as ccxt
import logging
from typing import Dict, Optional, Tuple, Any
from datetime import datetime
import os

from src.core.config import Settings, get_settings
from src.utils.safe_logger import SafeLogger

logger = SafeLogger.get_logger(__name__)


class RealTradeExecutor:
    """
    Executes real live trades on Binance Futures or Spot via CCXT.
    Converts PortfolioManager decisions into live exchange orders.
    Supports:
    - Real-time order execution (Market/Limit)
    - SL/TP conditional orders (Futures)
    - Leverage and margin configuration
    - Order verification and timeout
    - Paper trading mode for testing
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.exchange_type = getattr(self.settings, 'EXCHANGE_TYPE', 'future').lower()
        self.use_testnet = getattr(self.settings, 'USE_TESTNET', False)
        self.enable_real_trading = getattr(self.settings, 'ENABLE_REAL_TRADING', False)
        self.leverage = getattr(self.settings, 'DEFAULT_LEVERAGE', 3)
        self.margin_type = getattr(self.settings, 'MARGIN_TYPE', 'ISOLATED').upper()
        self.order_timeout = getattr(self.settings, 'ORDER_TIMEOUT', 30)
        self.max_retries = getattr(self.settings, 'MAX_ORDER_RETRIES', 3)
        
        self.api_key = getattr(self.settings, 'BINANCE_API_KEY', '')
        self.api_secret = getattr(self.settings, 'BINANCE_API_SECRET', '')
        
        self.exchange = None
        self.is_initialized = False
        self.active_orders = {}  # symbol -> dict of order IDs
        self._order_cache = {}  # order_id -> order details

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # INITIALIZATION
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def initialize(self) -> bool:
        """Initialize CCXT exchange connection and test API keys"""
        if not self.enable_real_trading:
            logger.info("ℹ️ RealTradeExecutor: ENABLE_REAL_TRADING is False. Running in Paper Trading Mode.")
            return False

        if not self.api_key or not self.api_secret:
            logger.error("❌ Cannot initialize RealTradeExecutor: Missing BINANCE_API_KEY or BINANCE_API_SECRET")
            return False
        
        try:
            config = {
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future' if self.exchange_type == 'future' else 'spot',
                    'adjustForTimeDifference': True
                }
            }
            
            self.exchange = ccxt.binance(config)
            
            if self.use_testnet:
                self.exchange.set_sandbox_mode(True)
                logger.info("🧪 RealTradeExecutor initialized in BINANCE TESTNET mode")
            else:
                logger.info("🚀 RealTradeExecutor initialized in LIVE REAL-MONEY MODE!")
                
            # Test connection
            balance = await self.exchange.fetch_balance()
            usdt_free = balance.get('free', {}).get('USDT', 0)
            logger.info(f"💰 Real Binance Balance Connected: ${usdt_free:.2f} USDT Available")
            
            self.is_initialized = True
            return True
            
        except ccxt.AuthenticationError as e:
            logger.error(f"❌ Binance Authentication Error: {e}")
            self.is_initialized = False
            return False
        except Exception as e:
            logger.error(f"❌ RealTradeExecutor initialization failed: {e}")
            self.is_initialized = False
            return False

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # HELPER METHODS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _get_ccxt_symbol(self, symbol: str) -> str:
        """Format symbol for CCXT Binance Futures / Spot"""
        symbol_clean = symbol.replace('USDT', '/USDT')
        if self.exchange_type == 'future':
            return f"{symbol_clean}:USDT"
        return symbol_clean

    def _get_action_side(self, action: str) -> str:
        """Convert action to exchange side"""
        return 'buy' if action.upper() == 'BUY' else 'sell'

    def _get_close_side(self, action: str) -> str:
        """Get opposite side for closing position"""
        return 'sell' if action.upper() == 'BUY' else 'buy'

    async def _check_balance(self, symbol: str, amount: float, price: float) -> Tuple[bool, float]:
        """Check if we have enough balance to execute order"""
        try:
            balance = await self.exchange.fetch_balance()
            usdt_free = balance.get('free', {}).get('USDT', 0)
            
            # Get estimated cost with buffer
            estimated_cost = amount * price * 1.01  # 1% buffer for slippage
            
            if estimated_cost > usdt_free * 0.95:  # 5% safety buffer
                logger.error(f"❌ Insufficient funds: Need ${estimated_cost:.2f}, Have ${usdt_free:.2f}")
                return False, usdt_free
            
            return True, usdt_free
            
        except Exception as e:
            logger.error(f"❌ Failed to check balance: {e}")
            return False, 0

    async def _verify_order_filled(self, order_id: str, symbol: str) -> bool:
        """Verify an order was actually filled"""
        try:
            order = await self.exchange.fetch_order(order_id, symbol)
            status = order.get('status')
            
            if status == 'closed':
                return True
            elif status == 'open':
                logger.warning(f"Order {order_id} still open")
                return False
            elif status == 'canceled':
                logger.warning(f"Order {order_id} was canceled")
                return False
            elif status == 'expired':
                logger.warning(f"Order {order_id} expired")
                return False
            else:
                logger.warning(f"Order {order_id} status: {status}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to verify order: {e}")
            return False

    def _format_amount(self, symbol: str, amount: float) -> float:
        """Format amount to exchange precision"""
        try:
            amount_precision = self.exchange.amount_to_precision(symbol, amount)
            return float(amount_precision)
        except:
            return round(amount, 8)

    def _format_price(self, symbol: str, price: float) -> float:
        """Format price to exchange precision"""
        try:
            price_precision = self.exchange.price_to_precision(symbol, price)
            return float(price_precision)
        except:
            return round(price, 2)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # OPEN POSITION
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def execute_open_position(self, position: Any) -> Optional[Dict]:
        """
        Executes a real market open position on Binance.
        Sets leverage, sets margin type, places entry order, and attaches SL/TP orders.
        """
        if not self.enable_real_trading:
            logger.info("ℹ️ Real trading disabled - skipping position open")
            return None

        if not self.is_initialized:
            initialized = await self.initialize()
            if not initialized:
                return None

        symbol_ccxt = self._get_ccxt_symbol(position.symbol)
        action = position.action.upper()
        side = self._get_action_side(action)
        amount = position.quantity
        entry_price = position.entry_price

        logger.info(f"⚡ EXECUTING REAL TRADE ON BINANCE: {action} {amount:.4f} {position.symbol} at ~${entry_price:.4f}")

        try:
            # 1. Configure Futures Margin & Leverage
            if self.exchange_type == 'future':
                try:
                    await self.exchange.set_margin_mode(self.margin_type, symbol_ccxt)
                except Exception as e:
                    logger.debug(f"Margin mode notice: {e}")

                try:
                    await self.exchange.set_leverage(self.leverage, symbol_ccxt)
                    logger.info(f"🔧 Set {self.leverage}x Leverage ({self.margin_type}) for {position.symbol}")
                except Exception as e:
                    logger.debug(f"Leverage notice: {e}")

            # 2. Format Amount Precision
            amount_float = self._format_amount(symbol_ccxt, amount)

            if amount_float <= 0:
                logger.error(f"❌ Position quantity {amount} below Binance minimum step size.")
                return None

            # 3. Check Balance
            has_funds, usdt_balance = await self._check_balance(symbol_ccxt, amount_float, entry_price)
            if not has_funds:
                return None

            # 4. Place Market Entry Order (with timeout)
            logger.info(f"📊 Placing {action} order: {amount_float} {position.symbol}")
            
            entry_order = await asyncio.wait_for(
                self.exchange.create_order(
                    symbol=symbol_ccxt,
                    type='market',
                    side=side,
                    amount=amount_float
                ),
                timeout=self.order_timeout
            )
            
            entry_id = entry_order.get('id')
            filled_price = float(entry_order.get('average', entry_price) or entry_price)
            
            # 5. Verify order was filled
            if not await self._verify_order_filled(entry_id, symbol_ccxt):
                logger.error(f"❌ Entry order {entry_id} was not filled")
                return None
            
            logger.info(f"✅ REAL ENTRY ORDER FILLED: ID={entry_id} | Symbol={position.symbol} | Filled Price=${filled_price:.4f}")

            # 6. Attach Stop Loss and Take Profit Conditional Orders (Futures)
            sl_id, tp_id = None, None
            if self.exchange_type == 'future':
                try:
                    sl_side = self._get_close_side(action)
                    
                    # Format prices
                    sl_price_formatted = self._format_price(symbol_ccxt, position.stop_loss)
                    tp_price_formatted = self._format_price(symbol_ccxt, position.take_profit)

                    # Stop Loss (STOP_MARKET)
                    sl_order = await self.exchange.create_order(
                        symbol=symbol_ccxt,
                        type='STOP_MARKET',
                        side=sl_side,
                        amount=amount_float,
                        params={
                            'stopPrice': sl_price_formatted,
                            'reduceOnly': True,
                            'closePosition': True
                        }
                    )
                    sl_id = sl_order.get('id')
                    
                    # Take Profit (TAKE_PROFIT_MARKET)
                    tp_order = await self.exchange.create_order(
                        symbol=symbol_ccxt,
                        type='TAKE_PROFIT_MARKET',
                        side=sl_side,
                        amount=amount_float,
                        params={
                            'stopPrice': tp_price_formatted,
                            'reduceOnly': True,
                            'closePosition': True
                        }
                    )
                    tp_id = tp_order.get('id')
                    logger.info(f"🛡️ Attached Real SL (${sl_price_formatted}) & TP (${tp_price_formatted}) Orders on Binance")

                except Exception as e:
                    logger.warning(f"⚠️ Could not set automated SL/TP conditional orders on exchange: {e}")

            order_record = {
                'entry_order_id': entry_id,
                'sl_order_id': sl_id,
                'tp_order_id': tp_id,
                'filled_price': filled_price,
                'amount': amount_float,
                'usdt_balance': usdt_balance,
                'timestamp': datetime.now().isoformat()
            }
            self.active_orders[position.symbol] = order_record
            self._order_cache[entry_id] = order_record
            
            logger.info(f"🎯 POSITION OPENED: {position.symbol} | Entry: ${filled_price:.2f} | Size: {amount_float}")
            return order_record

        except asyncio.TimeoutError:
            logger.error(f"❌ Order execution timeout for {position.symbol}")
            return None
        except ccxt.InsufficientFunds as e:
            logger.error(f"❌ Insufficient Funds on Binance for {position.symbol}: {e}")
            return None
        except ccxt.InvalidOrder as e:
            logger.error(f"❌ Invalid Order parameters for {position.symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Error executing real open position for {position.symbol}: {e}", exc_info=True)
            return None

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # CLOSE POSITION
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def execute_close_position(self, position: Any, reason: str = "MANUAL") -> Optional[Dict]:
        """
        Executes a real close position on Binance (Market order in opposite direction).
        Cancels remaining SL/TP conditional orders.
        """
        if not self.enable_real_trading or not self.is_initialized:
            return None

        symbol_ccxt = self._get_ccxt_symbol(position.symbol)
        close_side = self._get_close_side(position.action)
        amount = position.quantity

        logger.info(f"⚡ CLOSING REAL POSITION ON BINANCE: {position.symbol} ({reason}) | Amount: {amount:.4f}")

        try:
            # 1. Cancel active SL/TP orders for this symbol
            if self.exchange_type == 'future':
                try:
                    await self.exchange.cancel_all_orders(symbol_ccxt)
                    logger.info(f"🧹 Cancelled remaining SL/TP conditional orders for {position.symbol}")
                except Exception as e:
                    logger.debug(f"Cancel orders notice: {e}")

            # 2. Format amount
            amount_float = self._format_amount(symbol_ccxt, amount)
            
            if amount_float <= 0:
                logger.error(f"❌ Close quantity {amount} invalid")
                return None

            # 3. Execute Market Close Order (with timeout)
            close_order = await asyncio.wait_for(
                self.exchange.create_order(
                    symbol=symbol_ccxt,
                    type='market',
                    side=close_side,
                    amount=amount_float,
                    params={'reduceOnly': True} if self.exchange_type == 'future' else {}
                ),
                timeout=self.order_timeout
            )

            close_id = close_order.get('id')
            exit_price = float(close_order.get('average', position.current_price) or position.current_price)
            
            # 4. Verify order was filled
            if not await self._verify_order_filled(close_id, symbol_ccxt):
                logger.error(f"❌ Close order {close_id} was not filled")
                return None
            
            logger.info(f"✅ REAL CLOSE ORDER FILLED: ID={close_id} | Symbol={position.symbol} | Exit Price=${exit_price:.4f}")

            # 5. Clean up tracking
            if position.symbol in self.active_orders:
                del self.active_orders[position.symbol]
            self._order_cache[close_id] = {'type': 'close', 'symbol': position.symbol}

            return {
                'close_order_id': close_id,
                'exit_price': exit_price,
                'reason': reason,
                'timestamp': datetime.now().isoformat()
            }

        except asyncio.TimeoutError:
            logger.error(f"❌ Close order timeout for {position.symbol}")
            return None
        except Exception as e:
            logger.error(f"❌ Error executing real close position for {position.symbol}: {e}", exc_info=True)
            return None

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # ORDER MANAGEMENT
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an open order"""
        if not self.is_initialized:
            return False
        
        try:
            symbol_ccxt = self._get_ccxt_symbol(symbol)
            await self.exchange.cancel_order(order_id, symbol_ccxt)
            logger.info(f"Order {order_id} cancelled")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to cancel order: {e}")
            return False

    async def cancel_all_orders(self, symbol: str) -> bool:
        """Cancel all open orders for a symbol"""
        if not self.is_initialized:
            return False
        
        try:
            symbol_ccxt = self._get_ccxt_symbol(symbol)
            await self.exchange.cancel_all_orders(symbol_ccxt)
            logger.info(f"All orders cancelled for {symbol}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to cancel all orders: {e}")
            return False

    async def get_order_status(self, order_id: str, symbol: str) -> Optional[Dict]:
        """Get status of a specific order"""
        if not self.is_initialized:
            return None
        
        try:
            symbol_ccxt = self._get_ccxt_symbol(symbol)
            order = await self.exchange.fetch_order(order_id, symbol_ccxt)
            return {
                'id': order.get('id'),
                'symbol': order.get('symbol'),
                'side': order.get('side'),
                'type': order.get('type'),
                'status': order.get('status'),
                'price': float(order.get('price', 0)),
                'average': float(order.get('average', 0)),
                'filled': float(order.get('filled', 0)),
                'remaining': float(order.get('remaining', 0)),
                'cost': float(order.get('cost', 0)),
                'fee': order.get('fee'),
                'timestamp': order.get('timestamp')
            }
        except Exception as e:
            logger.error(f"❌ Failed to get order status: {e}")
            return None

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # BALANCE & POSITION QUERIES
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def get_balance(self, currency: str = 'USDT') -> Dict:
        """Get account balance for a currency"""
        if not self.is_initialized:
            return {
                'free': 0,
                'used': 0,
                'total': 0,
                'is_live': False
            }
        
        try:
            balance = await self.exchange.fetch_balance()
            if currency in balance:
                return {
                    'free': float(balance[currency]['free']),
                    'used': float(balance[currency]['used']),
                    'total': float(balance[currency]['total']),
                    'is_live': True
                }
            return {
                'free': 0,
                'used': 0,
                'total': 0,
                'is_live': True
            }
        except Exception as e:
            logger.error(f"❌ Failed to fetch balance: {e}")
            return {
                'free': 0,
                'used': 0,
                'total': 0,
                'is_live': False,
                'error': str(e)
            }

    async def get_position(self, symbol: str) -> Optional[Dict]:
        """Get current position for a symbol (futures only)"""
        if self.exchange_type != 'future' or not self.is_initialized:
            return None
        
        try:
            symbol_ccxt = self._get_ccxt_symbol(symbol)
            positions = await self.exchange.fetch_positions([symbol_ccxt])
            for pos in positions:
                if pos.get('symbol') == symbol_ccxt:
                    return {
                        'symbol': pos.get('symbol'),
                        'side': pos.get('side'),
                        'contracts': float(pos.get('contracts', 0)),
                        'entry_price': float(pos.get('entryPrice', 0)),
                        'mark_price': float(pos.get('markPrice', 0)),
                        'pnl': float(pos.get('unrealizedPnl', 0)),
                        'percentage': float(pos.get('percentage', 0)),
                        'leverage': float(pos.get('leverage', 1))
                    }
            return None
        except Exception as e:
            logger.error(f"❌ Failed to fetch position: {e}")
            return None

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # HEALTH & STATUS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def health_check(self) -> Dict:
        """Check exchange connection health"""
        if not self.is_initialized:
            return {
                'status': 'simulation' if self.enable_real_trading else 'disabled',
                'message': 'Running in paper trading mode' if not self.enable_real_trading else 'Not initialized'
            }
        
        try:
            ticker = await self.exchange.fetch_ticker('BTC/USDT')
            return {
                'status': 'healthy',
                'btc_price': ticker.get('last', 0),
                'exchange_type': self.exchange_type,
                'leverage': self.leverage,
                'timestamp': datetime.now().isoformat(),
                'is_live': True
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # CLEANUP
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def close(self):
        """Close exchange connection cleanly"""
        if self.exchange:
            try:
                await self.exchange.close()
                logger.info("✅ Binance exchange connection closed")
            except Exception as e:
                logger.error(f"❌ Error closing exchange: {e}")