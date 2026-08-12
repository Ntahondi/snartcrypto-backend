"""
Enhanced Portfolio Manager with AI Integration
"""

import json
import uuid
import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import os

from src.utils.safe_logger import SafeLogger
from src.services.history_manager import HistoryManager
from src.services.real_trade_excutor import RealTradeExecutor
from src.core.trading_profiles import TradingProfile, get_profile, RiskTolerance
from src.core.config import get_settings

logger = SafeLogger.get_logger(__name__)


@dataclass
class Position:
    """Position data class with AI-enhanced fields"""
    id: str
    symbol: str
    action: str
    entry_price: float
    current_price: float
    quantity: float
    entry_time: datetime
    stop_loss: float
    take_profit: float
    pnl: Optional[float] = None
    pnl_percentage: Optional[float] = None
    status: str = "OPEN"
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    max_holding_hours: int = 8
    session_id: str = "default"
    signal_id: Optional[str] = None
    timeframe: str = "1h"
    profile_name: str = "day_trader"
    
    # AI-Enhanced Fields
    ai_confidence: float = 0.0
    ai_signal_strength: float = 0.0
    expected_return: float = 0.0
    expected_time_to_profit: float = 0.0
    ensemble_agreement: float = 0.0
    market_regime: str = "UNKNOWN"


class PortfolioManager:
    """
    AI-Integrated Portfolio Manager with Dynamic Risk Management
    """
    
    def __init__(
            self,
            initial_capital: float = 10000.0,
            profile_name: str = "day_trader",
            profile: Optional[Any] = None,  # Added for 100% backward compatibility
            positions_dir: str = "positions",
            history_manager: Optional[HistoryManager] = None,
            telegram_service: Optional[Any] = None
        ):
            self.initial_capital = initial_capital
            self.available_capital = initial_capital
            self.positions_dir = positions_dir
            self.history_manager = history_manager
            self.telegram_service = telegram_service
            
            # Accept profile object if passed, otherwise load by profile_name string
            if profile is not None:
                self.profile = profile
            else:
                self.profile = get_profile(profile_name)

            # Handle both Enum and string representations safely
            style_val = getattr(self.profile.trading_style, 'value', str(self.profile.trading_style))
            risk_val = getattr(self.profile.risk_tolerance, 'value', str(self.profile.risk_tolerance))

            logger.info(f"📊 Trading Profile: {style_val} (Risk: {risk_val})")
            
            self.market_data = {}
            self.atr_values = {}
            
            self.last_trade_time = {}
            self.daily_trades = 0
            self.daily_pnl = 0.0
            self.today = datetime.now().date()
            
            self.peak_portfolio_value = initial_capital
            self.current_drawdown = 0.0
            self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            try:
                self.settings = get_settings()
                self.trade_executor = RealTradeExecutor(self.settings)
                if getattr(self.trade_executor, 'enable_real_trading', False):
                    asyncio.create_task(self.trade_executor.initialize())
            except Exception as e:
                logger.warning(f"RealTradeExecutor disabled: {e}")
                self.trade_executor = None
            
            os.makedirs(positions_dir, exist_ok=True)
            self.open_positions: Dict[str, Position] = {}
            self.closed_positions: List[Position] = []
            self.trade_history_file = f"{positions_dir}/trade_history.jsonl"
            
            self._load_positions()
            logger.info(f"💰 Portfolio Manager initialized with ${initial_capital:,.2f}")
    
    def calculate_position_size(self, signal: Dict) -> Dict:
        """Calculate position size using AI confidence and Kelly Criterion"""
        confidence = signal.get('confidence', 0.5)
        signal_strength = signal.get('signal_strength', 0.5)
        
        if self.profile.use_kelly_sizing:
            adjusted_win_rate = self.profile.expected_win_rate * (1 + (confidence - 0.4) * 2)
            adjusted_win_rate = float(np.clip(adjusted_win_rate, 0.35, 0.75))
            position_pct = self.profile.calculate_kelly_optimal_position(adjusted_win_rate)
        else:
            position_pct = self.profile.position_size_pct
        
        confidence_scalar = float(np.clip(0.5 + (confidence - 0.3) * 2.5, 0.5, 1.5))
        position_pct *= confidence_scalar
        
        strength_scalar = float(0.8 + signal_strength * 0.4)
        position_pct *= strength_scalar
        
        max_position = self.profile.position_size_pct * 1.5
        min_position = self.profile.position_size_pct * 0.4
        position_pct = float(np.clip(position_pct, min_position, max_position))
        
        allocation_usd = self.available_capital * position_pct
        entry_price = float(signal['price'])
        quantity = allocation_usd / entry_price
        
        atr = self._get_atr(signal['symbol'])
        if atr is None:
            stop_loss_pct = self.profile.stop_loss_pct
            take_profit_pct = self.profile.take_profit_pct
            if signal['action'] == 'BUY':
                stop_loss = entry_price * (1.0 - stop_loss_pct)
                take_profit = entry_price * (1.0 + take_profit_pct)
            else:
                stop_loss = entry_price * (1.0 + stop_loss_pct)
                take_profit = entry_price * (1.0 - take_profit_pct)
        else:
            if signal['action'] == 'BUY':
                stop_loss = entry_price - (atr * self.profile.stop_loss_atr_mult)
                take_profit = entry_price + (atr * self.profile.take_profit_atr_mult)
            else:
                stop_loss = entry_price + (atr * self.profile.stop_loss_atr_mult)
                take_profit = entry_price - (atr * self.profile.take_profit_atr_mult)

        return {
            'position_size_pct': position_pct,
            'allocation_usd': allocation_usd,
            'quantity': quantity,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'max_holding_hours': self.profile.max_holding_hours,
            'atr_used': atr,
            'kelly_size': position_pct
        }
    
    def _get_atr(self, symbol: str) -> Optional[float]:
        if symbol in self.atr_values:
            return self.atr_values[symbol]
        
        if symbol in self.market_data:
            df = self.market_data[symbol]
            if 'atr14' in df.columns and not df.empty:
                atr = float(df['atr14'].iloc[-1])
                self.atr_values[symbol] = atr
                return atr
        
        default_atr = {'BTCUSDT': 310.0, 'ETHUSDT': 12.0, 'ADAUSDT': 0.0019, 'LINKUSDT': 0.06, 'SOLUSDT': 0.56, 'DOTUSDT': 0.007}
        return default_atr.get(symbol, None)
    
    def should_open_position(self, signal: Dict) -> Tuple[bool, str]:
        symbol = signal.get('symbol')
        confidence = signal.get('confidence', 0)
        signal_strength = signal.get('signal_strength', 0)
        action = signal.get('action', 'HOLD')
        
        if action == 'HOLD':
            return False, "Signal action is HOLD"
        
        min_conf = max(self.profile.min_confidence, 0.35)
        if confidence < min_conf:
            return False, f"Confidence too low: {confidence:.2%} < {min_conf:.2%}"
        
        min_strength = max(self.profile.min_signal_strength, 0.30)
        if signal_strength < min_strength:
            return False, f"Strength too low: {signal_strength:.2%} < {min_strength:.2%}"
        
        open_positions = [p for p in self.open_positions.values() if p.status == 'OPEN']
        
        if symbol in self.open_positions and not self.profile.allow_multiple_positions:
            return False, f"Already have open position for {symbol}"
        
        if len(open_positions) >= self.profile.max_total_positions:
            return False, f"Max positions reached ({len(open_positions)}/{self.profile.max_total_positions})"
        
        if self.daily_trades >= self.profile.max_daily_trades:
            return False, f"Daily trade limit reached ({self.daily_trades})"
        
        if self.daily_pnl <= -self.profile.max_daily_loss_pct * self.initial_capital:
            return False, f"Daily loss limit reached (${abs(self.daily_pnl):.2f})"
        
        if symbol in self.last_trade_time:
            time_since = (datetime.now() - self.last_trade_time[symbol]).total_seconds()
            if time_since < self.profile.min_time_between_trades:
                return False, f"Cooldown active ({int(time_since)}s < {self.profile.min_time_between_trades}s)"
        
        return True, "Signal meets all criteria"
    
    def open_position(self, signal: Dict) -> Optional[Position]:
        """Open position with AI-enhanced parameters (Clean Async Execution)"""
        symbol = signal['symbol']
        action = signal['action']
        
        # Smart Reversal: Close opposite direction position if running
        active_positions = {k: v for k, v in self.open_positions.items() if v.status == 'OPEN'}
        if symbol in active_positions:
            existing = active_positions[symbol]
            if existing.action != action:
                logger.info(f"🔄 TREND REVERSAL DETECTED: Closing old {existing.action} to open new {action}")
                self.close_position(symbol, float(signal['price']), reason="SIGNAL_REVERSAL")
            elif not self.profile.allow_multiple_positions:
                logger.info(f"⏭️ Skipping {symbol}: Already have open {action} position")
                return None

        should_trade, reason = self.should_open_position(signal)
        if not should_trade:
            logger.info(f"⏭️ Skipping {symbol}: {reason}")
            return None
        
        pos_info = self.calculate_position_size(signal)
        
        position = Position(
            id=str(uuid.uuid4())[:8],
            symbol=symbol,
            action=action,
            entry_price=pos_info['entry_price'],
            current_price=pos_info['entry_price'],
            quantity=pos_info['quantity'],
            entry_time=datetime.now(),
            stop_loss=pos_info['stop_loss'],
            take_profit=pos_info['take_profit'],
            max_holding_hours=pos_info['max_holding_hours'],
            session_id=self.session_id,
            signal_id=signal.get('signal_id'),
            timeframe=self.profile.signal_timeframe.value,
            profile_name=self.profile.trading_style.value,
            ai_confidence=signal.get('confidence', 0),
            ai_signal_strength=signal.get('signal_strength', 0),
            expected_return=float(signal.get('expected_returns', {}).get('4h_return', '+0.0%').replace('%', '')) / 100.0,
            expected_time_to_profit=signal.get('strategy', {}).get('max_holding_hours', 8),
            ensemble_agreement=1.0,
            market_regime=signal.get('market_regime', 'UNKNOWN')
        )
        
        self.available_capital -= pos_info['allocation_usd']
        self.open_positions[symbol] = position
        self.last_trade_time[symbol] = datetime.now()
        self.daily_trades += 1
        
        self._save_position(position)
        
        # Real Execution on Binance (Non-blocking async task)
        if hasattr(self, 'trade_executor') and self.trade_executor and getattr(self.trade_executor, 'enable_real_trading', False):
            asyncio.create_task(self.trade_executor.execute_open_position(position))
        
        # Telegram Alert
        #if hasattr(self, 'telegram_service') and self.telegram_service and getattr(self.telegram_service, 'enable_telegram', False):
        #    asyncio.create_task(self.telegram_service.broadcast_signal(signal))
        
        logger.info(f"🎯 OPENED: {symbol} {action} | Size: ${pos_info['allocation_usd']:.2f} | Entry: ${position.entry_price:.4f}")
        return position

    def monitor_position(self, symbol: str, current_price: float):
        if symbol not in self.open_positions:
            return
        
        position = self.open_positions[symbol]
        if position.status != 'OPEN':
            return
        
        position.current_price = current_price
        
        if position.action == "BUY":
            position.pnl = (current_price - position.entry_price) * position.quantity
        else:
            position.pnl = (position.entry_price - current_price) * position.quantity
        
        position.pnl_percentage = (position.pnl / (position.entry_price * position.quantity)) * 100
        
        exit_reason = None
        exit_price = None
        
        if position.action == "BUY":
            if current_price <= position.stop_loss:
                exit_reason = "STOP_LOSS"
                exit_price = position.stop_loss
            elif current_price >= position.take_profit:
                exit_reason = "TAKE_PROFIT"
                exit_price = position.take_profit
        else:
            if current_price >= position.stop_loss:
                exit_reason = "STOP_LOSS"
                exit_price = position.stop_loss
            elif current_price <= position.take_profit:
                exit_reason = "TAKE_PROFIT"
                exit_price = position.take_profit
        
        if not exit_reason:
            hold_hours = (datetime.now() - position.entry_time).total_seconds() / 3600
            if hold_hours >= position.max_holding_hours:
                exit_reason = "TIMEOUT"
                exit_price = current_price
        
        if exit_reason:
            self.close_position(symbol, exit_price, exit_reason)

    def close_position(self, symbol: str, exit_price: float, reason: str):
        """Close an open position cleanly"""
        if symbol not in self.open_positions:
            return
        
        position = self.open_positions.pop(symbol)
        exit_time = datetime.now()
        
        position.exit_price = exit_price
        position.exit_time = exit_time
        position.status = "CLOSED"
        position.current_price = exit_price
        
        if position.action == "BUY":
            final_pnl = (exit_price - position.entry_price) * position.quantity
        else:
            final_pnl = (position.entry_price - exit_price) * position.quantity
        
        position.pnl = final_pnl
        position.pnl_percentage = (final_pnl / (position.entry_price * position.quantity)) * 100
        
        self.daily_pnl += final_pnl
        allocation = position.entry_price * position.quantity
        self.available_capital += allocation + final_pnl
        
        self.closed_positions.append(position)
        self._save_position(position)
        
        if self.history_manager and position.signal_id:
            outcome = "WIN" if final_pnl > 0 else "LOSS"
            self.history_manager.update_signal_outcome(
                position.signal_id,
                outcome,
                pnl=final_pnl,
                pnl_percentage=position.pnl_percentage,
                exit_price=exit_price,
                exit_time=exit_time,
                position_id=position.id
            )
        
        # Real Binance Close Order (Non-blocking)
        if hasattr(self, 'trade_executor') and self.trade_executor and getattr(self.trade_executor, 'enable_real_trading', False):
            asyncio.create_task(self.trade_executor.execute_close_position(position, reason))
        
        # Telegram Alert
        if hasattr(self, 'telegram_service') and self.telegram_service and getattr(self.telegram_service, 'enable_telegram', False):
            asyncio.create_task(self.telegram_service.broadcast_trade_closed(position))
        
        result = "WIN" if final_pnl > 0 else "LOSS"
        logger.info(f"📊 CLOSED: {symbol} | {result} | PnL: ${final_pnl:.2f} ({position.pnl_percentage:.2f}%) | Reason: {reason}")

    def get_portfolio_value(self) -> float:
        open_val = sum(p.current_price * p.quantity for p in self.open_positions.values() if p.status == 'OPEN')
        return self.available_capital + open_val

    def _save_position(self, position: Position):
        try:
            data = {
                'id': position.id,
                'symbol': position.symbol,
                'action': position.action,
                'entry_price': position.entry_price,
                'current_price': position.current_price,
                'quantity': position.quantity,
                'entry_time': position.entry_time.isoformat() + 'Z',
                'stop_loss': position.stop_loss,
                'take_profit': position.take_profit,
                'pnl': position.pnl,
                'pnl_percentage': position.pnl_percentage,
                'status': position.status,
                'exit_price': position.exit_price,
                'exit_time': position.exit_time.isoformat() + 'Z' if position.exit_time else None,
                'max_holding_hours': position.max_holding_hours,
                'session_id': self.session_id,
                'signal_id': position.signal_id,
                'timeframe': position.timeframe,
                'profile_name': position.profile_name,
                'ai_confidence': position.ai_confidence,
                'ai_signal_strength': position.ai_signal_strength,
                'expected_return': position.expected_return,
                'expected_time_to_profit': position.expected_time_to_profit,
                'ensemble_agreement': position.ensemble_agreement,
                'market_regime': position.market_regime
            }
            with open(self.trade_history_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(data) + '\n')
        except Exception as e:
            logger.error(f"❌ Error saving position: {e}")

    def _load_positions(self):
        try:
            if os.path.exists(self.trade_history_file):
                with open(self.trade_history_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line.strip())
                            if data.get('status') == 'OPEN':
                                position = Position(
                                    id=data['id'],
                                    symbol=data['symbol'],
                                    action=data['action'],
                                    entry_price=data['entry_price'],
                                    current_price=data.get('current_price', data['entry_price']),
                                    quantity=data['quantity'],
                                    entry_time=datetime.fromisoformat(data['entry_time'].replace('Z', '')),
                                    stop_loss=data['stop_loss'],
                                    take_profit=data['take_profit'],
                                    status=data['status'],
                                    session_id=data.get('session_id', 'default'),
                                    signal_id=data.get('signal_id'),
                                    max_holding_hours=data.get('max_holding_hours', 8),
                                    timeframe=data.get('timeframe', '1h'),
                                    profile_name=data.get('profile_name', self.profile.trading_style.value),
                                    ai_confidence=data.get('ai_confidence', 0),
                                    ai_signal_strength=data.get('ai_signal_strength', 0),
                                    expected_return=data.get('expected_return', 0),
                                    expected_time_to_profit=data.get('expected_time_to_profit', 4),
                                    ensemble_agreement=data.get('ensemble_agreement', 1.0),
                                    market_regime=data.get('market_regime', 'UNKNOWN')
                                )
                                if (data.get('session_id') == self.session_id or 
                                    (datetime.now() - position.entry_time).total_seconds() < 86400):
                                    self.open_positions[position.symbol] = position
                                else:
                                    position.status = 'CLOSED'
                                    position.exit_price = position.current_price
                                    position.exit_time = datetime.now()
                                    position.pnl = 0
                                    position.pnl_percentage = 0
                                    self._save_position(position)
            logger.info(f"📂 Loaded {len(self.open_positions)} active positions")
        except Exception as e:
            logger.error(f"❌ Error loading positions: {e}")

    async def start_monitoring(self, market_analyzer):
        logger.info("🔍 Portfolio position monitoring started")
        while True:
            try:
                for symbol in list(self.open_positions.keys()):
                    if symbol in market_analyzer.market_data:
                        df = market_analyzer.market_data[symbol]
                        if not df.empty and 'atr14' in df.columns:
                            self.atr_values[symbol] = float(df['atr14'].iloc[-1])
                
                for symbol, position in list(self.open_positions.items()):
                    if position.status == 'OPEN' and symbol in market_analyzer.market_data:
                        current_data = market_analyzer.market_data[symbol]
                        if not current_data.empty:
                            current_price = float(current_data['close'].iloc[-1])
                            self.monitor_position(symbol, current_price)
                
                await asyncio.sleep(10)
            except Exception as e:
                logger.error(f"❌ Portfolio monitoring error: {e}")
                await asyncio.sleep(30)