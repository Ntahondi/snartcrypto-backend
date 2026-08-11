"""
Advanced risk management system with ATR-based stops
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging

from src.core.config import Settings
from src.utils.logger import get_logger

from src.utils.safe_logger import SafeLogger
logger = SafeLogger.get_logger(__name__)

class RiskManager:
    """Advanced risk management with dynamic ATR-based stops"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.portfolio = {}
        
        # Risk parameters
        self.atr_multiplier_sl = 1.5
        self.atr_multiplier_tp = 3.0
        self.max_position_size = 0.15
        self.min_position_size = 0.01
        
    def calculate_position_size(self, signal: Dict, portfolio_value: float, 
                              risk_tolerance: str = "MODERATE") -> Dict:
        """Calculate optimal position size using Kelly Criterion"""
        
        risk_multipliers = {
            "CONSERVATIVE": 0.5,
            "MODERATE": 1.0,
            "AGGRESSIVE": 1.5
        }
        
        confidence = signal.get('confidence', 0.5)
        win_probability = self.estimate_win_probability(signal)
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 🔥 USE ATR FOR RISK CALCULATION
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        atr = signal.get('strategy', {}).get('atr_used', None)
        
        if atr and atr > 0:
            # Use ATR to determine risk per unit
            risk_per_unit = atr * self.atr_multiplier_sl
            expected_return = atr * self.atr_multiplier_tp
        else:
            # Fallback to fixed percentages
            risk_per_unit = signal['price'] * 0.02
            expected_return = signal['price'] * 0.04
        
        # Kelly Criterion: f* = (bp - q) / b
        b = expected_return / risk_per_unit  # payoff ratio
        kelly_fraction = (b * win_probability - (1 - win_probability)) / b
        
        # Apply risk tolerance and constraints
        max_kelly = kelly_fraction * risk_multipliers.get(risk_tolerance, 1.0)
        position_size = min(max_kelly, self.max_position_size)
        position_size = max(position_size, self.min_position_size)
        
        position_value = portfolio_value * position_size
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 🔥 USE ATR FOR SL/TP CALCULATION
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        stop_loss, take_profit = self.calculate_atr_stop_loss(signal, atr)
        
        return {
            'position_size': position_size,
            'position_value': position_value,
            'units': position_value / signal['price'],
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_reward_ratio': b,
            'atr_used': atr,
            'risk_per_unit': risk_per_unit,
            'expected_return': expected_return,
        }
    
    def calculate_atr_stop_loss(self, signal: Dict, atr: Optional[float] = None) -> Tuple[float, float]:
        """Calculate dynamic stop loss and take profit using ATR"""
        price = signal['price']
        action = signal.get('action', signal.get('direction_1h', 'HOLD'))
        
        # Get ATR from signal or use fallback
        if atr is None or atr <= 0:
            atr = signal.get('strategy', {}).get('atr_used', price * 0.01)
        
        if action == 'BUY':
            stop_loss = price - self.atr_multiplier_sl * atr
            take_profit = price + self.atr_multiplier_tp * atr
        elif action == 'SELL':
            stop_loss = price + self.atr_multiplier_sl * atr
            take_profit = price - self.atr_multiplier_tp * atr
        else:  # HOLD
            stop_loss = price
            take_profit = price
        
        return stop_loss, take_profit
    
    def estimate_win_probability(self, signal: Dict) -> float:
        """Estimate win probability based on signal strength and confidence"""
        base_prob = 0.5
        signal_strength = signal.get('signal_strength', 0.5)
        confidence = signal.get('confidence', 0.5)
        
        # Combine factors with higher weight on confidence
        win_prob = base_prob + (signal_strength * 0.2) + (confidence * 0.3)
        return min(max(win_prob, 0.3), 0.85)  # Clamp between 30% and 85%
    
    def calculate_stop_loss(self, signal: Dict, current_data: Optional[pd.DataFrame] = None) -> float:
        """Calculate dynamic stop loss using ATR if available"""
        if current_data is not None and 'atr14' in current_data.columns:
            atr = current_data['atr14'].iloc[-1]
            stop_loss, _ = self.calculate_atr_stop_loss(signal, atr)
            return stop_loss
        
        # Fallback to fixed percentage
        if signal.get('action', signal.get('direction_1h', 'HOLD')) == 'BUY':
            return signal['price'] * (1 - self.settings.STOP_LOSS_PCT)
        else:
            return signal['price'] * (1 + self.settings.STOP_LOSS_PCT)
    
    def calculate_take_profit(self, signal: Dict, current_data: Optional[pd.DataFrame] = None) -> float:
        """Calculate dynamic take profit using ATR if available"""
        if current_data is not None and 'atr14' in current_data.columns:
            atr = current_data['atr14'].iloc[-1]
            _, take_profit = self.calculate_atr_stop_loss(signal, atr)
            return take_profit
        
        # Fallback to fixed percentage
        if signal.get('action', signal.get('direction_1h', 'HOLD')) == 'BUY':
            return signal['price'] * (1 + self.settings.TAKE_PROFIT_PCT)
        else:
            return signal['price'] * (1 - self.settings.TAKE_PROFIT_PCT)
    
    def validate_portfolio_risk(self, portfolio: Dict) -> bool:
        """Validate overall portfolio risk"""
        total_exposure = sum(
            pos.get('position_value', 0) for pos in portfolio.values() 
            if pos.get('position_value', 0) > 0
        )
        
        max_portfolio_risk = self.settings.INITIAL_CAPITAL * 0.3
        return total_exposure <= max_portfolio_risk
    
    def get_atr_from_data(self, current_data: pd.DataFrame) -> Optional[float]:
        """Extract ATR from data if available"""
        if 'atr14' in current_data.columns and not current_data['atr14'].isna().all():
            return float(current_data['atr14'].iloc[-1])
        return None