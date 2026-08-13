"""
Trading Profiles - Single source of truth for all trading configurations
Includes 'test' profile for unrestricted empirical data collection.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Optional, List
import json
import numpy as np


class TradingStyle(str, Enum):
    SCALPER = "scalper"
    DAY_TRADER = "day_trader"
    SWING = "swing"
    POSITION = "position"
    TEST = "test"


class RiskTolerance(str, Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    EXTREME = "extreme"


class SignalTimeframe(str, Enum):
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    ALL = "all"


@dataclass
class TradingProfile:
    """Complete trading profile with AI-optimized parameters"""
    
    trading_style: TradingStyle = TradingStyle.DAY_TRADER
    risk_tolerance: RiskTolerance = RiskTolerance.MODERATE
    signal_timeframe: SignalTimeframe = SignalTimeframe.H1
    
    max_positions_per_symbol: int = 1
    max_total_positions: int = 10
    position_size_pct: float = 0.08
    max_holding_hours: int = 8
    
    # AI Signal Filters
    min_confidence: float = 0.40  
    min_signal_strength: float = 0.35
    require_timeframe_alignment: bool = True
    require_ensemble_agreement: bool = True  
    
    # Risk Management
    stop_loss_atr_mult: float = 1.5  
    take_profit_atr_mult: float = 3.0  
    max_daily_loss_pct: float = 0.08
    max_drawdown_pct: float = 0.20
    use_trailing_stop: bool = True
    
    # Kelly Criterion
    use_kelly_sizing: bool = True
    kelly_fraction: float = 0.25
    expected_win_rate: float = 0.58
    avg_win_loss_ratio: float = 2.0
    
    # Execution
    allow_multiple_positions: bool = False
    auto_compound: bool = True
    max_daily_trades: int = 20
    min_time_between_trades: int = 300
    
    def to_dict(self) -> Dict:
        return {
            'trading_style': self.trading_style.value,
            'risk_tolerance': self.risk_tolerance.value,
            'signal_timeframe': self.signal_timeframe.value,
            'max_positions_per_symbol': self.max_positions_per_symbol,
            'max_total_positions': self.max_total_positions,
            'position_size_pct': self.position_size_pct,
            'max_holding_hours': self.max_holding_hours,
            'min_confidence': self.min_confidence,
            'min_signal_strength': self.min_signal_strength,
            'require_timeframe_alignment': self.require_timeframe_alignment,
            'require_ensemble_agreement': self.require_ensemble_agreement,
            'stop_loss_atr_mult': self.stop_loss_atr_mult,
            'take_profit_atr_mult': self.take_profit_atr_mult,
            'max_daily_loss_pct': self.max_daily_loss_pct,
            'max_drawdown_pct': self.max_drawdown_pct,
            'use_trailing_stop': self.use_trailing_stop,
            'use_kelly_sizing': self.use_kelly_sizing,
            'kelly_fraction': self.kelly_fraction,
            'expected_win_rate': self.expected_win_rate,
            'avg_win_loss_ratio': self.avg_win_loss_ratio,
            'allow_multiple_positions': self.allow_multiple_positions,
            'auto_compound': self.auto_compound,
            'max_daily_trades': self.max_daily_trades,
            'min_time_between_trades': self.min_time_between_trades
        }
    
    def calculate_kelly_optimal_position(self, win_rate: float = None) -> float:
        wr = win_rate or self.expected_win_rate
        b = self.avg_win_loss_ratio
        kelly = (wr * b - (1.0 - wr)) / b
        optimal = kelly * self.kelly_fraction
        max_position = self.position_size_pct * 1.5
        min_position = self.position_size_pct * 0.4
        return float(np.clip(optimal, min_position, max_position))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# UNRESTRICTED PAPER TESTING PROFILE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_profile_test() -> TradingProfile:
    """Unrestricted Paper Testing Profile - Accepts all AI signals for empirical data collection"""
    return TradingProfile(
        trading_style=TradingStyle.TEST,
        risk_tolerance=RiskTolerance.EXTREME,
        signal_timeframe=SignalTimeframe.H1,
        max_positions_per_symbol=50,       # Allows multiple paper trades per symbol
        max_total_positions=200,           # High position cap
        position_size_pct=0.02,            # Small 2% size per trade
        max_holding_hours=8,               # 8h hold duration
        min_confidence=0.35,               # Opens paper trade on all approved ensemble signals
        min_signal_strength=0.30,
        require_timeframe_alignment=False,
        require_ensemble_agreement=False,
        stop_loss_atr_mult=1.5,
        take_profit_atr_mult=3.0,
        max_daily_loss_pct=0.50,
        max_drawdown_pct=0.50,
        use_trailing_stop=True,
        use_kelly_sizing=False,            # Fixed 2% sizing during data collection
        expected_win_rate=0.58,
        avg_win_loss_ratio=2.0,
        allow_multiple_positions=True,     # ✅ Allows opening paper trades on every signal
        auto_compound=True,
        max_daily_trades=500,              # High trade limit
        min_time_between_trades=0          # Zero cooldown
    )


def get_profile_scalper() -> TradingProfile:
    return TradingProfile(
        trading_style=TradingStyle.SCALPER,
        risk_tolerance=RiskTolerance.AGGRESSIVE,
        signal_timeframe=SignalTimeframe.H1,
        max_positions_per_symbol=5,
        max_total_positions=50,
        position_size_pct=0.03,
        max_holding_hours=2,
        min_confidence=0.35,
        min_signal_strength=0.30,
        require_timeframe_alignment=False,
        require_ensemble_agreement=False,
        stop_loss_atr_mult=1.0,
        take_profit_atr_mult=1.5,
        max_daily_loss_pct=0.05,
        max_drawdown_pct=0.15,
        use_trailing_stop=False,
        use_kelly_sizing=True,
        kelly_fraction=0.2,
        expected_win_rate=0.58,
        avg_win_loss_ratio=1.5,
        allow_multiple_positions=True,
        auto_compound=True,
        max_daily_trades=50,
        min_time_between_trades=120
    )


def get_profile_day_trader() -> TradingProfile:
    return TradingProfile(
        trading_style=TradingStyle.DAY_TRADER,
        risk_tolerance=RiskTolerance.MODERATE,
        signal_timeframe=SignalTimeframe.H1,
        max_positions_per_symbol=1,
        max_total_positions=10,
        position_size_pct=0.08,
        max_holding_hours=8,
        min_confidence=0.40,
        min_signal_strength=0.35,
        require_timeframe_alignment=True,
        require_ensemble_agreement=True,
        stop_loss_atr_mult=1.5,
        take_profit_atr_mult=3.0,
        max_daily_loss_pct=0.08,
        max_drawdown_pct=0.20,
        use_trailing_stop=True,
        use_kelly_sizing=True,
        kelly_fraction=0.25,
        expected_win_rate=0.58,
        avg_win_loss_ratio=2.0,
        allow_multiple_positions=False,
        auto_compound=True,
        max_daily_trades=20,
        min_time_between_trades=300
    )


def get_profile_swing() -> TradingProfile:
    return TradingProfile(
        trading_style=TradingStyle.SWING,
        risk_tolerance=RiskTolerance.CONSERVATIVE,
        signal_timeframe=SignalTimeframe.H4,
        max_positions_per_symbol=1,
        max_total_positions=5,
        position_size_pct=0.12,
        max_holding_hours=24,
        min_confidence=0.50,
        min_signal_strength=0.45,
        require_timeframe_alignment=True,
        require_ensemble_agreement=True,
        stop_loss_atr_mult=2.0,
        take_profit_atr_mult=4.0,
        max_daily_loss_pct=0.05,
        max_drawdown_pct=0.15,
        use_trailing_stop=True,
        use_kelly_sizing=True,
        kelly_fraction=0.3,
        expected_win_rate=0.58,
        avg_win_loss_ratio=2.5,
        allow_multiple_positions=False,
        auto_compound=True,
        max_daily_trades=5,
        min_time_between_trades=3600
    )


def get_profile_position() -> TradingProfile:
    return TradingProfile(
        trading_style=TradingStyle.POSITION,
        risk_tolerance=RiskTolerance.CONSERVATIVE,
        signal_timeframe=SignalTimeframe.D1,
        max_positions_per_symbol=1,
        max_total_positions=3,
        position_size_pct=0.15,
        max_holding_hours=168,
        min_confidence=0.60,
        min_signal_strength=0.55,
        require_timeframe_alignment=True,
        require_ensemble_agreement=True,
        stop_loss_atr_mult=2.5,
        take_profit_atr_mult=5.0,
        max_daily_loss_pct=0.03,
        max_drawdown_pct=0.25,
        use_trailing_stop=True,
        use_kelly_sizing=True,
        kelly_fraction=0.35,
        expected_win_rate=0.58,
        avg_win_loss_ratio=3.0,
        allow_multiple_positions=False,
        auto_compound=False,
        max_daily_trades=2,
        min_time_between_trades=86400
    )


PROFILES = {
    'scalper': get_profile_scalper,
    'day_trader': get_profile_day_trader,
    'swing': get_profile_swing,
    'position': get_profile_position,
    'test': get_profile_test
}


def get_profile(name: str) -> TradingProfile:
    if name in PROFILES:
        return PROFILES[name]()
    return get_profile_day_trader()