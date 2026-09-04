"""
SmartCrypto AI - Trading Profiles
=================================

Single source of truth for trading-policy configuration.

This module defines:
    - Trading styles
    - Risk tolerances
    - Signal timeframes
    - Model participation rules
    - Model 4 strategy rules
    - Risk-management limits
    - Position-sizing policy
    - Execution limits

IMPORTANT ARCHITECTURE RULE
---------------------------
TradingProfile describes WHAT the trading system is allowed to do.

It does NOT:
    - place orders
    - modify exchange positions
    - track account balance
    - track realized PnL
    - track open positions
    - maintain portfolio state

Those responsibilities belong to the portfolio/risk/execution layers.

This separation is important because the portfolio manager must remain
the final authority over portfolio-level constraints.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import math


# ============================================================================
# CONSTANTS
# ============================================================================

MODEL4_STRATEGIES: tuple[str, ...] = (
    "momentum_reversal",
    "ma_crossover",
    "heikin_ashi",
    "swing_trading",
    "candlestick",
    "role_reversal",
    "bollinger_squeeze",
    "narrow_range",
    "rsi_2",
)


# ============================================================================
# ENUMS
# ============================================================================

class TradingStyle(str, Enum):
    """High-level trading style."""

    SCALPER = "scalper"
    DAY_TRADER = "day_trader"
    SWING = "swing"
    POSITION = "position"
    TEST = "test"


class RiskTolerance(str, Enum):
    """Qualitative risk category."""

    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    EXTREME = "extreme"


class SignalTimeframe(str, Enum):
    """Primary decision timeframe."""

    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    ALL = "all"


# ============================================================================
# TRADING PROFILE
# ============================================================================

@dataclass
class TradingProfile:
    """
    Complete trading policy.

    Percentages represented as floats use decimal notation:

        0.01 = 1%
        0.05 = 5%
        0.10 = 10%

    Position-size percentages refer to portfolio equity allocation,
    not leverage-adjusted notional exposure.
    """

    # ------------------------------------------------------------------------
    # CORE PROFILE
    # ------------------------------------------------------------------------

    trading_style: TradingStyle = TradingStyle.DAY_TRADER
    risk_tolerance: RiskTolerance = RiskTolerance.MODERATE
    signal_timeframe: SignalTimeframe = SignalTimeframe.H1

    max_positions_per_symbol: int = 1
    max_total_positions: int = 10

    position_size_pct: float = 0.08
    max_holding_hours: int = 8

    # ------------------------------------------------------------------------
    # SIGNAL ACCEPTANCE
    # ------------------------------------------------------------------------

    min_confidence: float = 0.40
    min_signal_strength: float = 0.35

    require_timeframe_alignment: bool = True
    require_ensemble_agreement: bool = True

    # ------------------------------------------------------------------------
    # CONTINUOUS RETURN REGRESSION AI
    # ------------------------------------------------------------------------

    use_regression_model: bool = True

    min_expected_return: float = 0.005

    regression_weight_1h: float = 1.0
    regression_weight_4h: float = 1.5
    regression_weight_12h: float = 1.5

    # ------------------------------------------------------------------------
    # SMART TRADER AI
    # ------------------------------------------------------------------------

    use_smart_trader_model: bool = True

    smart_trader_weight: float = 1.0

    require_4h_direction: bool = True
    require_1d_confirmation: bool = False

    # ------------------------------------------------------------------------
    # MARKET GPT WORLD MODEL
    # ------------------------------------------------------------------------

    use_market_gpt_model: bool = True

    market_gpt_weight: float = 1.0

    min_market_gpt_probability: float = 0.55

    # ------------------------------------------------------------------------
    # MODEL 4 - STRATEGY DETECTOR
    # ------------------------------------------------------------------------

    use_model4_strategy_detector: bool = True

    model4_min_strategy_probability: float = 0.50
    model4_strong_strategy_probability: float = 0.75

    model4_min_active_strategies: int = 1
    model4_max_active_strategies: int = 9

    model4_require_strong_strategy: bool = False

    model4_use_as_direction_vote: bool = False

    model4_require_strategy_confirmation: bool = False

    model4_enabled_strategies: List[str] = field(
        default_factory=lambda: list(MODEL4_STRATEGIES)
    )

    model4_strategy_weights: Dict[str, float] = field(
        default_factory=lambda: {
            strategy: 1.0
            for strategy in MODEL4_STRATEGIES
        }
    )

    # ------------------------------------------------------------------------
    # RISK MANAGEMENT
    # ------------------------------------------------------------------------

    stop_loss_atr_mult: float = 1.5
    take_profit_atr_mult: float = 3.0

    max_daily_loss_pct: float = 0.08
    max_drawdown_pct: float = 0.20

    use_trailing_stop: bool = True

    # ------------------------------------------------------------------------
    # KELLY POSITION SIZING
    # ------------------------------------------------------------------------

    use_kelly_sizing: bool = True

    kelly_fraction: float = 0.25

    expected_win_rate: float = 0.58
    avg_win_loss_ratio: float = 2.0

    # ------------------------------------------------------------------------
    # EXECUTION LIMITS
    # ------------------------------------------------------------------------

    allow_multiple_positions: bool = False

    auto_compound: bool = True

    max_daily_trades: int = 20

    min_time_between_trades: int = 300

    # ------------------------------------------------------------------------
    # AI PROFIT SHIELD & DYNAMIC RECOVERY
    # ------------------------------------------------------------------------

    enable_profit_shield: bool = True
    tier1_breakeven_trigger_pct: float = 2.0
    tier1_fee_buffer_pct: float = 0.8
    tier2_lock_trigger_pct: float = 3.5
    tier2_profit_lock_pct: float = 1.8
    tier3_trail_trigger_pct: float = 6.0
    tier3_trail_distance_pct: float = 1.5
    enable_smart_recovery: bool = True
    recovery_shallow_dip_max_loss_pct: float = 1.5
    recovery_extension_hours: int = 2
    max_recovery_capped_hours: int = 6

    # =========================================================================
    # NORMALIZATION
    # =========================================================================

    def __post_init__(self) -> None:
        """Normalize and validate profile values."""

        # Normalize enum-like string inputs.
        if isinstance(self.trading_style, str):
            self.trading_style = TradingStyle(
                self.trading_style.strip().lower()
            )

        if isinstance(self.risk_tolerance, str):
            self.risk_tolerance = RiskTolerance(
                self.risk_tolerance.strip().lower()
            )

        if isinstance(self.signal_timeframe, str):
            self.signal_timeframe = SignalTimeframe(
                self.signal_timeframe.strip().lower()
            )

        # Normalize strategy names.
        self.model4_enabled_strategies = [
            str(strategy).strip().lower()
            for strategy in self.model4_enabled_strategies
        ]

        # Remove duplicates while preserving order.
        self.model4_enabled_strategies = list(
            dict.fromkeys(
                self.model4_enabled_strategies
            )
        )

        # Normalize strategy weights.
        self.model4_strategy_weights = {
            str(strategy).strip().lower(): float(weight)
            for strategy, weight
            in self.model4_strategy_weights.items()
        }

        self.validate(raise_on_error=True)

    @property
    def name(self) -> str:
        """Return profile name as string."""
        return self.trading_style.value

    # =========================================================================
    # VALIDATION
    # =========================================================================

    def validate(
        self,
        raise_on_error: bool = False,
    ) -> List[str]:
        """
        Validate the complete profile.

        Parameters
        ----------
        raise_on_error:
            Raise ValueError immediately when invalid.

        Returns
        -------
        list[str]
            Validation errors.
        """

        errors: List[str] = []

        # --------------------------------------------------------------------
        # CORE LIMITS
        # --------------------------------------------------------------------

        if self.max_positions_per_symbol < 1:
            errors.append(
                "max_positions_per_symbol must be >= 1."
            )

        if self.max_total_positions < 1:
            errors.append(
                "max_total_positions must be >= 1."
            )

        if self.position_size_pct <= 0:
            errors.append(
                "position_size_pct must be greater than 0."
            )

        if self.position_size_pct > 1.0:
            errors.append(
                "position_size_pct cannot exceed 1.0."
            )

        if self.max_holding_hours <= 0:
            errors.append(
                "max_holding_hours must be greater than 0."
            )

        # --------------------------------------------------------------------
        # SIGNAL FILTERS
        # --------------------------------------------------------------------

        self._validate_probability(
            self.min_confidence,
            "min_confidence",
            errors,
        )

        self._validate_probability(
            self.min_signal_strength,
            "min_signal_strength",
            errors,
        )

        if self.min_expected_return < 0:
            errors.append(
                "min_expected_return cannot be negative."
            )

        # --------------------------------------------------------------------
        # MODEL WEIGHTS
        # --------------------------------------------------------------------

        weights = {
            "regression_weight_1h": self.regression_weight_1h,
            "regression_weight_4h": self.regression_weight_4h,
            "regression_weight_12h": self.regression_weight_12h,
            "smart_trader_weight": self.smart_trader_weight,
            "market_gpt_weight": self.market_gpt_weight,
        }

        for name, weight in weights.items():

            if weight < 0:
                errors.append(
                    f"{name} cannot be negative."
                )

        # --------------------------------------------------------------------
        # MARKET GPT
        # --------------------------------------------------------------------

        self._validate_probability(
            self.min_market_gpt_probability,
            "min_market_gpt_probability",
            errors,
        )

        # --------------------------------------------------------------------
        # MODEL 4
        # --------------------------------------------------------------------

        self._validate_probability(
            self.model4_min_strategy_probability,
            "model4_min_strategy_probability",
            errors,
        )

        self._validate_probability(
            self.model4_strong_strategy_probability,
            "model4_strong_strategy_probability",
            errors,
        )

        if (
            self.model4_strong_strategy_probability
            < self.model4_min_strategy_probability
        ):
            errors.append(
                "model4_strong_strategy_probability must be >= "
                "model4_min_strategy_probability."
            )

        if self.model4_min_active_strategies < 0:
            errors.append(
                "model4_min_active_strategies cannot be negative."
            )

        if (
            self.model4_min_active_strategies
            > len(MODEL4_STRATEGIES)
        ):
            errors.append(
                "model4_min_active_strategies cannot exceed "
                f"{len(MODEL4_STRATEGIES)}."
            )

        if self.model4_max_active_strategies < 1:
            errors.append(
                "model4_max_active_strategies must be >= 1."
            )

        if (
            self.model4_max_active_strategies
            > len(MODEL4_STRATEGIES)
        ):
            errors.append(
                "model4_max_active_strategies cannot exceed "
                f"{len(MODEL4_STRATEGIES)}."
            )

        if (
            self.model4_min_active_strategies
            > self.model4_max_active_strategies
        ):
            errors.append(
                "model4_min_active_strategies cannot exceed "
                "model4_max_active_strategies."
            )

        unknown_strategies = [
            strategy
            for strategy in self.model4_enabled_strategies
            if strategy not in MODEL4_STRATEGIES
        ]

        if unknown_strategies:
            errors.append(
                "Unknown Model 4 strategies: "
                + ", ".join(unknown_strategies)
            )

        for strategy, weight in self.model4_strategy_weights.items():

            if strategy not in MODEL4_STRATEGIES:
                errors.append(
                    f"Unknown Model 4 strategy weight: {strategy}"
                )

            if weight < 0:
                errors.append(
                    f"Model 4 strategy weight cannot be negative: "
                    f"{strategy}"
                )

        # --------------------------------------------------------------------
        # RISK
        # --------------------------------------------------------------------

        if self.stop_loss_atr_mult <= 0:
            errors.append(
                "stop_loss_atr_mult must be greater than 0."
            )

        if self.take_profit_atr_mult <= 0:
            errors.append(
                "take_profit_atr_mult must be greater than 0."
            )

        if (
            self.take_profit_atr_mult
            <= self.stop_loss_atr_mult
        ):
            errors.append(
                "take_profit_atr_mult should be greater than "
                "stop_loss_atr_mult."
            )

        self._validate_probability(
            self.max_daily_loss_pct,
            "max_daily_loss_pct",
            errors,
        )

        self._validate_probability(
            self.max_drawdown_pct,
            "max_drawdown_pct",
            errors,
        )

        # --------------------------------------------------------------------
        # KELLY
        # --------------------------------------------------------------------

        self._validate_probability(
            self.kelly_fraction,
            "kelly_fraction",
            errors,
        )

        self._validate_probability(
            self.expected_win_rate,
            "expected_win_rate",
            errors,
        )

        if self.avg_win_loss_ratio <= 0:
            errors.append(
                "avg_win_loss_ratio must be greater than 0."
            )

        # --------------------------------------------------------------------
        # EXECUTION
        # --------------------------------------------------------------------

        if self.max_daily_trades < 0:
            errors.append(
                "max_daily_trades cannot be negative."
            )

        if self.min_time_between_trades < 0:
            errors.append(
                "min_time_between_trades cannot be negative."
            )

        # --------------------------------------------------------------------
        # RESULT
        # --------------------------------------------------------------------

        if raise_on_error and errors:

            raise ValueError(
                "Invalid TradingProfile:\n"
                + "\n".join(
                    f"  - {error}"
                    for error in errors
                )
            )

        return errors

    @staticmethod
    def _validate_probability(
        value: float,
        field_name: str,
        errors: List[str],
    ) -> None:
        """Validate a decimal probability/percentage field."""

        if not math.isfinite(value):
            errors.append(
                f"{field_name} must be finite."
            )
            return

        if not 0.0 <= value <= 1.0:
            errors.append(
                f"{field_name} must be between 0 and 1."
            )

    # =========================================================================
    # SERIALIZATION
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the profile into a JSON-compatible dictionary.
        """

        data = asdict(self)

        data["trading_style"] = self.trading_style.value
        data["risk_tolerance"] = self.risk_tolerance.value
        data["signal_timeframe"] = self.signal_timeframe.value

        return data

    # =========================================================================
    # KELLY SIZING
    # =========================================================================

    def calculate_kelly_optimal_position(
        self,
        win_rate: Optional[float] = None,
        win_loss_ratio: Optional[float] = None,
    ) -> float:
        """
        Calculate fractional Kelly position size.

        Kelly formula:

            f* = (p*b - q) / b

        where:
            p = probability of winning
            q = probability of losing
            b = average win/loss ratio

        The configured kelly_fraction applies a fractional-Kelly reduction.

        The final value is bounded by the profile's base position size.
        """

        if not self.use_kelly_sizing:
            return float(self.position_size_pct)

        p = (
            self.expected_win_rate
            if win_rate is None
            else float(win_rate)
        )

        b = (
            self.avg_win_loss_ratio
            if win_loss_ratio is None
            else float(win_loss_ratio)
        )

        if not math.isfinite(p):
            return float(self.position_size_pct)

        if not math.isfinite(b) or b <= 0:
            return float(self.position_size_pct)

        p = min(max(p, 0.0), 1.0)

        q = 1.0 - p

        full_kelly = (
            (p * b) - q
        ) / b

        fractional_kelly = (
            full_kelly
            * self.kelly_fraction
        )

        # Never allow Kelly to produce a negative position.
        fractional_kelly = max(
            fractional_kelly,
            0.0,
        )

        # Keep sizing bounded relative to the configured profile.
        max_position = self.position_size_pct * 1.5

        return float(
            min(
                fractional_kelly,
                max_position,
            )
        )

    # =========================================================================
    # MODEL 4 HELPERS
    # =========================================================================

    def is_strategy_enabled(
        self,
        strategy: str,
    ) -> bool:
        """Return whether a Model 4 strategy is enabled."""

        if not self.use_model4_strategy_detector:
            return False

        normalized = str(strategy).strip().lower()

        return normalized in self.model4_enabled_strategies

    def strategy_probability_is_valid(
        self,
        probability: float,
    ) -> bool:
        """Return whether a Model 4 probability passes the minimum threshold."""

        if not math.isfinite(probability):
            return False

        return (
            probability
            >= self.model4_min_strategy_probability
        )

    def strategy_is_strong(
        self,
        probability: float,
    ) -> bool:
        """Return whether a strategy reaches the strong threshold."""

        if not math.isfinite(probability):
            return False

        return (
            probability
            >= self.model4_strong_strategy_probability
        )

    def accepts_model4_signal(
        self,
        active_strategy_count: int,
        strongest_probability: float,
    ) -> bool:
        """
        Determine whether Model 4 passes its profile-level acceptance rules.

        This method does NOT determine trade direction.
        It only determines whether the Model 4 result satisfies policy.
        """

        if not self.use_model4_strategy_detector:
            return True

        if active_strategy_count < 0:
            return False

        if (
            active_strategy_count
            < self.model4_min_active_strategies
        ):
            return False

        if (
            active_strategy_count
            > self.model4_max_active_strategies
        ):
            return False

        if not math.isfinite(strongest_probability):
            return False

        if self.model4_require_strong_strategy:

            if (
                strongest_probability
                < self.model4_strong_strategy_probability
            ):
                return False

        return True

    # =========================================================================
    # PROFILE SUMMARY
    # =========================================================================

    def summary(self) -> Dict[str, Any]:
        """
        Return a compact profile summary useful for logs/API responses.
        """

        return {
            "trading_style": self.trading_style.value,
            "risk_tolerance": self.risk_tolerance.value,
            "signal_timeframe": self.signal_timeframe.value,

            "position_size_pct": self.position_size_pct,
            "max_total_positions": self.max_total_positions,
            "max_positions_per_symbol": (
                self.max_positions_per_symbol
            ),

            "min_confidence": self.min_confidence,
            "min_signal_strength": self.min_signal_strength,
            "min_expected_return": self.min_expected_return,

            "models": {
                "regression": self.use_regression_model,
                "smart_trader": self.use_smart_trader_model,
                "market_gpt": self.use_market_gpt_model,
                "model4": self.use_model4_strategy_detector,
            },

            "model4": {
                "min_probability": (
                    self.model4_min_strategy_probability
                ),
                "strong_probability": (
                    self.model4_strong_strategy_probability
                ),
                "min_active_strategies": (
                    self.model4_min_active_strategies
                ),
                "max_active_strategies": (
                    self.model4_max_active_strategies
                ),
                "require_confirmation": (
                    self.model4_require_strategy_confirmation
                ),
                "use_as_direction_vote": (
                    self.model4_use_as_direction_vote
                ),
            },

            "risk": {
                "stop_loss_atr": self.stop_loss_atr_mult,
                "take_profit_atr": self.take_profit_atr_mult,
                "max_daily_loss_pct": self.max_daily_loss_pct,
                "max_drawdown_pct": self.max_drawdown_pct,
                "trailing_stop": self.use_trailing_stop,
            },

            "execution": {
                "allow_multiple_positions": (
                    self.allow_multiple_positions
                ),
                "max_daily_trades": self.max_daily_trades,
                "min_time_between_trades": (
                    self.min_time_between_trades
                ),
            },
        }


# ============================================================================
# PROFILE FACTORIES
# ============================================================================

def get_profile_test() -> TradingProfile:
    """
    Paper-testing profile.

    This profile is intended for empirical testing and data collection.

    It deliberately relaxes ensemble/timeframe confirmation rules, but
    portfolio-level safety remains the responsibility of the portfolio
    manager.
    """

    return TradingProfile(
        trading_style=TradingStyle.TEST,
        risk_tolerance=RiskTolerance.EXTREME,
        signal_timeframe=SignalTimeframe.H1,

        max_positions_per_symbol=50,
        max_total_positions=200,

        position_size_pct=0.02,
        max_holding_hours=8,

        min_confidence=0.35,
        min_signal_strength=0.30,

        require_timeframe_alignment=False,
        require_ensemble_agreement=False,

        # Regression
        use_regression_model=True,
        min_expected_return=0.005,

        # Smart Trader
        use_smart_trader_model=True,
        require_4h_direction=False,
        require_1d_confirmation=False,

        # Market GPT
        use_market_gpt_model=True,
        min_market_gpt_probability=0.55,

        # Model 4
        use_model4_strategy_detector=True,
        model4_min_strategy_probability=0.50,
        model4_strong_strategy_probability=0.75,
        model4_min_active_strategies=1,
        model4_max_active_strategies=9,
        model4_require_strong_strategy=False,
        model4_use_as_direction_vote=False,
        model4_require_strategy_confirmation=False,

        # Risk
        stop_loss_atr_mult=1.5,
        take_profit_atr_mult=3.0,

        max_daily_loss_pct=0.50,
        max_drawdown_pct=0.50,

        use_trailing_stop=True,

        # Kelly disabled during empirical collection
        use_kelly_sizing=False,

        expected_win_rate=0.58,
        avg_win_loss_ratio=2.0,

        # Execution
        allow_multiple_positions=True,
        auto_compound=True,

        max_daily_trades=500,
        min_time_between_trades=0,
    )


def get_profile_scalper() -> TradingProfile:
    """Calibrated short-duration high-probability scalping profile for live exchange execution."""

    return TradingProfile(
        trading_style=TradingStyle.SCALPER,
        risk_tolerance=RiskTolerance.AGGRESSIVE,
        signal_timeframe=SignalTimeframe.H1,

        max_positions_per_symbol=1,
        max_total_positions=9,

        position_size_pct=0.05,
        max_holding_hours=2,

        min_confidence=0.65,
        min_signal_strength=0.60,

        require_timeframe_alignment=True,
        require_ensemble_agreement=True,

        # Regression
        use_regression_model=True,
        min_expected_return=0.004,

        # Smart Trader
        use_smart_trader_model=True,
        require_4h_direction=False,
        require_1d_confirmation=False,

        # Market GPT
        use_market_gpt_model=True,
        min_market_gpt_probability=0.55,

        # Model 4
        use_model4_strategy_detector=True,
        model4_min_strategy_probability=0.50,
        model4_strong_strategy_probability=0.75,
        model4_min_active_strategies=1,
        model4_max_active_strategies=9,
        model4_require_strong_strategy=False,
        model4_use_as_direction_vote=False,
        model4_require_strategy_confirmation=False,

        # Risk
        stop_loss_atr_mult=1.0,
        take_profit_atr_mult=1.8,

        max_daily_loss_pct=0.05,
        max_drawdown_pct=0.15,

        use_trailing_stop=True,

        # Kelly
        use_kelly_sizing=True,
        kelly_fraction=0.20,

        expected_win_rate=0.62,
        avg_win_loss_ratio=1.6,

        # AI Profit Shield
        enable_profit_shield=True,
        tier1_breakeven_trigger_pct=1.2,
        tier1_fee_buffer_pct=0.5,
        tier2_lock_trigger_pct=2.0,
        tier2_profit_lock_pct=1.0,
        tier3_trail_trigger_pct=3.0,
        tier3_trail_distance_pct=0.6,
        enable_smart_recovery=True,
        recovery_shallow_dip_max_loss_pct=1.0,
        recovery_extension_hours=1,
        max_recovery_capped_hours=3,

        # Execution
        allow_multiple_positions=False,
        auto_compound=True,

        max_daily_trades=30,
        min_time_between_trades=60,
    )


def get_profile_day_trader() -> TradingProfile:
    """Primary balanced H1 day-trading profile."""

    return TradingProfile(
        trading_style=TradingStyle.DAY_TRADER,
        risk_tolerance=RiskTolerance.MODERATE,
        signal_timeframe=SignalTimeframe.H1,

        max_positions_per_symbol=1,
        max_total_positions=9,

        position_size_pct=0.10,
        max_holding_hours=8,

        min_confidence=0.60,
        min_signal_strength=0.50,

        require_timeframe_alignment=True,
        require_ensemble_agreement=True,

        # Regression
        use_regression_model=True,
        min_expected_return=0.005,

        # Smart Trader
        use_smart_trader_model=True,
        require_4h_direction=True,
        require_1d_confirmation=False,

        # Market GPT
        use_market_gpt_model=True,
        min_market_gpt_probability=0.55,

        # Model 4
        use_model4_strategy_detector=True,
        model4_min_strategy_probability=0.50,
        model4_strong_strategy_probability=0.75,
        model4_min_active_strategies=1,
        model4_max_active_strategies=9,
        model4_require_strong_strategy=False,
        model4_use_as_direction_vote=False,
        model4_require_strategy_confirmation=False,

        # Risk
        stop_loss_atr_mult=1.5,
        take_profit_atr_mult=3.0,

        max_daily_loss_pct=0.08,
        max_drawdown_pct=0.20,

        use_trailing_stop=True,

        # Kelly
        use_kelly_sizing=True,
        kelly_fraction=0.25,

        expected_win_rate=0.58,
        avg_win_loss_ratio=2.0,

        # AI Profit Shield
        enable_profit_shield=True,
        tier1_breakeven_trigger_pct=2.0,
        tier1_fee_buffer_pct=0.8,
        tier2_lock_trigger_pct=3.5,
        tier2_profit_lock_pct=1.8,
        tier3_trail_trigger_pct=6.0,
        tier3_trail_distance_pct=1.5,
        enable_smart_recovery=True,
        recovery_shallow_dip_max_loss_pct=1.5,
        recovery_extension_hours=2,
        max_recovery_capped_hours=6,

        # Execution
        allow_multiple_positions=False,
        auto_compound=True,

        max_daily_trades=20,
        min_time_between_trades=300,
    )


def get_profile_swing() -> TradingProfile:
    """Conservative multi-hour/multi-day swing profile."""

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

        # Regression
        use_regression_model=True,
        min_expected_return=0.0075,

        # Smart Trader
        use_smart_trader_model=True,
        require_4h_direction=True,
        require_1d_confirmation=True,

        # Market GPT
        use_market_gpt_model=True,
        min_market_gpt_probability=0.55,

        # Model 4
        use_model4_strategy_detector=True,
        model4_min_strategy_probability=0.55,
        model4_strong_strategy_probability=0.75,
        model4_min_active_strategies=1,
        model4_max_active_strategies=9,
        model4_require_strong_strategy=False,
        model4_use_as_direction_vote=False,
        model4_require_strategy_confirmation=True,

        # Risk
        stop_loss_atr_mult=2.0,
        take_profit_atr_mult=4.0,

        max_daily_loss_pct=0.05,
        max_drawdown_pct=0.15,

        use_trailing_stop=True,

        # Kelly
        use_kelly_sizing=True,
        kelly_fraction=0.30,

        expected_win_rate=0.58,
        avg_win_loss_ratio=2.5,

        # AI Profit Shield
        enable_profit_shield=True,
        tier1_breakeven_trigger_pct=3.5,
        tier1_fee_buffer_pct=1.5,
        tier2_lock_trigger_pct=7.0,
        tier2_profit_lock_pct=3.5,
        tier3_trail_trigger_pct=12.0,
        tier3_trail_distance_pct=4.0,
        enable_smart_recovery=True,
        recovery_shallow_dip_max_loss_pct=2.5,
        recovery_extension_hours=12,
        max_recovery_capped_hours=48,

        # Execution
        allow_multiple_positions=False,
        auto_compound=True,

        max_daily_trades=5,
        min_time_between_trades=3600,
    )


def get_profile_position() -> TradingProfile:
    """Longer-horizon position-trading profile."""

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

        # Regression
        use_regression_model=True,
        min_expected_return=0.01,

        # Smart Trader
        use_smart_trader_model=True,
        require_4h_direction=True,
        require_1d_confirmation=True,

        # Market GPT
        use_market_gpt_model=True,
        min_market_gpt_probability=0.60,

        # Model 4
        use_model4_strategy_detector=True,
        model4_min_strategy_probability=0.60,
        model4_strong_strategy_probability=0.80,
        model4_min_active_strategies=1,
        model4_max_active_strategies=9,
        model4_require_strong_strategy=True,
        model4_use_as_direction_vote=False,
        model4_require_strategy_confirmation=True,

        # Risk
        stop_loss_atr_mult=2.5,
        take_profit_atr_mult=5.0,

        max_daily_loss_pct=0.03,
        max_drawdown_pct=0.25,

        use_trailing_stop=True,

        # Kelly
        use_kelly_sizing=True,
        kelly_fraction=0.35,

        expected_win_rate=0.58,
        avg_win_loss_ratio=3.0,

        # AI Profit Shield
        enable_profit_shield=True,
        tier1_breakeven_trigger_pct=5.0,
        tier1_fee_buffer_pct=2.5,
        tier2_lock_trigger_pct=12.0,
        tier2_profit_lock_pct=6.0,
        tier3_trail_trigger_pct=20.0,
        tier3_trail_distance_pct=6.0,
        enable_smart_recovery=True,
        recovery_shallow_dip_max_loss_pct=3.5,
        recovery_extension_hours=24,
        max_recovery_capped_hours=216,

        # Execution
        allow_multiple_positions=False,
        auto_compound=False,

        max_daily_trades=2,
        min_time_between_trades=86400,
    )


# ============================================================================
# PROFILE REGISTRY
# ============================================================================

PROFILES = {
    TradingStyle.SCALPER.value: get_profile_scalper,
    TradingStyle.DAY_TRADER.value: get_profile_day_trader,
    TradingStyle.SWING.value: get_profile_swing,
    TradingStyle.POSITION.value: get_profile_position,
    TradingStyle.TEST.value: get_profile_test,
}


# ============================================================================
# PROFILE LOADER
# ============================================================================

def get_profile(
    name: str,
) -> TradingProfile:
    """
    Return a fresh TradingProfile instance.

    Unknown profile names intentionally fall back to day_trader.
    """

    normalized_name = (
        str(name)
        .strip()
        .lower()
    )

    factory = PROFILES.get(
        normalized_name,
        get_profile_day_trader,
    )

    return factory()


# ============================================================================
# PROFILE VALIDATION API
# ============================================================================

def validate_profile(
    profile: TradingProfile,
) -> Dict[str, Any]:
    """
    Validate an existing TradingProfile.

    Returns a structured validation result.
    """

    errors = profile.validate(
        raise_on_error=False
    )

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "profile": profile.trading_style.value,
        "risk_tolerance": profile.risk_tolerance.value,
        "signal_timeframe": profile.signal_timeframe.value,
    }


# ============================================================================
# PROFILE LISTING
# ============================================================================

def available_profiles() -> List[str]:
    """Return available profile names."""

    return list(PROFILES.keys())


# ============================================================================
# DEBUG / SELF-TEST
# ============================================================================

def _run_self_test() -> None:
    """Run basic profile integrity checks."""

    print("=" * 78)
    print("SMARTCRYPTO AI - TRADING PROFILE SELF TEST")
    print("=" * 78)

    for profile_name in available_profiles():

        profile = get_profile(profile_name)

        validation = validate_profile(profile)

        print()
        print(f"PROFILE: {profile_name.upper()}")
        print("-" * 78)

        print(
            f"Valid                    : "
            f"{validation['valid']}"
        )

        print(
            f"Risk                     : "
            f"{profile.risk_tolerance.value}"
        )

        print(
            f"Timeframe                : "
            f"{profile.signal_timeframe.value}"
        )

        print(
            f"Position Size            : "
            f"{profile.position_size_pct:.2%}"
        )

        print(
            f"Max Positions            : "
            f"{profile.max_total_positions}"
        )

        print(
            f"Max Holding              : "
            f"{profile.max_holding_hours}h"
        )

        print(
            f"Model 4                  : "
            f"{profile.use_model4_strategy_detector}"
        )

        print(
            f"Model 4 Min Probability  : "
            f"{profile.model4_min_strategy_probability:.2%}"
        )

        print(
            f"Model 4 Strong Probability: "
            f"{profile.model4_strong_strategy_probability:.2%}"
        )

        print(
            f"Model 4 Confirmation     : "
            f"{profile.model4_require_strategy_confirmation}"
        )

        print(
            f"Model 4 Direction Vote   : "
            f"{profile.model4_use_as_direction_vote}"
        )

        if validation["errors"]:

            print("Errors:")

            for error in validation["errors"]:
                print(f"  - {error}")

    print()
    print("=" * 78)
    print("PROFILE SELF TEST COMPLETE")
    print("=" * 78)


# ============================================================================
# MODULE ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    _run_self_test()