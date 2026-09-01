"""
SmartCrypto AI - Unified Configuration
======================================

Central configuration layer for SmartCrypto AI v3.1.x.

Responsibilities
----------------
- Load environment variables from .env.
- Provide strongly typed application settings.
- Support optional config.yaml values.
- Keep secrets out of source-code defaults.
- Provide centralized paths for models, data, databases and logs.
- Validate critical configuration at startup.
- Provide one cached settings instance.

Configuration precedence
------------------------
1. Environment variables / .env
2. config.yaml
3. Safe application defaults

IMPORTANT
---------
Never place real exchange API keys, API secrets, Telegram tokens,
or other credentials directly in this file.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# ============================================================================
# PYDANTIC COMPATIBILITY
# ============================================================================

try:
    # Pydantic v2 with pydantic-settings installed.
    from pydantic_settings import BaseSettings, SettingsConfigDict
    PYDANTIC_V2 = True
except ImportError:
    try:
        # Pydantic v1.
        from pydantic import BaseSettings
        PYDANTIC_V2 = False
    except ImportError as exc:
        raise ImportError(
            "Pydantic settings support is required. "
            "Install pydantic-settings for Pydantic v2."
        ) from exc

from pydantic import Field


# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

ENV_FILE = PROJECT_ROOT / ".env"
YAML_CONFIG_FILE = PROJECT_ROOT / "config.yaml"


# ============================================================================
# YAML LOADER
# ============================================================================

def load_yaml_config(
    config_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """
    Load optional YAML configuration.

    YAML is treated as a secondary configuration source.

    Parameters
    ----------
    config_path:
        Optional path to config.yaml.

    Returns
    -------
    dict
        Parsed YAML configuration or an empty dictionary.
    """

    path = Path(config_path) if config_path else YAML_CONFIG_FILE

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.exists():
        return {}

    try:
        with path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file)

        if data is None:
            return {}

        if not isinstance(data, dict):
            raise ValueError(
                "config.yaml root must contain a mapping/object."
            )

        return data

    except yaml.YAMLError as exc:
        raise RuntimeError(
            f"Invalid YAML configuration: {path}"
        ) from exc

    except OSError as exc:
        raise RuntimeError(
            f"Unable to read configuration file: {path}"
        ) from exc


# ============================================================================
# PYDANTIC SETTINGS
# ============================================================================

class Settings(BaseSettings):
    """
    Strongly typed SmartCrypto AI configuration.

    Environment variables are the primary runtime configuration source.
    """

    # ------------------------------------------------------------------------
    # APPLICATION
    # ------------------------------------------------------------------------

    ENVIRONMENT: str = Field(
        default="production",
        description="Application environment.",
    )

    APP_NAME: str = Field(
        default="SmartCrypto",
        description="Application name.",
    )

    APP_VERSION: str = Field(
        default="3.1.0",
        description="Application version.",
    )

    APP_API_KEY: str = Field(
        default="",
        description="Internal API authentication key.",
    )

    JWT_SECRET_KEY: str = Field(
        default="",
        description="Secret key for JWT HS256 token signing. Must be set via .env in production.",
    )

    JWT_ALGORITHM: str = Field(
        default="HS256",
        description="JWT cryptographic algorithm.",
    )

    ACCESS_TOKEN_EXPIRE_DAYS: int = Field(
        default=30,
        description="JWT access token lifetime in days.",
    )

    REDIS_URL: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL for high-performance caching.",
    )

    CACHE_TTL_SECONDS: int = Field(
        default=10,
        description="Default caching time-to-live for API responses.",
    )

    RATE_LIMIT_PER_MINUTE: int = Field(
        default=120,
        description="General API request rate limit per minute.",
    )

    RATE_LIMIT_AUTH_PER_MINUTE: int = Field(
        default=10,
        description="Strict auth request rate limit per minute to prevent brute-force.",
    )

    SNAILGUARD_ENABLED: bool = Field(
        default=True,
        description="Enable SnailGuard AI WAF and SQL/XSS threat shielding.",
    )

    SNAILGUARD_ECONOMIC_WARFARE: bool = Field(
        default=True,
        description="Enable SnailGuard economic warfare penalties for malicious bots.",
    )

    API_PREFIX: str = Field(
        default="/api/v1",
        description="REST API prefix.",
    )

    DEBUG: bool = Field(
        default=False,
        description="Enable debug mode.",
    )

    # ------------------------------------------------------------------------
    # EXCHANGE
    # ------------------------------------------------------------------------

    EXCHANGE_TYPE: str = Field(
        default="future",
        description="Trading market type.",
    )

    ENABLE_REAL_TRADING: bool = Field(
        default=False,
        description="Master switch for real-money trading.",
    )

    USE_TESTNET: bool = Field(
        default=True,
        description="Use exchange testnet/sandbox where supported.",
    )

    BINANCE_API_BASE: str = Field(
        default="https://api.binance.com",
        description="Binance REST API base URL.",
    )

    BINANCE_WS_BASE: str = Field(
        default="wss://stream.binance.com:9443/ws",
        description="Binance WebSocket base URL.",
    )

    BINANCE_RATE_LIMIT: int = Field(
        default=1200,
        ge=1,
        description="Configured Binance request-weight limit.",
    )

    BINANCE_API_KEY: str = Field(
        default="",
        description="Binance API key.",
    )

    BINANCE_API_SECRET: str = Field(
        default="",
        description="Binance API secret.",
    )

    ENABLE_BINANCE: bool = Field(
        default=True,
        description="Enable Binance exchange execution.",
    )

    BINANCE_USE_TESTNET: Optional[bool] = Field(
        default=None,
        description="Override testnet mode specifically for Binance.",
    )

    ENABLE_BYBIT: bool = Field(
        default=True,
        description="Enable Bybit exchange execution.",
    )

    BYBIT_API_KEY: str = Field(
        default="",
        description="Bybit API key.",
    )

    BYBIT_API_SECRET: str = Field(
        default="",
        description="Bybit API secret.",
    )

    BYBIT_API_BASE: str = Field(
        default="https://api.bybit.com",
        description="Bybit REST API base URL.",
    )

    BYBIT_USE_TESTNET: Optional[bool] = Field(
        default=None,
        description="Override testnet mode specifically for Bybit.",
    )

    EXCHANGE_PROXY_URL: Optional[str] = Field(
        default=None,
        description="Optional HTTP/SOCKS proxy for exchange API requests.",
    )

    DEFAULT_LEVERAGE: int = Field(
        default=3,
        ge=1,
        description="Default futures leverage.",
    )

    MARGIN_TYPE: str = Field(
        default="ISOLATED",
        description="Default futures margin mode.",
    )

    # ------------------------------------------------------------------------
    # MONETIZATION & CRYPTO PAYMENT WALLETS
    # ------------------------------------------------------------------------

    WALLET_TRC20_ADDRESS: str = Field(
        default="TYDzsYUb4r8ZJ3pA4rXvWzR8G9cK8v1a2b",
        description="Your personal USDT (TRC20) deposit wallet address.",
    )

    WALLET_BSC_ADDRESS: str = Field(
        default="0x742d35Cc6634C0532925a3b844Bc454e4438f44e",
        description="Your personal USDT / USDC / BNB (BSC BEP20) wallet address.",
    )

    WALLET_POLYGON_ADDRESS: str = Field(
        default="0x742d35Cc6634C0532925a3b844Bc454e4438f44e",
        description="Your personal Polygon (MATIC) wallet address.",
    )

    WALLET_ERC20_ADDRESS: str = Field(
        default="0x742d35Cc6634C0532925a3b844Bc454e4438f44e",
        description="Your personal Ethereum (ERC20) wallet address.",
    )

    # ------------------------------------------------------------------------
    # TELEGRAM
    # ------------------------------------------------------------------------

    ENABLE_TELEGRAM: bool = Field(
        default=False,
    )

    TELEGRAM_BOT_TOKEN: str = Field(
        default="",
    )

    TELEGRAM_CHANNEL_ID: str = Field(
        default="",
    )

    TELEGRAM_ADMIN_CHAT_ID: str = Field(
        default="",
    )

    TELEGRAM_API_BASE: str = Field(
        default="https://api.telegram.org",
    )

    TELEGRAM_PROXY_URL: Optional[str] = Field(
        default=None,
    )

    # ------------------------------------------------------------------------
    # MARKET DATA
    # ------------------------------------------------------------------------

    SYMBOLS: List[str] = Field(
        default_factory=lambda: [
            "BTCUSDT",
            "ETHUSDT",
            "ADAUSDT",
            "LINKUSDT",
            "SOLUSDT",
            "DOTUSDT",
        ],
    )

    DEFAULT_INTERVAL: str = Field(
        default="1h",
    )

    DEFAULT_LIMIT: int = Field(
        default=500,
        ge=1,
    )

    MAX_HISTORICAL_DATA: int = Field(
        default=1000,
        ge=1,
    )

    # ------------------------------------------------------------------------
    # MODEL 1 - CONTINUOUS RETURN REGRESSION
    # ------------------------------------------------------------------------

    MODEL_REGRESSION_PATH: str = Field(
        default="smartcrypto_ai_models/continuous_regression_ai.keras",
    )

    SCALER_REGRESSION_PATH: str = Field(
        default="smartcrypto_ai_models/unconstrained_scaler.joblib",
    )

    FEATURES_REGRESSION_PATH: str = Field(
        default="smartcrypto_ai_models/unconstrained_features.joblib",
    )

    # ------------------------------------------------------------------------
    # MODEL 2 - SMART TRADER
    # ------------------------------------------------------------------------

    MODEL_SMART_PATH: str = Field(
        default="models/smart_trader_ai_final.keras",
    )

    SCALER_SMART_PATH: str = Field(
        default="models/robust_scaler.joblib",
    )

    TRANSFORMER_SMART_PATH: str = Field(
        default="models/power_transformer.joblib",
    )

    FEATURES_SMART_PATH: str = Field(
        default="models/feature_columns.joblib",
    )

    # ------------------------------------------------------------------------
    # MODEL 3 - MARKET GPT
    # ------------------------------------------------------------------------

    MODEL_GPT_PATH: str = Field(
        default="smartcrypto_ai_models/market_gpt_world_model.keras",
    )

    # ------------------------------------------------------------------------
    # MODEL 4 - STRATEGY DETECTOR
    # ------------------------------------------------------------------------

    MODEL4_ENABLED: bool = Field(
        default=True,
    )

    MODEL4_PACKAGE_PATH: str = Field(
        default=(
            "data/model4/"
            "MODEL4_STRATEGY_DETECTOR_V1/"
            "MODEL4_STRATEGY_DETECTOR_V1_PACKAGE.joblib"
        ),
    )

    MODEL4_MODELS_DIR: str = Field(
        default=(
            "data/model4/"
            "MODEL4_STRATEGY_DETECTOR_V1/"
            "models"
        ),
    )

    MODEL4_FEATURE_FILE: str = Field(
        default="data/model4/MODEL4_FEATURES_V3_1.parquet",
    )

    MODEL4_LABEL_FILE: str = Field(
        default=(
            "data/model4/"
            "MODEL4_STRATEGY_LABELS_V2/"
            "MODEL4_STRATEGY_LABELS_V2.parquet"
        ),
    )

    MODEL4_CONTROLLED_TEST_DIR: str = Field(
        default=(
            "data/model4/"
            "MODEL4_STRATEGY_DETECTOR_V1/"
            "CONTROLLED_TEST_V1"
        ),
    )

    # ------------------------------------------------------------------------
    # MODEL 4 - INDIVIDUAL DETECTORS
    # ------------------------------------------------------------------------

    MODEL4_MOMENTUM_REVERSAL_PATH: str = Field(
        default=(
            "data/model4/"
            "MODEL4_STRATEGY_DETECTOR_V1/models/"
            "momentum_reversal_detector.joblib"
        ),
    )

    MODEL4_MA_CROSSOVER_PATH: str = Field(
        default=(
            "data/model4/"
            "MODEL4_STRATEGY_DETECTOR_V1/models/"
            "ma_crossover_detector.joblib"
        ),
    )

    MODEL4_HEIKIN_ASHI_PATH: str = Field(
        default=(
            "data/model4/"
            "MODEL4_STRATEGY_DETECTOR_V1/models/"
            "heikin_ashi_detector.joblib"
        ),
    )

    MODEL4_SWING_TRADING_PATH: str = Field(
        default=(
            "data/model4/"
            "MODEL4_STRATEGY_DETECTOR_V1/models/"
            "swing_trading_detector.joblib"
        ),
    )

    MODEL4_CANDLESTICK_PATH: str = Field(
        default=(
            "data/model4/"
            "MODEL4_STRATEGY_DETECTOR_V1/models/"
            "candlestick_detector.joblib"
        ),
    )

    MODEL4_ROLE_REVERSAL_PATH: str = Field(
        default=(
            "data/model4/"
            "MODEL4_STRATEGY_DETECTOR_V1/models/"
            "role_reversal_detector.joblib"
        ),
    )

    MODEL4_BOLLINGER_SQUEEZE_PATH: str = Field(
        default=(
            "data/model4/"
            "MODEL4_STRATEGY_DETECTOR_V1/models/"
            "bollinger_squeeze_detector.joblib"
        ),
    )

    MODEL4_NARROW_RANGE_PATH: str = Field(
        default=(
            "data/model4/"
            "MODEL4_STRATEGY_DETECTOR_V1/models/"
            "narrow_range_detector.joblib"
        ),
    )

    MODEL4_RSI2_PATH: str = Field(
        default=(
            "data/model4/"
            "MODEL4_STRATEGY_DETECTOR_V1/models/"
            "rsi_2_detector.joblib"
        ),
    )

    MODEL4_STRATEGIES: List[str] = Field(
        default_factory=lambda: [
            "momentum_reversal",
            "ma_crossover",
            "heikin_ashi",
            "swing_trading",
            "candlestick",
            "role_reversal",
            "bollinger_squeeze",
            "narrow_range",
            "rsi_2",
        ],
    )

    # ------------------------------------------------------------------------
    # MODEL 4 - THRESHOLDS
    # ------------------------------------------------------------------------

    MODEL4_THRESHOLD_MOMENTUM_REVERSAL: float = Field(
        default=0.47,
        ge=0.0,
        le=1.0,
    )

    MODEL4_THRESHOLD_MA_CROSSOVER: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
    )

    MODEL4_THRESHOLD_HEIKIN_ASHI: float = Field(
        default=0.36,
        ge=0.0,
        le=1.0,
    )

    MODEL4_THRESHOLD_SWING_TRADING: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
    )

    MODEL4_THRESHOLD_CANDLESTICK: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
    )

    MODEL4_THRESHOLD_ROLE_REVERSAL: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
    )

    MODEL4_THRESHOLD_BOLLINGER_SQUEEZE: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
    )

    MODEL4_THRESHOLD_NARROW_RANGE: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
    )

    MODEL4_THRESHOLD_RSI2: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
    )

    MODEL4_MIN_ACTIVE_STRATEGIES: int = Field(
        default=1,
        ge=0,
    )

    MODEL4_MIN_STRATEGY_PROBABILITY: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
    )

    MODEL4_STRONG_STRATEGY_PROBABILITY: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
    )

    MODEL4_REQUIRE_STRONG_STRATEGY: bool = Field(
        default=False,
    )

    MODEL4_USE_AS_DIRECTION_VOTE: bool = Field(
        default=False,
    )

    # ------------------------------------------------------------------------
    # LEGACY MODEL PATHS
    # ------------------------------------------------------------------------
    # Kept for backwards compatibility with older modules.
    # New code should use the explicit MODEL_SMART_* settings.

    MODEL_PATH: str = Field(
        default="models/smart_trader_ai_final.keras",
    )

    SCALER_PATH: str = Field(
        default="models/robust_scaler.joblib",
    )

    POWER_TRANSFORMER_PATH: str = Field(
        default="models/power_transformer.joblib",
    )

    FEATURE_COLUMNS_PATH: str = Field(
        default="models/feature_columns.joblib",
    )

    # ------------------------------------------------------------------------
    # SIGNAL QUALITY
    # ------------------------------------------------------------------------

    MIN_EXPECTED_RETURN_THRESHOLD: float = Field(
        default=0.005,
        ge=0.0,
    )

    PERFORMANCE_THRESHOLD: float = Field(
        default=0.55,
        ge=0.0,
        le=1.0,
    )

    CONFIDENCE_THRESHOLD: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
    )

    MIN_CONFIDENCE: float = Field(
        default=0.40,
        ge=0.0,
        le=1.0,
    )

    MIN_SIGNAL_STRENGTH: float = Field(
        default=0.40,
        ge=0.0,
        le=1.0,
    )

    MAX_POSITION_SIZE: float = Field(
        default=0.15,
        gt=0.0,
        le=1.0,
    )

    # ------------------------------------------------------------------------
    # TRADING / RISK
    # ------------------------------------------------------------------------

    TRADING_PROFILE: str = Field(
        default="test",
    )

    INITIAL_CAPITAL: float = Field(
        default=10000.0,
        gt=0.0,
    )

    COMMISSION_RATE: float = Field(
        default=0.001,
        ge=0.0,
    )

    RISK_TOLERANCE: str = Field(
        default="MODERATE",
    )

    STOP_LOSS_PCT: float = Field(
        default=0.02,
        gt=0.0,
        lt=1.0,
    )

    TAKE_PROFIT_PCT: float = Field(
        default=0.04,
        gt=0.0,
        lt=1.0,
    )

    ATR_MULTIPLIER_SL: float = Field(
        default=1.5,
        gt=0.0,
    )

    ATR_MULTIPLIER_TP: float = Field(
        default=3.0,
        gt=0.0,
    )

    MAX_HOLDING_HOURS: int = Field(
        default=8,
        gt=0,
    )

    # ------------------------------------------------------------------------
    # TRAINING
    # ------------------------------------------------------------------------

    TRAINING_MODE: str = Field(
        default="fine_tune",
    )

    FINE_TUNE_DAYS: int = Field(
        default=60,
        gt=0,
    )

    FINE_TUNE_LEARNING_RATE: float = Field(
        default=0.0001,
        gt=0.0,
    )

    FINE_TUNE_EPOCHS: int = Field(
        default=5,
        gt=0,
    )

    FINE_TUNE_MIN_IMPROVEMENT: float = Field(
        default=0.015,
        ge=0.0,
    )

    FULL_RETRAIN_DAYS: int = Field(
        default=730,
        gt=0,
    )

    FULL_RETRAIN_INTERVAL_MONTHS: int = Field(
        default=6,
        gt=0,
    )

    AUTO_RETRAIN: bool = Field(
        default=True,
    )

    RETRAIN_INTERVAL_HOURS: int = Field(
        default=24,
        gt=0,
    )

    # ------------------------------------------------------------------------
    # DATABASE / FILES
    # ------------------------------------------------------------------------

    SIGNAL_DB_PATH: str = Field(
        default="signal_history/signals.db",
    )

    PORTFOLIO_DB_PATH: str = Field(
        default="portfolio.db",
    )

    TRADE_HISTORY_PATH: str = Field(
        default="positions/trade_history.jsonl",
    )

    # ------------------------------------------------------------------------
    # LOGGING
    # ------------------------------------------------------------------------

    LOG_LEVEL: str = Field(
        default="INFO",
    )

    LOG_FILE: str = Field(
        default="logs/trading.log",
    )

    # ------------------------------------------------------------------------
    # PYDANTIC V2 CONFIGURATION
    # ------------------------------------------------------------------------

    if PYDANTIC_V2:
        model_config = SettingsConfigDict(
            env_file=str(ENV_FILE),
            env_file_encoding="utf-8",
            extra="ignore",
            case_sensitive=False,
        )

    else:

        class Config:
            env_file = str(ENV_FILE)
            env_file_encoding = "utf-8"
            extra = "ignore"
            case_sensitive = False


# ============================================================================
# YAML FLATTENING
# ============================================================================

def _flatten_dict(
    data: Dict[str, Any],
    parent_key: str = "",
) -> Dict[str, Any]:
    """
    Flatten nested YAML configuration.

    Example
    -------
    {
        "binance": {
            "api_key": "..."
        }
    }

    becomes:

    {
        "BINANCE_API_KEY": "..."
    }
    """

    flattened: Dict[str, Any] = {}

    for key, value in data.items():

        current_key = (
            f"{parent_key}_{key}".upper()
            if parent_key
            else str(key).upper()
        )

        if isinstance(value, dict):
            flattened.update(
                _flatten_dict(
                    value,
                    current_key,
                )
            )
        else:
            flattened[current_key] = value

    return flattened


# ============================================================================
# UNIFIED SETTINGS
# ============================================================================

class UnifiedSettings:
    """
    Unified SmartCrypto configuration interface.

    The object exposes the typed Settings object while allowing optional
    YAML fallback values for fields that were not explicitly supplied
    through the environment.

    Environment variables always have priority.
    """

    def __init__(self) -> None:
        self.env = Settings()
        self.yaml = load_yaml_config()
        self._apply_yaml_fallbacks()

    # ------------------------------------------------------------------------
    # YAML FALLBACK
    # ------------------------------------------------------------------------

    def _apply_yaml_fallbacks(self) -> None:
        """
        Apply YAML values only where an environment value was not supplied.

        This is intentionally different from blindly overwriting Settings
        after initialization.

        That prevents config.yaml from silently overriding .env.
        """

        if not self.yaml:
            return

        flattened = _flatten_dict(self.yaml)

        if not flattened:
            return

        # Determine which environment variables were explicitly provided.
        environment_keys = {
            key.upper()
            for key in os.environ.keys()
        }

        for key, value in flattened.items():

            if key in environment_keys:
                continue

            if not hasattr(self.env, key):
                continue

            try:
                setattr(self.env, key, value)
            except (AttributeError, TypeError, ValueError):
                # Invalid YAML values should not destroy startup.
                # Validation is handled separately.
                continue

    # ------------------------------------------------------------------------
    # ATTRIBUTE ACCESS
    # ------------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        """
        Delegate unknown attributes to the typed Settings object.
        """

        try:
            return getattr(self.env, name)
        except AttributeError as exc:
            raise AttributeError(
                f"Unknown SmartCrypto setting: {name}"
            ) from exc

    # ------------------------------------------------------------------------
    # PATH HELPERS
    # ------------------------------------------------------------------------

    def resolve_path(self, configured_path: str) -> Path:
        """
        Resolve a configured relative path against the project root.
        """

        path = Path(configured_path)

        if path.is_absolute():
            return path

        return PROJECT_ROOT / path

    def model_paths(self) -> Dict[str, Path]:
        """
        Return all primary model-related paths.
        """

        return {
            "regression_model": self.resolve_path(
                self.MODEL_REGRESSION_PATH
            ),
            "regression_scaler": self.resolve_path(
                self.SCALER_REGRESSION_PATH
            ),
            "regression_features": self.resolve_path(
                self.FEATURES_REGRESSION_PATH
            ),
            "smart_model": self.resolve_path(
                self.MODEL_SMART_PATH
            ),
            "smart_scaler": self.resolve_path(
                self.SCALER_SMART_PATH
            ),
            "smart_transformer": self.resolve_path(
                self.TRANSFORMER_SMART_PATH
            ),
            "smart_features": self.resolve_path(
                self.FEATURES_SMART_PATH
            ),
            "market_gpt": self.resolve_path(
                self.MODEL_GPT_PATH
            ),
            "model4_package": self.resolve_path(
                self.MODEL4_PACKAGE_PATH
            ),
            "model4_models_dir": self.resolve_path(
                self.MODEL4_MODELS_DIR
            ),
        }

    # ------------------------------------------------------------------------
    # DIRECTORY INITIALIZATION
    # ------------------------------------------------------------------------

    def ensure_runtime_directories(self) -> None:
        """
        Create directories required for runtime output.

        Existing model/data directories are NOT created automatically.
        """

        runtime_paths = [
            self.resolve_path(self.SIGNAL_DB_PATH).parent,
            self.resolve_path(self.PORTFOLIO_DB_PATH).parent,
            self.resolve_path(self.TRADE_HISTORY_PATH).parent,
            self.resolve_path(self.LOG_FILE).parent,
        ]

        for path in runtime_paths:
            path.mkdir(
                parents=True,
                exist_ok=True,
            )


# ============================================================================
# SINGLETON
# ============================================================================

@lru_cache(maxsize=1)
def get_settings() -> UnifiedSettings:
    """
    Return the cached application settings instance.
    """

    return UnifiedSettings()


def get_settings_legacy() -> Settings:
    """
    Backwards-compatible access to the underlying Pydantic Settings object.
    """

    return get_settings().env


# ============================================================================
# VALIDATION
# ============================================================================

def validate_settings(
    settings: Optional[UnifiedSettings] = None,
) -> List[str]:
    """
    Validate cross-field configuration constraints.

    Returns
    -------
    list[str]
        Empty list means configuration is valid.
    """

    settings = settings or get_settings()

    errors: List[str] = []

    # ------------------------------------------------------------------------
    # REAL TRADING SAFETY
    # ------------------------------------------------------------------------

    if settings.ENABLE_REAL_TRADING:

        if not settings.BINANCE_API_KEY:
            errors.append(
                "ENABLE_REAL_TRADING=True but BINANCE_API_KEY is empty."
            )

        if not settings.BINANCE_API_SECRET:
            errors.append(
                "ENABLE_REAL_TRADING=True but BINANCE_API_SECRET is empty."
            )

        if settings.USE_TESTNET:
            errors.append(
                "ENABLE_REAL_TRADING=True while USE_TESTNET=True. "
                "Explicitly choose the intended execution environment."
            )

    # ------------------------------------------------------------------------
    # MARKET DATA
    # ------------------------------------------------------------------------

    if not settings.SYMBOLS:
        errors.append(
            "SYMBOLS cannot be empty."
        )

    if settings.DEFAULT_LIMIT > settings.MAX_HISTORICAL_DATA:
        errors.append(
            "DEFAULT_LIMIT cannot exceed MAX_HISTORICAL_DATA."
        )

    # ------------------------------------------------------------------------
    # MODEL 4
    # ------------------------------------------------------------------------

    if (
        settings.MODEL4_STRONG_STRATEGY_PROBABILITY
        < settings.MODEL4_MIN_STRATEGY_PROBABILITY
    ):
        errors.append(
            "MODEL4_STRONG_STRATEGY_PROBABILITY must be greater than "
            "or equal to MODEL4_MIN_STRATEGY_PROBABILITY."
        )

    if (
        settings.MODEL4_MIN_ACTIVE_STRATEGIES
        > len(settings.MODEL4_STRATEGIES)
    ):
        errors.append(
            "MODEL4_MIN_ACTIVE_STRATEGIES cannot exceed the number "
            "of configured Model 4 strategies."
        )

    # ------------------------------------------------------------------------
    # RISK
    # ------------------------------------------------------------------------

    if settings.ATR_MULTIPLIER_TP <= settings.ATR_MULTIPLIER_SL:
        errors.append(
            "ATR_MULTIPLIER_TP should normally be greater than "
            "ATR_MULTIPLIER_SL."
        )

    # ------------------------------------------------------------------------
    # LEVERAGE
    # ------------------------------------------------------------------------

    if settings.DEFAULT_LEVERAGE < 1:
        errors.append(
            "DEFAULT_LEVERAGE must be at least 1."
        )

    # ------------------------------------------------------------------------
    # ENUM-LIKE VALUES
    # ------------------------------------------------------------------------

    valid_environments = {
        "development",
        "testing",
        "staging",
        "production",
    }

    if settings.ENVIRONMENT.lower() not in valid_environments:
        errors.append(
            f"Unsupported ENVIRONMENT: {settings.ENVIRONMENT}"
        )

    valid_margin_types = {
        "ISOLATED",
        "CROSSED",
    }

    if settings.MARGIN_TYPE.upper() not in valid_margin_types:
        errors.append(
            f"Unsupported MARGIN_TYPE: {settings.MARGIN_TYPE}"
        )

    return errors


def assert_valid_settings() -> UnifiedSettings:
    """
    Validate configuration and raise a clear exception if invalid.
    """

    settings = get_settings()
    errors = validate_settings(settings)

    if errors:

        formatted = "\n".join(
            f"  - {error}"
            for error in errors
        )

        raise RuntimeError(
            "Invalid SmartCrypto configuration:\n"
            f"{formatted}"
        )

    return settings


# ============================================================================
# SAFE DEBUG OUTPUT
# ============================================================================

def print_settings() -> None:
    """
    Print non-secret runtime configuration.

    Secrets are intentionally never printed.
    """

    settings = get_settings()

    print("=" * 78)
    print("SMARTCRYPTO AI - CURRENT SETTINGS")
    print("=" * 78)

    print(f"Environment              : {settings.ENVIRONMENT}")
    print(f"Version                  : {settings.APP_VERSION}")
    print(f"Trading Profile          : {settings.TRADING_PROFILE}")
    print(f"Exchange Type             : {settings.EXCHANGE_TYPE}")
    print(f"Real Trading              : {settings.ENABLE_REAL_TRADING}")
    print(f"Testnet                   : {settings.USE_TESTNET}")
    print(f"Symbols                   : {settings.SYMBOLS}")
    print(f"Default Interval          : {settings.DEFAULT_INTERVAL}")

    print("\nAI MODELS")
    print("-" * 78)

    print(
        f"Regression               : "
        f"{settings.MODEL_REGRESSION_PATH}"
    )

    print(
        f"Smart Trader             : "
        f"{settings.MODEL_SMART_PATH}"
    )

    print(
        f"Market GPT               : "
        f"{settings.MODEL_GPT_PATH}"
    )

    print(
        f"Model 4 Package          : "
        f"{settings.MODEL4_PACKAGE_PATH}"
    )

    print("\nMODEL 4")
    print("-" * 78)

    print(
        f"Enabled                  : "
        f"{settings.MODEL4_ENABLED}"
    )

    print(
        f"Strategies               : "
        f"{len(settings.MODEL4_STRATEGIES)}"
    )

    print(
        f"Minimum Probability      : "
        f"{settings.MODEL4_MIN_STRATEGY_PROBABILITY:.2%}"
    )

    print(
        f"Strong Probability       : "
        f"{settings.MODEL4_STRONG_STRATEGY_PROBABILITY:.2%}"
    )

    print(
        f"Minimum Active Strategies: "
        f"{settings.MODEL4_MIN_ACTIVE_STRATEGIES}"
    )

    print(
        f"Direction Vote           : "
        f"{settings.MODEL4_USE_AS_DIRECTION_VOTE}"
    )

    print("\nSIGNAL FILTERS")
    print("-" * 78)

    print(
        f"Expected Return          : "
        f"{settings.MIN_EXPECTED_RETURN_THRESHOLD:.2%}"
    )

    print(
        f"Confidence Threshold     : "
        f"{settings.CONFIDENCE_THRESHOLD:.2%}"
    )

    print(
        f"Minimum Confidence       : "
        f"{settings.MIN_CONFIDENCE:.2%}"
    )

    print(
        f"Signal Strength          : "
        f"{settings.MIN_SIGNAL_STRENGTH:.2%}"
    )

    print("\nRISK")
    print("-" * 78)

    print(
        f"ATR Stop Loss            : "
        f"{settings.ATR_MULTIPLIER_SL:.2f}x"
    )

    print(
        f"ATR Take Profit          : "
        f"{settings.ATR_MULTIPLIER_TP:.2f}x"
    )

    print(
        f"Maximum Holding          : "
        f"{settings.MAX_HOLDING_HOURS} hours"
    )

    print(
        f"Maximum Position Size    : "
        f"{settings.MAX_POSITION_SIZE:.2%}"
    )

    print("=" * 78)


# ============================================================================
# MODULE ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    try:
        settings = assert_valid_settings()
        settings.ensure_runtime_directories()
        print_settings()

    except Exception as exc:
        print("=" * 78)
        print("SMARTCRYPTO CONFIGURATION ERROR")
        print("=" * 78)
        print(str(exc))
        raise