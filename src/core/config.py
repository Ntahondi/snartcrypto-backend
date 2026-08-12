"""
Configuration management for SmartCrypto - Unified settings
Loads from .env (primary) and config.yaml (fallback)
"""

import os
import yaml
from typing import List, Optional
from functools import lru_cache

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PYDANTIC IMPORTS (Compatible with v1 and v2)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
try:
    from pydantic import BaseSettings, Field
except ImportError:
    try:
        from pydantic.v1 import BaseSettings, Field
    except ImportError:
        from pydantic import Field
        from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings - Unified configuration.
    Reads from .env first, falls back to config.yaml.
    """
    ENVIRONMENT: str = "production"

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # API SETTINGS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    APP_NAME: str = "SmartCrypto"
    APP_VERSION: str = "3.0.0"
    API_PREFIX: str = "/api/v1"
    DEBUG: bool = Field(False, env="DEBUG")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # BINANCE SETTINGS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    BINANCE_API_BASE: str = Field("https://api.binance.com", env="BINANCE_API_BASE")
    BINANCE_WS_BASE: str = Field("wss://stream.binance.com:9443/ws", env="BINANCE_WS_BASE")
    BINANCE_RATE_LIMIT: int = Field(1200, env="BINANCE_RATE_LIMIT")
    EXCHANGE_TYPE: str = Field("future", env="EXCHANGE_TYPE")
    
    # Live Real Trading & Testnet
    ENABLE_REAL_TRADING: bool = Field(False, env="ENABLE_REAL_TRADING")
    USE_TESTNET: bool = Field(False, env="USE_TESTNET")
    BINANCE_API_KEY: str = Field("", env="BINANCE_API_KEY")
    BINANCE_API_SECRET: str = Field("", env="BINANCE_API_SECRET")
    DEFAULT_LEVERAGE: int = Field(3, env="DEFAULT_LEVERAGE")
    MARGIN_TYPE: str = Field("ISOLATED", env="MARGIN_TYPE")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TELEGRAM VIP CHANNEL & BOT SETTINGS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ENABLE_TELEGRAM: bool = Field(False, env="ENABLE_TELEGRAM")
    TELEGRAM_BOT_TOKEN: str = Field("", env="TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHANNEL_ID: str = Field("", env="TELEGRAM_CHANNEL_ID")
    TELEGRAM_ADMIN_CHAT_ID: str = Field("", env="TELEGRAM_ADMIN_CHAT_ID")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # DATA SETTINGS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    SYMBOLS: List[str] = Field(
        default=["BTCUSDT", "ETHUSDT", "ADAUSDT", "LINKUSDT", "SOLUSDT", "DOTUSDT"],
        env="SYMBOLS"
    )
    DEFAULT_INTERVAL: str = Field("1h", env="DEFAULT_INTERVAL")
    DEFAULT_LIMIT: int = Field(500, env="DEFAULT_LIMIT")
    MAX_HISTORICAL_DATA: int = Field(1000, env="MAX_HISTORICAL_DATA")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 3-TIER ENSEMBLE MODEL PATHS 🏆
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    MODEL_REGRESSION_PATH: str = Field("smartcrypto_ai_models/continuous_regression_ai.keras", env="MODEL_REGRESSION_PATH")
    MODEL_SMART_PATH: str = Field("models/smart_trader_ai_final.keras", env="MODEL_SMART_PATH")
    MODEL_GPT_PATH: str = Field("smartcrypto_ai_models/market_gpt_world_model.keras", env="MODEL_GPT_PATH")
    
    SCALER_REGRESSION_PATH: str = Field("smartcrypto_ai_models/unconstrained_scaler.joblib", env="SCALER_REGRESSION_PATH")
    FEATURES_REGRESSION_PATH: str = Field("smartcrypto_ai_models/unconstrained_features.joblib", env="FEATURES_REGRESSION_PATH")
    
    # Legacy Paths
    MODEL_PATH: str = Field("models/smart_trader_ai_final.keras", env="MODEL_PATH")
    SCALER_PATH: str = Field("models/robust_scaler.joblib", env="SCALER_PATH")
    POWER_TRANSFORMER_PATH: str = Field("models/power_transformer.joblib", env="POWER_TRANSFORMER_PATH")
    FEATURE_COLUMNS_PATH: str = Field("models/feature_columns.joblib", env="FEATURE_COLUMNS_PATH")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # SIGNAL QUALITY FILTERS ✅
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    MIN_EXPECTED_RETURN_THRESHOLD: float = Field(0.007, env="MIN_EXPECTED_RETURN_THRESHOLD")  # 1.0% High-Conviction Move
    PERFORMANCE_THRESHOLD: float = Field(0.55, env="PERFORMANCE_THRESHOLD")
    CONFIDENCE_THRESHOLD: float = Field(0.50, env="CONFIDENCE_THRESHOLD")   
    MIN_CONFIDENCE: float = Field(0.40, env="MIN_CONFIDENCE")               
    MIN_SIGNAL_STRENGTH: float = Field(0.40, env="MIN_SIGNAL_STRENGTH")     
    MAX_POSITION_SIZE: float = Field(0.15, env="MAX_POSITION_SIZE")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TRADING & RISK SETTINGS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    INITIAL_CAPITAL: float = Field(10000.0, env="INITIAL_CAPITAL")
    COMMISSION_RATE: float = Field(0.001, env="COMMISSION_RATE")
    RISK_TOLERANCE: str = Field("MODERATE", env="RISK_TOLERANCE")
    
    STOP_LOSS_PCT: float = Field(0.02, env="STOP_LOSS_PCT")
    TAKE_PROFIT_PCT: float = Field(0.04, env="TAKE_PROFIT_PCT")
    
    # ATR-Based (Primary)
    ATR_MULTIPLIER_SL: float = Field(1.5, env="ATR_MULTIPLIER_SL")
    ATR_MULTIPLIER_TP: float = Field(3.0, env="ATR_MULTIPLIER_TP")
    MAX_HOLDING_HOURS: int = Field(8, env="MAX_HOLDING_HOURS")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TRAINING SETTINGS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    TRAINING_MODE: str = Field("fine_tune", env="TRAINING_MODE")
    
    FINE_TUNE_DAYS: int = Field(60, env="FINE_TUNE_DAYS")
    FINE_TUNE_LEARNING_RATE: float = Field(0.0001, env="FINE_TUNE_LEARNING_RATE")
    FINE_TUNE_EPOCHS: int = Field(5, env="FINE_TUNE_EPOCHS")
    FINE_TUNE_MIN_IMPROVEMENT: float = Field(0.015, env="FINE_TUNE_MIN_IMPROVEMENT")
    
    FULL_RETRAIN_DAYS: int = Field(730, env="FULL_RETRAIN_DAYS")
    FULL_RETRAIN_INTERVAL_MONTHS: int = Field(6, env="FULL_RETRAIN_INTERVAL_MONTHS")
    
    AUTO_RETRAIN: bool = Field(True, env="AUTO_RETRAIN")
    RETRAIN_INTERVAL_HOURS: int = Field(24, env="RETRAIN_INTERVAL_HOURS")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PATHS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    SIGNAL_DB_PATH: str = Field("signal_history/signals.db", env="SIGNAL_DB_PATH")
    PORTFOLIO_DB_PATH: str = Field("portfolio.db", env="PORTFOLIO_DB_PATH")
    TRADE_HISTORY_PATH: str = Field("positions/trade_history.jsonl", env="TRADE_HISTORY_PATH")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # LOGGING
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    LOG_LEVEL: str = Field("INFO", env="LOG_LEVEL")
    LOG_FILE: str = Field("logs/trading.log", env="LOG_FILE")
    
    class Config:
        env_file = ".env"
        extra = "ignore"
        case_sensitive = False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG.YAML LOADER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_yaml_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file safely with UTF-8 encoding"""
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        return {}
    except Exception as e:
        print(f"⚠️ Could not load config.yaml: {e}")
        return {}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# UNIFIED SETTINGS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class UnifiedSettings:
    """
    Unified settings combining .env and config.yaml.
    .env takes precedence over config.yaml.
    """
    
    def __init__(self):
        self.env = Settings()
        self.yaml = load_yaml_config()
        self._merge()
    
    def _merge(self):
        """Safely merge config parameters without throwing Pydantic Setattr ValueErrors"""
        if not self.yaml:
            return

        config_dict = self._flatten_dict(self.yaml)
        for key, value in config_dict.items():
            try:
                if hasattr(self.env, key):
                    setattr(self.env, key, value)
            except (ValueError, AttributeError):
                pass
    
    def _flatten_dict(self, d: dict, parent_key: str = "") -> dict:
        """Flatten nested dicts (e.g., model.min_confidence)"""
        items = {}
        if not isinstance(d, dict):
            return items

        for k, v in d.items():
            new_key = f"{parent_key}_{k}".upper() if parent_key else k.upper()
            if isinstance(v, dict):
                items.update(self._flatten_dict(v, new_key))
            else:
                items[new_key] = v
        return items
    
    def __getattr__(self, name):
        return getattr(self.env, name)


_settings: Optional[UnifiedSettings] = None


@lru_cache(maxsize=1)
def get_settings() -> UnifiedSettings:
    """Get application settings (singleton)"""
    global _settings
    if _settings is None:
        _settings = UnifiedSettings()
    return _settings


def get_settings_legacy() -> Settings:
    return get_settings().env


def print_settings():
    """Print current settings for debugging"""
    settings = get_settings()
    print("=" * 60)
    print("📊 SMARTCRYPTO AI - CURRENT SETTINGS")
    print("=" * 60)
    print(f"  Mode: {settings.TRAINING_MODE}")
    print(f"  Symbols: {settings.SYMBOLS}")
    print(f"  Min Expected Return: {settings.MIN_EXPECTED_RETURN_THRESHOLD:.1%}")
    print(f"  ATR SL: {settings.ATR_MULTIPLIER_SL}x, TP: {settings.ATR_MULTIPLIER_TP}x")
    print("=" * 60)


if __name__ == "__main__":
    print_settings()