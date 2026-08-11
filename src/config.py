"""
Configuration management for SmartCrypto AI
"""
import os
from pathlib import Path
from dotenv import load_dotenv
import yaml

# Load environment variables
load_dotenv()

class Config:
    """Global configuration"""
    
    # Paths - Use absolute path from .env
    PROJECT_ROOT = Path(os.getenv('PROJECT_ROOT', Path(__file__).parent.parent))
    
    # If PROJECT_ROOT is relative, make it absolute
    if not PROJECT_ROOT.is_absolute():
        PROJECT_ROOT = Path.cwd() / PROJECT_ROOT
    
    DATA_PATH = PROJECT_ROOT / os.getenv('DATA_PATH', 'data/raw')
    PROCESSED_PATH = PROJECT_ROOT / os.getenv('PROCESSED_DATA_PATH', 'data/processed')
    MODELS_PATH = PROJECT_ROOT / os.getenv('MODELS_PATH', 'models')
    LOGS_PATH = PROJECT_ROOT / 'logs'
    
    # Create directories
    for path in [DATA_PATH, PROCESSED_PATH, MODELS_PATH, LOGS_PATH]:
        path.mkdir(parents=True, exist_ok=True)
    
    # Model parameters
    RANDOM_SEED = int(os.getenv('RANDOM_SEED', 42))
    BATCH_SIZE = 1024
    EPOCHS = 100
    LEARNING_RATE = 0.001
    
    # Trading parameters
    INITIAL_CAPITAL = 10000
    COMMISSION = 0.001
    SLIPPAGE = 0.0005
    RISK_FREE_RATE = 0.02
    
    # Feature parameters
    VWAP_WINDOW = 24
    VOLATILITY_WINDOW = 20
    FUNDING_WINDOW = 90
    
    @classmethod
    def load_yaml(cls, path='config.yaml'):
        """Load configuration from YAML file"""
        with open("config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        for key, value in config.items():
            if hasattr(cls, key):
                setattr(cls, key, value)
        return cls

# Singleton instance
config = Config()
print(f"📁 Models path: {config.MODELS_PATH}")
print(f"📁 Data path: {config.DATA_PATH}")