"""
Data loading and preprocessing
"""
import pandas as pd
import numpy as np
from pathlib import Path
from src.config import config
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    """Load and preprocess cryptocurrency data"""
    
    def __init__(self, data_path=None):
        self.data_path = Path(data_path) if data_path else config.DATA_PATH
        logger.info(f"Data path: {self.data_path}")
    
    def load_data(self, filename='combined_multi_horizon_1h.parquet'):
        """Load data from parquet file"""
        filepath = self.data_path / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        logger.info(f"Loading data from {filepath}")
        df = pd.read_parquet(filepath)
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        
        return df
    
    def validate_data(self, df):
        """Validate data quality"""
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol']
        missing_cols = set(required_cols) - set(df.columns)
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check for nulls
        null_counts = df.isnull().sum()
        if null_counts.sum() > 0:
            logger.warning(f"Found {null_counts.sum()} null values")
            logger.warning(f"Null columns: {null_counts[null_counts > 0].to_dict()}")
        
        return True