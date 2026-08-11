#!/usr/bin/env python
"""
Main training script
"""
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np

from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.train import Trainer
from src.config import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Run training pipeline"""
    logger.info("🚀 Starting SmartCrypto AI Training")
    
    # Print paths for debugging
    logger.info(f"📁 Project root: {config.PROJECT_ROOT}")
    logger.info(f"📁 Data path: {config.DATA_PATH}")
    logger.info(f"📁 Models path: {config.MODELS_PATH}")
    
    # Load data
    loader = DataLoader()
    df = loader.load_data()
    loader.validate_data(df)
    
    # Feature engineering
    engineer = FeatureEngineer()
    
    # Process each symbol
    all_symbols = []
    for symbol in df['symbol'].unique():
        logger.info(f"Processing symbol: {symbol}")
        symbol_data = df[df['symbol'] == symbol].copy()
        
        # Fetch derivatives data
        symbol_ccxt = symbol.replace('USDT', '/USDT')
        df_deriv = engineer.fetch_derivatives_data(symbol_ccxt)
        if df_deriv is not None:
            symbol_data = engineer.add_derivatives_features(symbol_data, df_deriv)
        
        # Add technical indicators
        symbol_data = engineer.add_technical_indicators(symbol_data)
        
        # Create targets
        symbol_data = engineer.generate_targets(symbol_data)
        symbol_data = engineer.create_risk_targets(symbol_data)
        symbol_data = engineer.create_regime_targets(symbol_data)
        symbol_data = engineer.create_confidence_target(symbol_data)
        
        all_symbols.append(symbol_data)
    
    df_enhanced = pd.concat(all_symbols, ignore_index=True)
    
    # Get stationary features
    features = engineer.get_stationary_features(df_enhanced)
    logger.info(f"Selected {len(features)} stationary features")
    
    # Prepare data
    X = df_enhanced[features].fillna(0).replace([np.inf, -np.inf], 0)
    
    # Split data
    n = len(df_enhanced)
    train_size = int(0.7 * n)
    val_size = int(0.1 * n)
    
    X_train, X_val = X.iloc[:train_size], X.iloc[train_size:train_size+val_size]
    
    y_train = {
        'direction_1h': df_enhanced['target_1h'].iloc[:train_size].values,
        'direction_4h': df_enhanced['target_4h'].iloc[:train_size].values,
        'direction_1d': df_enhanced['target_1d'].iloc[:train_size].values,
        'confidence_score': df_enhanced['confidence_score'].iloc[:train_size].values,
        'risk_level': df_enhanced['risk_level'].iloc[:train_size].values,
        'market_regime': df_enhanced['market_regime'].iloc[:train_size].values
    }
    
    y_val = {
        'direction_1h': df_enhanced['target_1h'].iloc[train_size:train_size+val_size].values,
        'direction_4h': df_enhanced['target_4h'].iloc[train_size:train_size+val_size].values,
        'direction_1d': df_enhanced['target_1d'].iloc[train_size:train_size+val_size].values,
        'confidence_score': df_enhanced['confidence_score'].iloc[train_size:train_size+val_size].values,
        'risk_level': df_enhanced['risk_level'].iloc[train_size:train_size+val_size].values,
        'market_regime': df_enhanced['market_regime'].iloc[train_size:train_size+val_size].values
    }
    
    # Train model
    trainer = Trainer(X_train, y_train, X_val, y_val)
    trainer.build_model()
    history = trainer.train()
    trainer.save_artifacts()
    
    # Verify model saved
    model_path = config.MODELS_PATH / 'smart_trader_ai_final.keras'
    if model_path.exists():
        logger.info(f"✅ Model successfully saved to {model_path}")
    else:
        logger.error(f"❌ Model NOT saved to {model_path}")
    
    logger.info("✅ Training completed!")

if __name__ == '__main__':
    main()