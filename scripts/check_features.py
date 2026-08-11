#!/usr/bin/env python
"""
Check which features are missing between training and test data
"""
import sys
from pathlib import Path
import joblib
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import config
from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer

def check_missing_features():
    """Compare training features vs test data features"""
    
    print("="*60)
    print("🔍 CHECKING MISSING FEATURES")
    print("="*60)
    
    # Load model features
    features_path = config.MODELS_PATH / "feature_columns.joblib"
    if features_path.exists():
        model_features = joblib.load(features_path)
        print(f"✅ Model expects {len(model_features)} features")
    else:
        print("❌ Model features file not found")
        return
    
    # Load data
    loader = DataLoader()
    df = loader.load_data()
    
    # Process data with features
    engineer = FeatureEngineer()
    all_symbols = []
    for symbol in df["symbol"].unique():
        symbol_data = df[df["symbol"] == symbol].copy()
        symbol_data = engineer.add_technical_indicators(symbol_data)
        all_symbols.append(symbol_data)
    df_enhanced = pd.concat(all_symbols, ignore_index=True)
    
    # Get test data features
    n = len(df_enhanced)
    test_size = int(0.2 * n)
    df_test = df_enhanced.iloc[-test_size:].copy()
    test_features = df_test.columns.tolist()
    
    # Find missing features
    missing_features = [f for f in model_features if f not in test_features]
    
    print(f"\n📊 Test data has {len(test_features)} columns")
    print(f"📊 Model expects {len(model_features)} features")
    print(f"\n❌ Missing {len(missing_features)} features:")
    
    if missing_features:
        print("\n" + "-"*40)
        for i, feat in enumerate(missing_features, 1):
            print(f"  {i:2d}. {feat}")
        print("-"*40)
        
        # Categorize missing features
        print("\n📂 CATEGORIZATION:")
        
        funding = [f for f in missing_features if 'funding' in f or 'oi_' in f]
        if funding:
            print(f"  Derivatives features: {len(funding)}")
            print(f"    {funding}")
        
        bollinger = [f for f in missing_features if 'BB' in f or 'bb' in f]
        if bollinger:
            print(f"  Bollinger features: {len(bollinger)}")
            print(f"    {bollinger}")
        
        other = [f for f in missing_features if f not in funding and f not in bollinger]
        if other:
            print(f"  Other features: {len(other)}")
            print(f"    {other}")
        
        # Suggest fixes
        print("\n💡 SUGGESTED FIXES:")
        print("  Option 1: Add dummy columns for missing features")
        print("  Option 2: Retrain model without derivatives features")
        print("  Option 3: Collect derivatives data for test period")
    else:
        print("✅ No features missing! All 58 features are present.")
    
    return missing_features

if __name__ == "__main__":
    check_missing_features()