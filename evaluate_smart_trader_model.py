"""
Evaluate smart_trader_ai_final.keras Model Metrics
Run: python evaluate_smart_trader_model.py
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, mean_absolute_error
import joblib

from src.data.processors import DataProcessor


def main():
    print("=" * 80)
    print("📊 EVALUATING: models/smart_trader_ai_final.keras")
    print("=" * 80)

    model_path = "models/smart_trader_ai_final.keras"
    scaler_path = "models/robust_scaler.joblib"
    transformer_path = "models/power_transformer.joblib"
    features_path = "models/feature_columns.joblib"

    if not os.path.exists(model_path):
        print(f"❌ Model file not found at {model_path}")
        return

    # Load Model & Scalers
    print("📥 Loading model weights, scalers, and feature columns...")
    custom_objects = {
        'GlorotUniform': tf.keras.initializers.GlorotUniform,
        'AdamW': tf.keras.optimizers.AdamW,
    }
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
    scaler = joblib.load(scaler_path)
    transformer = joblib.load(transformer_path)
    feature_cols = joblib.load(features_path)

    # Load Master Dataset
    parquet_path = "data/raw/combined_multi_horizon_1h.parquet"
    if not os.path.exists(parquet_path):
        print(f"❌ Parquet file not found at {parquet_path}")
        return

    df_raw = pd.read_parquet(parquet_path)
    processor = DataProcessor()

    print("🔬 Engineering features for evaluation...")
    df_featured = processor.engineer_features(df_raw)

    # Ensure all feature columns exist
    for col in feature_cols:
        if col not in df_featured.columns:
            df_featured[col] = 0.0

    X_raw = df_featured[feature_cols].fillna(0.0).values
    
    # Scale & Transform Features
    X_scaled = scaler.transform(X_raw)
    X_trans = transformer.transform(X_scaled)

    # Unseen Test Split (Recent 20%)
    split_idx = int(len(X_trans) * 0.8)
    X_test = X_trans[split_idx:]

    # Predict
    print(f"⚡ Running Model Predictions on {len(X_test):,} unseen test samples...")
    preds = model.predict(X_test, verbose=0)

    # Extract Predictions
    pred_1h = np.argmax(preds[0], axis=1)
    pred_4h = np.argmax(preds[1], axis=1)
    pred_1d = np.argmax(preds[2], axis=1)
    conf_scores = preds[3].flatten()
    risk_levels = np.argmax(preds[4], axis=1)
    regimes = np.argmax(preds[5], axis=1)

    # Map labels: 0=SELL, 1=HOLD, 2=BUY
    bincount_1h = np.bincount(pred_1h, minlength=3)
    bincount_4h = np.bincount(pred_4h, minlength=3)
    bincount_1d = np.bincount(pred_1d, minlength=3)

    print("\n" + "=" * 80)
    print("🎯 PURE METRICS REPORT FOR smart_trader_ai_final.keras:")
    print("=" * 80)
    print(f"   • Total Unseen Test Samples Evaluated: {len(X_test):,}")
    print(f"   • 1H Predictions (SELL / HOLD / BUY):  {bincount_1h[0]:,} / {bincount_1h[1]:,} / {bincount_1h[2]:,}")
    print(f"   • 4H Predictions (SELL / HOLD / BUY):  {bincount_4h[0]:,} / {bincount_4h[1]:,} / {bincount_4h[2]:,}")
    print(f"   • 1D Predictions (SELL / HOLD / BUY):  {bincount_1d[0]:,} / {bincount_1d[1]:,} / {bincount_1d[2]:,}")
    print(f"   • Average Predicted Confidence:        {np.mean(conf_scores):.2%}")
    print(f"   • Market Regime Distribution (T/R/V/X):{np.bincount(regimes, minlength=4).tolist()}")
    print(f"   • Risk Level Distribution (L/M/H):    {np.bincount(risk_levels, minlength=3).tolist()}")
    print("=" * 80)


if __name__ == "__main__":
    main()