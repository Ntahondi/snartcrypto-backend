"""
Ensemble Reliability Evaluator with Dynamic Conviction Thresholds
Run: python test_ensemble_reliability.py
"""

import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import RobustScaler

from smartcrypto_ai_models.candle_microstructure import UnconstrainedCandleExtractor
from smartcrypto_ai_models.regression_targets import ContinuousRegressionLabeler
from smartcrypto_ai_models.conv1d_attention_model import ResNetBlock1D


def main():
    print("=" * 80)
    print("🛡️ DUAL-AI HIGH-CONVICTION CONVERGENCE BENCHMARK")
    print("=" * 80)

    parquet_path = "data/raw/combined_multi_horizon_1h.parquet"
    if not os.path.exists(parquet_path):
        print(f"❌ Parquet file not found at {parquet_path}")
        return

    df_raw = pd.read_parquet(parquet_path)
    extractor = UnconstrainedCandleExtractor()
    df_featured = extractor.extract_features(df_raw)
    feature_cols = extractor.get_feature_columns()

    labeler_reg = ContinuousRegressionLabeler(horizons={'1h': 1, '4h': 4, '12h': 12})
    df_labeled = labeler_reg.label_dataset(df_featured)

    scaler = RobustScaler()
    sequence_length = 48

    X_test_list, y1_te_list, y4_te_list = [], [], []

    for symbol in df_labeled['symbol'].unique():
        s_df = df_labeled[df_labeled['symbol'] == symbol].sort_values('timestamp').reset_index(drop=True)
        if len(s_df) < 200:
            continue

        feat_scaled = scaler.fit_transform(s_df[feature_cols].values)
        y1_v = s_df['target_ret_1h'].values
        y4_v = s_df['target_ret_4h'].values

        n_samples = len(s_df) - sequence_length - 12
        X_s = np.zeros((n_samples, sequence_length, len(feature_cols)), dtype=np.float32)
        
        for i in range(n_samples):
            X_s[i] = feat_scaled[i:i + sequence_length]

        s_split = int(len(X_s) * 0.8)

        X_test_list.append(X_s[s_split:])
        y1_te_list.append(y1_v[sequence_length + s_split:sequence_length + s_split + len(X_s[s_split:])])
        y4_te_list.append(y4_v[sequence_length + s_split:sequence_length + s_split + len(X_s[s_split:])])

    X_test = np.concatenate(X_test_list, axis=0)
    y1_test = np.concatenate(y1_te_list, axis=0)
    y4_test = np.concatenate(y4_te_list, axis=0)

    # Load Model 1 (Regression AI)
    reg_path = "smartcrypto_ai_models/continuous_regression_ai.keras"
    custom_objects = {'ResNetBlock1D': ResNetBlock1D}
    model_reg = tf.keras.models.load_model(reg_path, custom_objects=custom_objects)

    print("⚡ Running Model 1 Predictions...")
    preds_reg = model_reg.predict(X_test, verbose=0)
    pred_1h = preds_reg[0].flatten()
    pred_4h = preds_reg[1].flatten()

    print("\n" + "=" * 80)
    print("📊 CONVICTION THRESHOLD VS WIN RATE BENCHMARK")
    print("=" * 80)
    print(f"{'Threshold':<15} | {'Triggers':<15} | {'Trigger %':<12} | {'Win Rate':<12} | {'Avg 4H Move':<12}")
    print("-" * 80)

    # Test thresholds from 0.6% up to 2.0% with 1H & 4H directional agreement
    thresholds = [0.006, 0.008, 0.010, 0.012, 0.015, 0.018, 0.020]

    for thresh in thresholds:
        dir_agreement = (np.sign(pred_1h) == np.sign(pred_4h))
        magnitude_pass = (np.abs(pred_4h) >= thresh)
        
        mask = dir_agreement & magnitude_pass
        idx = np.where(mask)[0]

        if len(idx) > 0:
            correct = (np.sign(pred_4h[idx]) == np.sign(y4_test[idx]))
            win_rate = np.mean(correct)
            avg_move = np.mean(np.abs(y4_test[idx]))
            pct_triggered = (len(idx) / len(X_test)) * 100
            
            print(f"≥ {thresh*100:.1f}% Return   | {len(idx):<15,} | {pct_triggered:<11.1f}% | {win_rate:<11.2%} | {avg_move*100:<11.2f}%")
        else:
            print(f"≥ {thresh*100:.1f}% Return   | 0               | 0.0%        | N/A         | N/A")

    print("=" * 80)


if __name__ == "__main__":
    main()