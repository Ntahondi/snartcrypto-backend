"""
Classification AI Rescue Benchmark
Tests unconstrained_candle_ai.keras with a Softmax Confidence Margin Filter (>= 55%)
Run: python test_classification_rescue.py
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score

from smartcrypto_ai_models.candle_microstructure import UnconstrainedCandleExtractor
from smartcrypto_ai_models.triple_barrier_targets import TripleBarrierTargetLabeler
from smartcrypto_ai_models.conv1d_attention_model import ResNetBlock1D


def main():
    print("=" * 80)
    print("🚑 CLASSIFICATION AI RESCUE BENCHMARK (CONFIDENCE FILTER)")
    print("=" * 80)

    parquet_path = "data/raw/combined_multi_horizon_1h.parquet"
    if not os.path.exists(parquet_path):
        print(f"❌ Parquet file not found at {parquet_path}")
        return

    df_raw = pd.read_parquet(parquet_path)
    extractor = UnconstrainedCandleExtractor()
    df_featured = extractor.extract_features(df_raw)
    feature_cols = extractor.get_feature_columns()

    labeler = TripleBarrierTargetLabeler(tp_atr_mult=2.0, sl_atr_mult=2.0, max_holding_bars=12)
    df_labeled = labeler.label_dataset(df_featured)

    scaler = RobustScaler()
    sequence_length = 48

    X_test_list, y_dir_test_list = [], []

    for symbol in df_labeled['symbol'].unique():
        s_df = df_labeled[df_labeled['symbol'] == symbol].sort_values('timestamp').reset_index(drop=True)
        if len(s_df) < 200:
            continue

        feat_scaled = scaler.fit_transform(s_df[feature_cols].values)
        dirs = s_df['target_direction'].values

        n_samples = len(s_df) - sequence_length - 12
        X_s = np.zeros((n_samples, sequence_length, len(feature_cols)), dtype=np.float32)
        
        for i in range(n_samples):
            X_s[i] = feat_scaled[i:i + sequence_length]

        s_split = int(len(X_s) * 0.8)

        X_test_list.append(X_s[s_split:])
        y_dir_test_list.append(dirs[sequence_length + s_split:sequence_length + s_split + len(X_s[s_split:])])

    X_test = np.concatenate(X_test_list, axis=0)
    y_dir_test = np.concatenate(y_dir_test_list, axis=0)

    # Load Model 2 (Classification AI)
    cls_path = "smartcrypto_ai_models/unconstrained_candle_ai.keras"
    custom_objects = {'ResNetBlock1D': ResNetBlock1D}
    model_cls = tf.keras.models.load_model(cls_path, custom_objects=custom_objects)

    print("⚡ Predicting 3-Class Probabilities on 74,123 test samples...")
    preds_cls = model_cls.predict(X_test, verbose=0)
    dir_probs = preds_cls[0]  # Shape: (74123, 3)

    print("\n" + "=" * 80)
    print("📊 SOFTMAX CONFIDENCE FILTER VS WIN RATE BENCHMARK")
    print("=" * 80)
    print(f"{'Min Confidence':<18} | {'Triggers':<15} | {'Trigger %':<12} | {'Win Rate':<12}")
    print("-" * 80)

    for min_conf in [0.33, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        max_probs = np.max(dir_probs, axis=1)
        pred_classes = np.argmax(dir_probs, axis=1)

        # Filter: Max probability >= min_conf AND class != 1 (Exclude HOLD)
        mask = (max_probs >= min_conf) & (pred_classes != 1)
        idx = np.where(mask)[0]

        if len(idx) > 0:
            correct = (pred_classes[idx] == y_dir_test[idx])
            win_rate = np.mean(correct)
            pct_triggered = (len(idx) / len(X_test)) * 100
            print(f"≥ {min_conf*100:.0f}% Confidence    | {len(idx):<15,} | {pct_triggered:<11.1f}% | {win_rate:<11.2%}")
        else:
            print(f"≥ {min_conf*100:.0f}% Confidence    | 0               | 0.0%        | N/A")

    print("=" * 80)


if __name__ == "__main__":
    main()