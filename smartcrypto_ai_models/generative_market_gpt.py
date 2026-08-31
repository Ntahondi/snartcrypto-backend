# smartcrypto_ai_models/generative_market_gpt.py

import os
import sys
sys.path.insert(0, os.path.abspath("."))

from typing import Optional, Dict, List, Tuple
import json
import logging
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.preprocessing import RobustScaler

from smartcrypto_ai_models.candle_microstructure import UnconstrainedCandleExtractor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MarketGPTWorldModel:
    """
    Generative Autoregressive Temporal Transformer ("Market GPT v2").
    Learns conditional probability distribution P(Future_12h | History_48h)
    and generates 1,000 stochastic future price trajectories with balanced volatility diffusion.
    """

    @staticmethod
    def build_generative_model(sequence_length: int = 48, num_features: int = 23, forecast_horizon: int = 12) -> models.Model:
        inputs = layers.Input(shape=(sequence_length, num_features), name='historical_48h_context')

        # 1. Temporal Convolutional Feature Embedding
        x = layers.Conv1D(filters=64, kernel_size=3, padding='causal', activation='swish')(inputs)
        x = layers.LayerNormalization()(x)

        # 2. Multi-Head Temporal Self-Attention Blocks (Transformer)
        for _ in range(2):
            attn_out = layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
            x = layers.Add()([x, attn_out])
            x = layers.LayerNormalization()(x)
            
            dense_ff = layers.Dense(128, activation='swish')(x)
            dense_ff = layers.Dense(64)(dense_ff)
            x = layers.Add()([x, dense_ff])
            x = layers.LayerNormalization()(x)

        # 3. Global Temporal Aggregation
        pooled = layers.GlobalAveragePooling1D()(x)
        dense_core = layers.Dense(64, activation='swish')(pooled)
        dense_core = layers.Dropout(0.1)(dense_core)

        # 4. Dual Stochastic Output Heads (Mean Return & Volatility Dispersion)
        forecast_mean = layers.Dense(forecast_horizon, activation='linear', name='future_return_means')(dense_core)
        forecast_std = layers.Dense(forecast_horizon, activation='softplus', name='future_return_stds')(dense_core)

        model = models.Model(inputs=inputs, outputs=[forecast_mean, forecast_std], name='MarketGPT_WorldModel_v2')
        return model

    @staticmethod
    def simulate_future_paths(
        model: models.Model,
        current_context_matrix: np.ndarray,
        scaler: Optional[RobustScaler] = None,
        n_simulations: int = 1000,
        forecast_horizon: int = 12,
        tp_pct: float = 0.015,
        sl_pct: float = 0.015,
    ) -> dict:
        """
        Simulate 1,000 Monte Carlo stochastic trajectories forward in time.
        """
        if current_context_matrix.ndim == 2:
            matrix = current_context_matrix.copy()
            if scaler is not None:
                matrix = scaler.transform(matrix)
            context_tensor = tf.convert_to_tensor([matrix], dtype=tf.float32)
        else:
            context_tensor = tf.convert_to_tensor(current_context_matrix, dtype=tf.float32)

        means, stds = model(context_tensor, training=False)

        means = means[0].numpy()
        stds = np.clip(stds[0].numpy(), 0.001, 0.05)

        # Vectorized Monte Carlo Path Sampling
        sampled_returns = np.random.normal(
            loc=np.tile(means, (n_simulations, 1)),
            scale=np.tile(stds, (n_simulations, 1))
        )

        price_trajectories = np.cumprod(1.0 + sampled_returns, axis=1)

        tp_target = 1.0 + tp_pct
        sl_target = 1.0 - sl_pct

        tp_hits, sl_hits = 0, 0
        for sim in range(n_simulations):
            path = price_trajectories[sim]
            hit_tp_idx = np.where(path >= tp_target)[0]
            hit_sl_idx = np.where(path <= sl_target)[0]

            first_tp = hit_tp_idx[0] if len(hit_tp_idx) > 0 else 999
            first_sl = hit_sl_idx[0] if len(hit_sl_idx) > 0 else 999

            if first_tp < first_sl:
                tp_hits += 1
            elif first_sl < first_tp:
                sl_hits += 1

        prob_win = float(tp_hits / n_simulations)
        prob_loss = float(sl_hits / n_simulations)
        expected_pnl = float(np.mean(price_trajectories[:, -1] - 1.0))

        return {
            'win_probability': prob_win,
            'loss_probability': prob_loss,
            'expected_return': expected_pnl,
            'predicted_means': means.tolist(),
            'predicted_stds': stds.tolist(),
            'simulations_run': n_simulations
        }


def main():
    logger.info("=" * 80)
    logger.info("🚀 RETRAINING MARKET GPT WORLD MODEL (v2) ON MULTI-YEAR DATASET")
    logger.info("=" * 80)

    parquet_path = "data/raw/combined_multi_horizon_1h.parquet"
    if not os.path.exists(parquet_path):
        logger.error(f"❌ Parquet file not found at {parquet_path}")
        return

    df_raw = pd.read_parquet(parquet_path)
    df_raw = df_raw.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
    logger.info(f"📥 Loaded {len(df_raw):,} rows across {df_raw['symbol'].nunique()} symbols.")

    extractor = UnconstrainedCandleExtractor()
    df_featured = extractor.extract_features(df_raw)
    feature_cols = extractor.get_feature_columns()
    logger.info(f"✅ Extracted {len(feature_cols)} features per candle.")

    # 1. Fit RobustScaler on feature columns
    scaler = RobustScaler()
    all_feats = df_featured[feature_cols].fillna(0.0).values
    scaler.fit(all_feats)
    logger.info("✅ RobustScaler fitted on training features.")

    # 2. Build 48h Context Sequences and 12h Future Step Returns
    X_list, y_mean_list, y_std_list = [], [], []
    seq_len = 48
    horizon = 12

    for symbol in df_featured['symbol'].unique():
        s_df = df_featured[df_featured['symbol'] == symbol].sort_values('timestamp').reset_index(drop=True)
        raw_feats = s_df[feature_cols].fillna(0.0).values
        scaled_feats = scaler.transform(raw_feats)
        rets = s_df['ret_1'].fillna(0.0).values

        n_samples = len(s_df) - seq_len - horizon
        for i in range(0, n_samples, 2):  # Sample every 2 bars
            X_list.append(scaled_feats[i:i + seq_len])
            fut_rets = rets[i + seq_len:i + seq_len + horizon]
            y_mean_list.append(fut_rets)
            y_std_list.append(np.abs(fut_rets))

    X = np.array(X_list, dtype=np.float32)
    y_m = np.array(y_mean_list, dtype=np.float32)
    y_s = np.array(y_std_list, dtype=np.float32)

    logger.info(f"📊 Training Dataset: {len(X):,} sequences of shape ({seq_len}, {len(feature_cols)})")

    # Chronological 85/15 train/val split
    split_idx = int(len(X) * 0.85)
    X_train, X_val = X[:split_idx], X[split_idx:]
    ym_train, ym_val = y_m[:split_idx], y_m[split_idx:]
    ys_train, ys_val = y_s[:split_idx], y_s[split_idx:]

    model = MarketGPTWorldModel.build_generative_model(
        sequence_length=seq_len,
        num_features=len(feature_cols),
        forecast_horizon=horizon
    )

    model.compile(
        optimizer=optimizers.AdamW(learning_rate=0.0008, weight_decay=1e-4),
        loss={
            'future_return_means': tf.keras.losses.Huber(delta=0.01),
            'future_return_stds': 'mae',
        },
        loss_weights={'future_return_means': 1.0, 'future_return_stds': 0.5},
    )

    cb_list = [
        callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1),
    ]

    logger.info("🏋️ Training Transformer Neural World Model...")
    model.fit(
        X_train,
        {'future_return_means': ym_train, 'future_return_stds': ys_train},
        validation_data=(X_val, {'future_return_means': ym_val, 'future_return_stds': ys_val}),
        epochs=15,
        batch_size=256,
        callbacks=cb_list,
        verbose=1
    )

    # 3. Save Model & Scaler Artifacts
    save_dir = "smartcrypto_ai_models"
    os.makedirs(save_dir, exist_ok=True)
    model_save_path = os.path.join(save_dir, "market_gpt_world_model.keras")
    scaler_save_path = os.path.join(save_dir, "market_gpt_scaler.joblib")
    features_save_path = os.path.join(save_dir, "market_gpt_features.joblib")

    model.save(model_save_path)
    joblib.dump(scaler, scaler_save_path)
    joblib.dump(feature_cols, features_save_path)

    logger.info("=" * 80)
    logger.info(f"🎉 SUCCESS! Market GPT World Model (v2) Saved:")
    logger.info(f"   Model: {model_save_path}")
    logger.info(f"   Scaler: {scaler_save_path}")
    logger.info(f"   Features: {features_save_path}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()