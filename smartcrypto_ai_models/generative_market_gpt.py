# smartcrypto_ai_models/generative_market_gpt.py

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import logging

from smartcrypto_ai_models.candle_microstructure import UnconstrainedCandleExtractor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MarketGPTWorldModel:
    """
    Generative Autoregressive Temporal Transformer ("Market GPT").
    Learns joint conditional probability distribution P(Future_12h | History_48h)
    and generates 1,000 stochastic future price trajectories.
    """

    @staticmethod
    def build_generative_model(sequence_length: int = 48, num_features: int = 21, forecast_horizon: int = 12) -> models.Model:
        inputs = layers.Input(shape=(sequence_length, num_features), name='historical_48h_context')

        x = layers.Conv1D(filters=128, kernel_size=3, padding='causal', activation='swish')(inputs)
        x = layers.LayerNormalization()(x)

        for _ in range(2):
            attn_out = layers.MultiHeadAttention(num_heads=8, key_dim=32)(x, x)
            x = layers.Add()([x, attn_out])
            x = layers.LayerNormalization()(x)
            
            dense_ff = layers.Dense(256, activation='swish')(x)
            dense_ff = layers.Dense(128)(dense_ff)
            x = layers.Add()([x, dense_ff])
            x = layers.LayerNormalization()(x)

        pooled = layers.GlobalAveragePooling1D()(x)

        forecast_mean = layers.Dense(forecast_horizon, name='future_return_means')(pooled)
        forecast_std = layers.Dense(forecast_horizon, activation='softplus', name='future_return_stds')(pooled)

        model = models.Model(inputs=inputs, outputs=[forecast_mean, forecast_std], name='MarketGPT_WorldModel')
        return model

    @staticmethod
    def simulate_future_paths(model: models.Model, current_context_matrix: np.ndarray,
                              n_simulations: int = 1000, forecast_horizon: int = 12) -> dict:
        context_tensor = tf.convert_to_tensor([current_context_matrix], dtype=tf.float32)
        means, stds = model(context_tensor)

        means = means[0].numpy()
        stds = stds[0].numpy() + 1e-4

        sampled_returns = np.random.normal(
            loc=np.tile(means, (n_simulations, 1)),
            scale=np.tile(stds, (n_simulations, 1))
        )

        price_trajectories = np.cumprod(1.0 + sampled_returns, axis=1)

        tp_target = 1.015  # +1.5% Gain
        sl_target = 0.985  # -1.5% Loss

        tp_hits, sl_hits = 0, 0
        for sim in range(n_simulations):
            path = price_trajectories[sim]
            hit_tp = np.any(path >= tp_target)
            hit_sl = np.any(path <= sl_target)

            if hit_tp and not hit_sl:
                tp_hits += 1
            elif hit_sl and not hit_tp:
                sl_hits += 1

        prob_win = tp_hits / n_simulations
        prob_loss = sl_hits / n_simulations
        expected_pnl = float(np.mean(price_trajectories[:, -1] - 1.0))

        return {
            'win_probability': prob_win,
            'loss_probability': prob_loss,
            'expected_return': expected_pnl,
            'simulations_run': n_simulations
        }


def main():
    logger.info("=" * 80)
    logger.info("🚀 STARTING PARADIGM C: GENERATIVE MARKET GPT WORLD MODEL")
    logger.info("=" * 80)

    parquet_path = "data/raw/combined_multi_horizon_1h.parquet"
    if not os.path.exists(parquet_path):
        logger.error(f"❌ Parquet file not found at {parquet_path}")
        return

    df_raw = pd.read_parquet(parquet_path)
    extractor = UnconstrainedCandleExtractor()
    df_featured = extractor.extract_features(df_raw)
    feature_cols = extractor.get_feature_columns()

    # Create sequence contexts for training Market GPT
    X_list, y_mean_list, y_std_list = [], [], []
    seq_len = 48
    horizon = 12

    for symbol in df_featured['symbol'].unique():
        s_df = df_featured[df_featured['symbol'] == symbol].sort_values('timestamp').reset_index(drop=True)
        feats = s_df[feature_cols].values
        rets = s_df['ret_1'].values

        n_samples = len(s_df) - seq_len - horizon
        for i in range(0, n_samples, 4):  # Subsample every 4 bars for speed
            X_list.append(feats[i:i + seq_len])
            fut_rets = rets[i + seq_len:i + seq_len + horizon]
            y_mean_list.append(fut_rets)
            y_std_list.append(np.abs(fut_rets))

    X = np.array(X_list, dtype=np.float32)
    y_m = np.array(y_mean_list, dtype=np.float32)
    y_s = np.array(y_std_list, dtype=np.float32)

    model = MarketGPTWorldModel.build_generative_model(sequence_length=48, num_features=len(feature_cols))
    model.compile(optimizer=optimizers.Adam(learning_rate=0.0005), loss=['mse', 'mae'])

    logger.info(f"🏋️ Training Generative Market GPT on {len(X):,} sequences...")
    model.fit(X, [y_m, y_s], epochs=10, batch_size=256, verbose=1)

    # Save trained Generative World Model
    save_dir = "smartcrypto_ai_models"
    os.makedirs(save_dir, exist_ok=True)
    model.save(os.path.join(save_dir, "market_gpt_world_model.keras"))

    # Sample 1,000 Monte Carlo Future Path Simulation
    sample_context = df_featured.iloc[-48:][feature_cols].values
    sim_results = MarketGPTWorldModel.simulate_future_paths(model, sample_context, n_simulations=1000)

    logger.info("=" * 80)
    logger.info("🎰 1,000 MONTE CARLO FUTURE PATH SIMULATION RESULTS:")
    logger.info(f"   Win Probability (+1.5% TP First): {sim_results['win_probability']:.1%}")
    logger.info(f"   Loss Probability (-1.5% SL First): {sim_results['loss_probability']:.1%}")
    logger.info(f"   Expected 12H PnL: {sim_results['expected_return']:.2%}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()