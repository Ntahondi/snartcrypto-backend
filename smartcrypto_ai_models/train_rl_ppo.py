# smartcrypto_ai_models/train_rl_ppo.py

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import logging

from smartcrypto_ai_models.candle_microstructure import UnconstrainedCandleExtractor
from smartcrypto_ai_models.rl_trading_env import RLMarketEnvironment

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ActorCriticNetwork(models.Model):
    """Deep Actor-Critic Network for PPO RL Trading"""
    def __init__(self, action_dim: int = 1):
        super().__init__()
        self.dense1 = layers.Dense(256, activation='swish')
        self.bn1 = layers.BatchNormalization()
        self.dense2 = layers.Dense(128, activation='swish')
        
        # Actor Head: Outputs mean action (-1.0 to +1.0)
        self.actor_mean = layers.Dense(action_dim, activation='tanh', name='actor_mean')
        
        # Critic Head: Outputs estimated state value
        self.critic_value = layers.Dense(1, name='critic_value')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.bn1(x)
        x = self.dense2(x)
        return self.actor_mean(x), self.critic_value(x)


def main():
    logger.info("=" * 80)
    logger.info("🚀 STARTING PARADIGM B: DEEP REINFORCEMENT LEARNING (PPO AGENT)")
    logger.info("=" * 80)

    parquet_path = "data/raw/combined_multi_horizon_1h.parquet"
    if not os.path.exists(parquet_path):
        logger.error(f"❌ Parquet file not found at {parquet_path}")
        return

    # 1. Load Data & Extract Features
    df_raw = pd.read_parquet(parquet_path)
    extractor = UnconstrainedCandleExtractor()
    df_featured = extractor.extract_features(df_raw)
    feature_cols = extractor.get_feature_columns()

    # Use BTCUSDT history for RL training
    btc_df = df_featured[df_featured['symbol'] == 'BTCUSDT'].sort_values('timestamp').reset_index(drop=True)

    # 2. Initialize Market Environment
    env = RLMarketEnvironment(btc_df, feature_cols, initial_balance=10000.0)
    
    # 3. Initialize Actor-Critic Network
    model = ActorCriticNetwork(action_dim=1)
    optimizer = optimizers.Adam(learning_rate=0.0003)

    episodes = 5
    logger.info(f"🏋️ Training PPO Agent across {len(btc_df):,} historical hours...")

    for ep in range(1, episodes + 1):
        obs = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            obs_tensor = tf.convert_to_tensor([obs], dtype=tf.float32)
            
            # Execute step in market environment
            with tf.GradientTape() as tape:
                action_mean, value = model(obs_tensor)
                
                action_val = float(action_mean[0][0]) + np.random.normal(0, 0.1)
                action_val = float(np.clip(action_val, -1.0, 1.0))

                next_obs, reward, done, info = env.step(action_val)
                
                # Keep reward and advantage in TensorFlow Graph for Critic Gradients!
                reward_tensor = tf.convert_to_tensor([[reward]], dtype=tf.float32)
                advantage = reward_tensor - value
                
                # Policy Actor Loss + Value Critic Loss (Full Gradient Flow)
                action_diff = tf.abs(action_mean - action_val) + 1e-5
                actor_loss = -tf.math.log(action_diff) * tf.stop_gradient(advantage)
                critic_loss = tf.square(advantage)
                
                total_loss = tf.reduce_mean(actor_loss + 0.5 * critic_loss)

            # Apply gradients to BOTH Actor and Critic
            grads = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            obs = next_obs
            total_reward += reward
            steps += 1

            if steps % 10000 == 0:
                logger.info(f"   Episode {ep} | Step {steps:,}/{env.n_steps:,} | Portfolio: ${info['balance']:,.2f}")

        logger.info(f"🎉 Episode {ep} Finished | Final Portfolio: ${env.balance:,.2f} | Total Reward: {total_reward:.2f}")

# FIX: Keras 3 requires filenames to end in .weights.h5
    save_dir = "smartcrypto_ai_models"
    os.makedirs(save_dir, exist_ok=True)
    model.save_weights(os.path.join(save_dir, "rl_ppo_agent.weights.h5"))
    logger.info("✅ RL PPO Agent Trained & Weights Saved!")


if __name__ == "__main__":
    main()