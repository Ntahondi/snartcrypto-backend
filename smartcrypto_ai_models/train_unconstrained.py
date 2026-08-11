"""
Training pipeline for the unconstrained market learner
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import logging
import json

from smartcrypto_ai_models.unconstrained_learner import UnconstrainedMarketLearner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UnconstrainedTrainingPipeline:
    """
    Complete training pipeline for the unconstrained learner
    """
    
    def __init__(self):
        self.learner = UnconstrainedMarketLearner(sequence_length=48, latent_dim=64)
        self.data = None
        self.labels = None
    
    def prepare_raw_data(self, parquet_path: str) -> np.ndarray:
        """
        Prepare raw OHLCV data WITHOUT feature engineering
        """
        logger.info("📥 Loading raw market data...")
        df = pd.read_parquet(parquet_path)
        
        # Only use raw OHLCV data
        raw_cols = ['open', 'high', 'low', 'close', 'volume']
        
        # Normalize each symbol independently
        normalized_data = []
        
        for symbol in df['symbol'].unique():
            symbol_df = df[df['symbol'] == symbol].sort_values('timestamp')
            
            # Raw OHLCV - let the model discover what matters
            raw_data = symbol_df[raw_cols].values
            
            # Normalize to zero mean, unit variance (simple)
            mean = raw_data.mean(axis=0, keepdims=True)
            std = raw_data.std(axis=0, keepdims=True) + 1e-8
            raw_data_norm = (raw_data - mean) / std
            
            normalized_data.append(raw_data_norm)
        
        # Combine all symbols
        combined_data = np.concatenate(normalized_data, axis=0)
        
        # Create sequences
        sequences = []
        seq_length = 48
        
        for i in range(len(combined_data) - seq_length):
            sequences.append(combined_data[i:i+seq_length])
        
        self.data = np.array(sequences)
        
        logger.info(f"✅ Prepared {len(self.data)} sequences of length {seq_length}")
        logger.info(f"   Shape: {self.data.shape}")
        
        return self.data
    
    def create_ground_truth_labels(self, data: np.ndarray) -> np.ndarray:
        """
        Create simple future returns labels (not teaching the model patterns,
        just giving it a goal to optimize)
        """
        # Use the close prices
        closes = data[:, -1, 3]  # Close is index 3
        
        # Future returns for different horizons
        future_returns = []
        
        for horizon in [1, 4, 12, 24]:
            returns = np.zeros(len(closes))
            for i in range(len(closes) - horizon):
                returns[i] = (closes[i+horizon] / closes[i] - 1)
            future_returns.append(returns)
        
        # Create directional labels (BUY/SELL/HOLD) from future returns
        direction_labels = np.zeros(len(closes), dtype=int)
        
        # Let the model learn what constitutes a "good" trade
        # We'll use the 4h return to label
        returns_4h = future_returns[1]  # 4h returns
        
        # Simple labeling - the model will learn the optimal threshold
        threshold = np.std(returns_4h) * 0.3  # Dynamic threshold
        
        direction_labels[returns_4h > threshold] = 0  # BUY
        direction_labels[returns_4h < -threshold] = 2  # SELL
        direction_labels[abs(returns_4h) <= threshold] = 1  # HOLD
        
        self.labels = direction_labels
        
        logger.info(f"✅ Created labels with dynamic threshold: {threshold:.4f}")
        logger.info(f"   BUY: {np.sum(direction_labels==0)}")
        logger.info(f"   HOLD: {np.sum(direction_labels==1)}")
        logger.info(f"   SELL: {np.sum(direction_labels==2)}")
        
        return direction_labels
    
    def train_unsupervised(self, epochs: int = 100):
        """
        Phase 1: Unsupervised pretraining
        """
        logger.info("=" * 80)
        logger.info("🧠 PHASE 1: Unsupervised Pretraining")
        logger.info("=" * 80)
        
        # Split for validation
        train_data, val_data = train_test_split(
            self.data, test_size=0.2, random_state=42
        )
        
        # Train the encoder
        history = self.learner.pretrain_unsupervised(train_data, epochs=epochs)
        
        logger.info("✅ Unsupervised pretraining complete!")
        
        return history
    
    def discover_patterns(self):
        """
        Phase 2: Let the model discover patterns
        """
        logger.info("=" * 80)
        logger.info("🔍 PHASE 2: Pattern Discovery")
        logger.info("=" * 80)
        
        # Use the full dataset for pattern discovery
        patterns = self.learner.discover_market_patterns(self.data)
        
        # Save discovered patterns
        with open('discovered_patterns.json', 'w') as f:
            # Convert numpy arrays to lists for JSON
            patterns_serializable = {
                'num_patterns': len(patterns['patterns']),
                'patterns': {
                    k: {
                        'size': v['size'],
                        'center': v['center'][:10],  # Truncate for display
                        'latent_center': v['latent_center'][:10]
                    }
                    for k, v in patterns['patterns'].items()
                },
                'cluster_summary': {
                    f'cluster_{i}': {
                        'size': np.sum(patterns['cluster_labels'] == i),
                        'percentage': np.sum(patterns['cluster_labels'] == i) / len(patterns['cluster_labels']) * 100
                    }
                    for i in range(len(patterns['patterns']))
                }
            }
            json.dump(patterns_serializable, f, indent=2)
        
        logger.info(f"✅ Discovered {len(patterns['patterns'])} natural patterns")
        
        return patterns
    
    def train_supervised(self, epochs: int = 30):
        """
        Phase 3: Supervised fine-tuning on the model's own representations
        """
        logger.info("=" * 80)
        logger.info("🎯 PHASE 3: Supervised Fine-tuning")
        logger.info("=" * 80)
        
        # Get latent representations
        latent_reps = self.learner.encoder.predict(self.data)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            latent_reps, self.labels, test_size=0.2, random_state=42
        )
        
        # Build predictor
        predictor = self.learner.build_predictor(latent_dim=self.learner.latent_dim)
        
        # Compile
        predictor.compile(
            optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-4),
            loss={
                'signal': 'sparse_categorical_crossentropy',
                'expected_movement': 'mse',
                'confidence': 'binary_crossentropy'
            },
            loss_weights={
                'signal': 0.6,
                'expected_movement': 0.2,
                'confidence': 0.2
            },
            metrics={'signal': ['accuracy']}
        )
        
        # Train
        history = predictor.fit(
            X_train,
            {
                'signal': y_train,
                'expected_movement': np.zeros_like(y_train, dtype=float),
                'confidence': np.ones_like(y_train, dtype=float)
            },
            validation_data=(
                X_test,
                {
                    'signal': y_test,
                    'expected_movement': np.zeros_like(y_test, dtype=float),
                    'confidence': np.ones_like(y_test, dtype=float)
                }
            ),
            epochs=epochs,
            batch_size=256,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
                tf.keras.callbacks.ModelCheckpoint('discovered_predictor.keras', save_best_only=True)
            ],
            verbose=1
        )
        
        # Save the predictor
        self.learner.predictor.save('discovered_predictor_final.keras')
        
        # Evaluate
        eval_results = predictor.evaluate(X_test, {
            'signal': y_test,
            'expected_movement': np.zeros_like(y_test, dtype=float),
            'confidence': np.ones_like(y_test, dtype=float)
        })
        
        logger.info(f"✅ Supervised training complete!")
        logger.info(f"   Test Loss: {eval_results[0]:.4f}")
        logger.info(f"   Signal Accuracy: {eval_results[3]:.4f}")
        
        return history
    
    def evaluate_unconstrained(self):
        """
        Evaluate the unconstrained learner
        """
        logger.info("=" * 80)
        logger.info("📊 EVALUATING UNCONSTRAINED LEARNER")
        logger.info("=" * 80)
        
        # Get predictions
        predictions = self.learner.predict_unconstrained(self.data)
        
        # Calculate metrics
        accuracy = np.mean(predictions['signal'] == self.labels)
        
        # Calculate win rate on active signals
        active_mask = predictions['signal'] != 1
        active_accuracy = np.mean(
            predictions['signal'][active_mask] == self.labels[active_mask]
        ) if np.sum(active_mask) > 0 else 0
        
        logger.info(f"📊 Results:")
        logger.info(f"   Overall Accuracy: {accuracy:.2%}")
        logger.info(f"   Active Signal Win Rate: {active_accuracy:.2%}")
        logger.info(f"   Active Signals: {np.sum(active_mask)} / {len(predictions['signal'])}")
        
        # Analyze discovered patterns
        logger.info(f"\n📈 Pattern Analysis:")
        for i in range(3):  # BUY/HOLD/SELL
            mask = predictions['signal'] == i
            if np.sum(mask) > 0:
                avg_movement = np.mean(predictions['expected_movement'][mask])
                avg_confidence = np.mean(predictions['confidence'][mask])
                logger.info(f"   Class {i}: {np.sum(mask)} predictions, "
                          f"Avg Movement: {avg_movement:.4f}, "
                          f"Avg Confidence: {avg_confidence:.2f}")
        
        # Analyze predictions by discovered pattern
        logger.info(f"\n🎯 Pattern-Specific Performance:")
        patterns = self.learner.discover_market_patterns(self.data)
        
        for i in range(len(patterns['patterns'])):
            mask = patterns['cluster_labels'] == i
            if np.sum(mask) > 0:
                pattern_accuracy = np.mean(
                    predictions['signal'][mask] == self.labels[mask]
                )
                logger.info(f"   Pattern {i}: {np.sum(mask)} samples, "
                          f"Accuracy: {pattern_accuracy:.2%}")
        
        return {
            'accuracy': accuracy,
            'active_win_rate': active_accuracy,
            'active_signals': np.sum(active_mask),
            'pattern_performance': {
                f'pattern_{i}': np.mean(
                    predictions['signal'][patterns['cluster_labels'] == i] == 
                    self.labels[patterns['cluster_labels'] == i]
                )
                for i in range(len(patterns['patterns']))
            }
        }


def main():
    """
    Complete unconstrained training pipeline
    """
    print("=" * 80)
    print("🚀 UNCONSTRAINED MARKET LEARNER")
    print("   The AI discovers its own market understanding")
    print("=" * 80)
    
    # Initialize pipeline
    pipeline = UnconstrainedTrainingPipeline()
    
    # 1. Load raw data (NO feature engineering)
    print("\n📥 Loading raw market data...")
    data = pipeline.prepare_raw_data("data/raw/combined_multi_horizon_1h.parquet")
    
    # 2. Create simple ground truth (only for evaluation)
    print("\n🏷️ Creating ground truth labels...")
    labels = pipeline.create_ground_truth_labels(data)
    
    # 3. Phase 1: Unsupervised pretraining
    print("\n🧠 Phase 1: Let AI learn market structure unsupervised...")
    unsupervised_history = pipeline.train_unsupervised(epochs=50)
    
    # 4. Phase 2: Pattern discovery
    print("\n🔍 Phase 2: AI discovers its own patterns...")
    discovered_patterns = pipeline.discover_patterns()
    
    # 5. Phase 3: Supervised fine-tuning
    print("\n🎯 Phase 3: AI learns to predict using its own features...")
    supervised_history = pipeline.train_supervised(epochs=30)
    
    # 6. Evaluation
    print("\n📊 Final evaluation...")
    results = pipeline.evaluate_unconstrained()
    
    print("\n" + "=" * 80)
    print("🎉 UNCONSTRAINED LEARNING COMPLETE!")
    print("=" * 80)
    print(f"📊 Final Results:")
    print(f"   Overall Accuracy: {results['accuracy']:.2%}")
    print(f"   Active Signal Win Rate: {results['active_win_rate']:.2%}")
    print(f"   Active Signals: {results['active_signals']}")
    print("=" * 80)


if __name__ == "__main__":
    main()