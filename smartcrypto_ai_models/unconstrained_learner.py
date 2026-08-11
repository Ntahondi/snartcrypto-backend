"""
Unconstrained Market Learner - Learns market structure without human features
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class UnconstrainedMarketLearner:
    """
    Learns market structure through self-supervision without human features
    """
    
    def __init__(self, sequence_length: int = 48, latent_dim: int = 64):
        self.sequence_length = sequence_length
        self.latent_dim = latent_dim
        self.encoder = None
        self.decoder = None
        self.predictor = None
        
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 1. CONTRASTIVE LEARNING (SimCLR Style)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    def build_contrastive_learner(self, input_shape: Tuple) -> models.Model:
        """
        Build a contrastive learning model that learns by comparing
        different views of the same market data
        """
        
        # Input: Raw OHLCV data (5 channels)
        inputs = layers.Input(shape=input_shape)
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # ENCODER: Learns representations without human bias
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        # Raw price normalization (learned)
        x = layers.BatchNormalization()(inputs)
        
        # Multi-scale temporal convolutions (learns its own timeframes)
        conv_3 = layers.Conv1D(32, 3, padding='same', activation='swish')(x)
        conv_7 = layers.Conv1D(32, 7, padding='same', activation='swish')(x)
        conv_15 = layers.Conv1D(32, 15, padding='same', activation='swish')(x)
        
        # Let model learn which timeframes matter
        concatenated = layers.Concatenate()([conv_3, conv_7, conv_15])
        
        # Self-attention to discover relationships
        attn = layers.MultiHeadAttention(num_heads=4, key_dim=16)(concatenated, concatenated)
        attn = layers.Add()([concatenated, attn])
        attn = layers.LayerNormalization()(attn)
        
        # Learn hierarchical features (unsupervised)
        x = layers.Conv1D(64, 3, padding='same', activation='swish')(attn)
        x = layers.MaxPooling1D(2)(x)
        
        x = layers.Conv1D(128, 3, padding='same', activation='swish')(x)
        x = layers.GlobalAveragePooling1D()(x)
        
        # Latent representation (the model's own understanding)
        latent = layers.Dense(self.latent_dim, activation='swish', name='latent')(x)
        latent = layers.BatchNormalization()(latent)
        
        # Projection head for contrastive learning
        projection = layers.Dense(128, activation='swish')(latent)
        projection = layers.Dense(64, activation='swish')(projection)
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # DECODER: Reconstruct original data (self-supervision)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        # Decode from latent
        d = layers.Dense(128, activation='swish')(latent)
        d = layers.Dense(256, activation='swish')(d)
        d = layers.Dense(512, activation='swish')(d)
        d = layers.Dense(np.prod(input_shape), activation='linear')(d)
        
        reconstruction = layers.Reshape(input_shape)(d)
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # BUILD MODEL
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        # Contrastive learning uses two augmented views
        # We'll handle this in the training loop
        
        self.encoder = models.Model(inputs, latent, name='encoder')
        self.decoder = models.Model(inputs, reconstruction, name='decoder')
        
        return models.Model(
            inputs, 
            [projection, reconstruction], 
            name='unconstrained_learner'
        )
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 2. PREDICTIVE MODEL (Learns from its own representations)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    def build_predictor(self, latent_dim: int = 64) -> models.Model:
        """
        Build a predictor that uses the learned representations
        """
        latent_input = layers.Input(shape=(latent_dim,))
        
        # Let the model discover its own patterns
        x = layers.Dense(128, activation='swish')(latent_input)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(64, activation='swish')(x)
        x = layers.Dropout(0.2)(x)
        
        # Discovered patterns output
        pattern_embedding = layers.Dense(32, activation='swish', name='discovered_pattern')(x)
        
        # Signal output (BUY/HOLD/SELL)
        signal = layers.Dense(3, activation='softmax', name='signal')(pattern_embedding)
        
        # Expected movement (learned from data)
        movement = layers.Dense(1, activation='linear', name='expected_movement')(pattern_embedding)
        
        # Confidence (learned calibration)
        confidence = layers.Dense(1, activation='sigmoid', name='confidence')(pattern_embedding)
        
        self.predictor = models.Model(
            latent_input,
            [signal, movement, confidence],
            name='discovered_predictor'
        )
        
        return self.predictor
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 3. SELF-SUPERVISED TRAINING
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    def pretrain_unsupervised(self, data: np.ndarray, epochs: int = 100):
        """
        Pretrain on raw data without labels
        """
        logger.info("🧠 Starting unsupervised pretraining...")
        
        # Build contrastive model
        input_shape = (self.sequence_length, data.shape[-1])
        model = self.build_contrastive_learner(input_shape)
        
        # Use both reconstruction and contrastive losses
        def contrastive_loss(projections, temperature=0.1):
            """Contrastive loss for unlabeled data"""
            # Normalize projections
            projections = K.l2_normalize(projections, axis=1)
            
            # Compute similarity matrix
            similarity = K.dot(projections, K.transpose(projections))
            
            # Mask diagonal (self-similarity)
            batch_size = K.shape(projections)[0]
            mask = 1 - K.eye(batch_size)
            similarity = similarity * mask
            
            # Temperature scaling
            similarity = similarity / temperature
            
            # Compute loss (positive pairs are augmented views)
            # We'll use a simplified version here
            exp_sim = K.exp(similarity)
            sum_exp = K.sum(exp_sim, axis=1, keepdims=True)
            
            # Positive pairs are diagonal of augmented views
            # For simplicity, we'll use all pairs as positive
            loss = -K.log(exp_sim / (sum_exp + K.epsilon()))
            
            return K.mean(loss)
        
        # Reconstruction loss
        def reconstruction_loss(y_true, y_pred):
            return K.mean(K.square(y_true - y_pred))
        
        # Compile with combined loss
        model.compile(
            optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-4),
            loss=[contrastive_loss, reconstruction_loss],
            loss_weights=[0.5, 0.5]
        )
        
        # Data augmentation for contrastive learning
        def augment_data(x):
            # Noise augmentation
            noise = tf.random.normal(shape=tf.shape(x), stddev=0.01)
            x_aug = x + noise
            
            # Random scaling
            scale = 1.0 + tf.random.uniform([], -0.05, 0.05)
            x_aug = x_aug * scale
            
            return x_aug
        
        # Training
        history = model.fit(
            data,
            [data, data],  # Reconstruction target is same data
            epochs=epochs,
            batch_size=256,
            validation_split=0.2,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=10),
                tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
            ],
            verbose=1
        )
        
        logger.info("✅ Unsupervised pretraining complete!")
        
        # Save encoder
        self.encoder.save('discovered_encoder.keras')
        
        return history
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 4. DISCOVERY & PATTERN EXTRACTION
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    def discover_market_patterns(self, data: np.ndarray) -> Dict:
        """
        Let the model discover its own patterns in the data
        """
        logger.info("🔍 Discovering market patterns...")
        
        # Get latent representations
        latent_reps = self.encoder.predict(data)
        
        # Discover patterns using the model's own representations
        # Use clustering to find natural market states
        from sklearn.cluster import KMeans
        
        # Find optimal number of clusters
        inertias = []
        for k in range(2, 15):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(latent_reps)
            inertias.append(kmeans.inertia_)
        
        # Use elbow method to find natural number of market regimes
        from kneed import KneeLocator
        knee = KneeLocator(range(2, 15), inertias, curve='convex', direction='decreasing')
        optimal_k = knee.knee + 2
        
        # Cluster discovered patterns
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(latent_reps)
        
        # Analyze discovered patterns
        discovered_patterns = {}
        for i in range(optimal_k):
            cluster_mask = clusters == i
            cluster_data = data[cluster_mask]
            
            if len(cluster_data) > 0:
                # Extract pattern characteristics (model's own discovery)
                pattern = {
                    'size': len(cluster_data),
                    'center': kmeans.cluster_centers_[i].tolist(),
                    'latent_center': latent_reps[cluster_mask].mean(axis=0).tolist()
                }
                
                # The model discovers what this pattern means
                discovered_patterns[f'pattern_{i}'] = pattern
        
        logger.info(f"✅ Discovered {optimal_k} natural market patterns")
        
        return {
            'patterns': discovered_patterns,
            'cluster_labels': clusters,
            'latent_representations': latent_reps
        }
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 5. PREDICT WITH DISCOVERED KNOWLEDGE
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    def predict_unconstrained(self, data: np.ndarray) -> Dict:
        """
        Make predictions using the model's own discovered knowledge
        """
        # Get latent representation
        latent = self.encoder.predict(data)
        
        # Get predictions
        signal, movement, confidence = self.predictor.predict(latent)
        
        # Convert to trading signal
        signal_class = np.argmax(signal, axis=1)
        confidence_score = confidence.flatten()
        
        return {
            'signal': signal_class,
            'signal_probs': signal,
            'expected_movement': movement.flatten(),
            'confidence': confidence_score,
            'latent_state': latent
        }