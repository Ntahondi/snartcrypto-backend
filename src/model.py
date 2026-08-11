"""
Multi-head neural network model
"""
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import logging
from src.config import config

logger = logging.getLogger(__name__)

class SmartTraderModel:
    """Multi-head neural network for trading"""
    
    def __init__(self, input_shape, n_features, learning_rate=0.001):
        self.input_shape = input_shape
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.model = None
        self._build_model()
    
    def _build_model(self):
        """Build the multi-head architecture"""
        inputs = layers.Input(shape=(self.input_shape,), name='market_features')
        
        # Feature attention
        attention_weights = layers.Dense(
            self.n_features, 
            activation='softmax', 
            name='feature_attention'
        )(inputs)
        weighted_features = layers.Multiply()([inputs, attention_weights])
        
        # Shared representation
        x = layers.Dense(512, activation='swish')(weighted_features)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(256, activation='swish')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.25)(x)
        
        shared = layers.Dense(64, activation='swish')(x)
        
        # Directional heads
        def create_dir_head(name):
            h = layers.Dense(32, activation='swish')(shared)
            h = layers.Dropout(0.2)(h)
            return layers.Dense(3, activation='softmax', name=name)(h)
        
        # Auxiliary heads
        conf_output = layers.Dense(1, activation='sigmoid', name='confidence_score')(
            layers.Dense(16, activation='swish')(shared)
        )
        risk_output = layers.Dense(3, activation='softmax', name='risk_level')(
            layers.Dense(16, activation='swish')(shared)
        )
        regime_output = layers.Dense(4, activation='softmax', name='market_regime')(
            layers.Dense(16, activation='swish')(shared)
        )
        
        self.model = models.Model(
            inputs=inputs,
            outputs=[
                create_dir_head('direction_1h'),
                create_dir_head('direction_4h'),
                create_dir_head('direction_1d'),
                conf_output,
                risk_output,
                regime_output
            ],
            name='SmartTraderAI'
        )
        
        return self.model
    
    def compile_model(self):
        """Compile the model with loss weights"""
        self.model.compile(
            optimizer=optimizers.AdamW(learning_rate=self.learning_rate, weight_decay=0.01),
            loss={
                'direction_1h': 'sparse_categorical_crossentropy',
                'direction_4h': 'sparse_categorical_crossentropy',
                'direction_1d': 'sparse_categorical_crossentropy',
                'confidence_score': 'mse',
                'risk_level': 'sparse_categorical_crossentropy',
                'market_regime': 'sparse_categorical_crossentropy'
            },
            loss_weights={
                'direction_1h': 0.25,
                'direction_4h': 0.25,
                'direction_1d': 0.25,
                'confidence_score': 0.1,
                'risk_level': 0.075,
                'market_regime': 0.075
            },
            metrics={
                'direction_1h': 'sparse_categorical_accuracy',
                'direction_4h': 'sparse_categorical_accuracy',
                'direction_1d': 'sparse_categorical_accuracy',
                'risk_level': 'sparse_categorical_accuracy',
                'market_regime': 'sparse_categorical_accuracy',
                'confidence_score': 'mae'
            }
        )
        return self.model
    
    def summary(self):
        """Print model summary"""
        return self.model.summary()
    
    def save(self, path):
        """Save model"""
        path = str(path)
        self.model.save(path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path):
        """Load model"""
        self.model = models.load_model(path)
        logger.info(f"Model loaded from {path}")
        return self.model