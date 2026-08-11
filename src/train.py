"""
Training pipeline
"""
import numpy as np
import joblib
from sklearn.preprocessing import RobustScaler, PowerTransformer
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from src.model import SmartTraderModel
from src.config import config
import logging

logger = logging.getLogger(__name__)

class Trainer:
    """Model training pipeline"""
    
    def __init__(self, X_train, y_train, X_val, y_val):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.scaler = RobustScaler()
        self.power_transformer = PowerTransformer(method='yeo-johnson')
        self.model = None
    
    def preprocess_features(self):
        """Scale and transform features"""
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        X_val_scaled = self.scaler.transform(self.X_val)
        
        X_train_transformed = self.power_transformer.fit_transform(X_train_scaled)
        X_val_transformed = self.power_transformer.transform(X_val_scaled)
        
        return X_train_transformed, X_val_transformed
    
    def build_model(self):
        """Build and compile model"""
        input_shape = self.X_train.shape[1]
        n_features = self.X_train.shape[1]
        
        self.model = SmartTraderModel(input_shape, n_features)
        self.model.compile_model()
        
        return self.model
    
    def train(self, epochs=100, batch_size=1024):
        """Train the model"""
        X_train, X_val = self.preprocess_features()
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_lr=1e-6),
            ModelCheckpoint(
                str(config.MODELS_PATH / 'smart_trader_ai_best.keras'),
                monitor='val_direction_1h_sparse_categorical_accuracy',
                mode='max', save_best_only=True
            )
        ]
        
        history = self.model.model.fit(
            X_train, self.y_train,
            validation_data=(X_val, self.y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save scalers
        joblib.dump(self.scaler, config.MODELS_PATH / 'robust_scaler.joblib')
        joblib.dump(self.power_transformer, config.MODELS_PATH / 'power_transformer.joblib')
        
        return history
    
    def save_artifacts(self):
        """Save all artifacts"""
        # Save model with absolute path
        model_path = config.MODELS_PATH / 'smart_trader_ai_final.keras'
        self.model.save(str(model_path))
        logger.info(f"Model saved to {model_path}")
        
        # Save scalers
        joblib.dump(self.scaler, config.MODELS_PATH / 'robust_scaler.joblib')
        joblib.dump(self.power_transformer, config.MODELS_PATH / 'power_transformer.joblib')
        
        # Save features
        joblib.dump(self.X_train.columns.tolist(), config.MODELS_PATH / 'feature_columns.joblib')