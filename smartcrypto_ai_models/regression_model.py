# smartcrypto_ai_models/regression_model.py

import os
import tensorflow as tf
from tensorflow.keras import layers, models

# Universal Keras 2/3 serialization decorator
try:
    from keras.saving import register_keras_serializable
except ImportError:
    try:
        from tensorflow.keras.utils import register_keras_serializable
    except ImportError:
        from keras.utils import register_keras_serializable


@register_keras_serializable(package="smartcrypto_ai_models")
class ResNetBlock1D(layers.Layer):
    """
    1D Residual Block with Projection Shortcut and Swish Activations.
    Designed for high-frequency financial time series feature extraction.
    """
    def __init__(self, filters: int = 64, kernel_size: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.conv1 = layers.Conv1D(filters, kernel_size, padding='same', activation='swish')
        self.ln1 = layers.LayerNormalization()
        self.conv2 = layers.Conv1D(filters, kernel_size, padding='same')
        self.ln2 = layers.LayerNormalization()
        self.shortcut = layers.Conv1D(filters, 1, padding='same')

    def build(self, input_shape):
        self.conv1.build(input_shape)
        self.ln1.build([input_shape[0], input_shape[1], self.filters])
        self.conv2.build([input_shape[0], input_shape[1], self.filters])
        self.ln2.build([input_shape[0], input_shape[1], self.filters])
        self.shortcut.build(input_shape)
        self.built = True
        super().build(input_shape)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.ln1(x)
        x = self.conv2(x)
        x = self.ln2(x)
        shortcut = self.shortcut(inputs)
        return tf.nn.swish(layers.add([x, shortcut]))

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
        })
        return config


@register_keras_serializable(package="smartcrypto_ai_models")
class DirectionalHuberLoss(tf.keras.losses.Loss):
    """
    Direction-Aware Composite Loss Function:
    Combines Huber Loss (magnitude accuracy) with Directional Sign Penalty.
    
    Prevents the neural network from collapsing into the unconditional sample mean
    by heavily penalizing wrong-sign directional forecasts.
    """
    def __init__(self, delta: float = 1.0, sign_penalty: float = 0.5, name: str = "directional_huber_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.delta = float(delta)
        self.sign_penalty = float(sign_penalty)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # 1. Huber magnitude loss
        error = y_true - y_pred
        abs_error = tf.abs(error)
        linear_mask = abs_error > self.delta
        huber = tf.where(
            linear_mask,
            self.delta * (abs_error - 0.5 * self.delta),
            0.5 * tf.square(error)
        )
        
        # 2. Directional sign penalty (non-zero when sign(y_pred) != sign(y_true))
        # -y_true * y_pred is positive whenever signs disagree
        sign_mismatch = tf.maximum(0.0, -y_true * y_pred)
        
        total_loss = huber + self.sign_penalty * sign_mismatch
        return tf.reduce_mean(total_loss)

    def get_config(self):
        config = super().get_config()
        config.update({
            "delta": self.delta,
            "sign_penalty": self.sign_penalty,
        })
        return config


@register_keras_serializable(package="smartcrypto_ai_models")
def directional_accuracy(y_true, y_pred):
    """Computes the percentage of predictions with the correct directional sign."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    correct_direction = tf.equal(tf.sign(y_true), tf.sign(y_pred))
    return tf.reduce_mean(tf.cast(correct_direction, tf.float32))


class ContinuousRegressionAIModelBuilder:
    """
    Builds and loads the 1D ResNet + Multi-Head Self-Attention model
    for multi-horizon continuous crypto return forecasting.
    """

    @staticmethod
    def build_model(sequence_length: int = 48, num_features: int = 23) -> models.Model:
        """
        Constructs the neural architecture:
        - 1D Temporal Convolution Stem
        - 3x ResNet Residual Blocks (64 -> 128 -> 128)
        - Multi-Head Self-Attention with Residual Skip
        - Dual Global Pooling (Avg + Max)
        - Regularized Dense Core with LayerNorm
        - 3 Output Regression Heads (1h, 4h, 12h normalized returns)
        """
        inputs = layers.Input(shape=(sequence_length, num_features), name='candle_48h_matrix')

        # 1. Feature Representation Stem
        x = layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='swish')(inputs)
        x = layers.LayerNormalization()(x)

        # 2. Deep Residual Feature Extraction
        x = ResNetBlock1D(filters=64)(x)
        x = ResNetBlock1D(filters=128)(x)
        x = ResNetBlock1D(filters=128)(x)

        # 3. Temporal Self-Attention Mechanism
        attn_out = layers.MultiHeadAttention(num_heads=8, key_dim=32)(x, x)
        added = layers.Add()([x, attn_out])
        norm_attn = layers.LayerNormalization()(added)

        # 4. Dual Temporal Pooling (Captures both general trend and extreme spike wicks)
        gap = layers.GlobalAveragePooling1D()(norm_attn)
        gmp = layers.GlobalMaxPooling1D()(norm_attn)
        pooled = layers.Concatenate()([gap, gmp])

        # 5. Regularized Regression Core (LayerNorm + Dropout avoids variance shift)
        dense = layers.Dropout(0.25)(pooled)
        dense = layers.Dense(256, activation='swish')(dense)
        dense = layers.LayerNormalization()(dense)
        dense = layers.Dropout(0.25)(dense)

        shared = layers.Dense(128, activation='swish', name='regression_core')(dense)
        shared = layers.LayerNormalization()(shared)
        shared = layers.Dropout(0.15)(shared)

        # 6. Multi-Horizon Output Heads (Normalized Return Predictions)
        ret_1h_out = layers.Dense(1, activation='linear', name='ret_1h_head')(shared)
        ret_4h_out = layers.Dense(1, activation='linear', name='ret_4h_head')(shared)
        ret_12h_out = layers.Dense(1, activation='linear', name='ret_12h_head')(shared)

        model = models.Model(
            inputs=inputs,
            outputs=[ret_1h_out, ret_4h_out, ret_12h_out],
            name='ContinuousRegressionAI_v2'
        )

        model._model_config = {
            "sequence_length": sequence_length,
            "num_features": num_features
        }

        return model

    @staticmethod
    def load_model(model_path: str) -> models.Model:
        """
        Safely load a trained model with custom deserialization.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model artifact not found at: {model_path}")
            
        return models.load_model(
            model_path,
            custom_objects={
                "ResNetBlock1D": ResNetBlock1D,
                "DirectionalHuberLoss": DirectionalHuberLoss,
                "directional_accuracy": directional_accuracy,
            }
        )