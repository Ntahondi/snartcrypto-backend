# smartcrypto_ai_models/conv1d_attention_model.py

import tensorflow as tf
from tensorflow.keras import layers, models

# Universal Keras 2/3 decorator import
try:
    from keras.saving import register_keras_serializable
except ImportError:
    try:
        from tensorflow.keras.utils import register_keras_serializable
    except ImportError:
        from keras.utils import register_keras_serializable


@register_keras_serializable(package="smartcrypto_ai_models")
class ResNetBlock1D(layers.Layer):
    """1D Residual Conv Block with Skip Connection"""
    def __init__(self, filters: int = 64, kernel_size: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.conv1 = layers.Conv1D(filters, kernel_size, padding='same', activation='swish')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv1D(filters, kernel_size, padding='same')
        self.bn2 = layers.BatchNormalization()
        self.shortcut = layers.Conv1D(filters, 1, padding='same')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        shortcut = self.shortcut(inputs)
        return tf.nn.swish(layers.add([x, shortcut]))

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
        })
        return config


class UnconstrainedAIModelBuilder:
    @staticmethod
    def build_model(sequence_length: int = 48, num_features: int = 21) -> models.Model:
        inputs = layers.Input(shape=(sequence_length, num_features), name='unconstrained_48h_matrix')

        x = layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='swish')(inputs)
        x = layers.BatchNormalization()(x)

        x = ResNetBlock1D(filters=64)(x)
        x = ResNetBlock1D(filters=128)(x)

        attn_out = layers.MultiHeadAttention(num_heads=8, key_dim=32)(x, x)
        added = layers.Add()([x, attn_out])
        norm_attn = layers.LayerNormalization()(added)

        gap = layers.GlobalAveragePooling1D()(norm_attn)
        gmp = layers.GlobalMaxPooling1D()(norm_attn)
        pooled = layers.Concatenate()([gap, gmp])

        dense_core = layers.Dense(256, activation='swish')(pooled)
        dense_core = layers.BatchNormalization()(dense_core)
        dense_core = layers.Dropout(0.3)(dense_core)

        shared_repr = layers.Dense(128, activation='swish', name='unconstrained_pattern_core')(dense_core)

        dir_h = layers.Dense(64, activation='swish')(shared_repr)
        dir_h = layers.Dropout(0.2)(dir_h)
        dir_output = layers.Dense(3, activation='softmax', name='direction_head')(dir_h)

        time_h = layers.Dense(32, activation='swish')(shared_repr)
        time_output = layers.Dense(1, activation='relu', name='time_to_profit_head')(time_h)

        model = models.Model(
            inputs=inputs,
            outputs=[dir_output, time_output],
            name='UnconstrainedCandleAI_v2'
        )
        return model