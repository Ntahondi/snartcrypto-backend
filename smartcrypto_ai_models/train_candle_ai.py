# smartcrypto_ai_models/train_candle_ai.py

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score
import joblib
import logging

from smartcrypto_ai_models.candle_microstructure import UnconstrainedCandleExtractor
from smartcrypto_ai_models.triple_barrier_targets import TripleBarrierTargetLabeler
from smartcrypto_ai_models.conv1d_attention_model import UnconstrainedAIModelBuilder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_48h_sequences(features: np.ndarray, directions: np.ndarray, times: np.ndarray, sequence_length: int = 48):
    """Reshapes 2D feature array into 48-hour 3D sequence tensors"""
    num_samples = len(features) - sequence_length
    num_features = features.shape[1]

    X_seq = np.zeros((num_samples, sequence_length, num_features), dtype=np.float32)
    y_dir = directions[sequence_length:]
    y_time = times[sequence_length:]

    for i in range(num_samples):
        X_seq[i] = features[i:i + sequence_length]

    return X_seq, y_dir, y_time


def main():
    logger.info("=" * 80)
    logger.info("🚀 STARTING REGULARIZED RESNET-1D + ATTENTION AI TRAINING")
    logger.info("=" * 80)

    parquet_path = "data/raw/combined_multi_horizon_1h.parquet"
    if not os.path.exists(parquet_path):
        logger.error(f"❌ Master Parquet file not found at {parquet_path}")
        return

    # 1. Load Master Dataset
    df_raw = pd.read_parquet(parquet_path)
    logger.info(f"📥 Step 1: Loaded {len(df_raw):,} records across {df_raw['symbol'].nunique()} symbols.")

    # 2. Extract Unconstrained 21-Channel Matrix
    extractor = UnconstrainedCandleExtractor()
    df_featured = extractor.extract_features(df_raw)
    feature_cols = extractor.get_feature_columns()

    # 3. Path-Dependent Triple Barrier Target Labeling (2.0x ATR Barriers)
    labeler = TripleBarrierTargetLabeler(tp_atr_mult=2.0, sl_atr_mult=2.0, max_holding_bars=12)
    df_labeled = labeler.label_dataset(df_featured)

    # 4. Per-Symbol Chronological Train / Test Splitting
    logger.info("🧱 Step 4: Constructing 48-hour sequence matrices with per-symbol time split...")
    scaler = RobustScaler()

    X_train_list, X_test_list = [], []
    y_dir_train_list, y_dir_test_list = [], []
    y_time_train_list, y_time_test_list = [], []

    sequence_length = 48

    for symbol in df_labeled['symbol'].unique():
        symbol_df = df_labeled[df_labeled['symbol'] == symbol].sort_values('timestamp').reset_index(drop=True)
        if len(symbol_df) < 200:
            continue

        feat_matrix = symbol_df[feature_cols].values
        feat_scaled = scaler.fit_transform(feat_matrix)

        dirs = symbol_df['target_direction'].values
        times = symbol_df['time_to_profit_hours'].values

        X_s, y_d_s, y_t_s = create_48h_sequences(feat_scaled, dirs, times, sequence_length=sequence_length)

        # Per-symbol chronological split (80% Train, 20% Test)
        s_split = int(len(X_s) * 0.8)

        X_train_list.append(X_s[:s_split])
        X_test_list.append(X_s[s_split:])

        y_dir_train_list.append(y_d_s[:s_split])
        y_dir_test_list.append(y_d_s[s_split:])

        y_time_train_list.append(y_t_s[:s_split])
        y_time_test_list.append(y_t_s[s_split:])

    X_train = np.concatenate(X_train_list, axis=0)
    X_test = np.concatenate(X_test_list, axis=0)

    y_dir_train = np.concatenate(y_dir_train_list, axis=0)
    y_dir_test = np.concatenate(y_dir_test_list, axis=0)

    y_time_train = np.concatenate(y_time_train_list, axis=0)
    y_time_test = np.concatenate(y_time_test_list, axis=0)

    logger.info(f"   Train Tensors: {X_train.shape} | Test Tensors: {X_test.shape}")

    # 5. Build ResNet-1D + Attention Model
    model = UnconstrainedAIModelBuilder.build_model(sequence_length=sequence_length, num_features=len(feature_cols))

    optimizer = tf.keras.optimizers.AdamW(learning_rate=0.00015, weight_decay=0.03)
    
    # FIX A: Use ordered lists matching output 0 (direction_head) and output 1 (time_to_profit_head)
    model.compile(
        optimizer=optimizer,
        loss=['sparse_categorical_crossentropy', 'mae'],
        loss_weights=[0.8, 0.2],
        metrics=[['accuracy'], ['mae']]
    )

    save_dir = "smartcrypto_ai_models"
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "unconstrained_candle_ai.keras")
    scaler_path = os.path.join(save_dir, "unconstrained_scaler.joblib")

    callbacks = [
        EarlyStopping(monitor='val_compile_metrics', mode='min', patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.5, patience=4, min_lr=1e-6, verbose=1),
        ModelCheckpoint(model_path, monitor='val_loss', mode='min', save_best_only=True, verbose=1)
    ]

    # FIX B: Calculate sample weights for output 0 (direction_head)
    classes = np.unique(y_dir_train)
    cw = compute_class_weight(class_weight='balanced', classes=classes, y=y_dir_train)
    cw_map = {int(c): float(cw[i]) for i, c in enumerate(classes)}
    
    sample_weights_direction = np.array([cw_map[y] for y in y_dir_train], dtype=np.float32)
    sample_weights_time = np.ones_like(y_time_train, dtype=np.float32)

    logger.info(f"⚖️ Applied Keras 3 Sample Weights for Direction Head: {cw_map}")

    # 6. Train Model using ordered lists
    history = model.fit(
        X_train,
        [y_dir_train, y_time_train],
        sample_weight=[sample_weights_direction, sample_weights_time],
        validation_data=(X_test, [y_dir_test, y_time_test]),
        epochs=30,
        batch_size=256,
        callbacks=callbacks,
        verbose=1
    )

    # Save scaler & feature metadata
    joblib.dump(scaler, scaler_path)
    joblib.dump(feature_cols, os.path.join(save_dir, "unconstrained_features.joblib"))

    # 7. Evaluate Accuracy
    logger.info("📊 Step 7: Evaluating on unseen validation test data...")
    predictions = model.predict(X_test, verbose=0)
    pred_dirs = np.argmax(predictions[0], axis=1)
    acc = accuracy_score(y_dir_test, pred_dirs)

    logger.info("=" * 80)
    logger.info(f"🎉 SUCCESS! Regularized Candle AI Trained & Saved!")
    logger.info(f"   Model Location: {model_path}")
    logger.info(f"   3-Class Validation Accuracy: {acc:.2%} (vs 33.3% Random Chance)")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()