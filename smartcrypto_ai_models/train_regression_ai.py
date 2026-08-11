# smartcrypto_ai_models/train_regression_ai.py

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import logging

from smartcrypto_ai_models.candle_microstructure import UnconstrainedCandleExtractor
from smartcrypto_ai_models.regression_targets import ContinuousRegressionLabeler
from smartcrypto_ai_models.regression_model import ContinuousRegressionAIModelBuilder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_regression_sequences(features: np.ndarray, y_1h: np.ndarray, y_4h: np.ndarray, y_12h: np.ndarray, sequence_length: int = 48):
    """Reshapes 2D feature array into 3D sequence tensors for regression"""
    num_samples = len(features) - sequence_length - 12
    num_features = features.shape[1]

    X_seq = np.zeros((num_samples, sequence_length, num_features), dtype=np.float32)
    y1 = y_1h[sequence_length:sequence_length + num_samples]
    y4 = y_4h[sequence_length:sequence_length + num_samples]
    y12 = y_12h[sequence_length:sequence_length + num_samples]

    for i in range(num_samples):
        X_seq[i] = features[i:i + sequence_length]

    return X_seq, y1, y4, y12


def main():
    logger.info("=" * 80)
    logger.info("🚀 STARTING UNCONSTRAINED CONTINUOUS RETURN REGRESSION AI TRAINING")
    logger.info("=" * 80)

    parquet_path = "data/raw/combined_multi_horizon_1h.parquet"
    if not os.path.exists(parquet_path):
        logger.error(f"❌ Parquet file not found at {parquet_path}")
        return

    # 1. Load Master Dataset
    df_raw = pd.read_parquet(parquet_path)
    logger.info(f"📥 Step 1: Loaded {len(df_raw):,} records across {df_raw['symbol'].nunique()} symbols.")

    # 2. Extract 21-Channel Features
    extractor = UnconstrainedCandleExtractor()
    df_featured = extractor.extract_features(df_raw)
    feature_cols = extractor.get_feature_columns()

    # 3. Continuous Regression Target Labeling
    logger.info("📈 Step 2: Generating pure continuous percentage return targets...")
    labeler = ContinuousRegressionLabeler(horizons={'1h': 1, '4h': 4, '12h': 12})
    df_labeled = labeler.label_dataset(df_featured)

    # 4. Construct 48-Hour Sequence Matrices per Symbol
    logger.info("🧱 Step 3: Constructing 48-hour sequence matrices with per-symbol time split...")
    scaler = RobustScaler()

    X_train_list, X_test_list = [], []
    y1_tr_list, y1_te_list = [], []
    y4_tr_list, y4_te_list = [], []
    y12_tr_list, y12_te_list = [], []

    sequence_length = 48

    for symbol in df_labeled['symbol'].unique():
        s_df = df_labeled[df_labeled['symbol'] == symbol].sort_values('timestamp').reset_index(drop=True)
        if len(s_df) < 200:
            continue

        feat_matrix = s_df[feature_cols].values
        feat_scaled = scaler.fit_transform(feat_matrix)

        y_1h_vals = s_df['target_ret_1h'].values
        y_4h_vals = s_df['target_ret_4h'].values
        y_12h_vals = s_df['target_ret_12h'].values

        X_s, y1_s, y4_s, y12_s = create_regression_sequences(
            feat_scaled, y_1h_vals, y_4h_vals, y_12h_vals, sequence_length=sequence_length
        )

        s_split = int(len(X_s) * 0.8)

        X_train_list.append(X_s[:s_split])
        X_test_list.append(X_s[s_split:])

        y1_tr_list.append(y1_s[:s_split]); y1_te_list.append(y1_s[s_split:])
        y4_tr_list.append(y4_s[:s_split]); y4_te_list.append(y4_s[s_split:])
        y12_tr_list.append(y12_s[:s_split]); y12_te_list.append(y12_s[s_split:])

    X_train = np.concatenate(X_train_list, axis=0)
    X_test = np.concatenate(X_test_list, axis=0)

    y1_train, y1_test = np.concatenate(y1_tr_list, axis=0), np.concatenate(y1_te_list, axis=0)
    y4_train, y4_test = np.concatenate(y4_tr_list, axis=0), np.concatenate(y4_te_list, axis=0)
    y12_train, y12_test = np.concatenate(y12_tr_list, axis=0), np.concatenate(y12_te_list, axis=0)

    logger.info(f"   Train Tensors: {X_train.shape} | Test Tensors: {X_test.shape}")

    # 5. Build Regression Model with Huber Loss
    logger.info("🧠 Step 4: Building ResNet-1D + Attention Regression Neural Network...")
    model = ContinuousRegressionAIModelBuilder.build_model(sequence_length=sequence_length, num_features=len(feature_cols))

    optimizer = tf.keras.optimizers.AdamW(learning_rate=0.00015, weight_decay=0.03)
    
    # Huber Loss protects against extreme liquidation wicks
    model.compile(
        optimizer=optimizer,
        loss={
            'ret_1h_head': tf.keras.losses.Huber(delta=0.01),
            'ret_4h_head': tf.keras.losses.Huber(delta=0.02),
            'ret_12h_head': tf.keras.losses.Huber(delta=0.04)
        },
        loss_weights={'ret_1h_head': 0.3, 'ret_4h_head': 0.3, 'ret_12h_head': 0.4},
        metrics={'ret_1h_head': ['mae'], 'ret_4h_head': ['mae'], 'ret_12h_head': ['mae']}
    )

    save_dir = "smartcrypto_ai_models"
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "continuous_regression_ai.keras")
    scaler_path = os.path.join(save_dir, "regression_scaler.joblib")

    callbacks = [
        EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.5, patience=4, min_lr=1e-6, verbose=1),
        ModelCheckpoint(model_path, monitor='val_loss', mode='min', save_best_only=True, verbose=1)
    ]

    # 6. Train Regression Model
    logger.info("🏋️ Step 5: Training Continuous Return Regression AI...")
    history = model.fit(
        X_train,
        {'ret_1h_head': y1_train, 'ret_4h_head': y4_train, 'ret_12h_head': y12_train},
        validation_data=(X_test, {'ret_1h_head': y1_test, 'ret_4h_head': y4_test, 'ret_12h_head': y12_test}),
        epochs=30,
        batch_size=256,
        callbacks=callbacks,
        verbose=1
    )

    # Save scaler
    joblib.dump(scaler, scaler_path)
    joblib.dump(feature_cols, os.path.join(save_dir, "regression_features.joblib"))

    # 7. Evaluate Performance
    logger.info("📊 Step 6: Evaluating on unseen validation test data...")
    predictions = model.predict(X_test, verbose=0)
    
    mae_1h = mean_absolute_error(y1_test, predictions[0])
    mae_4h = mean_absolute_error(y4_test, predictions[1])
    mae_12h = mean_absolute_error(y12_test, predictions[2])

    logger.info("=" * 80)
    logger.info("🎉 SUCCESS! Continuous Return Regression AI Trained & Saved!")
    logger.info(f"   Model Location: {model_path}")
    logger.info(f"   1H Return Prediction MAE:  {mae_1h:.4%}")
    logger.info(f"   4H Return Prediction MAE:  {mae_4h:.4%}")
    logger.info(f"   12H Return Prediction MAE: {mae_12h:.4%}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()