# smartcrypto_ai_models/train_regression_ai.py

import sys
import os
sys.path.insert(0, os.path.abspath("."))

import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import logging

from smartcrypto_ai_models.candle_microstructure import UnconstrainedCandleExtractor
from smartcrypto_ai_models.regression_targets import ContinuousRegressionLabeler, validate_targets, trim_tail_nans
from smartcrypto_ai_models.regression_model import (
    ContinuousRegressionAIModelBuilder,
    DirectionalHuberLoss,
    directional_accuracy
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_symbol_sequences(
    df: pd.DataFrame,
    feature_cols: list,
    sequence_length: int = 48,
):
    """
    Generate 3D sequences PER SYMBOL to prevent cross-symbol contamination.
    Target index corresponds strictly to the last candle of the input window.
    """
    X_list, y1_list, y4_list, y12_list = [], [], [], []
    
    for symbol, group in df.groupby('symbol', sort=False):
        group = group.sort_values('timestamp').reset_index(drop=True)
        
        feats = group[feature_cols].values.astype(np.float32)
        y1 = group['target_ret_1h'].values.astype(np.float32)
        y4 = group['target_ret_4h'].values.astype(np.float32)
        y12 = group['target_ret_12h'].values.astype(np.float32)
        
        num_samples = len(group) - sequence_length
        if num_samples <= 0:
            logger.warning(f"Symbol {symbol} has insufficient data: {len(group)} rows")
            continue
        
        for i in range(num_samples):
            X_list.append(feats[i : i + sequence_length])
            target_idx = i + sequence_length - 1
            y1_list.append(y1[target_idx])
            y4_list.append(y4[target_idx])
            y12_list.append(y12[target_idx])
        
        logger.info(f"   {symbol}: Generated {num_samples:,} sequences")
    
    if not X_list:
        raise RuntimeError("No sequences generated. Check data lengths.")
    
    X = np.array(X_list, dtype=np.float32)
    y1 = np.array(y1_list, dtype=np.float32)
    y4 = np.array(y4_list, dtype=np.float32)
    y12 = np.array(y12_list, dtype=np.float32)
    
    logger.info(f"   Total sequences: {len(X):,} | Shape: {X.shape}")
    return X, y1, y4, y12


def make_tf_dataset(X, y1, y4, y12, batch_size: int = 256, is_training: bool = True):
    dataset = tf.data.Dataset.from_tensor_slices(
        (X, {'ret_1h_head': y1, 'ret_4h_head': y4, 'ret_12h_head': y12})
    )
    if is_training:
        buffer_size = min(len(X), 50000)
        dataset = dataset.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)
    
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def main():
    logger.info("=" * 80)
    logger.info("🚀 STARTING RETRAINING: CONTINUOUS RETURN REGRESSION AI (v2 WITH DIRECTIONAL LOSS)")
    logger.info("=" * 80)

    parquet_path = "data/raw/combined_multi_horizon_1h.parquet"
    if not os.path.exists(parquet_path):
        logger.error(f"❌ Parquet file not found at {parquet_path}")
        return

    # 1. Load Master Dataset
    df_raw = pd.read_parquet(parquet_path)
    logger.info(f"📥 Step 1: Loaded {len(df_raw):,} records across {df_raw['symbol'].nunique()} symbols.")

    # 2. Extract 23-Channel Features (PER SYMBOL)
    logger.info("📊 Step 2: Extracting 23 microstructure features per symbol...")
    extractor = UnconstrainedCandleExtractor()
    df_featured = extractor.extract_features(df_raw)
    feature_cols = extractor.get_feature_columns()
    logger.info(f"   Feature Count: {len(feature_cols)}")

    # 3. Continuous Regression Target Labeling (PER SYMBOL)
    logger.info("📈 Step 3: Generating continuous percentage return targets...")
    labeler = ContinuousRegressionLabeler(horizons={'1h': 1, '4h': 4, '12h': 12})
    df_labeled = labeler.label_dataset(df_featured)

    # 4. Trim NaN targets
    logger.info("✂️ Step 4: Trimming NaN target boundaries...")
    df_trimmed = trim_tail_nans(df_labeled)
    logger.info(f"   Rows after trimming: {len(df_trimmed):,}")

    # 5. Temporal Train/Test Split (80/20 with lookback warmup)
    logger.info("🕐 Step 5: Creating temporal train/test split...")
    df_trimmed = df_trimmed.sort_values('timestamp').reset_index(drop=True)
    sequence_length = 48
    batch_size = 256

    cutoff_timestamp = df_trimmed['timestamp'].quantile(0.80)
    logger.info(f"   Cutoff timestamp: {cutoff_timestamp}")
    
    warmup_timestamp = cutoff_timestamp - pd.Timedelta(hours=sequence_length - 1)
    train_df = df_trimmed[df_trimmed['timestamp'] <= cutoff_timestamp].copy()
    test_df = df_trimmed[df_trimmed['timestamp'] > warmup_timestamp].copy()

    logger.info(f"   Train rows: {len(train_df):,} | Test rows: {len(test_df):,}")

    # 6. Feature Scaling (Fit on Train Only)
    logger.info("📊 Step 6: Scaling input features with RobustScaler...")
    scaler = RobustScaler()
    scaler.fit(train_df[feature_cols].values)
    
    train_df[feature_cols] = scaler.transform(train_df[feature_cols].values)
    test_df[feature_cols] = scaler.transform(test_df[feature_cols].values)

    # 7. Create Sequences PER SYMBOL
    logger.info("🧬 Step 7: Creating sequences per symbol...")
    X_train, y1_tr_raw, y4_tr_raw, y12_tr_raw = create_symbol_sequences(train_df, feature_cols, sequence_length)
    X_test, y1_ts_raw, y4_ts_raw, y12_ts_raw = create_symbol_sequences(test_df, feature_cols, sequence_length)

    # 8. Target Standardization (Z-score Scaling to prevent Mean Collapse)
    logger.info("🎯 Step 8: Standardizing targets to unit variance (Z-Score Scaling)...")
    target_scalers = {
        '1h': {
            'mean': float(np.mean(y1_tr_raw)),
            'std': float(np.std(y1_tr_raw)) if np.std(y1_tr_raw) > 1e-6 else 0.01
        },
        '4h': {
            'mean': float(np.mean(y4_tr_raw)),
            'std': float(np.std(y4_tr_raw)) if np.std(y4_tr_raw) > 1e-6 else 0.02
        },
        '12h': {
            'mean': float(np.mean(y12_tr_raw)),
            'std': float(np.std(y12_tr_raw)) if np.std(y12_tr_raw) > 1e-6 else 0.035
        },
    }
    
    logger.info(f"   1H Target  | Mean: {target_scalers['1h']['mean']:+.6f}, Std: {target_scalers['1h']['std']:.6f}")
    logger.info(f"   4H Target  | Mean: {target_scalers['4h']['mean']:+.6f}, Std: {target_scalers['4h']['std']:.6f}")
    logger.info(f"   12H Target | Mean: {target_scalers['12h']['mean']:+.6f}, Std: {target_scalers['12h']['std']:.6f}")

    # Standardize train targets
    y1_tr = (y1_tr_raw - target_scalers['1h']['mean']) / target_scalers['1h']['std']
    y4_tr = (y4_tr_raw - target_scalers['4h']['mean']) / target_scalers['4h']['std']
    y12_tr = (y12_tr_raw - target_scalers['12h']['mean']) / target_scalers['12h']['std']

    # Standardize test targets with train statistics (No leakage)
    y1_ts = (y1_ts_raw - target_scalers['1h']['mean']) / target_scalers['1h']['std']
    y4_ts = (y4_ts_raw - target_scalers['4h']['mean']) / target_scalers['4h']['std']
    y12_ts = (y12_ts_raw - target_scalers['12h']['mean']) / target_scalers['12h']['std']

    # 9. Create TF Datasets
    logger.info("🧬 Step 9: Creating TF Datasets...")
    train_dataset = make_tf_dataset(X_train, y1_tr, y4_tr, y12_tr, batch_size, is_training=True)
    test_dataset = make_tf_dataset(X_test, y1_ts, y4_ts, y12_ts, batch_size, is_training=False)

    # 10. Build Architecture
    logger.info("🧠 Step 10: Building 1D ResNet + Multi-Head Attention Model...")
    model = ContinuousRegressionAIModelBuilder.build_model(
        sequence_length=sequence_length,
        num_features=len(feature_cols)
    )

    optimizer = tf.keras.optimizers.AdamW(learning_rate=0.0002, weight_decay=0.01)

    # Direction-Aware Compound Loss
    model.compile(
        optimizer=optimizer,
        loss={
            'ret_1h_head': DirectionalHuberLoss(delta=1.0, sign_penalty=0.4, name='loss_1h'),
            'ret_4h_head': DirectionalHuberLoss(delta=1.0, sign_penalty=0.4, name='loss_4h'),
            'ret_12h_head': DirectionalHuberLoss(delta=1.0, sign_penalty=0.5, name='loss_12h'),
        },
        loss_weights={'ret_1h_head': 0.30, 'ret_4h_head': 0.35, 'ret_12h_head': 0.35},
        metrics={
            'ret_1h_head': ['mae', directional_accuracy],
            'ret_4h_head': ['mae', directional_accuracy],
            'ret_12h_head': ['mae', directional_accuracy],
        }
    )

    save_dir = "smartcrypto_ai_models"
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "continuous_regression_ai.keras")
    scaler_path = os.path.join(save_dir, "regression_scaler.joblib")
    target_scaler_path = os.path.join(save_dir, "regression_target_scaler.joblib")
    features_path = os.path.join(save_dir, "regression_features.joblib")

    callbacks = [
        EarlyStopping(monitor='val_loss', mode='min', patience=7, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.5, patience=3, min_lr=1e-6, verbose=1),
        ModelCheckpoint(model_path, monitor='val_loss', mode='min', save_best_only=True, verbose=1)
    ]

    # 11. Train Model
    logger.info("🏋️ Step 11: Training with Directional Huber Loss...")
    history = model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=25,
        callbacks=callbacks,
        verbose=1
    )

    # Save artifacts
    joblib.dump(scaler, scaler_path)
    joblib.dump(target_scalers, target_scaler_path)
    joblib.dump(feature_cols, features_path)
    logger.info("💾 Saved scaler, target scaler, and feature list artifacts.")

    # 12. Evaluate Real Performance on Test Set
    logger.info("📊 Step 12: Evaluating dynamic range and real return metrics on test set...")
    raw_preds = model.predict(test_dataset, verbose=0)

    # De-scale predictions back to actual market percentage returns
    pred_1h_raw = raw_preds[0].flatten() * target_scalers['1h']['std'] + target_scalers['1h']['mean']
    pred_4h_raw = raw_preds[1].flatten() * target_scalers['4h']['std'] + target_scalers['4h']['mean']
    pred_12h_raw = raw_preds[2].flatten() * target_scalers['12h']['std'] + target_scalers['12h']['mean']

    # Metrics
    mae_1h = mean_absolute_error(y1_ts_raw, pred_1h_raw)
    mae_4h = mean_absolute_error(y4_ts_raw, pred_4h_raw)
    mae_12h = mean_absolute_error(y12_ts_raw, pred_12h_raw)

    dir_acc_1h = np.mean(np.sign(y1_ts_raw) == np.sign(pred_1h_raw))
    dir_acc_4h = np.mean(np.sign(y4_ts_raw) == np.sign(pred_4h_raw))
    dir_acc_12h = np.mean(np.sign(y12_ts_raw) == np.sign(pred_12h_raw))

    logger.info("=" * 80)
    logger.info("🎉 SUCCESS! Continuous Return Regression AI (v2) Trained & Verified!")
    logger.info(f"   Model Location: {model_path}")
    logger.info(f"   1H Return  | MAE: {mae_1h:.4%} | Dir Acc: {dir_acc_1h:.2%} | Pred Std: {np.std(pred_1h_raw):.4%} | Range: [{np.min(pred_1h_raw):+.2%}, {np.max(pred_1h_raw):+.2%}]")
    logger.info(f"   4H Return  | MAE: {mae_4h:.4%} | Dir Acc: {dir_acc_4h:.2%} | Pred Std: {np.std(pred_4h_raw):.4%} | Range: [{np.min(pred_4h_raw):+.2%}, {np.max(pred_4h_raw):+.2%}]")
    logger.info(f"   12H Return | MAE: {mae_12h:.4%} | Dir Acc: {dir_acc_12h:.2%} | Pred Std: {np.std(pred_12h_raw):.4%} | Range: [{np.min(pred_12h_raw):+.2%}, {np.max(pred_12h_raw):+.2%}]")
    logger.info("=" * 80)

    # Save training history
    history_path = os.path.join(save_dir, "regression_history.json")
    with open(history_path, 'w') as f:
        json.dump(history.history, f, indent=2)


if __name__ == "__main__":
    main()