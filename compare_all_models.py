"""
AI Model Comparison & Pure Metrics Evaluator
Optimized version with fixed warnings and improved performance
Run: python compare_all_models.py
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, mean_absolute_error

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SUPPRESS WARNINGS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN warnings

# For Windows GPU warning suppression
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU if GPU not available

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CUSTOM IMPORTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
from smartcrypto_ai_models.candle_microstructure import UnconstrainedCandleExtractor
from smartcrypto_ai_models.triple_barrier_targets import TripleBarrierTargetLabeler
from smartcrypto_ai_models.regression_targets import ContinuousRegressionLabeler
from smartcrypto_ai_models.generative_market_gpt import MarketGPTWorldModel
from smartcrypto_ai_models.conv1d_attention_model import ResNetBlock1D


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HELPER FUNCTIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def print_header(text: str, char: str = "="):
    """Print formatted header"""
    print("\n" + char * 80)
    print(f" {text}")
    print(char * 80)


def print_metrics(name: str, metrics: dict, indent: int = 3):
    """Pretty print metrics"""
    indent_str = " " * indent
    for key, value in metrics.items():
        if isinstance(value, float):
            if "accuracy" in key.lower() or "rate" in key.lower():
                print(f"{indent_str}• {key}: {value:.2%}")
            elif "latency" in key.lower() or "speed" in key.lower():
                print(f"{indent_str}• {key}: {value:.3f} ms")
            elif "mae" in key.lower():
                print(f"{indent_str}• {key}: {value:.4%}")
            else:
                print(f"{indent_str}• {key}: {value:.2f}")
        else:
            print(f"{indent_str}• {key}: {value}")


def load_model_with_fallback(model_path: str, custom_objects: dict = None):
    """Load model with fallback options"""
    try:
        if custom_objects:
            return tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        else:
            return tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
        return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN EVALUATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    print_header("📊 SMARTCRYPTO AI MODEL COMPARISON & PURE METRICS EVALUATOR")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 1. LOAD DATA
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    parquet_path = "data/raw/combined_multi_horizon_1h.parquet"
    if not os.path.exists(parquet_path):
        print(f"❌ Parquet file not found at {parquet_path}")
        return

    print("\n📥 Step 1: Loading 7.5-Year Master Dataset...")
    df_raw = pd.read_parquet(parquet_path)
    print(f"   ✅ Loaded {len(df_raw):,} rows")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 2. EXTRACT FEATURES
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("🔧 Step 2: Extracting features...")
    extractor = UnconstrainedCandleExtractor()
    df_featured = extractor.extract_features(df_raw)
    feature_cols = extractor.get_feature_columns()
    print(f"   ✅ Extracted {len(feature_cols)} features")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 3. LABEL TARGETS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("🏷️ Step 3: Labeling targets...")
    
    # Classification targets (Triple Barrier)
    labeler_class = TripleBarrierTargetLabeler(
        tp_atr_mult=2.0, 
        sl_atr_mult=2.0, 
        max_holding_bars=12
    )
    df_class = labeler_class.label_dataset(df_featured)
    
    # Regression targets (Continuous Returns)
    labeler_reg = ContinuousRegressionLabeler(
        horizons={'1h': 1, '4h': 4, '12h': 12}
    )
    df_labeled = labeler_reg.label_dataset(df_class)
    
    print(f"   ✅ Labels generated")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 4. BUILD TEST SET (48-hour sequences)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("🧱 Step 4: Building test sequences (48-hour windows)...")
    
    scaler = RobustScaler()
    sequence_length = 48
    batch_size = 1000  # For memory efficiency

    # Use lists to collect data
    X_test_list, y_dir_test_list = [], []
    y1_test_list, y4_test_list, y12_test_list = [], [], []

    symbols = df_labeled['symbol'].unique()
    
    for idx, symbol in enumerate(symbols):
        print(f"   Processing {symbol} ({idx+1}/{len(symbols)})...")
        
        s_df = df_labeled[df_labeled['symbol'] == symbol].sort_values('timestamp').reset_index(drop=True)
        if len(s_df) < 200:
            continue

        # Scale features
        feat_scaled = scaler.fit_transform(s_df[feature_cols].values)
        
        # Get targets
        dirs = s_df['target_direction'].values
        y1_v = s_df['target_ret_1h'].values
        y4_v = s_df['target_ret_4h'].values
        y12_v = s_df['target_ret_12h'].values

        # Build sequences
        n_samples = len(s_df) - sequence_length - 12
        if n_samples <= 0:
            continue
            
        X_s = np.zeros((n_samples, sequence_length, len(feature_cols)), dtype=np.float32)
        
        for i in range(n_samples):
            X_s[i] = feat_scaled[i:i + sequence_length]

        # Split (80/20)
        s_split = int(len(X_s) * 0.8)
        
        # Store test data
        X_test_list.append(X_s[s_split:])
        y_dir_test_list.append(dirs[sequence_length + s_split:sequence_length + s_split + len(X_s[s_split:])])
        y1_test_list.append(y1_v[sequence_length + s_split:sequence_length + s_split + len(X_s[s_split:])])
        y4_test_list.append(y4_v[sequence_length + s_split:sequence_length + s_split + len(X_s[s_split:])])
        y12_test_list.append(y12_v[sequence_length + s_split:sequence_length + s_split + len(X_s[s_split:])])

    # Concatenate all symbols
    X_test = np.concatenate(X_test_list, axis=0)
    y_dir_test = np.concatenate(y_dir_test_list, axis=0)
    y1_test = np.concatenate(y1_test_list, axis=0)
    y4_test = np.concatenate(y4_test_list, axis=0)
    y12_test = np.concatenate(y12_test_list, axis=0)

    print(f"   ✅ Test set: {len(X_test):,} sequences")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 5. EVALUATE MODELS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    custom_objects = {'ResNetBlock1D': ResNetBlock1D}
    
    # ────────────────────────────────────────────
    # MODEL 1: CONTINUOUS REGRESSION AI
    # ────────────────────────────────────────────
    print_header("🔍 [MODEL 1] Continuous Return Regression AI (continuous_regression_ai.keras)")
    
    reg_path = "smartcrypto_ai_models/continuous_regression_ai.keras"
    if os.path.exists(reg_path):
        model_reg = load_model_with_fallback(reg_path, custom_objects)
        
        if model_reg:
            # Warm up
            _ = model_reg.predict(X_test[:10], verbose=0)
            
            # Benchmark
            t0 = time.time()
            preds_reg = model_reg.predict(X_test, verbose=0)
            latency_ms = ((time.time() - t0) / len(X_test)) * 1000

            pred_1h = preds_reg[0].flatten()
            pred_4h = preds_reg[1].flatten()
            pred_12h = preds_reg[2].flatten()

            # Calculate metrics
            metrics = {
                "1H Return MAE": mean_absolute_error(y1_test, pred_1h),
                "1H Directional Accuracy": np.mean(np.sign(pred_1h) == np.sign(y1_test)),
                "4H Return MAE": mean_absolute_error(y4_test, pred_4h),
                "4H Directional Accuracy": np.mean(np.sign(pred_4h) == np.sign(y4_test)),
                "12H Return MAE": mean_absolute_error(y12_test, pred_12h),
                "12H Directional Accuracy": np.mean(np.sign(pred_12h) == np.sign(y12_test)),
                "Inference Speed (ms)": latency_ms
            }
            
            print_metrics("", metrics)
            
            # Free memory
            del preds_reg, pred_1h, pred_4h, pred_12h
    else:
        print("   ❌ Model file not found.")

    # ────────────────────────────────────────────
    # MODEL 2: TRIPLE BARRIER CLASSIFICATION
    # ────────────────────────────────────────────
    print_header("🔍 [MODEL 2] Triple Barrier Classification AI (unconstrained_candle_ai.keras)")
    
    cls_path = "smartcrypto_ai_models/unconstrained_candle_ai.keras"
    if os.path.exists(cls_path):
        model_cls = load_model_with_fallback(cls_path, custom_objects)
        
        if model_cls:
            # Warm up
            _ = model_cls.predict(X_test[:10], verbose=0)
            
            # Benchmark
            t0 = time.time()
            preds_cls = model_cls.predict(X_test, verbose=0)
            latency_ms = ((time.time() - t0) / len(X_test)) * 1000

            pred_dirs = np.argmax(preds_cls[0], axis=1)
            acc_3class = accuracy_score(y_dir_test, pred_dirs)
            
            trade_indices = np.where(pred_dirs != 1)[0]
            if len(trade_indices) > 0:
                win_rate = np.mean(y_dir_test[trade_indices] == pred_dirs[trade_indices])
            else:
                win_rate = 0.0

            metrics = {
                "3-Class Overall Accuracy": acc_3class,
                "Active Signal Win Rate": win_rate,
                "Active Signal Triggers": f"{len(trade_indices):,} / {len(X_test):,}",
                "Inference Speed (ms)": latency_ms
            }
            
            print_metrics("", metrics)
            
            del preds_cls, pred_dirs
    else:
        print("   ❌ Model file not found.")

    # ────────────────────────────────────────────
    # MODEL 3: MARKET GPT WORLD MODEL
    # ────────────────────────────────────────────
    print_header("🔍 [MODEL 3] Market GPT World Model (market_gpt_world_model.keras)")
    
    gpt_path = "smartcrypto_ai_models/market_gpt_world_model.keras"
    if os.path.exists(gpt_path):
        model_gpt = load_model_with_fallback(gpt_path)
        
        if model_gpt:
            # Use a sample sequence
            sample_context = X_test[-1]
            
            # Warm up
            _ = MarketGPTWorldModel.simulate_future_paths(model_gpt, sample_context, n_simulations=100)
            
            # Benchmark with 1000 simulations
            t0 = time.time()
            sim_res = MarketGPTWorldModel.simulate_future_paths(
                model_gpt, 
                sample_context, 
                n_simulations=1000
            )
            time_sim = time.time() - t0

            metrics = {
                "Win Probability": sim_res.get('win_probability', 0),
                "Loss Probability": sim_res.get('loss_probability', 0),
                "Expected 12H PnL": sim_res.get('expected_return', 0),
                "Simulation Speed (1000 paths)": f"{time_sim:.2f}s"
            }
            
            print_metrics("", metrics)
            
            del model_gpt, sample_context
    else:
        print("   ❌ Model file not found.")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 6. SUMMARY
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print_header("🎉 MODEL EVALUATION COMPLETE!")
    print("\n📊 Summary:")
    print(f"   • Test samples: {len(X_test):,}")
    print(f"   • Features: {len(feature_cols)}")
    print(f"   • Symbols: {len(symbols)}")
    print("\n" + "=" * 80)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ENTRY POINT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Evaluation interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()