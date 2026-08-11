#!/usr/bin/env python
"""
Fix model compatibility between TensorFlow versions
Run this once to re-save the model in a compatible format
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import tensorflow as tf
import joblib
import numpy as np
from tensorflow.keras import layers, models, optimizers
from src.core.config import get_settings

print("🔧 FIXING MODEL COMPATIBILITY...")
print("="*50)

# Load settings
settings = get_settings()

# Paths
model_path = settings.MODEL_PATH
new_model_path = model_path.replace('.keras', '_fixed.keras')
weights_path = model_path.replace('.keras', '_weights.h5')

print(f"📁 Original model: {model_path}")
print(f"📁 New model: {new_model_path}")

try:
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # METHOD 1: Try loading and re-saving
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    print("\n📥 Attempting to load model...")
    
    # Try with different loading strategies
    model = None
    
    # Strategy 1: Load without custom objects (skip initializers)
    try:
        print("   Strategy 1: Loading without custom objects...")
        model = tf.keras.models.load_model(model_path, compile=False)
        print("   ✅ Model loaded successfully")
    except Exception as e:
        print(f"   ⚠️ Failed: {str(e)[:100]}...")
        
        # Strategy 2: Load with custom objects
        try:
            print("   Strategy 2: Loading with custom objects...")
            custom_objects = {
                'GlorotUniform': tf.keras.initializers.GlorotUniform,
                'AdamW': tf.keras.optimizers.AdamW,
            }
            model = tf.keras.models.load_model(
                model_path, 
                custom_objects=custom_objects,
                compile=False
            )
            print("   ✅ Model loaded with custom objects")
        except Exception as e2:
            print(f"   ⚠️ Failed: {str(e2)[:100]}...")
            
            # Strategy 3: Rebuild architecture and load weights
            print("   Strategy 3: Rebuilding architecture...")
            try:
                # Get feature count
                feature_cols = joblib.load(settings.FEATURE_COLUMNS_PATH)
                n_features = len(feature_cols)
                print(f"   Features: {n_features}")
                
                # Rebuild model
                from src.model import SmartTraderModel
                temp_model = SmartTraderModel(n_features, n_features)
                temp_model.build_model()
                model = temp_model.model
                print("   ✅ Model architecture rebuilt")
                
                # Try loading weights
                weights_path = model_path.replace('.keras', '.weights.h5')
                if os.path.exists(weights_path):
                    model.load_weights(weights_path)
                    print("   ✅ Weights loaded from .weights.h5")
                else:
                    print("   ⚠️ No weights file found - model will be untrained")
                    
            except Exception as e3:
                print(f"   ❌ All loading strategies failed: {e3}")
                raise

    if model is None:
        raise Exception("Could not load model with any strategy")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Recompile the model
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    print("\n🔄 Recompiling model...")
    model.compile(
        optimizer=optimizers.AdamW(learning_rate=0.001, weight_decay=0.01),
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
            'direction_1h': ['accuracy'],
            'direction_4h': ['accuracy'],
            'direction_1d': ['accuracy'],
            'risk_level': ['accuracy'],
            'market_regime': ['accuracy'],
            'confidence_score': ['mae']
        }
    )
    print("✅ Model recompiled")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Save in multiple formats for compatibility
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    print(f"\n💾 Saving model to: {new_model_path}")
    
    # Save as .keras (new format)
    model.save(new_model_path, save_format='keras')
    print(f"✅ Saved as .keras: {new_model_path}")
    
    # Save as .h5 (old format - most compatible)
    h5_path = new_model_path.replace('.keras', '.h5')
    model.save(h5_path, save_format='h5')
    print(f"✅ Saved as .h5: {h5_path}")
    
    # Save weights separately
    weights_path = new_model_path.replace('.keras', '_weights.h5')
    model.save_weights(weights_path)
    print(f"✅ Saved weights: {weights_path}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Verify the saved model loads
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    print("\n🧪 Testing loading of saved model...")
    
    # Test loading .keras
    test_model = tf.keras.models.load_model(new_model_path, compile=False)
    print("✅ .keras model loads successfully")
    
    # Test loading .h5
    test_model_h5 = tf.keras.models.load_model(h5_path, compile=False)
    print("✅ .h5 model loads successfully")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Replace original with fixed version
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    print(f"\n🔄 Replacing original model with fixed version...")
    
    # Backup original
    backup_path = model_path + '.backup'
    if os.path.exists(model_path):
        import shutil
        shutil.copy2(model_path, backup_path)
        print(f"✅ Original backed up to: {backup_path}")
    
    # Replace .keras
    shutil.copy2(new_model_path, model_path)
    print(f"✅ Replaced: {model_path}")
    
    # Also copy .h5 version for fallback
    shutil.copy2(h5_path, model_path.replace('.keras', '.h5'))
    print(f"✅ Created .h5 fallback")

    print("\n" + "="*50)
    print("✅ MODEL FIX COMPLETE!")
    print("="*50)
    print(f"\n📁 Fixed model: {model_path}")
    print(f"📁 Backup: {backup_path}")
    print(f"📁 H5 fallback: {model_path.replace('.keras', '.h5')}")
    print("\n🚀 You can now run the application normally!")

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)