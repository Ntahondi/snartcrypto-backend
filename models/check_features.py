# Quick feature viewer (one-liner)
import joblib
features = joblib.load("models/feature_columns.joblib")
print(f"📊 Total features: {len(features)}")
for i, f in enumerate(sorted(features), 1):
    print(f"{i:3d}. {f}")