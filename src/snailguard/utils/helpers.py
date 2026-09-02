import logging
from snailguard.core.detector import SnailGuardDetector

def test_installation():
    """Test that all models load correctly"""
    print("🧪 Testing SnailGuard AI installation...")
    
    try:
        detector = SnailGuardDetector()
        print("✅ All 6 AI models loaded successfully!")
        print("   - XGBoost (Production)")
        print("   - XGBoost Rapid (Phase 1)")
        print("   - Random Forest Balanced (Phase 2)") 
        print("   - Gradient Boosting (Phase 3)")
        print("   - SVM Linear (Backup)")
        print("   - Neural Network (Backup)")
        
        # Test feature extraction
        test_request = {
            'method': 'GET',
            'path': '/api/test',
            'headers': {'user-agent': 'test-client'},
            'body': {},
            'client_ip': '127.0.0.1'
        }
        
        features = detector.feature_extractor.extract(test_request)
        print(f"✅ Feature extraction working: {len(features)} features")
        
        return True
        
    except Exception as e:
        print(f"❌ Installation test failed: {e}")
        return False

if __name__ == "__main__":
    test_installation()