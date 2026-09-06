from typing import Dict, Any

DEFAULT_CONFIG = {
    # Cascade thresholds
    'phase1_threshold': 0.1,
    'phase2_threshold': 0.7,
    'phase3_threshold': 0.95, 
    'phase4_threshold': 0.999,
    'zero_fp_threshold': 0.999,
    
    # Economic warfare
    'enable_economic_warfare': True,
    
    # Performance
    'max_processing_time_ms': 100,
    
    # Logging
    'log_level': 'INFO',
    'log_threats': True,
    
    # Model paths
    'model_path': None,  # Auto-detect
}

def load_config(custom_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Load configuration with custom overrides"""
    config = DEFAULT_CONFIG.copy()
    
    if custom_config:
        config.update(custom_config)
    
    return config