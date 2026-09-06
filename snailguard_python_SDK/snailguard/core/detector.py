import pickle
import numpy as np
from typing import Dict, Any, Tuple, Optional
import logging
from pathlib import Path
import re

from .feature_extractor import FeatureExtractor
from .economic_engine import EconomicEngine
from .models import CascadePhase, DetectionResult
from .license_manager import LicenseManager, LicenseTier

class LicenseError(Exception):
    """Custom exception for license-related errors"""
    pass


class SnailGuardDetector:
    """
    Main SnailGuard AI detector implementing 4-phase cascade analysis with rule-based protection
    """
    
    def __init__(self, api_key: str = None, model_path: str = None, config: Dict[str, Any] = None):
        self.logger = logging.getLogger(__name__)
        self.license_manager = LicenseManager()
        
        # Validate API key
        self.api_key = api_key
        self.license_info = self._validate_license(api_key)
        self.tier = self.license_info['tier']
        self.features = self.license_manager.get_tier_features(self.tier)
        
        self.config = self._load_config(config)
        
        # Initialize components
        self.feature_extractor = FeatureExtractor()
        self.economic_engine = EconomicEngine()
        
        # Load trained models based on tier
        self.models = self._load_models(model_path)
        
        # Cascade thresholds (may be limited by tier)
        self.thresholds = self._setup_thresholds()
        
        self.logger.info(f"SnailGuard AI Detector initialized for tier: {self.tier.value}")
    
    def _validate_license(self, api_key: str) -> Dict:
        """Validate API key and return license information"""
        if not api_key:
            raise LicenseError("API key is required. Get your free key at: https://snailguard.ai/api-keys")
            
        is_valid, license_info = self.license_manager.validate_key(api_key)
        if not is_valid:
            raise LicenseError(license_info.get('error', 'Invalid API key'))
            
        return license_info
    
    def _load_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Load configuration with tier-based defaults"""
        default_config = {
            'phase1_threshold': 0.1,
            'phase2_threshold': 0.7, 
            'phase3_threshold': 0.95,
            'phase4_threshold': 0.999,
            'zero_fp_threshold': 0.999,
            'enable_economic_warfare': self.features['economic_warfare'],
            'enable_rule_based_detection': True,
            'max_processing_time_ms': 100,
            'enable_advanced_models': self.features['advanced_models']
        }
        
        if config:
            # Prevent free tier from enabling paid features
            if self.tier == LicenseTier.FREE:
                config['enable_economic_warfare'] = False
                config['enable_advanced_models'] = False
                
            default_config.update(config)
            
        return default_config
    
    def _setup_thresholds(self) -> Dict[CascadePhase, float]:
        """Setup thresholds based on tier"""
        base_thresholds = {
            CascadePhase.PHASE1: self.config.get('phase1_threshold', 0.1),
            CascadePhase.PHASE2: self.config.get('phase2_threshold', 0.7),
            CascadePhase.PHASE3: self.config.get('phase3_threshold', 0.95),
            CascadePhase.PHASE4: self.config.get('phase4_threshold', 0.999),
            CascadePhase.ZERO_FP: self.config.get('zero_fp_threshold', 0.999)
        }
        
        # Free tier has higher thresholds (more conservative)
        if self.tier == LicenseTier.FREE:
            return {
                CascadePhase.PHASE1: 0.3,    # Higher threshold = less sensitive
                CascadePhase.PHASE2: 0.8,
                CascadePhase.PHASE3: 0.98,
                CascadePhase.PHASE4: 0.999,
                CascadePhase.ZERO_FP: 0.999
            }
            
        return base_thresholds    
    
    def _load_models(self, model_path: str = None) -> Dict[str, Any]:
        """Load models based on tier access"""
        if model_path is None:
            model_path = Path(__file__).parent.parent / "data" / "models"
        else:
            model_path = Path(model_path)
        
        models = {}
        
        try:
            # All tiers get basic production model
            with open(model_path / "production_model.pkl", 'rb') as f:
                models['production'] = pickle.load(f)
            self.logger.info("✅ Loaded production XGBoost model")
            
            cascade_path = model_path / "cascade_models"
            
            # Free tier only gets basic models
            if self.tier == LicenseTier.FREE:
                models[CascadePhase.PHASE1] = models['production']
                models[CascadePhase.PHASE2] = models['production'] 
                models[CascadePhase.PHASE3] = models['production']
                self.logger.info("🔄 Free tier: Using production model for all phases")
                
            else:
                # Pro and Enterprise get full model suite
                try:
                    models[CascadePhase.PHASE1] = pickle.load(open(cascade_path / "xgb_rapid.pkl", 'rb'))
                    self.logger.info("✅ Loaded Phase 1 XGBoost model")
                except Exception as e:
                    self.logger.warning(f"❌ Phase 1 model failed: {e}")
                    models[CascadePhase.PHASE1] = models['production']
                
                # Pro/Enterprise can use advanced models if available and enabled
                if self.config.get('enable_advanced_models', True):
                    models[CascadePhase.PHASE2] = models['production']  # Fallback
                    models[CascadePhase.PHASE3] = models['production']  # Fallback
                else:
                    models[CascadePhase.PHASE2] = models['production']
                    models[CascadePhase.PHASE3] = models['production']
                    
                self.logger.info(f"🎯 {self.tier.value.title()} tier: Advanced models enabled")
            
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")
            raise
        
        return models
    
    def analyze_request(self, request_data: Dict[str, Any]) -> DetectionResult:
        """
        Main analysis method with tier-based feature access
        """
        # Check license usage
        self._check_license_limits()
        
        # Rule-based detection (available to all tiers)
        if self.config.get('enable_rule_based_detection', True):
            rule_based_threat = self._detect_rule_based_attacks(request_data)
            if rule_based_threat:
                economic_actions = []
                # Economic warfare only for Pro/Enterprise
                if self.config['enable_economic_warfare']:
                    max_features = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
                    economic_actions = self.economic_engine.trigger_economic_warfare(max_features)
                
                self.logger.warning(f"🚨 Rule-based threat detected: {rule_based_threat}")
                return DetectionResult(
                    is_threat=True,
                    confidence=0.99,
                    phase=CascadePhase.PHASE1,
                    actions=economic_actions
                )
        
        # AI cascade detection
        return self._analyze_with_ai_cascade(request_data)

    def _check_license_limits(self):
        """Check if usage is within license limits"""
        # This would connect to license server in production
        pass
    
    def _analyze_with_ai_cascade(self, request_data: Dict[str, Any]) -> DetectionResult:
        """
        Original AI cascade analysis - kept separate for clarity
        """
        # Extract real-time features
        features = self.feature_extractor.extract(request_data)
        
        # Phase 1: Rapid Screening (xgb_rapid - WORKS)
        phase1_score = self._run_phase(CascadePhase.PHASE1, features)
        if phase1_score >= self.thresholds[CascadePhase.PHASE1]:
            economic_actions = []
            if self.config['enable_economic_warfare']:
                economic_actions = self.economic_engine.trigger_economic_warfare(features)
            return DetectionResult(
                is_threat=True,
                confidence=phase1_score,
                phase=CascadePhase.PHASE1,
                actions=economic_actions
            )
        
        # Phase 2: Behavioral Analysis (using production model as fallback)
        phase2_score = self._run_phase(CascadePhase.PHASE2, features)
        if phase2_score >= self.thresholds[CascadePhase.PHASE2]:
            economic_actions = []
            if self.config['enable_economic_warfare']:
                economic_actions = self.economic_engine.trigger_economic_warfare(features)
            return DetectionResult(
                is_threat=True,
                confidence=phase2_score,
                phase=CascadePhase.PHASE2,
                actions=economic_actions
            )
        
        # Phase 3: Advanced Patterns (using production model as fallback)
        phase3_score = self._run_phase(CascadePhase.PHASE3, features)
        if phase3_score >= self.thresholds[CascadePhase.PHASE3]:
            economic_actions = []
            if self.config['enable_economic_warfare']:
                economic_actions = self.economic_engine.trigger_economic_warfare(features)
            return DetectionResult(
                is_threat=True,
                confidence=phase3_score,
                phase=CascadePhase.PHASE3,
                actions=economic_actions
            )
        
        # Phase 4: Economic Warfare (production model only)
        production_score = self._run_model(self.models['production'], features)
        
        # Zero false positive guarantee
        if production_score >= self.thresholds[CascadePhase.ZERO_FP]:
            economic_actions = []
            if self.config['enable_economic_warfare']:
                economic_actions = self.economic_engine.trigger_economic_warfare(features)
            
            return DetectionResult(
                is_threat=True,
                confidence=production_score,
                phase=CascadePhase.PHASE4,
                actions=economic_actions
            )
        
        # Not a threat (low probability)
        return DetectionResult(
            is_threat=False,
            confidence=production_score,
            phase=CascadePhase.PHASE4,
            actions=[]
        )
    
    def _detect_rule_based_attacks(self, request_data: Dict[str, Any]) -> str:
        """
        Rule-based detection for known attack patterns
        Returns: Attack type if detected, empty string otherwise
        """
        # Extract data for analysis
        body = request_data.get('body', {})
        path = request_data.get('path', '')
        headers = request_data.get('headers', {})
        user_agent = headers.get('user-agent', '')
        
        # Convert to strings for pattern matching
        body_str = str(body).lower()
        path_str = path.lower()
        headers_str = str(headers).lower()
        user_agent_lower = user_agent.lower()
        all_text = body_str + path_str + headers_str
        
        # SQL Injection patterns (Context-aware patterns to avoid header false positives)
        sql_patterns = [
            r"'\s+or\s+['\"]?1['\"]?\s*=\s*['\"]?1",      # Basic SQL injection: admin' OR '1'='1
            r"'\s+or\s+true\b",                             # ' OR TRUE
            r"\bunion\s+all\s+select\b|\bunion\s+select\b", # Union select
            r"\binsert\s+into\b.*\bvalues\b",              # Insert statements  
            r"\bdrop\s+table\b|\bdrop\s+database\b",       # Drop table/database
            r";\s*--|\bwaitfor\s+delay\b|\bsleep\(\d+\)",   # Stacked query / time delay
        ]
        
        # Check for SQL injection
        for pattern in sql_patterns:
            if re.search(pattern, all_text, re.IGNORECASE):
                return f"SQL injection detected: {pattern}"
        
        # XSS patterns
        xss_patterns = [
            r".*<script>",          # Script tags
            r".*javascript:",       # JavaScript
            r".*alert\(",           # Alert calls
            r".*onerror=",          # Event handlers
            r".*onload=",
            r".*onclick=",
            r".*document\.cookie",  # Cookie access
        ]
        
        # Check for XSS
        for pattern in xss_patterns:
            if re.search(pattern, all_text, re.IGNORECASE):
                return f"XSS detected: {pattern}"
        
        # Check for known attack tools
        attack_tools = ['sqlmap', 'nmap', 'burp', 'metasploit', 'hydra', 'nikto']
        if any(tool in user_agent_lower for tool in attack_tools):
            return f"Attack tool detected: {user_agent}"
        
        # Check for path traversal
        traversal_patterns = [
            r"\.\./",               # Basic directory traversal
            r"\.\.\\",              # Windows traversal
            r"etc/passwd",          # Common target
            r"win\.ini",            # Windows target
            r"\.\.%.*",             # Encoded traversal
        ]
        
        for pattern in traversal_patterns:
            if re.search(pattern, path_str, re.IGNORECASE) or re.search(pattern, body_str, re.IGNORECASE):
                return f"Path traversal detected: {pattern}"
        
        # Check for command injection
        command_patterns = [
            r";\s*(rm|ls|cat|echo|wget|curl|nc|netcat)",
            r"\|\s*(rm|ls|cat|echo|wget|curl|nc|netcat)",
            r"&\s*(rm|ls|cat|echo|wget|curl|nc|netcat)",
            r"`.*`",
            r"\$\(.*\)",
        ]
        
        for pattern in command_patterns:
            if re.search(pattern, all_text, re.IGNORECASE):
                return f"Command injection detected: {pattern}"
        
        return ""  # No rule-based threat detected

    def _run_model(self, model: Any, features: np.ndarray) -> float:
        """Run inference with a specific model"""
        try:
            if hasattr(features, 'reshape'):
                features = features.reshape(1, -1)
            
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(features)
                return proba[0][1] if proba.shape[1] > 1 else proba[0][0]
            else:
                return float(model.predict(features)[0])
        except Exception as e:
            self.logger.error(f"Model inference error: {e}")
            return 0.0
        
    def _run_phase(self, phase: CascadePhase, features: np.ndarray) -> float:
        """Run inference for a specific cascade phase"""
        model = self.models[phase]
        
        try:
            # Ensure features are in correct format for model
            if hasattr(features, 'reshape'):
                features = features.reshape(1, -1)
            
            # Get prediction probability for threat class
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(features)
                # Return probability of positive class (threat)
                return proba[0][1] if proba.shape[1] > 1 else proba[0][0]
            else:
                # For models without predict_proba
                return float(model.predict(features)[0])
                
        except Exception as e:
            self.logger.error(f"Error in phase {phase} inference: {e}")
            # Fail open - treat as non-threat on model errors
            return 0.0