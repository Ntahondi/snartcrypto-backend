import hashlib
import time
from typing import Dict, Optional, Tuple
import logging
from enum import Enum
import re
import secrets
import string

class LicenseTier(Enum):
    FREE = "free"
    PRO = "pro" 
    ENTERPRISE = "enterprise"

class LicenseManager:
    """
    Manages API key validation and tier-based feature access with secure keys
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.valid_keys = self._load_predefined_keys()
        
    def _load_predefined_keys(self) -> Dict[str, Dict]:
        """Load predefined valid API keys with secure format"""
        return {
            # Predefined keys for testing (32 characters)
            "SG-FREE-aB3xY7pQ9rT1vW5zK8mN2cF4hJ6gL0dS": {"tier": LicenseTier.FREE, "active": True, "requests_today": 0},
            "SG-FREE-qW9eR5tY7uI3oP1aS2dF4gH6jK8lZ0xC": {"tier": LicenseTier.FREE, "active": True, "requests_today": 0},
            "SG-PRO-bN4vM8qR2tW6yZ9xE1cF7hJ3kL5pO0uA": {"tier": LicenseTier.PRO, "active": True, "requests_today": 0},
            "SG-PRO-mK8jH4gF7dS1aP3oI6uY9tR5eW2qQ0zX": {"tier": LicenseTier.PRO, "active": True, "requests_today": 0},
            "SG-ENT-cX7zL1pO9mN3bV5qW8tR0yU4iI6kK2jH": {"tier": LicenseTier.ENTERPRISE, "active": True, "requests_today": 0},
            "SG-ENT-pL0oK9iJ8uH7yG6tF5rD4eS3wQ2aZ1xV": {"tier": LicenseTier.ENTERPRISE, "active": True, "requests_today": 0},
        }
    
    def validate_key(self, api_key: str) -> Tuple[bool, Optional[Dict]]:
        """Validate API key - accepts both predefined and generated keys"""
        if not api_key:
            return False, {"error": "API key is required"}
        
        # Check if it's a predefined key
        if api_key in self.valid_keys:
            key_info = self.valid_keys[api_key]
            if not key_info.get('active', True):
                return False, {"error": "API key is deactivated"}
            return True, key_info
        
        # Check if it's a valid generated key format
        key_validation = self._validate_generated_key(api_key)
        if key_validation[0]:
            return key_validation
        
        return False, {"error": "Invalid API key"}
    
    def _validate_generated_key(self, api_key: str) -> Tuple[bool, Optional[Dict]]:
        """Validate dynamically generated API keys with secure format"""
        # Check key format: SG-{TIER}-{32CHARCODE}
        pattern = r'^SG-(FREE|PRO|ENTERPRISE)-[A-Za-z0-9]{32}$'
        if not re.match(pattern, api_key):
            return False, {"error": "Invalid API key format. Must be: SG-{TIER}-{32CHARCODE}"}
        
        # Extract tier from key
        tier_part = api_key.split('-')[1]
        tier_map = {
            'FREE': LicenseTier.FREE,
            'PRO': LicenseTier.PRO,
            'ENTERPRISE': LicenseTier.ENTERPRISE
        }
        
        tier = tier_map.get(tier_part)
        if not tier:
            return False, {"error": "Invalid tier in API key"}
        
        # Create key info for generated keys
        key_info = {
            'tier': tier,
            'active': True,
            'requests_today': 0,
            'is_generated': True,
            'last_used': time.time()
        }
        
        # Store the generated key for future use
        self.valid_keys[api_key] = key_info
        
        return True, key_info
    
    def _exceeded_daily_limit(self, api_key: str, key_info: Dict) -> bool:
        """Check if daily request limit is exceeded"""
        tier = key_info['tier']
        requests_today = key_info.get('requests_today', 0)
        
        limits = {
            LicenseTier.FREE: 1000,        # 1,000 requests/day
            LicenseTier.PRO: 10000,        # 10,000 requests/day  
            LicenseTier.ENTERPRISE: 100000 # 100,000 requests/day
        }
        
        return requests_today >= limits.get(tier, 1000)
    
    def get_tier_features(self, tier: LicenseTier) -> Dict:
        """Get features available for each tier"""
        features = {
            LicenseTier.FREE: {
                'max_requests_per_day': 1000,
                'economic_warfare': False,
                'advanced_models': False,
                'custom_thresholds': False,
                'priority_support': False,
                'concurrent_requests': 1
            },
            LicenseTier.PRO: {
                'max_requests_per_day': 10000,
                'economic_warfare': True,
                'advanced_models': True, 
                'custom_thresholds': True,
                'priority_support': False,
                'concurrent_requests': 5
            },
            LicenseTier.ENTERPRISE: {
                'max_requests_per_day': 100000,
                'economic_warfare': True,
                'advanced_models': True,
                'custom_thresholds': True,
                'priority_support': True,
                'concurrent_requests': 50
            }
        }
        return features.get(tier, features[LicenseTier.FREE])