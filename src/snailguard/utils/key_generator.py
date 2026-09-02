import secrets
import string
from typing import Dict
import time

class APIKeyGenerator:
    """Generates API keys for different tiers"""
    
    @staticmethod
    def generate_key(tier: str) -> Dict:
        """Generate a new API key for specified tier"""
        prefix = f"SG-{tier.upper()}"
        random_part = ''.join(secrets.choice(string.ascii_uppercase + string.digits) 
                            for _ in range(8))
        
        api_key = f"{prefix}-{random_part}"
        
        return {
            'api_key': api_key,
            'tier': tier,
            'created_at': time.time(),
            'is_active': True
        }
    
    @staticmethod
    def generate_batch(tier: str, count: int = 10) -> list:
        """Generate multiple API keys"""
        return [APIKeyGenerator.generate_key(tier) for _ in range(count)]