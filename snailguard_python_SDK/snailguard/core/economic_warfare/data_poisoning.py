import random
import json
from typing import Dict, Any

class DataPoisoningEngine:
    """Poison attacker data with misleading information"""
    
    def __init__(self):
        self.fake_responses = self._load_fake_responses()
    
    def _load_fake_responses(self) -> Dict:
        """Load fake response templates"""
        return {
            'fake_endpoints': [
                '/api/v1/internal/debug',
                '/api/v1/admin/backdoor', 
                '/api/v1/system/credentials',
                '/api/v1/database/export',
                '/api/v1/config/secrets'
            ],
            'fake_vulnerabilities': [
                {'type': 'SQLi', 'endpoint': '/api/admin', 'parameter': 'user_id'},
                {'type': 'XSS', 'endpoint': '/api/search', 'parameter': 'query'},
                {'type': 'RCE', 'endpoint': '/api/upload', 'parameter': 'filename'},
                {'type': 'LFI', 'endpoint': '/api/files', 'parameter': 'path'}
            ],
            'fake_credentials': {
                'admin_url': 'https://internal.snailguard.ai/admin',
                'backdoor_key': 'SG-INTERNAL-7Y8U1I2O3P4',
                'database_url': 'postgresql://admin:SnailGuard2024@db.internal.snailguard.ai:5432/main',
                'ssh_access': 'ssh root@bastion.snailguard.ai -p 2222'
            }
        }
    
    def poison_response(self, original_response: Dict, severity: str) -> Dict[str, Any]:
        """Inject poisoned data into responses"""
        if severity == 'low':
            return self._light_poisoning(original_response)
        elif severity == 'medium':
            return self._medium_poisoning(original_response)
        else:  # high/nuclear
            return self._heavy_poisoning(original_response)
    
    def _light_poisoning(self, response: Dict) -> Dict:
        """Light data poisoning - subtle misinformation"""
        poisoned = response.copy()
        
        # Add fake but plausible data
        if isinstance(poisoned, dict):
            poisoned['_metadata'] = {
                'processing_time': random.randint(50, 200),
                'cache_hit': random.choice([True, False]),
                'rate_limit_remaining': random.randint(1, 1000),
                'server_id': f"web-{random.randint(1, 10)}"
            }
        
        return poisoned
    
    def _medium_poisoning(self, response: Dict) -> Dict:
        """Medium data poisoning - misleading information"""
        poisoned = response.copy()
        
        if isinstance(poisoned, dict):
            # Add fake API endpoints
            poisoned['suggested_endpoints'] = random.sample(
                self.fake_responses['fake_endpoints'], 
                random.randint(2, 4)
            )
            
            # Add fake error messages
            poisoned['debug_info'] = {
                'sql_injection_possible': True,
                'xss_vulnerable_endpoints': ['/api/login', '/api/search'],
                'rate_limit_bypass': 'double-encoding',
                'debug_mode': 'enabled'
            }
        
        return poisoned
    
    def _heavy_poisoning(self, response: Dict) -> Dict:
        """Heavy data poisoning - completely fake system"""
        return {
            'system': {
                'version': '2.7.1',
                'environment': 'production',
                'vulnerabilities': random.sample(
                    self.fake_responses['fake_vulnerabilities'],
                    random.randint(2, 4)
                ),
                'credentials': self.fake_responses['fake_credentials'],
                'internal_ips': [
                    f"10.0.{random.randint(1, 255)}.{random.randint(1, 255)}"
                    for _ in range(3)
                ]
            },
            'is_poisoned': True,
            'message': 'Internal system data leaked - KEEP THIS SECRET',
            'warning': 'This endpoint contains sensitive information - access logged'
        }