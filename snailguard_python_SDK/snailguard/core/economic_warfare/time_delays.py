import time
import random
from typing import Dict

class TimeDelayEngine:
    """Implements progressive time delays to waste attacker time"""
    
    def __init__(self):
        self.delay_patterns = {
            'low': {'base_delay': 1, 'max_delay': 5, 'jitter': 2},
            'medium': {'base_delay': 5, 'max_delay': 15, 'jitter': 5},
            'high': {'base_delay': 10, 'max_delay': 30, 'jitter': 10},
            'nuclear': {'base_delay': 30, 'max_delay': 120, 'jitter': 30}
        }
    
    def apply_delay(self, severity: str, client_ip: str) -> Dict[str, any]:
        """Apply progressive time delay based on severity and IP"""
        pattern = self.delay_patterns[severity]
        
        # Base delay + random jitter
        delay_seconds = pattern['base_delay'] + random.randint(0, pattern['jitter'])
        delay_seconds = min(delay_seconds, pattern['max_delay'])
        
        # Progressive delays for repeat offenders
        repeat_multiplier = self._get_repeat_offender_multiplier(client_ip)
        final_delay = delay_seconds * repeat_multiplier
        
        # Actually sleep (this blocks the request)
        time.sleep(final_delay)
        
        return {
            'type': 'time_delay',
            'delay_seconds': final_delay,
            'estimated_time_cost': final_delay * 10,  # $10/hour attacker time
            'repeat_offender_multiplier': repeat_multiplier
        }
    
    def _get_repeat_offender_multiplier(self, client_ip: str) -> float:
        """Increase delays for repeat attackers"""
        # In production, this would check a database
        # For now, simulate with random
        return random.choice([1.0, 1.5, 2.0, 3.0])