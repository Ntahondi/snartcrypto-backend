from typing import List, Dict, Any
import logging
import numpy as np
import time
import threading
from .economic_warfare.deceptive_responses import DeceptiveResponseEngine
from .economic_warfare.computational_waste import ComputationalWaste
from .economic_warfare.time_delays import TimeDelayEngine

class EconomicEngine:
    """
    Enhanced economic warfare with real-time deceptive responses
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.deceptive_engine = DeceptiveResponseEngine()
        self.computational_waste = ComputationalWaste()
        self.time_delays = TimeDelayEngine()
    
    def trigger_economic_warfare(self, features: np.ndarray, client_ip: str = None, original_request: Dict = None) -> List[Dict[str, Any]]:
        """
        Trigger economic warfare with real-time deceptive responses
        """
        actions = []
        threat_level = self._calculate_threat_level(features)
        
        # Always start with computational waste (background suffering)
        if threat_level > 0.5:
            waste_action = self.computational_waste.trigger_waste_cycles('medium')
            waste_action['estimated_attacker_cost'] = 25000
            actions.append(waste_action)
        
        # Tier 1: Time delays + Light deception
        if threat_level > 0.7:
            # Intentional time delays
            if client_ip:
                delay_action = self.time_delays.apply_delay('medium', client_ip)
                delay_action['estimated_attacker_cost'] = 15000
                actions.append(delay_action)
            
            # Light deceptive response
            deception_action = {
                'type': 'light_deception',
                'severity': 'medium',
                'description': 'Generate lightweight deceptive responses',
                'estimated_attacker_cost': 35000,
                'deceptive_response': self.deceptive_engine.generate_slow_deceptive_response(
                    original_request, 'medium'
                ) if original_request else None
            }
            actions.append(deception_action)
        
        # Tier 2: Heavy deception + Resource waste
        if threat_level > 0.85:
            # Heavy computational waste
            heavy_waste = self.computational_waste.trigger_waste_cycles('high')
            heavy_waste['estimated_attacker_cost'] = 100000
            actions.append(heavy_waste)
            
            # Heavy deceptive response
            heavy_deception = {
                'type': 'heavy_deception',
                'severity': 'high',
                'description': 'Generate complex deceptive responses with maximum delays',
                'estimated_attacker_cost': 150000,
                'deceptive_response': self.deceptive_engine.generate_slow_deceptive_response(
                    original_request, 'high'
                ) if original_request else None
            }
            actions.append(heavy_deception)
        
        # Tier 3: Nuclear options - Maximum suffering
        if threat_level > 0.999:
            # Maximum time delays
            if client_ip:
                nuclear_delay = self.time_delays.apply_delay('nuclear', client_ip)
                nuclear_delay['estimated_attacker_cost'] = 200000
                actions.append(nuclear_delay)
            
            # Nuclear deceptive response
            nuclear_deception = {
                'type': 'nuclear_deception',
                'severity': 'nuclear',
                'description': 'Generate extremely complex deceptive responses with massive delays',
                'estimated_attacker_cost': 294000,
                'deceptive_response': self.deceptive_engine.generate_slow_deceptive_response(
                    original_request, 'nuclear'
                ) if original_request else None
            }
            actions.append(nuclear_deception)
        
        total_cost = sum(action['estimated_attacker_cost'] for action in actions)
        if actions:
            suffering_level = 'MAXIMUM' if threat_level > 0.999 else 'HIGH' if threat_level > 0.85 else 'MEDIUM'
            self.logger.warning(f"💸 Economic warfare triggered: ${total_cost:,} cost | Suffering: {suffering_level}")
        
        return actions
    
    def _calculate_threat_level(self, features: np.ndarray) -> float:
        """Calculate threat level based on feature analysis"""
        sophistication = features[0]
        intelligence = features[1]
        edge_case = features[2]
        
        return min(1.0, (sophistication * 0.4 + intelligence * 0.4 + edge_case * 0.2))