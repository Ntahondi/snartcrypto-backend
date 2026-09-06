from enum import Enum
from typing import List, Dict, Any
from dataclasses import dataclass

class CascadePhase(Enum):
    """4-phase cascade detection system"""
    PHASE1 = "phase1_rapid_screening"
    PHASE2 = "phase2_behavioral_analysis" 
    PHASE3 = "phase3_advanced_patterns"
    PHASE4 = "phase4_economic_warfare"
    ZERO_FP = "zero_false_positive"

@dataclass
class DetectionResult:
    """Result from threat detection analysis"""
    is_threat: bool
    confidence: float
    phase: CascadePhase
    actions: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'is_threat': self.is_threat,
            'confidence': self.confidence,
            'phase': self.phase.value,
            'actions': self.actions
        }