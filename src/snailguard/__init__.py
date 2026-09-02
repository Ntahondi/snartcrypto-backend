"""
SnailGuard AI - Advanced AI-Powered API Protection
Zero false positives guaranteed. Nuclear economic warfare against sophisticated attacks.
"""

__version__ = "1.0.0"
__author__ = "SnailGuard AI Team"

# Import core detector
from snailguard.core.detector import SnailGuardDetector

# Import decorators
from snailguard.decorators import protect_flask, protect_fastapi

__all__ = [
    "SnailGuardDetector",
    "protect_flask", 
    "protect_fastapi"
]