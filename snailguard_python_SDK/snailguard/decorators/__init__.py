try:
    from .fastapi import protect_fastapi
except ImportError:
    protect_fastapi = None

try:
    from .flask import protect_flask
except ImportError:
    protect_flask = None

__all__ = [
    "protect_fastapi",
    "protect_flask"
]