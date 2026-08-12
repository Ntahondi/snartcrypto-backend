"""
API Security & Authentication Layer for SmartCrypto AI v3.0.0
Protects REST API endpoints with X-API-Key HTTP Header verification.
"""

from fastapi import Security, HTTPException, status
from fastapi.security.api_key import APIKeyHeader
from src.core.config import get_settings

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_header)):
    """
    Validates X-API-Key header against configured APP_API_KEY.
    Blocks unauthorized public access to AI trading signals and portfolio data.
    """
    settings = get_settings()
    expected_api_key = getattr(settings, 'APP_API_KEY', 'smartcrypto_live_secret_key_2026')

    # If key is blank, allow public access
    if not expected_api_key:
        return True

    if not api_key or api_key != expected_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: Missing or invalid X-API-Key header.",
            headers={"WWW-Authenticate": "APIKey"}
        )
    return True