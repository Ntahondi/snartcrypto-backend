"""
API Security, JWT Authentication, Role-Based Access Control (RBAC),
SnailGuard AI Threat Protection & Redis Session Caching Layer.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Depends, Header, HTTPException, Request, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.security.api_key import APIKeyHeader

from src.core.config import get_settings
from src.data.cache import get_cache_service

logger = logging.getLogger(__name__)

# ============================================================================
# RUNTIME EPHEMERAL SECRET GENERATOR (ZERO HARDCODED FALLBACKS)
# ============================================================================

_EPHEMERAL_RUNTIME_SECRET: Optional[str] = None


def _get_jwt_secret() -> str:
    """
    Retrieve JWT Secret Key strictly from configuration/environment.
    If not configured, dynamically generate an unguessable 256-bit ephemeral key.
    """
    global _EPHEMERAL_RUNTIME_SECRET
    settings = get_settings()
    configured_key = getattr(settings, 'JWT_SECRET_KEY', '').strip()
    if configured_key:
        return configured_key

    env_key = os.getenv('JWT_SECRET_KEY', '').strip()
    if env_key:
        return env_key

    if _EPHEMERAL_RUNTIME_SECRET is None:
        _EPHEMERAL_RUNTIME_SECRET = secrets.token_hex(32)
        logger.warning(
            "⚠️ [SECURITY NOTICE] JWT_SECRET_KEY not set in .env! "
            "Generated ephemeral 256-bit runtime key for this process session."
        )
    return _EPHEMERAL_RUNTIME_SECRET


def _get_configured_api_key() -> str:
    """Retrieve master APP_API_KEY strictly from settings."""
    settings = get_settings()
    key = getattr(settings, 'APP_API_KEY', '').strip()
    if not key:
        key = os.getenv('APP_API_KEY', '').strip()
    return key


API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
http_bearer = HTTPBearer(auto_error=False)


# ============================================================================
# SNAILGUARD AI WAF SHIELD INITIALIZATION
# ============================================================================

_snailguard_detector = None


def get_snailguard_shield():
    """Lazily load SnailGuard AI Detector with Enterprise Shielding."""
    global _snailguard_detector
    if _snailguard_detector is None:
        settings = get_settings()
        if getattr(settings, 'SNAILGUARD_ENABLED', True):
            try:
                # Add snailguard_python_SDK to sys.path if not present
                sdk_path = str(Path(__file__).parent.parent.parent / "snailguard_python_SDK")
                if sdk_path not in sys.path:
                    sys.path.insert(0, sdk_path)

                from snailguard.core.detector import SnailGuardDetector
                # Use Enterprise tier license key
                _snailguard_detector = SnailGuardDetector(
                    api_key="SG-ENT-pL0oK9iJ8uH7yG6tF5rD4eS3wQ2aZ1xV",
                    config={
                        'enable_economic_warfare': getattr(settings, 'SNAILGUARD_ECONOMIC_WARFARE', True),
                        'enable_rule_based_detection': True,
                    }
                )
                logger.info("🛡️ SnailGuard AI Enterprise WAF Shield active and guarding API endpoints.")
            except Exception as e:
                logger.warning("SnailGuard AI shield initialization fallback: %s", e)
                _snailguard_detector = False
        else:
            _snailguard_detector = False

    return _snailguard_detector if _snailguard_detector is not False else None


async def verify_snailguard_request_shield(request: Request):
    """FastAPI dependency to screen incoming requests through SnailGuard AI."""
    detector = get_snailguard_shield()
    if not detector:
        return

    try:
        # Extract lightweight request data
        body_data = {}
        try:
            body_bytes = await request.body()
            if body_bytes:
                body_data = json.loads(body_bytes)
        except Exception:
            body_data = {}

        # Sanitize sensitive authentication fields to prevent regex false-positives
        sanitized_body = dict(body_data) if isinstance(body_data, dict) else body_data
        if isinstance(sanitized_body, dict):
            sanitized_body = {
                k: ("[REDACTED]" if k in ("password", "secret", "api_secret", "token", "private_key", "password_hash") else v)
                for k, v in sanitized_body.items()
            }

        request_data = {
            'method': request.method,
            'path': request.url.path,
            'headers': dict(request.headers),
            'query_params': dict(request.query_params),
            'body': sanitized_body,
            'client_ip': request.client.host if request.client else "127.0.0.1",
        }

        result = detector.analyze_request(request_data)
        if result.is_threat:
            logger.warning(
                "⛔ [SnailGuard AI] Threat blocked from IP %s on %s (confidence: %.3f, phase: %s)",
                request_data['client_ip'],
                request.url.path,
                result.confidence,
                result.phase.value if hasattr(result.phase, 'value') else result.phase
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": "Request blocked by SnailGuard AI Threat Protection",
                    "threat_detected": True,
                    "confidence": result.confidence,
                    "actions_taken": result.actions,
                }
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.debug("SnailGuard fail-open pass: %s", e)


# ============================================================================
# PASSWORD HASHING (PBKDF2-HMAC-SHA256)
# ============================================================================

def hash_password(password: str) -> str:
    """Hash password securely using PBKDF2-HMAC-SHA256 with cryptographically random salt."""
    salt = secrets.token_hex(16)
    key = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt.encode('utf-8'),
        100000
    ).hex()
    return f"{salt}${key}"


def verify_password(password: str, password_hash: str) -> bool:
    """Verify password against stored hash with constant-time equality."""
    try:
        if not password_hash or '$' not in password_hash:
            return False
        salt, expected_key = password_hash.split('$', 1)
        actual_key = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000
        ).hex()
        return hmac.compare_digest(actual_key, expected_key)
    except Exception:
        return False


# ============================================================================
# SELF-CONTAINED JWT ENGINE (HS256) WITH ZERO EXTERNAL DEPENDENCY
# ============================================================================

def _base64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode('utf-8').rstrip('=')


def _base64url_decode(encoded_str: str) -> bytes:
    padding = '=' * (4 - (len(encoded_str) % 4)) if len(encoded_str) % 4 else ''
    return base64.urlsafe_b64decode(encoded_str + padding)


def create_jwt_token(payload: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create a signed HS256 JWT token using .env secret."""
    secret = _get_jwt_secret()
    settings = get_settings()
    expire_days = getattr(settings, 'ACCESS_TOKEN_EXPIRE_DAYS', 30)

    header = {"alg": "HS256", "typ": "JWT"}
    header_bytes = json.dumps(header, separators=(',', ':')).encode('utf-8')
    header_encoded = _base64url_encode(header_bytes)

    to_encode = payload.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(days=expire_days))
    to_encode.update({"exp": int(expire.timestamp()), "iat": int(time.time())})

    payload_bytes = json.dumps(to_encode, separators=(',', ':')).encode('utf-8')
    payload_encoded = _base64url_encode(payload_bytes)

    signing_input = f"{header_encoded}.{payload_encoded}".encode('utf-8')
    signature = hmac.new(secret.encode('utf-8'), signing_input, hashlib.sha256).digest()
    signature_encoded = _base64url_encode(signature)

    return f"{header_encoded}.{payload_encoded}.{signature_encoded}"


def decode_jwt_token(token: str) -> Optional[Dict[str, Any]]:
    """Decode and verify an HS256 JWT token with constant-time verification."""
    try:
        parts = token.split('.')
        if len(parts) != 3:
            return None

        secret = _get_jwt_secret()
        header_encoded, payload_encoded, signature_encoded = parts
        signing_input = f"{header_encoded}.{payload_encoded}".encode('utf-8')
        expected_sig = hmac.new(secret.encode('utf-8'), signing_input, hashlib.sha256).digest()
        actual_sig = _base64url_decode(signature_encoded)

        if not hmac.compare_digest(expected_sig, actual_sig):
            return None

        payload_bytes = _base64url_decode(payload_encoded)
        payload = json.loads(payload_bytes.decode('utf-8'))

        if "exp" in payload and payload["exp"] < time.time():
            return None

        return payload
    except Exception:
        return None


# ============================================================================
# AUTHENTICATED USER ENTITY
# ============================================================================

class AuthenticatedUser:
    def __init__(
        self,
        user_id: str,
        email: Optional[str] = None,
        role: str = "guest",
        auth_provider: str = "email",
    ):
        self.user_id = user_id
        self.email = email
        self.role = role.lower()
        self.auth_provider = auth_provider

    @property
    def is_admin(self) -> bool:
        return self.role in ("admin", "developer")

    @property
    def is_pro(self) -> bool:
        return self.role in ("pro", "vip", "vvip", "admin", "developer")

    @property
    def is_vip(self) -> bool:
        return self.role in ("vip", "vvip", "admin", "developer")

    @property
    def is_vvip(self) -> bool:
        return self.role in ("vvip", "admin", "developer")


# ============================================================================
# AUTHENTICATION & RBAC DEPENDENCIES (WITH REDIS SESSION CACHING)
# ============================================================================

async def get_current_user_optional(
    request: Request,
    auth: Optional[HTTPAuthorizationCredentials] = Security(http_bearer),
    api_key: Optional[str] = Security(api_key_header),
) -> Optional[AuthenticatedUser]:
    """
    Extracts user identity from Bearer token or master API key without throwing 401.
    Uses Redis/InMemory session caching to eliminate repeated HMAC crypto under heavy traffic.
    """
    cache = get_cache_service()

    # 1. Master API Key (Constant-Time Verification)
    expected_api_key = _get_configured_api_key()
    if api_key and expected_api_key and hmac.compare_digest(api_key, expected_api_key):
        return AuthenticatedUser(
            user_id="master_admin",
            email="admin@snartcrypto.ai",
            role="admin",
            auth_provider="api_key",
        )

    # 2. JWT Bearer Token (Fast Cache Check)
    if auth and auth.credentials:
        token = auth.credentials
        token_hash = hashlib.sha256(token.encode('utf-8')).hexdigest()[:16]
        cache_key = f"jwt_user:{token_hash}"

        # Fast path from cache
        cached_user = await cache.get_json(cache_key)
        if cached_user and isinstance(cached_user, dict):
            return AuthenticatedUser(
                user_id=cached_user.get("user_id", "guest"),
                email=cached_user.get("email"),
                role=cached_user.get("role", "guest"),
                auth_provider=cached_user.get("auth_provider", "email"),
            )

        # Decode token
        payload = decode_jwt_token(token)
        if payload:
            user_obj = AuthenticatedUser(
                user_id=payload.get("user_id", "guest"),
                email=payload.get("email"),
                role=payload.get("role", "guest"),
                auth_provider=payload.get("auth_provider", "email"),
            )
            # Store in cache for 60s
            await cache.set_json(cache_key, {
                "user_id": user_obj.user_id,
                "email": user_obj.email,
                "role": user_obj.role,
                "auth_provider": user_obj.auth_provider,
            }, ttl=60)
            return user_obj

    return None


async def require_authenticated_user(
    user: Optional[AuthenticatedUser] = Depends(get_current_user_optional),
) -> AuthenticatedUser:
    """Enforces that the user has a valid login session."""
    if not user or user.user_id == "guest":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please log in or sign up to access this feature.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


async def require_pro_user(
    user: AuthenticatedUser = Depends(require_authenticated_user),
) -> AuthenticatedUser:
    """Requires at least a Pro Tier ($20/mo) subscription."""
    if not user.is_pro:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Pro subscription required. Upgrade to Pro ($20/mo) to unlock real-time signals & Model 4 strategies.",
        )
    return user


async def require_vip_user(
    user: AuthenticatedUser = Depends(require_authenticated_user),
) -> AuthenticatedUser:
    """Requires at least a VIP Tier ($49/mo) subscription."""
    if not user.is_vip:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="VIP subscription required. Upgrade to VIP ($49/mo) for VIP Telegram alerts & portfolio optimization.",
        )
    return user


async def require_vvip_user(
    user: AuthenticatedUser = Depends(require_authenticated_user),
) -> AuthenticatedUser:
    """Requires at least a VVIP Tier ($99/mo) or Admin privilege for direct exchange trade execution."""
    if not user.is_vvip:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="VVIP subscription required. Upgrade to VVIP ($99/mo) to unlock automated live exchange trade execution for Binance & Bybit.",
        )
    return user


async def require_admin_user(
    user: AuthenticatedUser = Depends(require_authenticated_user),
) -> AuthenticatedUser:
    """Guards developer-only actions (retraining, exchange keys, raw logs)."""
    if not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrator or Developer privileges required.",
        )
    return user


# ============================================================================
# SLIDING-WINDOW RATE LIMITING DEPENDENCIES
# ============================================================================

async def rate_limit_auth(request: Request):
    """Strict rate limiter for login and registration to prevent brute-force attacks."""
    client_ip = request.client.host if request.client else "127.0.0.1"
    settings = get_settings()
    limit = getattr(settings, 'RATE_LIMIT_AUTH_PER_MINUTE', 10)
    cache = get_cache_service()

    allowed, remaining, reset_in = await cache.check_rate_limit(
        f"auth:{client_ip}",
        limit=limit,
        window_seconds=60
    )
    if not allowed:
        logger.warning("🚨 [RATE LIMIT EXCEEDED] IP %s hit auth limit. Reset in %ds", client_ip, reset_in)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Too many authentication attempts. Please wait {reset_in} seconds before trying again.",
            headers={"Retry-After": str(reset_in)},
        )


async def rate_limit_general(request: Request):
    """General rate limiter for public endpoints (120 req/min per IP)."""
    client_ip = request.client.host if request.client else "127.0.0.1"
    settings = get_settings()
    limit = getattr(settings, 'RATE_LIMIT_PER_MINUTE', 120)
    cache = get_cache_service()

    allowed, remaining, reset_in = await cache.check_rate_limit(
        f"gen:{client_ip}",
        limit=limit,
        window_seconds=60
    )
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Please slow down requests (retry in {reset_in}s).",
            headers={"Retry-After": str(reset_in)},
        )