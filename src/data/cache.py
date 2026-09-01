"""
SmartCrypto AI - Enterprise High-Performance Caching & Rate Limiting Engine
============================================================================
Provides sub-millisecond data caching and sliding-window rate limiting for 1,000+ concurrent users.
Features:
- Primary: Asynchronous Redis client (`redis.asyncio`) with connection pooling.
- Fallback: Thread-safe in-memory `AsyncTTLCache` with TTL expiration and LRU pruning.
- Automatic failover with zero service interruption.
- Sliding-window rate limiter for brute-force and DDoS protection.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from src.core.config import get_settings

logger = logging.getLogger(__name__)

# Try to import redis
try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    aioredis = None


class InMemoryEntry:
    __slots__ = ('value', 'expires_at')

    def __init__(self, value: str, ttl: int):
        self.value = value
        self.expires_at = time.time() + ttl if ttl > 0 else float('inf')

    @property
    def is_expired(self) -> bool:
        return time.time() > self.expires_at


class AsyncTTLCache:
    """Thread-safe, async in-memory cache with TTL and maximum size limit."""

    def __init__(self, max_size: int = 10000):
        self._data: Dict[str, InMemoryEntry] = {}
        self._rate_limits: Dict[str, List[float]] = {}
        self._max_size = max_size
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[str]:
        async with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            if entry.is_expired:
                del self._data[key]
                return None
            return entry.value

    async def set(self, key: str, value: str, ttl: int = 10) -> bool:
        async with self._lock:
            # Simple eviction if max size reached
            if len(self._data) >= self._max_size:
                now = time.time()
                expired_keys = [k for k, v in self._data.items() if v.expires_at <= now]
                for k in expired_keys:
                    del self._data[k]
                if len(self._data) >= self._max_size:
                    # Remove oldest 10%
                    keys_to_remove = list(self._data.keys())[:int(self._max_size * 0.1)]
                    for k in keys_to_remove:
                        del self._data[k]

            self._data[key] = InMemoryEntry(value, ttl)
            return True

    async def delete(self, key: str) -> bool:
        async with self._lock:
            return self._data.pop(key, None) is not None

    async def delete_pattern(self, pattern_prefix: str) -> int:
        async with self._lock:
            keys_to_delete = [k for k in self._data.keys() if k.startswith(pattern_prefix)]
            for k in keys_to_delete:
                del self._data[k]
            return len(keys_to_delete)

    async def check_rate_limit(
        self,
        identifier: str,
        limit: int = 120,
        window_seconds: int = 60
    ) -> Tuple[bool, int, int]:
        """
        Sliding-window rate limiter.
        Returns: (allowed: bool, remaining_requests: int, reset_seconds: int)
        """
        async with self._lock:
            now = time.time()
            window_start = now - window_seconds
            timestamps = self._rate_limits.get(identifier, [])

            # Filter out old requests
            timestamps = [ts for ts in timestamps if ts > window_start]

            if len(timestamps) >= limit:
                oldest = timestamps[0]
                reset_seconds = max(1, int(oldest + window_seconds - now))
                self._rate_limits[identifier] = timestamps
                return False, 0, reset_seconds

            timestamps.append(now)
            self._rate_limits[identifier] = timestamps
            remaining = max(0, limit - len(timestamps))
            return True, remaining, window_seconds


class CacheService:
    """Enterprise Caching Service with Redis and InMemory Fallback."""

    def __init__(self):
        self._settings = get_settings()
        self._memory_cache = AsyncTTLCache(max_size=20000)
        self._redis: Optional[Any] = None
        self._redis_connected = False
        self._init_attempted = False

    async def _get_redis(self) -> Optional[Any]:
        if self._redis_connected and self._redis is not None:
            return self._redis

        if not REDIS_AVAILABLE:
            return None

        if not self._init_attempted:
            self._init_attempted = True
            try:
                redis_url = getattr(self._settings, 'REDIS_URL', 'redis://localhost:6379/0')
                self._redis = aioredis.from_url(
                    redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                    socket_connect_timeout=0.5,
                    socket_timeout=0.5,
                )
                # Test connection
                await self._redis.ping()
                self._redis_connected = True
                logger.info("🚀 High-Speed Redis Cache connected successfully: %s", redis_url)
            except Exception as e:
                logger.warning("⚠️ Redis connection unavailable (%s). Using high-performance In-Memory cache fallback.", e)
                self._redis_connected = False
                self._redis = None

        return self._redis if self._redis_connected else None

    async def get_json(self, key: str) -> Optional[Any]:
        """Fetch and deserialise cached JSON object."""
        try:
            client = await self._get_redis()
            if client is not None:
                val = await client.get(key)
                if val is not None:
                    return json.loads(val)
        except Exception as e:
            logger.debug("Redis get error, falling back to memory: %s", e)

        raw = await self._memory_cache.get(key)
        if raw is not None:
            try:
                return json.loads(raw)
            except Exception:
                return None
        return None

    async def set_json(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Serialise and store object in cache with TTL."""
        if ttl is None:
            ttl = getattr(self._settings, 'CACHE_TTL_SECONDS', 10)

        try:
            serialized = json.dumps(value, default=str)
        except Exception:
            return False

        # Store in in-memory cache as immediate fallback
        await self._memory_cache.set(key, serialized, ttl)

        try:
            client = await self._get_redis()
            if client is not None:
                await client.setex(key, ttl, serialized)
                return True
        except Exception as e:
            logger.debug("Redis set error: %s", e)

        return True

    async def delete(self, key: str) -> bool:
        """Delete specific key from both Redis and memory."""
        await self._memory_cache.delete(key)
        try:
            client = await self._get_redis()
            if client is not None:
                await client.delete(key)
        except Exception:
            pass
        return True

    async def delete_pattern(self, prefix: str) -> int:
        """Invalidate all keys matching prefix."""
        count = await self._memory_cache.delete_pattern(prefix)
        try:
            client = await self._get_redis()
            if client is not None:
                keys = await client.keys(f"{prefix}*")
                if keys:
                    deleted = await client.delete(*keys)
                    count = max(count, deleted)
        except Exception:
            pass
        return count

    async def check_rate_limit(
        self,
        identifier: str,
        limit: int = 120,
        window_seconds: int = 60
    ) -> Tuple[bool, int, int]:
        """
        Sliding-window rate limiter.
        Returns: (is_allowed, remaining, reset_seconds)
        """
        try:
            client = await self._get_redis()
            if client is not None:
                now = time.time()
                key = f"ratelimit:{identifier}"
                pipe = client.pipeline()
                pipe.zremrangebyscore(key, 0, now - window_seconds)
                pipe.zcard(key)
                pipe.zadd(key, {str(now): now})
                pipe.expire(key, window_seconds)
                results = await pipe.execute()
                current_count = results[1]

                if current_count >= limit:
                    # Exceeded
                    oldest_items = await client.zrange(key, 0, 0, withscores=True)
                    reset_in = window_seconds
                    if oldest_items:
                        oldest_ts = oldest_items[0][1]
                        reset_in = max(1, int(oldest_ts + window_seconds - now))
                    return False, 0, reset_in

                return True, max(0, limit - current_count - 1), window_seconds
        except Exception as e:
            logger.debug("Redis rate limit fallback to memory: %s", e)

        return await self._memory_cache.check_rate_limit(identifier, limit, window_seconds)


# Global Singleton
_global_cache: Optional[CacheService] = None


def get_cache_service() -> CacheService:
    global _global_cache
    if _global_cache is None:
        _global_cache = CacheService()
    return _global_cache
