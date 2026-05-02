"""Shared cache layer: in-process lru_cache + persistent diskcache.

All cache keys are derived from **post-PII-scrubbing** content so that
raw user data never lands on disk and semantically identical queries
(same intent, different PII) share a single cache entry.

Usage
-----
::

    from .cache import lru, disk

    # In-process only (resets on restart)
    @lru(maxsize=256)
    def expensive(x: str) -> str: ...

    # Survives restarts; stored under .cache/diskcache/
    @disk("prefix")
    def really_expensive(x: str, y: str) -> str: ...

    # Plain access (for retriever vector cache)
    v = disk_cache["mykey"]
    disk_cache["mykey"] = v

The ``disk`` decorator computes a SHA-256 key from the tuple of positional
args so it works for any pickle-able return type.
"""

from __future__ import annotations

import functools
import hashlib
import pickle
from typing import Any, Callable, TypeVar

import diskcache

from . import config
from .logger import get_logger

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

# Singleton disk cache – all namespaces share one directory.
_CACHE_DIR = config.CACHE_DIR / "diskcache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

disk_cache: diskcache.Cache = diskcache.Cache(str(_CACHE_DIR))


# Decorators

def lru(maxsize: int = 256):
    """In-process LRU cache.  Drop-in for functools.lru_cache with a named maxsize."""
    def decorator(fn: F) -> F:
        return functools.lru_cache(maxsize=maxsize)(fn)  # type: ignore[return-value]
    return decorator


def disk(prefix: str, ttl: int | None = None):
    """Persistent disk cache backed by diskcache.

    The cache key is ``prefix:<sha256 of pickled positional args>``.
    Keyword arguments are intentionally not included — keep callers simple.

    Args:
        prefix: Logical namespace (e.g. ``"llm"``, ``"translate"``).
        ttl:    Optional expiry in seconds.  None means never expires.
    """
    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            raw = pickle.dumps(args)
            digest = hashlib.sha256(raw).hexdigest()
            key = f"{prefix}:{digest}"
            hit = disk_cache.get(key, default=_MISS)
            if hit is not _MISS:
                logger.debug("diskcache HIT  prefix=%s", prefix)
                return hit
            result = fn(*args, **kwargs)
            disk_cache.set(key, result, expire=ttl)
            logger.debug("diskcache MISS prefix=%s (stored)", prefix)
            return result
        return wrapper  # type: ignore[return-value]
    return decorator


# Sentinel that is never equal to a legitimate cached value.
class _MissType:
    __slots__ = ()
    def __repr__(self) -> str:
        return "<CACHE_MISS>"

_MISS = _MissType()
