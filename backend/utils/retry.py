"""Retry logic with exponential backoff and cache cleanup."""
import time
import logging
from functools import wraps
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar('T')

_TRANSIENT = ("timeout", "connection", "429", "temporary", "unavailable")


def is_transient(e: Exception) -> bool:
    msg = str(e).lower()
    # Rate limits are NOT transient — retrying in seconds won't help
    if "ratelimit" in msg or "too many requests" in msg:
        return False
    return any(k in msg for k in _TRANSIENT)


def retry(max_attempts: int = 3, base_delay: float = 1.0):
    """Decorator: retry on transient errors with exponential backoff."""
    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @wraps(fn)
        def wrapper(*args, **kwargs) -> T:
            last_exc = None
            for attempt in range(max_attempts):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    if not is_transient(e) or attempt == max_attempts - 1:
                        raise
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"Retry {attempt+1}/{max_attempts} for {fn.__name__}: {e} (wait {delay:.1f}s)")
                    time.sleep(delay)
            raise last_exc  # unreachable but satisfies type checker
        return wrapper
    return decorator


def cleanup_cache(max_age_hours: int = 24):
    """Remove cache entries older than max_age_hours."""
    from backend.data.real_providers.cache import CACHE_DIR
    if not CACHE_DIR.exists():
        return 0
    cutoff = time.time() - max_age_hours * 3600
    removed = 0
    for f in CACHE_DIR.glob("*.pkl"):
        try:
            if f.stat().st_mtime < cutoff:
                f.unlink()
                removed += 1
        except OSError:
            pass
    return removed
