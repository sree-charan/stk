"""Simple file-based cache with TTL and staleness tracking."""
import hashlib
import pickle
import time
from pathlib import Path
from typing import Any, Optional, Tuple

CACHE_DIR = Path.home() / '.stk' / 'cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_TTL = 300  # 5 minutes


def _key(name: str) -> Path:
    h = hashlib.md5(name.encode()).hexdigest()
    return CACHE_DIR / f"{h}.pkl"


def get(name: str, ttl: int = DEFAULT_TTL) -> Optional[Any]:
    p = _key(name)
    if not p.exists():
        return None
    try:
        with open(p, 'rb') as f:
            entry = pickle.load(f)
        if time.time() - entry['ts'] > ttl:
            return None
        return entry['data']
    except Exception:
        return None


def get_stale(name: str) -> Tuple[Optional[Any], float]:
    """Get cached data regardless of TTL. Returns (data, age_seconds)."""
    p = _key(name)
    if not p.exists():
        return None, 0
    try:
        with open(p, 'rb') as f:
            entry = pickle.load(f)
        return entry['data'], time.time() - entry['ts']
    except Exception:
        return None, 0


def put(name: str, data: Any):
    p = _key(name)
    with open(p, 'wb') as f:
        pickle.dump({'ts': time.time(), 'data': data}, f)
