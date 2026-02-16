"""Tests for the file-based cache module."""
import pytest
import time
from pathlib import Path
from unittest.mock import patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.data.real_providers import cache


class TestCache:
    @pytest.fixture(autouse=True)
    def tmp_cache(self, tmp_path):
        with patch.object(cache, 'CACHE_DIR', tmp_path):
            yield

    def test_put_and_get(self):
        cache.put("test_key", {"value": 42})
        assert cache.get("test_key") == {"value": 42}

    def test_get_missing_returns_none(self):
        assert cache.get("nonexistent") is None

    def test_ttl_expiry(self):
        cache.put("expire_me", "data")
        assert cache.get("expire_me", ttl=1) == "data"
        with patch('backend.data.real_providers.cache.time') as mock_time:
            mock_time.time.return_value = time.time() + 2
            assert cache.get("expire_me", ttl=1) is None

    def test_get_stale_returns_expired_data(self):
        cache.put("stale_key", "old_data")
        with patch('backend.data.real_providers.cache.time') as mock_time:
            mock_time.time.return_value = time.time() + 2
            data, age = cache.get_stale("stale_key")
            assert data == "old_data"
            assert age >= 1.0

    def test_get_stale_missing_returns_none(self):
        data, age = cache.get_stale("missing")
        assert data is None
        assert age == 0

    def test_overwrite(self):
        cache.put("k", "v1")
        cache.put("k", "v2")
        assert cache.get("k") == "v2"

    def test_different_keys_independent(self):
        cache.put("a", 1)
        cache.put("b", 2)
        assert cache.get("a") == 1
        assert cache.get("b") == 2

    def test_stores_various_types(self):
        cache.put("list", [1, 2, 3])
        cache.put("str", "hello")
        cache.put("none", None)
        assert cache.get("list") == [1, 2, 3]
        assert cache.get("str") == "hello"
        assert cache.get("none") is None  # None stored but get returns None for missing too

    def test_corrupted_file_returns_none(self, tmp_path):
        # Write garbage to a cache file
        h = cache._key("corrupt")
        h.write_bytes(b"not a pickle")
        assert cache.get("corrupt") is None
        data, age = cache.get_stale("corrupt")
        assert data is None
