"""Tests for retry logic and cache cleanup."""
import pytest
import time
from pathlib import Path
from unittest.mock import patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.utils.retry import retry, is_transient, cleanup_cache
from backend.data.real_providers import cache


class TestIsTransient:
    def test_timeout(self):
        assert is_transient(Exception("Connection timeout"))

    def test_429(self):
        # "429 Too Many Requests" contains "too many requests" which is a rate limit, not transient
        assert not is_transient(Exception("429 Too Many Requests"))

    def test_429_without_rate_limit_text(self):
        # Plain "429" without "too many requests" is transient
        assert is_transient(Exception("HTTP 429"))

    def test_not_transient(self):
        assert not is_transient(ValueError("No data found"))

    def test_temporary(self):
        assert is_transient(Exception("Service temporarily unavailable"))


class TestRetryDecorator:
    def test_succeeds_first_try(self):
        call_count = 0

        @retry(max_attempts=3, base_delay=0.01)
        def fn():
            nonlocal call_count
            call_count += 1
            return "ok"

        assert fn() == "ok"
        assert call_count == 1

    def test_retries_on_transient(self):
        call_count = 0

        @retry(max_attempts=3, base_delay=0.01)
        def fn():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Connection timeout")
            return "ok"

        assert fn() == "ok"
        assert call_count == 3

    def test_no_retry_on_non_transient(self):
        @retry(max_attempts=3, base_delay=0.01)
        def fn():
            raise ValueError("bad input")

        with pytest.raises(ValueError):
            fn()

    def test_exhausts_retries(self):
        @retry(max_attempts=2, base_delay=0.01)
        def fn():
            raise ConnectionError("timeout")

        with pytest.raises(ConnectionError):
            fn()


class TestProviderRetry:
    """Test that sentiment and macro providers use retry."""

    @patch('backend.data.real_providers.sentiment_data._fetch_articles')
    @patch('backend.data.real_providers.sentiment_data.cache')
    def test_sentiment_uses_retry(self, mock_cache, mock_fetch):
        mock_cache.get.return_value = None
        mock_cache.get_stale.return_value = (None, 0)
        mock_fetch.side_effect = ConnectionError("timeout")
        from backend.data.real_providers.sentiment_data import get_sentiment
        result = get_sentiment("TSLA")
        assert result is not None

    @patch('backend.data.real_providers.macro_data._fetch_fred_series')
    @patch('backend.data.real_providers.macro_data.cache')
    def test_macro_uses_retry(self, mock_cache, mock_fetch):
        mock_cache.get.return_value = None
        mock_cache.get_stale.return_value = (None, 0)
        mock_fetch.side_effect = ConnectionError("timeout")
        from backend.data.real_providers.macro_data import get_macro_data
        result = get_macro_data()
        assert result is not None


class TestCacheCleanup:
    @pytest.fixture(autouse=True)
    def tmp_cache(self, tmp_path):
        with patch.object(cache, 'CACHE_DIR', tmp_path):
            yield tmp_path

    def test_removes_old_entries(self, tmp_cache):
        cache.put("old", "data")
        old_file = list(tmp_cache.glob("*.pkl"))[0]
        import os
        old_time = time.time() - 25 * 3600
        os.utime(old_file, (old_time, old_time))

        removed = cleanup_cache(max_age_hours=24)
        assert removed == 1
        assert cache.get("old") is None

    def test_keeps_fresh_entries(self, tmp_cache):
        cache.put("fresh", "data")
        removed = cleanup_cache(max_age_hours=24)
        assert removed == 0
        assert cache.get("fresh") == "data"

    def test_empty_cache_dir(self, tmp_cache):
        assert cleanup_cache() == 0
