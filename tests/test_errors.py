"""Tests for custom error types and engine error classification."""
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from cli.errors import InvalidTickerError, NetworkError, RateLimitError, StkError, NoDataError, ConfigError


class TestErrorTypes:
    def test_invalid_ticker(self):
        e = InvalidTickerError("XYZ")
        assert "XYZ" in str(e)
        assert e.ticker == "XYZ"
        assert isinstance(e, StkError)

    def test_network_error(self):
        e = NetworkError("Yahoo Finance", "timeout")
        assert "Yahoo Finance" in str(e)
        assert "timeout" in str(e)

    def test_network_error_no_detail(self):
        e = NetworkError("FRED")
        assert "FRED" in str(e)

    def test_rate_limit_error(self):
        e = RateLimitError("NewsAPI")
        assert "NewsAPI" in str(e)
        assert "Rate limit" in str(e)

    def test_no_data_error(self):
        e = NoDataError("XYZ", "Yahoo Finance")
        assert "XYZ" in str(e)
        assert "Yahoo Finance" in str(e)
        assert e.ticker == "XYZ"
        assert isinstance(e, StkError)

    def test_no_data_error_no_source(self):
        e = NoDataError("XYZ")
        assert "XYZ" in str(e)

    def test_config_error(self):
        e = ConfigError("fred-key", "Run: stk config set fred-key YOUR_KEY")
        assert "fred-key" in str(e)
        assert "stk config set" in str(e)
        assert e.key == "fred-key"
        assert isinstance(e, StkError)

    def test_config_error_no_hint(self):
        e = ConfigError("news-key")
        assert "news-key" in str(e)


class TestErrorClassification:
    def test_classify_no_data(self):
        from cli.engine import _classify_error
        with pytest.raises(InvalidTickerError):
            _classify_error(ValueError("No price data found for 'XYZ'"), "XYZ")

    def test_classify_rate_limit(self):
        from cli.engine import _classify_error
        with pytest.raises(RateLimitError):
            _classify_error(Exception("429 Too Many Requests"), "TSLA")

    def test_classify_connection(self):
        from cli.engine import _classify_error
        with pytest.raises(NetworkError):
            _classify_error(ConnectionError("Connection timeout"), "TSLA")

    def test_classify_unknown_reraises(self):
        from cli.engine import _classify_error
        with pytest.raises(RuntimeError):
            _classify_error(RuntimeError("something else"), "TSLA")
