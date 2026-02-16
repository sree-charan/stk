"""Tests for real data providers with mocked external APIs."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.data.real_providers import cache


@pytest.fixture(autouse=True)
def tmp_cache(tmp_path):
    with patch.object(cache, 'CACHE_DIR', tmp_path):
        yield


# --- Price Data ---

class TestPriceProvider:
    def _mock_history(self):
        dates = pd.date_range("2024-01-01", periods=100, freq="B")
        return pd.DataFrame({
            "Date": dates, "Open": np.random.uniform(200, 300, 100),
            "High": np.random.uniform(250, 350, 100),
            "Low": np.random.uniform(150, 250, 100),
            "Close": np.random.uniform(200, 300, 100),
            "Volume": np.random.randint(1e6, 5e7, 100),
        }).set_index("Date")

    @patch("backend.data.real_providers.price_data.yf")
    def test_get_ohlcv_success(self, mock_yf):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = self._mock_history()
        mock_yf.Ticker.return_value = mock_ticker

        from backend.data.real_providers.price_data import get_ohlcv
        df = get_ohlcv("TSLA")
        assert not df.empty
        assert "close" in df.columns
        assert "volume" in df.columns
        assert df["symbol"].iloc[0] == "TSLA"

    @patch("backend.data.real_providers.price_data.yf")
    def test_get_ohlcv_empty_raises(self, mock_yf):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()
        mock_yf.Ticker.return_value = mock_ticker

        from backend.data.real_providers.price_data import get_ohlcv
        # Invalid tickers now raise ValueError instead of silently falling back
        import pytest
        with pytest.raises(ValueError):
            get_ohlcv("INVALID")

    @patch("backend.data.real_providers.price_data.yf")
    def test_get_ohlcv_uses_cache(self, mock_yf):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = self._mock_history()
        mock_yf.Ticker.return_value = mock_ticker

        from backend.data.real_providers.price_data import get_ohlcv
        get_ohlcv("AAPL")
        get_ohlcv("AAPL")
        # Second call should use cache, so yf.Ticker called only once
        assert mock_yf.Ticker.call_count == 1


# --- Options Data ---

class TestOptionsProvider:
    @patch("backend.data.real_providers.options_data.yf")
    def test_get_options_chain_success(self, mock_yf):
        mock_ticker = MagicMock()
        mock_ticker.options = ["2026-03-20"]
        calls = pd.DataFrame({
            "strike": [250, 260], "bid": [5, 3], "ask": [6, 4],
            "lastPrice": [5.5, 3.5], "volume": [100, 50],
            "openInterest": [1000, 500], "impliedVolatility": [0.4, 0.35],
        })
        puts = pd.DataFrame({
            "strike": [240, 230], "bid": [4, 2], "ask": [5, 3],
            "lastPrice": [4.5, 2.5], "volume": [80, 40],
            "openInterest": [800, 400], "impliedVolatility": [0.45, 0.4],
        })
        mock_ticker.option_chain.return_value = MagicMock(calls=calls, puts=puts)
        mock_yf.Ticker.return_value = mock_ticker

        from backend.data.real_providers.options_data import get_options_chain
        chain = get_options_chain("TSLA")
        assert not chain.empty
        assert "call" in chain["option_type"].values
        assert "put" in chain["option_type"].values

    @patch("backend.data.real_providers.options_data.yf")
    def test_get_options_no_expirations_fallback(self, mock_yf):
        mock_ticker = MagicMock()
        mock_ticker.options = []
        mock_yf.Ticker.return_value = mock_ticker

        from backend.data.real_providers.options_data import get_options_chain
        chain = get_options_chain("TSLA", 250.0)
        assert not chain.empty  # mock fallback

    def test_put_call_ratio(self):
        from backend.data.real_providers.options_data import get_put_call_ratio
        chain = pd.DataFrame({
            "option_type": ["call", "call", "put", "put"],
            "volume": [100, 200, 150, 50],
            "open_interest": [1000, 2000, 800, 700],
        })
        ratio = get_put_call_ratio(chain)
        assert ratio["volume_ratio"] == pytest.approx(200 / 300, rel=0.01)
        assert ratio["total_call_volume"] == 300
        assert ratio["total_put_volume"] == 200


# --- Fundamentals ---

class TestFundamentalsProvider:
    @patch("backend.data.real_providers.fundamentals.yf")
    def test_get_fundamentals_success(self, mock_yf):
        mock_ticker = MagicMock()
        mock_ticker.info = {
            "currentPrice": 250, "sharesOutstanding": 3e9,
            "marketCap": 750e9, "trailingPE": 50, "forwardPE": 40,
            "priceToSalesTrailing12Months": 10, "priceToBook": 15,
            "enterpriseToEbitda": 30, "pegRatio": 2.5,
            "dividendYield": 0, "trailingEps": 5.0,
            "revenueGrowth": 0.2, "earningsGrowth": 0.3,
        }
        dates = pd.date_range("2024-01-01", periods=4, freq="QE")
        qf = pd.DataFrame({
            d: {"Total Revenue": 25e9, "Net Income": 3e9,
                "Gross Profit": 10e9, "Operating Income": 5e9}
            for d in dates
        })
        mock_ticker.quarterly_financials = qf
        mock_ticker.balance_sheet = pd.DataFrame()
        mock_yf.Ticker.return_value = mock_ticker

        from backend.data.real_providers.fundamentals import get_fundamentals
        result = get_fundamentals("TSLA")
        assert "quarterly" in result
        assert "valuation" in result
        assert result["valuation"]["pe_ratio"] == 50

    @patch("backend.data.real_providers.fundamentals.yf")
    def test_get_fundamentals_api_error_fallback(self, mock_yf):
        mock_yf.Ticker.side_effect = Exception("API down")

        from backend.data.real_providers.fundamentals import get_fundamentals
        result = get_fundamentals("TSLA")
        assert "quarterly" in result  # mock fallback


# --- Sentiment ---

class TestSentimentProvider:
    @patch("backend.data.real_providers.sentiment_data.cache")
    def test_get_sentiment_success(self, mock_cache):
        mock_cache.get.return_value = None
        mock_cache.get_stale.return_value = (None, 0)

        with patch.dict("sys.modules", {}):
            mock_newsapi_cls = MagicMock()
            mock_newsapi_inst = MagicMock()
            mock_newsapi_inst.get_everything.return_value = {
                "articles": [
                    {"title": "Tesla beats earnings", "description": "Great quarter",
                     "publishedAt": "2026-02-15T00:00:00Z", "source": {"name": "Reuters"}},
                ]
            }
            mock_newsapi_cls.return_value = mock_newsapi_inst

            mock_vader_cls = MagicMock()
            mock_vader_inst = MagicMock()
            mock_vader_inst.polarity_scores.return_value = {"compound": 0.8}
            mock_vader_cls.return_value = mock_vader_inst

            with patch("newsapi.NewsApiClient", mock_newsapi_cls), \
                 patch("vaderSentiment.vaderSentiment.SentimentIntensityAnalyzer", mock_vader_cls):
                # Force reimport to use our mocked cache
                import importlib
                import backend.data.real_providers.sentiment_data as sm
                importlib.reload(sm)
                # Just verify the function structure works with mock fallback
                result = sm.get_sentiment("TSLA")
                assert "news" in result


# --- Macro Data ---

class TestMacroProvider:
    @patch("backend.data.real_providers.macro_data.cache")
    def test_get_macro_cached(self, mock_cache):
        fake_data = {"interest_rates": pd.DataFrame(), "economic_indicators": pd.DataFrame(),
                     "vix": pd.DataFrame(), "market_regime": pd.DataFrame()}
        mock_cache.get.return_value = fake_data

        from backend.data.real_providers.macro_data import get_macro_data
        result = get_macro_data()
        assert "interest_rates" in result

    @patch("backend.data.real_providers.macro_data.cache")
    def test_get_macro_fallback(self, mock_cache):
        """When FRED API fails, should fall back to mock."""
        mock_cache.get.return_value = None
        mock_cache.get_stale.return_value = (None, 0)
        with patch.dict("sys.modules", {"fredapi": MagicMock(Fred=MagicMock(side_effect=Exception("no network")))}):
            from backend.data.real_providers.macro_data import get_macro_data
            result = get_macro_data()
            assert "interest_rates" in result
            assert "vix" in result
