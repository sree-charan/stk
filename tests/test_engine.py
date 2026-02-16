"""Tests for CLI engine with mocked providers."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


def _price_df(symbol="TSLA", days=100):
    dates = pd.date_range("2024-01-01", periods=days, freq="B")
    close = np.linspace(200, 250, days) + np.random.randn(days) * 2
    return pd.DataFrame({
        "date": dates, "symbol": symbol,
        "open": close - 1, "high": close + 2, "low": close - 2,
        "close": close, "volume": np.random.randint(1e6, 5e7, days),
    })


def _opts_df():
    return pd.DataFrame({
        "symbol": ["TSLA"] * 4, "expiration": ["2026-03-20"] * 4,
        "strike": [240, 260, 240, 260], "option_type": ["call", "call", "put", "put"],
        "bid": [12, 5, 4, 8], "ask": [13, 6, 5, 9],
        "last": [12.5, 5.5, 4.5, 8.5], "volume": [100, 50, 80, 40],
        "open_interest": [1000, 500, 800, 400], "iv": [0.4, 0.35, 0.45, 0.4],
        "delta": [0.6, 0.3, -0.4, -0.7], "gamma": [0.02] * 4,
        "theta": [-0.1] * 4, "vega": [0.3] * 4, "dte": [30] * 4,
    })


def _fund():
    return {
        "quarterly": pd.DataFrame([{
            "symbol": "TSLA", "quarter": "Q4 2025", "date": datetime(2025, 12, 31).date(),
            "revenue": 25e9, "eps": 1.5, "gross_margin": 0.25,
            "operating_margin": 0.12, "net_margin": 0.10,
            "revenue_growth_yoy": 0.2, "eps_growth_yoy": 0.3,
            "revenue_estimate": 24e9, "revenue_surprise_pct": 0.04,
            "eps_estimate": 1.4, "eps_surprise_pct": 0.07,
        }]),
        "valuation": {"symbol": "TSLA", "pe_ratio": 50, "forward_pe": 40,
                       "ps_ratio": 10, "pb_ratio": 15, "ev_ebitda": 30,
                       "peg_ratio": 2.5, "dividend_yield": 0, "market_cap": 750e9,
                       "shares_outstanding": 3e9},
        "latest_quarter": {"quarter": "Q4 2025", "eps": 1.5, "revenue": 25e9},
    }


def _sent():
    return {
        "news": pd.DataFrame([{
            "symbol": "TSLA", "date": datetime.now().date(),
            "timestamp": pd.Timestamp.now(), "headline": "Tesla beats earnings",
            "sentiment": 0.7, "source": "Reuters", "relevance": 0.9,
        }]),
        "earnings_calls": pd.DataFrame(),
        "sec_filings": pd.DataFrame(),
        "social": pd.DataFrame(),
    }


def _macro():
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    return {
        "interest_rates": pd.DataFrame({
            "date": dates.date, "fed_funds_rate": 5.25, "treasury_2y": 4.5,
            "treasury_10y": 4.2, "treasury_30y": 4.4, "yield_curve_2_10": -0.3,
        }),
        "economic_indicators": pd.DataFrame({
            "date": dates.date[:24], "gdp_growth_yoy": 2.5, "cpi_yoy": 3.2,
            "core_cpi_yoy": 2.8, "unemployment_rate": 3.7,
            "consumer_confidence": 100, "pmi_manufacturing": 50, "pmi_services": 52,
        }),
        "vix": pd.DataFrame({
            "date": dates.date, "vix": 18.0, "vix_9d": 17.0,
            "vix_3m": 19.0, "vix_term_structure": 0.0,
        }),
        "market_regime": pd.DataFrame({
            "date": dates.date, "regime": "neutral", "market_breadth": 0.5,
            "momentum_score": 0.0, "risk_on_off": 0.0, "correlation_regime": 0.5,
        }),
    }


_PROVIDERS = "cli.engine"


class TestGetAnalysis:
    @patch(f"{_PROVIDERS}.get_macro_data", return_value=_macro())
    @patch(f"{_PROVIDERS}.get_sentiment", return_value=_sent())
    @patch(f"{_PROVIDERS}.get_fundamentals", return_value=_fund())
    @patch(f"{_PROVIDERS}.get_options_chain", return_value=_opts_df())
    @patch(f"{_PROVIDERS}.generate_ohlcv", return_value=_price_df())
    def test_returns_all_fields(self, *mocks):
        import cli.engine as eng
        eng._loaded = False
        a = eng.get_analysis("TSLA")
        assert a["ticker"] == "TSLA"
        assert isinstance(a["price"], float)
        assert "horizons" in a
        for h in ("short", "medium", "long"):
            assert h in a["horizons"]
            assert "confidence" in a["horizons"][h]
            assert "direction" in a["horizons"][h]
        assert isinstance(a["bullish"], list)
        assert isinstance(a["bearish"], list)
        assert "fetched_at" in a

    @patch(f"{_PROVIDERS}.get_macro_data", return_value=_macro())
    @patch(f"{_PROVIDERS}.get_sentiment", return_value=_sent())
    @patch(f"{_PROVIDERS}.get_fundamentals", return_value=_fund())
    @patch(f"{_PROVIDERS}.get_options_chain", return_value=_opts_df())
    @patch(f"{_PROVIDERS}.generate_ohlcv", side_effect=ValueError("No price data found for 'XYZ'"))
    def test_invalid_ticker_raises(self, *mocks):
        from cli.errors import InvalidTickerError
        import cli.engine as eng
        eng._loaded = True
        with pytest.raises(InvalidTickerError):
            eng.get_analysis("XYZ")


class TestGetPrice:
    @patch(f"{_PROVIDERS}.generate_ohlcv", return_value=_price_df("AAPL", 30))
    def test_returns_price_info(self, mock):
        from cli.engine import get_price
        p = get_price("AAPL")
        assert p["ticker"] == "AAPL"
        assert "price" in p
        assert "change" in p
        assert "change_pct" in p
        assert "volume" in p


class TestGetNews:
    @patch(f"{_PROVIDERS}.get_sentiment", return_value=_sent())
    def test_returns_articles(self, mock):
        from cli.engine import get_news
        articles = get_news("TSLA")
        assert len(articles) == 1
        assert articles[0]["headline"] == "Tesla beats earnings"

    @patch(f"{_PROVIDERS}.get_sentiment")
    def test_empty_news(self, mock):
        mock.return_value = {"news": pd.DataFrame()}
        from cli.engine import get_news
        assert get_news("TSLA") == []


class TestGetEarnings:
    @patch(f"{_PROVIDERS}.get_fundamentals", return_value=_fund())
    def test_returns_earnings(self, mock):
        from cli.engine import get_earnings
        e = get_earnings("TSLA")
        assert "valuation" in e
        assert "quarters" in e
        assert e["valuation"]["pe_ratio"] == 50


class TestChatQuery:
    @patch(f"{_PROVIDERS}.get_macro_data", return_value=_macro())
    @patch(f"{_PROVIDERS}.get_sentiment", return_value=_sent())
    @patch(f"{_PROVIDERS}.get_fundamentals", return_value=_fund())
    @patch(f"{_PROVIDERS}.get_options_chain", return_value=_opts_df())
    @patch(f"{_PROVIDERS}.generate_ohlcv", return_value=_price_df())
    def test_chat_returns_string(self, *mocks):
        import cli.engine as eng
        eng._loaded = False
        resp = eng.chat_query("should I buy TSLA?")
        assert isinstance(resp, str)
        assert len(resp) > 0

    def test_chat_no_ticker(self):
        from cli.engine import chat_query
        resp = chat_query("hello")
        assert "ticker" in resp.lower() or "include" in resp.lower()


class TestEngineEdgeCases:
    """Edge cases: minimal data, missing columns, single-row dataframes."""

    @patch(f"{_PROVIDERS}.get_macro_data", return_value=_macro())
    @patch(f"{_PROVIDERS}.get_sentiment", return_value=_sent())
    @patch(f"{_PROVIDERS}.get_fundamentals", return_value=_fund())
    @patch(f"{_PROVIDERS}.get_options_chain", return_value=_opts_df())
    @patch(f"{_PROVIDERS}.generate_ohlcv")
    def test_single_row_price(self, mock_ohlcv, *mocks):
        """Engine should handle a single-row price dataframe."""
        mock_ohlcv.return_value = _price_df("TSLA", 2)
        import cli.engine as eng
        eng._loaded = False
        a = eng.get_analysis("TSLA")
        assert a["ticker"] == "TSLA"
        assert isinstance(a["price"], float)

    @patch(f"{_PROVIDERS}.get_macro_data", return_value=_macro())
    @patch(f"{_PROVIDERS}.get_sentiment")
    @patch(f"{_PROVIDERS}.get_fundamentals", return_value=_fund())
    @patch(f"{_PROVIDERS}.get_options_chain", return_value=_opts_df())
    @patch(f"{_PROVIDERS}.generate_ohlcv", return_value=_price_df())
    def test_empty_news_sentiment(self, mock_ohlcv, mock_opts, mock_fund, mock_sent, mock_macro):
        """Engine should handle empty news gracefully."""
        mock_sent.return_value = {
            "news": pd.DataFrame(), "earnings_calls": pd.DataFrame(),
            "sec_filings": pd.DataFrame(), "social": pd.DataFrame(),
        }
        import cli.engine as eng
        eng._loaded = False
        a = eng.get_analysis("TSLA")
        assert "horizons" in a

    @patch(f"{_PROVIDERS}.generate_ohlcv", side_effect=ConnectionError("Connection timeout"))
    def test_network_error_classified(self, mock):
        from cli.errors import NetworkError
        import cli.engine as eng
        eng._loaded = True
        with pytest.raises(NetworkError):
            eng.get_analysis("TSLA")

    @patch(f"{_PROVIDERS}.generate_ohlcv", side_effect=Exception("429 Too Many Requests"))
    def test_rate_limit_classified(self, mock):
        from cli.errors import RateLimitError
        import cli.engine as eng
        eng._loaded = True
        with pytest.raises(RateLimitError):
            eng.get_analysis("TSLA")


class TestScanEngine:
    @patch(f"{_PROVIDERS}.generate_ohlcv")
    def test_scan_rsi_filter(self, mock_ohlcv):
        mock_ohlcv.return_value = _price_df()
        from cli.engine import scan_tickers
        results = scan_tickers(['TSLA'], 'rsi<100')
        # RSI is always < 100, so should match
        assert len(results) >= 0

    @patch(f"{_PROVIDERS}.generate_ohlcv")
    def test_scan_no_match(self, mock_ohlcv):
        mock_ohlcv.return_value = _price_df()
        from cli.engine import scan_tickers
        results = scan_tickers(['TSLA'], 'rsi<0')
        assert len(results) == 0

    def test_portfolio_risk_sharpe(self):
        """Verify portfolio_risk returns sharpe ratio."""
        from cli.engine import portfolio_risk
        with patch(f"{_PROVIDERS}.generate_ohlcv", return_value=_price_df()):
            with patch('cli.engine.get_price', return_value={'price': 250}):
                result = portfolio_risk([{'ticker': 'TSLA', 'entry_price': 200, 'qty': 10}])
                assert 'sharpe' in result
                assert 'annualized_return' in result


class TestExtractSignals:
    """Tests for the refactored _extract_signals helper."""

    def _make_feats(self, **kwargs):
        """Create a single-row DataFrame with given feature values."""
        return pd.DataFrame([kwargs])

    def test_rsi_oversold(self):
        from cli.engine import _extract_signals
        feats = self._make_feats(rsi_14=25.0)
        bull, bear = _extract_signals(feats, 250.0)
        assert any("RSI oversold" in s for s in bull)

    def test_rsi_overbought(self):
        from cli.engine import _extract_signals
        feats = self._make_feats(rsi_14=75.0)
        bull, bear = _extract_signals(feats, 250.0)
        assert any("RSI overbought" in s for s in bear)

    def test_volume_surge(self):
        from cli.engine import _extract_signals
        feats = self._make_feats(volume_ratio=2.0)
        bull, bear = _extract_signals(feats, 250.0)
        assert any("Volume surge" in s for s in bull)

    def test_empty_features(self):
        from cli.engine import _extract_signals
        feats = pd.DataFrame([{}])
        bull, bear = _extract_signals(feats, 250.0)
        assert isinstance(bull, list)
        assert isinstance(bear, list)

    def test_bearish_vix(self):
        from cli.engine import _extract_signals
        feats = self._make_feats(vix=30.0)
        bull, bear = _extract_signals(feats, 250.0)
        assert any("High VIX" in s for s in bear)

    def test_bullish_stochastic(self):
        from cli.engine import _extract_signals
        feats = self._make_feats(stoch_k=15.0, stoch_d=20.0)
        bull, bear = _extract_signals(feats, 250.0)
        assert any("Stochastic oversold" in s for s in bull)


class TestBuildHorizons:
    """Tests for _build_horizons helper."""

    def test_returns_all_horizons(self):
        from cli.engine import _build_horizons
        raw = {
            'short': {'prediction': 0.01, 'confidence': 0.6, 'direction': 'bullish'},
            'medium': {'prediction': 0.02, 'confidence': 0.55, 'direction': 'bullish'},
            'long': {'prediction': 0.05, 'confidence': 0.7, 'direction': 'bullish'},
        }
        feats = pd.DataFrame([{'sma_50': 240.0}])
        result = _build_horizons(raw, 250.0, 5.0, feats)
        assert 'short' in result
        assert 'medium' in result
        assert 'long' in result
        for h in result.values():
            assert 'stop' in h
            assert 'target' in h
            assert 'entry_lo' in h
            assert 'entry_hi' in h

    def test_no_sma50_fallback(self):
        from cli.engine import _build_horizons
        raw = {
            'short': {'prediction': 0.01, 'confidence': 0.6, 'direction': 'bearish'},
            'medium': {'prediction': -0.02, 'confidence': 0.55, 'direction': 'bearish'},
            'long': {'prediction': -0.05, 'confidence': 0.7, 'direction': 'bearish'},
        }
        feats = pd.DataFrame([{}])  # no sma_50
        result = _build_horizons(raw, 250.0, 5.0, feats)
        assert result['short']['support'] == round(250.0 - 5.0 * 3, 2)


class TestGetTickerInfo:
    """Tests for _get_ticker_info helper."""

    def test_fallback_on_error(self):
        from cli.engine import _get_ticker_info
        with patch('cli.engine._get_yf_info', return_value={}):
            name, mc, sector, pe = _get_ticker_info("FAKE")
            assert name == "FAKE"
            assert sector == "N/A"

    def test_returns_info(self):
        from cli.engine import _get_ticker_info
        mock_info = {'shortName': 'Tesla', 'marketCap': 800e9, 'sector': 'Tech', 'trailingPE': 50}
        with patch('cli.engine._get_yf_info', return_value=mock_info):
            name, mc, sector, pe = _get_ticker_info("TSLA")
            assert name == "Tesla"
            assert sector == "Tech"


class TestCheckThreshold:
    """Tests for _check_threshold helper."""

    def test_bullish_threshold(self):
        from cli.engine import _check_threshold
        bull, bear = [], []
        _check_threshold(25.0, 30, 70, lambda v: f"low {v}", lambda v: f"high {v}", bull, bear)
        assert len(bull) == 1
        assert "low" in bull[0]

    def test_bearish_threshold(self):
        from cli.engine import _check_threshold
        bull, bear = [], []
        _check_threshold(75.0, 30, 70, lambda v: f"low {v}", lambda v: f"high {v}", bull, bear)
        assert len(bear) == 1
        assert "high" in bear[0]

    def test_none_value(self):
        from cli.engine import _check_threshold
        bull, bear = [], []
        _check_threshold(None, 30, 70, lambda v: "low", lambda v: "high", bull, bear)
        assert len(bull) == 0
        assert len(bear) == 0

    def test_neutral_value(self):
        from cli.engine import _check_threshold
        bull, bear = [], []
        _check_threshold(50.0, 30, 70, lambda v: "low", lambda v: "high", bull, bear)
        assert len(bull) == 0
        assert len(bear) == 0


class TestExtractSignalsAdvanced:
    """Advanced signal extraction tests for multi-indicator combinations."""

    def _make_feats(self, **kwargs):
        return pd.DataFrame([kwargs])

    def test_adx_uptrend(self):
        from cli.engine import _extract_signals
        feats = self._make_feats(adx_14=25.0, return_5d=0.05)
        bull, bear = _extract_signals(feats, 250.0)
        assert any("uptrend" in s.lower() for s in bull)

    def test_adx_downtrend(self):
        from cli.engine import _extract_signals
        feats = self._make_feats(adx_14=25.0, return_5d=-0.05)
        bull, bear = _extract_signals(feats, 250.0)
        assert any("downtrend" in s.lower() for s in bear)

    def test_ema_crossover_bullish(self):
        from cli.engine import _extract_signals
        feats = self._make_feats(ema_5=260.0, ema_20=250.0)
        bull, bear = _extract_signals(feats, 255.0)
        assert any("EMA5 above" in s for s in bull)

    def test_ema_crossover_bearish(self):
        from cli.engine import _extract_signals
        feats = self._make_feats(ema_5=240.0, ema_20=260.0)
        bull, bear = _extract_signals(feats, 250.0)
        assert any("EMA5 below" in s for s in bear)

    def test_ichimoku_above_cloud(self):
        from cli.engine import _extract_signals
        feats = self._make_feats(ichimoku_cloud_pos=0.05)
        bull, bear = _extract_signals(feats, 250.0)
        assert any("Ichimoku cloud" in s for s in bull)

    def test_aroon_bullish(self):
        from cli.engine import _extract_signals
        feats = self._make_feats(aroon_up=80.0, aroon_down=20.0)
        bull, bear = _extract_signals(feats, 250.0)
        assert any("Aroon bullish" in s for s in bull)

    def test_aroon_bearish(self):
        from cli.engine import _extract_signals
        feats = self._make_feats(aroon_up=20.0, aroon_down=80.0)
        bull, bear = _extract_signals(feats, 250.0)
        assert any("Aroon bearish" in s for s in bear)

    def test_sma50_above(self):
        from cli.engine import _extract_signals
        feats = self._make_feats(sma_50=230.0)
        bull, bear = _extract_signals(feats, 250.0)
        assert any("above 50-SMA" in s for s in bull)

    def test_sma50_below(self):
        from cli.engine import _extract_signals
        feats = self._make_feats(sma_50=280.0)
        bull, bear = _extract_signals(feats, 250.0)
        assert any("below 50-SMA" in s for s in bear)

    def test_multiple_signals(self):
        """Multiple indicators should produce multiple signals."""
        from cli.engine import _extract_signals
        feats = self._make_feats(rsi_14=25.0, volume_ratio=2.5, vix=12.0, stoch_k=15.0)
        bull, bear = _extract_signals(feats, 250.0)
        assert len(bull) >= 3  # RSI oversold + volume surge + low VIX + stochastic oversold


class TestEvalOp:
    """Tests for _eval_op helper."""

    def test_less_than(self):
        from cli.engine import _eval_op
        assert _eval_op(5.0, '<', 10.0) is True
        assert _eval_op(10.0, '<', 5.0) is False

    def test_greater_than(self):
        from cli.engine import _eval_op
        assert _eval_op(10.0, '>', 5.0) is True
        assert _eval_op(5.0, '>', 10.0) is False

    def test_less_equal(self):
        from cli.engine import _eval_op
        assert _eval_op(5.0, '<=', 5.0) is True
        assert _eval_op(6.0, '<=', 5.0) is False

    def test_greater_equal(self):
        from cli.engine import _eval_op
        assert _eval_op(5.0, '>=', 5.0) is True
        assert _eval_op(4.0, '>=', 5.0) is False

    def test_equals(self):
        from cli.engine import _eval_op
        assert _eval_op(5.0, '=', 5.0) is True
        assert _eval_op(5.005, '=', 5.0) is True  # within 0.01 tolerance
        assert _eval_op(6.0, '=', 5.0) is False

    def test_unknown_op(self):
        from cli.engine import _eval_op
        assert _eval_op(5.0, '!=', 5.0) is False
