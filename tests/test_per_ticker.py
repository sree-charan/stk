"""Tests for per-ticker model training, after-hours price, retrain, signals, confidence."""
import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch
from datetime import datetime
import sys
import tempfile
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent))

_PROVIDERS = "cli.engine"


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


# ---- After-Hours Price Tests ----

class TestGetCurrentPrice:
    """Tests for _get_current_price helper."""

    def test_post_market_preferred(self):
        from cli.engine import _get_current_price
        info = {'postMarketPrice': 255.0, 'preMarketPrice': 252.0, 'regularMarketPrice': 250.0}
        with patch('cli.engine._get_yf_info', return_value=info):
            assert _get_current_price("TSLA") == 255.0

    def test_pre_market_when_no_post(self):
        from cli.engine import _get_current_price
        info = {'preMarketPrice': 252.0, 'regularMarketPrice': 250.0}
        with patch('cli.engine._get_yf_info', return_value=info):
            assert _get_current_price("TSLA") == 252.0

    def test_regular_market_fallback(self):
        from cli.engine import _get_current_price
        info = {'regularMarketPrice': 250.0}
        with patch('cli.engine._get_yf_info', return_value=info):
            assert _get_current_price("TSLA") == 250.0

    def test_fallback_close(self):
        from cli.engine import _get_current_price
        with patch('cli.engine._get_yf_info', return_value={}):
            assert _get_current_price("TSLA", fallback_close=245.0) == 245.0

    def test_fallback_to_ohlcv(self):
        from cli.engine import _get_current_price
        df = _price_df("TSLA", 5)
        with patch('cli.engine._get_yf_info', return_value={}):
            with patch('cli.engine.generate_ohlcv', return_value=df):
                price = _get_current_price("TSLA")
                assert isinstance(price, float)

    def test_case_insensitive(self):
        from cli.engine import _get_current_price
        info = {'regularMarketPrice': 250.0}
        with patch('cli.engine._get_yf_info', return_value=info):
            assert _get_current_price("tsla") == 250.0


class TestGetYfInfo:
    """Tests for _get_yf_info caching."""

    def test_caches_result(self):
        import cli.engine as eng
        eng._info_cache.clear()
        mock_info = {'regularMarketPrice': 250.0}
        with patch('yfinance.Ticker') as mock_yf:
            mock_yf.return_value.info = mock_info
            result1 = eng._get_yf_info("TSLA")
            result2 = eng._get_yf_info("TSLA")
            assert result1 == mock_info
            assert result2 == mock_info
            # Should only call yfinance once due to caching
            assert mock_yf.call_count == 1
        eng._info_cache.clear()

    def test_returns_empty_on_error(self):
        import cli.engine as eng
        eng._info_cache.clear()
        with patch('yfinance.Ticker', side_effect=Exception("network error")):
            result = eng._get_yf_info("BAD")
            assert result == {}
        eng._info_cache.clear()


# ---- Per-Ticker Model Tests ----

class TestTickerModelDir:
    def test_returns_correct_path(self):
        from cli.engine import _ticker_model_dir
        p = _ticker_model_dir("TSLA")
        assert p == Path.home() / '.stk' / 'models' / 'TSLA'

    def test_uppercases_ticker(self):
        from cli.engine import _ticker_model_dir
        p = _ticker_model_dir("tsla")
        assert p.name == "TSLA"


class TestTickerMeta:
    def test_save_and_load(self):
        import cli.engine as eng
        tmpdir = tempfile.mkdtemp()
        try:
            with patch.object(eng, '_MODELS_DIR', Path(tmpdir)):
                eng._ticker_meta.clear()
                meta = {'trained_at': '2026-01-01', 'accuracy': {'short': 0.6}}
                eng._save_ticker_meta("TEST", meta)
                eng._ticker_meta.clear()  # clear cache to force file read
                loaded = eng._load_ticker_meta("TEST")
                assert loaded is not None
                assert loaded['accuracy']['short'] == 0.6
        finally:
            shutil.rmtree(tmpdir)

    def test_load_nonexistent(self):
        import cli.engine as eng
        eng._ticker_meta.pop("NONEXIST", None)
        tmpdir = tempfile.mkdtemp()
        try:
            with patch.object(eng, '_MODELS_DIR', Path(tmpdir)):
                assert eng._load_ticker_meta("NONEXIST") is None
        finally:
            shutil.rmtree(tmpdir)


class TestTrainTickerModel:
    @patch(f"{_PROVIDERS}.get_macro_data", return_value=_macro())
    @patch(f"{_PROVIDERS}.get_sentiment", return_value=_sent())
    @patch(f"{_PROVIDERS}.get_fundamentals", return_value=_fund())
    @patch(f"{_PROVIDERS}.get_options_chain", return_value=_opts_df())
    def test_trains_and_saves(self, mock_opts, mock_fund, mock_sent, mock_macro):
        import cli.engine as eng
        tmpdir = tempfile.mkdtemp()
        try:
            with patch.object(eng, '_MODELS_DIR', Path(tmpdir)):
                # Mock yfinance history
                hist_df = pd.DataFrame({
                    'Open': np.linspace(200, 250, 200),
                    'High': np.linspace(202, 252, 200),
                    'Low': np.linspace(198, 248, 200),
                    'Close': np.linspace(200, 250, 200),
                    'Volume': np.random.randint(1e6, 5e7, 200),
                }, index=pd.date_range("2024-01-01", periods=200, freq="B"))
                hist_df.index.name = 'Date'

                with patch('yfinance.Ticker') as mock_yf:
                    mock_yf.return_value.history.return_value = hist_df
                    eng._ticker_ensembles.pop("TESTX", None)
                    eng._ticker_meta.pop("TESTX", None)
                    acc = eng.train_ticker_model("TESTX", verbose=False)

                assert 'short' in acc
                assert 'medium' in acc
                assert 'long' in acc
                assert 0 <= acc['short'] <= 1
                # Check model files saved
                model_dir = Path(tmpdir) / 'TESTX'
                assert (model_dir / 'xgb_short.pkl').exists()
                assert (model_dir / 'meta.json').exists()
                # Check meta
                meta = json.loads((model_dir / 'meta.json').read_text())
                assert 'accuracy' in meta
                assert 'trained_at' in meta
                assert meta['samples'] == 200
        finally:
            eng._ticker_ensembles.pop("TESTX", None)
            eng._ticker_meta.pop("TESTX", None)
            shutil.rmtree(tmpdir)

    def test_raises_on_empty_data(self):
        import cli.engine as eng
        with patch('yfinance.Ticker') as mock_yf:
            mock_yf.return_value.history.return_value = pd.DataFrame()
            with pytest.raises(ValueError, match="No data"):
                eng.train_ticker_model("EMPTY", verbose=False)


class TestEnsureTickerModel:
    @patch(f"{_PROVIDERS}.get_macro_data", return_value=_macro())
    @patch(f"{_PROVIDERS}.get_sentiment", return_value=_sent())
    @patch(f"{_PROVIDERS}.get_fundamentals", return_value=_fund())
    @patch(f"{_PROVIDERS}.get_options_chain", return_value=_opts_df())
    @patch(f"{_PROVIDERS}.generate_ohlcv", return_value=_price_df())
    def test_falls_back_to_generic(self, *mocks):
        """When per-ticker training fails, falls back to generic model."""
        import cli.engine as eng
        eng._loaded = False
        eng._ticker_ensembles.pop("FAILX", None)
        tmpdir = tempfile.mkdtemp()
        try:
            with patch.object(eng, '_MODELS_DIR', Path(tmpdir)):
                with patch('yfinance.Ticker', side_effect=Exception("fail")):
                    ens = eng._ensure_ticker_model("FAILX")
                    # Should return the generic ensemble
                    assert ens is eng._ensemble
        finally:
            shutil.rmtree(tmpdir)


class TestRetrainTicker:
    @patch(f"{_PROVIDERS}.get_macro_data", return_value=_macro())
    @patch(f"{_PROVIDERS}.get_sentiment", return_value=_sent())
    @patch(f"{_PROVIDERS}.get_fundamentals", return_value=_fund())
    @patch(f"{_PROVIDERS}.get_options_chain", return_value=_opts_df())
    def test_retrain_clears_cache(self, *mocks):
        import cli.engine as eng
        tmpdir = tempfile.mkdtemp()
        try:
            with patch.object(eng, '_MODELS_DIR', Path(tmpdir)):
                hist_df = pd.DataFrame({
                    'Open': np.linspace(200, 250, 200),
                    'High': np.linspace(202, 252, 200),
                    'Low': np.linspace(198, 248, 200),
                    'Close': np.linspace(200, 250, 200),
                    'Volume': np.random.randint(1e6, 5e7, 200),
                }, index=pd.date_range("2024-01-01", periods=200, freq="B"))
                hist_df.index.name = 'Date'
                with patch('yfinance.Ticker') as mock_yf:
                    mock_yf.return_value.history.return_value = hist_df
                    # Train first
                    eng.train_ticker_model("RTX", verbose=False)
                    assert "RTX" in eng._ticker_ensembles
                    # Retrain clears and retrains
                    acc = eng.retrain_ticker("RTX")
                    assert 'short' in acc
                    assert "RTX" in eng._ticker_ensembles
        finally:
            eng._ticker_ensembles.pop("RTX", None)
            eng._ticker_meta.pop("RTX", None)
            shutil.rmtree(tmpdir)


class TestRetrainAll:
    def test_retrain_all_empty(self):
        import cli.engine as eng
        tmpdir = tempfile.mkdtemp()
        try:
            with patch.object(eng, '_MODELS_DIR', Path(tmpdir)):
                results = eng.retrain_all()
                assert results == {}
        finally:
            shutil.rmtree(tmpdir)

    @patch(f"{_PROVIDERS}.get_macro_data", return_value=_macro())
    @patch(f"{_PROVIDERS}.get_sentiment", return_value=_sent())
    @patch(f"{_PROVIDERS}.get_fundamentals", return_value=_fund())
    @patch(f"{_PROVIDERS}.get_options_chain", return_value=_opts_df())
    def test_retrain_all_with_models(self, *mocks):
        import cli.engine as eng
        tmpdir = tempfile.mkdtemp()
        try:
            with patch.object(eng, '_MODELS_DIR', Path(tmpdir)):
                # Create a fake model dir with meta.json
                d = Path(tmpdir) / 'FAKEX'
                d.mkdir(parents=True)
                (d / 'meta.json').write_text('{"trained_at": "2026-01-01"}')

                hist_df = pd.DataFrame({
                    'Open': np.linspace(200, 250, 200),
                    'High': np.linspace(202, 252, 200),
                    'Low': np.linspace(198, 248, 200),
                    'Close': np.linspace(200, 250, 200),
                    'Volume': np.random.randint(1e6, 5e7, 200),
                }, index=pd.date_range("2024-01-01", periods=200, freq="B"))
                hist_df.index.name = 'Date'
                with patch('yfinance.Ticker') as mock_yf:
                    mock_yf.return_value.history.return_value = hist_df
                    results = eng.retrain_all()
                    assert 'FAKEX' in results
                    assert 'short' in results['FAKEX']
        finally:
            eng._ticker_ensembles.pop("FAKEX", None)
            eng._ticker_meta.pop("FAKEX", None)
            shutil.rmtree(tmpdir)


# ---- Confidence Calibration Tests ----

class TestCalibrateConfidence:
    def test_caps_at_accuracy(self):
        import cli.engine as eng
        eng._ticker_meta['LOWX'] = {'accuracy': {'short': 0.55, 'medium': 0.55, 'long': 0.55}}
        result = eng._calibrate_confidence(0.89, 'LOWX')
        # avg_acc=0.55, ceiling=0.60, allow float tolerance
        assert result < 0.61
        assert result < 0.89  # must be capped below raw
        eng._ticker_meta.pop('LOWX', None)

    def test_no_meta_returns_raw(self):
        import cli.engine as eng
        eng._ticker_meta.pop('NOMETA', None)
        tmpdir = tempfile.mkdtemp()
        try:
            with patch.object(eng, '_MODELS_DIR', Path(tmpdir)):
                result = eng._calibrate_confidence(0.75, 'NOMETA')
                assert result == 0.75
        finally:
            shutil.rmtree(tmpdir)

    def test_high_accuracy_allows_higher_confidence(self):
        import cli.engine as eng
        eng._ticker_meta['HIGHX'] = {'accuracy': {'short': 0.80, 'medium': 0.82, 'long': 0.85}}
        result = eng._calibrate_confidence(0.80, 'HIGHX')
        assert result <= 0.872  # avg=0.823, ceiling=0.873
        eng._ticker_meta.pop('HIGHX', None)

    def test_never_exceeds_95(self):
        import cli.engine as eng
        eng._ticker_meta['PERFX'] = {'accuracy': {'short': 0.99, 'medium': 0.99, 'long': 0.99}}
        result = eng._calibrate_confidence(0.99, 'PERFX')
        assert result <= 0.95
        eng._ticker_meta.pop('PERFX', None)


# ---- Feature Importance Signals Tests ----

class TestFeatureImportanceSignals:
    def test_generates_signals(self):
        from cli.engine import _feature_importance_signals
        from backend.models.ensemble import EnsembleModel
        ens = EnsembleModel()
        ens.xgb_medium._trained = True
        feats = pd.DataFrame([{'a': 0.5, 'b': -0.3, 'c': 0.1, 'd': -0.05, 'e': 0.0}])
        imp = np.array([0.3, 0.2, 0.15, 0.1, 0.05])
        with patch.object(type(ens.xgb_medium.model), 'feature_importances_', new_callable=lambda: property(lambda self: imp)):
            bull, bear = _feature_importance_signals(ens, feats)
            assert any("imp" in s for s in bull + bear)

    def test_no_importance_returns_empty(self):
        from cli.engine import _feature_importance_signals
        from backend.models.ensemble import EnsembleModel
        ens = EnsembleModel()
        feats = pd.DataFrame([{'a': 0.5}])
        bull, bear = _feature_importance_signals(ens, feats)
        assert bull == []
        assert bear == []

    def test_mismatched_columns_returns_empty(self):
        from cli.engine import _feature_importance_signals
        from backend.models.ensemble import EnsembleModel
        ens = EnsembleModel()
        ens.xgb_medium._trained = True
        feats = pd.DataFrame([{'a': 0.5, 'b': -0.3, 'c': 0.1}])  # 3 cols vs 2 importances
        imp = np.array([0.5, 0.5])
        with patch.object(type(ens.xgb_medium.model), 'feature_importances_', new_callable=lambda: property(lambda self: imp)):
            bull, bear = _feature_importance_signals(ens, feats)
            assert bull == []
            assert bear == []


# ---- CLI Command Tests ----

class TestRetrainCLI:
    def test_retrain_no_args(self):
        from click.testing import CliRunner
        from cli.main import cli
        runner = CliRunner()
        result = runner.invoke(cli, ['retrain'])
        assert 'Provide a TICKER or use --all' in result.output

    @patch(f"{_PROVIDERS}.retrain_ticker")
    def test_retrain_ticker_success(self, mock_retrain):
        mock_retrain.return_value = {'short': 0.583, 'medium': 0.621, 'long': 0.674}
        from click.testing import CliRunner
        from cli.main import cli
        runner = CliRunner()
        result = runner.invoke(cli, ['retrain', 'TSLA'])
        assert '58.3%' in result.output
        assert '62.1%' in result.output
        assert '67.4%' in result.output

    @patch(f"{_PROVIDERS}.retrain_ticker", side_effect=Exception("No data"))
    def test_retrain_ticker_error(self, mock_retrain):
        from click.testing import CliRunner
        from cli.main import cli
        runner = CliRunner()
        result = runner.invoke(cli, ['retrain', 'BAD'])
        assert 'Error' in result.output

    @patch('cli.engine.retrain_all')
    def test_retrain_all_empty(self, mock_all):
        mock_all.return_value = {}
        from click.testing import CliRunner
        from cli.main import cli
        runner = CliRunner()
        result = runner.invoke(cli, ['retrain', '--all'])
        assert 'No saved models' in result.output

    @patch('cli.engine.retrain_all')
    def test_retrain_all_success(self, mock_all):
        mock_all.return_value = {'TSLA': {'short': 0.6, 'medium': 0.65, 'long': 0.7}}
        from click.testing import CliRunner
        from cli.main import cli
        runner = CliRunner()
        result = runner.invoke(cli, ['retrain', '--all'])
        assert 'TSLA' in result.output
        assert '60.0%' in result.output


class TestAnalyzeSignalsFlag:
    @patch(f"{_PROVIDERS}.get_macro_data", return_value=_macro())
    @patch(f"{_PROVIDERS}.get_sentiment", return_value=_sent())
    @patch(f"{_PROVIDERS}.get_fundamentals", return_value=_fund())
    @patch(f"{_PROVIDERS}.get_options_chain", return_value=_opts_df())
    @patch(f"{_PROVIDERS}.generate_ohlcv", return_value=_price_df())
    def test_signals_flag_shows_only_signals(self, *mocks):
        import cli.engine as eng
        eng._loaded = False
        from click.testing import CliRunner
        from cli.main import cli
        runner = CliRunner()
        result = runner.invoke(cli, ['analyze', 'TSLA', '--signals', '--mock'])
        assert 'Bullish Signals' in result.output
        assert 'Bearish Signals' in result.output
        # Should NOT show price panel or horizon info
        assert 'SHORT-TERM' not in result.output

    @patch(f"{_PROVIDERS}.get_macro_data", return_value=_macro())
    @patch(f"{_PROVIDERS}.get_sentiment", return_value=_sent())
    @patch(f"{_PROVIDERS}.get_fundamentals", return_value=_fund())
    @patch(f"{_PROVIDERS}.get_options_chain", return_value=_opts_df())
    @patch(f"{_PROVIDERS}.generate_ohlcv", return_value=_price_df())
    def test_verbose_shows_all_signals(self, *mocks):
        import cli.engine as eng
        eng._loaded = False
        from click.testing import CliRunner
        from cli.main import cli
        runner = CliRunner()
        result = runner.invoke(cli, ['analyze', 'TSLA', '--verbose', '--mock'])
        assert 'Feature Values' in result.output


# ---- Analysis uses _get_current_price ----

class TestAnalysisUsesCurrentPrice:
    @patch(f"{_PROVIDERS}.get_macro_data", return_value=_macro())
    @patch(f"{_PROVIDERS}.get_sentiment", return_value=_sent())
    @patch(f"{_PROVIDERS}.get_fundamentals", return_value=_fund())
    @patch(f"{_PROVIDERS}.get_options_chain", return_value=_opts_df())
    @patch(f"{_PROVIDERS}.generate_ohlcv", return_value=_price_df())
    def test_analysis_uses_after_hours_price(self, *mocks):
        import cli.engine as eng
        eng._loaded = False
        info = {'postMarketPrice': 999.0}
        with patch('cli.engine._get_yf_info', return_value=info):
            a = eng.get_analysis("TSLA")
            assert a['price'] == 999.0

    @patch(f"{_PROVIDERS}.generate_ohlcv", return_value=_price_df("AAPL", 30))
    def test_get_price_uses_after_hours(self, mock):
        import cli.engine as eng
        info = {'postMarketPrice': 188.5}
        with patch('cli.engine._get_yf_info', return_value=info):
            p = eng.get_price("AAPL")
            assert p['price'] == 188.5


# ---- Confidence in analysis is calibrated ----

class TestAnalysisConfidenceCalibrated:
    @patch(f"{_PROVIDERS}.get_macro_data", return_value=_macro())
    @patch(f"{_PROVIDERS}.get_sentiment", return_value=_sent())
    @patch(f"{_PROVIDERS}.get_fundamentals", return_value=_fund())
    @patch(f"{_PROVIDERS}.get_options_chain", return_value=_opts_df())
    @patch(f"{_PROVIDERS}.generate_ohlcv", return_value=_price_df())
    def test_confidence_capped_by_accuracy(self, *mocks):
        import cli.engine as eng
        eng._loaded = False
        old_ens = eng._ticker_ensembles.pop('TSLA', None)
        eng._ticker_meta['TSLA'] = {'accuracy': {'short': 0.55, 'medium': 0.55, 'long': 0.55}}
        try:
            a = eng.get_analysis("TSLA")
            for h in ('short', 'medium', 'long'):
                # Confidence is now classifier probability (0-1), accuracy shown separately
                assert 0.0 <= a['horizons'][h]['confidence'] <= 1.0
                assert a['horizons'][h].get('wf_accuracy') is not None
        finally:
            eng._ticker_meta.pop('TSLA', None)
            eng._ticker_ensembles.pop('TSLA', None)
            if old_ens is not None:
                eng._ticker_ensembles['TSLA'] = old_ens


# ---- Additional Edge Case Tests ----

class TestGetYfInfoTTLExpiry:
    def test_cache_expires_after_ttl(self):
        import cli.engine as eng
        import time as _time
        eng._info_cache.clear()
        mock_info1 = {'regularMarketPrice': 250.0}
        mock_info2 = {'regularMarketPrice': 260.0}
        with patch('yfinance.Ticker') as mock_yf:
            mock_yf.return_value.info = mock_info1
            eng._get_yf_info("EXPX")
            # Manually expire the cache
            eng._info_cache["EXPX"] = (_time.time() - eng._INFO_TTL - 1, mock_info1)
            mock_yf.return_value.info = mock_info2
            result = eng._get_yf_info("EXPX")
            assert result == mock_info2
            assert mock_yf.call_count == 2
        eng._info_cache.clear()


class TestCalibrateConfidenceEdgeCases:
    def test_empty_accuracy_dict(self):
        import cli.engine as eng
        eng._ticker_meta['EMPTACC'] = {'accuracy': {}}
        result = eng._calibrate_confidence(0.80, 'EMPTACC')
        # Empty accuracy → avg_acc=0.6, ceiling=0.65
        assert result <= 0.65
        eng._ticker_meta.pop('EMPTACC', None)

    def test_single_horizon_accuracy(self):
        import cli.engine as eng
        eng._ticker_meta['SINGX'] = {'accuracy': {'medium': 0.70}}
        result = eng._calibrate_confidence(0.80, 'SINGX')
        # avg_acc=0.70, ceiling=0.75
        assert result <= 0.75
        eng._ticker_meta.pop('SINGX', None)


# ---- Zero-price edge case ----

class TestGetCurrentPriceZeroValues:
    """Test that zero-value prices (falsy but present) are skipped."""

    def test_zero_post_market_falls_through(self):
        from cli.engine import _get_current_price
        info = {'postMarketPrice': 0, 'preMarketPrice': 252.0, 'regularMarketPrice': 250.0}
        with patch('cli.engine._get_yf_info', return_value=info):
            assert _get_current_price("TSLA") == 252.0

    def test_zero_post_and_pre_falls_to_regular(self):
        from cli.engine import _get_current_price
        info = {'postMarketPrice': 0.0, 'preMarketPrice': 0, 'regularMarketPrice': 250.0}
        with patch('cli.engine._get_yf_info', return_value=info):
            assert _get_current_price("TSLA") == 250.0

    def test_all_zero_uses_fallback(self):
        from cli.engine import _get_current_price
        info = {'postMarketPrice': 0, 'preMarketPrice': 0, 'regularMarketPrice': 0}
        with patch('cli.engine._get_yf_info', return_value=info):
            assert _get_current_price("TSLA", fallback_close=245.0) == 245.0

    def test_none_values_skipped(self):
        from cli.engine import _get_current_price
        info = {'postMarketPrice': None, 'regularMarketPrice': 250.0}
        with patch('cli.engine._get_yf_info', return_value=info):
            assert _get_current_price("TSLA") == 250.0


# ---- Retrain --all with partial failures ----

class TestRetrainAllPartialFailure:
    @patch(f"{_PROVIDERS}.get_macro_data", return_value=_macro())
    @patch(f"{_PROVIDERS}.get_sentiment", return_value=_sent())
    @patch(f"{_PROVIDERS}.get_fundamentals", return_value=_fund())
    @patch(f"{_PROVIDERS}.get_options_chain", return_value=_opts_df())
    def test_retrain_all_one_fails_one_succeeds(self, *mocks):
        import cli.engine as eng
        tmpdir = tempfile.mkdtemp()
        try:
            with patch.object(eng, '_MODELS_DIR', Path(tmpdir)):
                # Create two ticker dirs with meta.json
                for t in ('GOODX', 'BADX'):
                    d = Path(tmpdir) / t
                    d.mkdir(parents=True)
                    (d / 'meta.json').write_text('{"trained_at": "2026-01-01"}')

                hist_df = pd.DataFrame({
                    'Open': np.linspace(200, 250, 200),
                    'High': np.linspace(202, 252, 200),
                    'Low': np.linspace(198, 248, 200),
                    'Close': np.linspace(200, 250, 200),
                    'Volume': np.random.randint(1e6, 5e7, 200),
                }, index=pd.date_range("2024-01-01", periods=200, freq="B"))
                hist_df.index.name = 'Date'

                call_count = [0]

                def mock_history(**kwargs):
                    call_count[0] += 1
                    # Fail on second ticker
                    if call_count[0] > 1:
                        return pd.DataFrame()
                    return hist_df

                with patch('yfinance.Ticker') as mock_yf:
                    mock_yf.return_value.history.side_effect = mock_history
                    results = eng.retrain_all()

                # One should succeed, one should have error
                assert len(results) == 2
                success = [t for t, r in results.items() if 'short' in r]
                errors = [t for t, r in results.items() if 'error' in r]
                assert len(success) == 1
                assert len(errors) == 1
        finally:
            eng._ticker_ensembles.pop("GOODX", None)
            eng._ticker_ensembles.pop("BADX", None)
            eng._ticker_meta.pop("GOODX", None)
            eng._ticker_meta.pop("BADX", None)
            shutil.rmtree(tmpdir)


# ---- Integration: analyze → retrain → re-analyze ----

class TestAnalyzeRetrainFlow:
    @patch(f"{_PROVIDERS}.get_macro_data", return_value=_macro())
    @patch(f"{_PROVIDERS}.get_sentiment", return_value=_sent())
    @patch(f"{_PROVIDERS}.get_fundamentals", return_value=_fund())
    @patch(f"{_PROVIDERS}.get_options_chain", return_value=_opts_df())
    @patch(f"{_PROVIDERS}.generate_ohlcv", return_value=_price_df("FLOWX", 100))
    def test_full_flow(self, *mocks):
        import cli.engine as eng
        tmpdir = tempfile.mkdtemp()
        try:
            with patch.object(eng, '_MODELS_DIR', Path(tmpdir)):
                hist_df = pd.DataFrame({
                    'Open': np.linspace(200, 250, 200),
                    'High': np.linspace(202, 252, 200),
                    'Low': np.linspace(198, 248, 200),
                    'Close': np.linspace(200, 250, 200),
                    'Volume': np.random.randint(1e6, 5e7, 200),
                }, index=pd.date_range("2024-01-01", periods=200, freq="B"))
                hist_df.index.name = 'Date'

                info = {'regularMarketPrice': 250.0}
                with patch('yfinance.Ticker') as mock_yf, \
                     patch('cli.engine._get_yf_info', return_value=info), \
                     patch('cli.db.save_prediction'):
                    mock_yf.return_value.history.return_value = hist_df
                    mock_yf.return_value.info = info

                    eng._ticker_ensembles.pop("FLOWX", None)
                    eng._ticker_meta.pop("FLOWX", None)
                    eng._loaded = False

                    # First analyze: trains model
                    a1 = eng.get_analysis("FLOWX")
                    assert a1['ticker'] == 'FLOWX'
                    assert 'FLOWX' in eng._ticker_ensembles

                    # Retrain
                    acc = eng.retrain_ticker("FLOWX")
                    assert 'short' in acc

                    # Second analyze: uses retrained model
                    a2 = eng.get_analysis("FLOWX")
                    assert a2['ticker'] == 'FLOWX'
        finally:
            eng._ticker_ensembles.pop("FLOWX", None)
            eng._ticker_meta.pop("FLOWX", None)
            shutil.rmtree(tmpdir)


class TestGetCurrentPriceNegativeFallback:
    """Negative fallback_close should be ignored, falling through to fetch."""

    def test_negative_fallback_skipped(self):
        import cli.engine as eng
        eng._info_cache.clear()
        with patch('cli.engine._get_yf_info', return_value={}):
            df = _price_df("TEST", 10)
            with patch('cli.engine.generate_ohlcv', return_value=df):
                price = eng._get_current_price("TEST", fallback_close=-5.0)
                assert price > 0

    def test_zero_fallback_skipped(self):
        import cli.engine as eng
        eng._info_cache.clear()
        with patch('cli.engine._get_yf_info', return_value={}):
            df = _price_df("TEST", 10)
            with patch('cli.engine.generate_ohlcv', return_value=df):
                price = eng._get_current_price("TEST", fallback_close=0.0)
                assert price > 0

    def test_positive_fallback_used(self):
        import cli.engine as eng
        eng._info_cache.clear()
        with patch('cli.engine._get_yf_info', return_value={}):
            price = eng._get_current_price("TEST", fallback_close=42.5)
            assert price == 42.5


class TestCorruptedModelFile:
    """_get_ticker_ensemble should fall back to generic on corrupted files."""

    def test_corrupted_load_returns_fallback(self):
        import cli.engine as eng
        eng._ticker_ensembles.pop("CORRUPT", None)
        tmpdir = Path(tempfile.mkdtemp())
        try:
            model_dir = tmpdir / "CORRUPT"
            model_dir.mkdir(parents=True)
            # Write garbage to model files
            (model_dir / "xgb_short.pkl").write_bytes(b"not a pickle")
            (model_dir / "xgb_medium.pkl").write_bytes(b"not a pickle")
            (model_dir / "xgb_long.pkl").write_bytes(b"not a pickle")
            (model_dir / "lstm.pkl").write_bytes(b"not a pickle")

            with patch.object(eng, '_MODELS_DIR', tmpdir):
                ens = eng._get_ticker_ensemble("CORRUPT")
                # Should return the fallback _ensemble, not crash
                assert ens is eng._ensemble
        finally:
            eng._ticker_ensembles.pop("CORRUPT", None)
            shutil.rmtree(tmpdir)


class TestTrainTickerModelShortHistory:
    """train_ticker_model should reject tickers with < 50 days of history."""

    def test_short_history_raises(self):
        import cli.engine as eng
        # Create a very short history (20 days)
        dates = pd.date_range(end="2025-01-20", periods=20, freq="B")
        hist_df = pd.DataFrame({
            "Date": dates, "Open": 100.0, "High": 105.0,
            "Low": 95.0, "Close": 102.0, "Volume": 1000000,
        }).set_index("Date")

        with patch('yfinance.Ticker') as mock_yf:
            mock_yf.return_value.history.return_value = hist_df
            with pytest.raises(ValueError, match="Insufficient history"):
                eng.train_ticker_model("SHORTY", verbose=False)

    def test_exactly_50_days_ok(self):
        import cli.engine as eng
        tmpdir = Path(tempfile.mkdtemp())
        try:
            dates = pd.bdate_range(end="2025-03-14", periods=50)
            n = len(dates)
            np.random.seed(42)
            closes = 100 + np.cumsum(np.random.randn(n) * 0.5)
            hist_df = pd.DataFrame({
                "Open": closes - 0.5, "High": closes + 1,
                "Low": closes - 1, "Close": closes, "Volume": np.full(n, 1000000),
            }, index=dates)
            hist_df.index.name = "Date"

            with patch.object(eng, '_MODELS_DIR', tmpdir), \
                 patch('yfinance.Ticker') as mock_yf, \
                 patch('cli.engine.get_options_chain', return_value=_opts_df()), \
                 patch('cli.engine.get_fundamentals', return_value=_fund()), \
                 patch('cli.engine.get_sentiment', return_value=_sent()), \
                 patch('cli.engine.get_macro_data', return_value=_macro()):
                mock_yf.return_value.history.return_value = hist_df
                acc = eng.train_ticker_model("FIFTY", verbose=False)
                assert 'short' in acc
                assert (tmpdir / "FIFTY" / "meta.json").exists()
        finally:
            eng._ticker_ensembles.pop("FIFTY", None)
            eng._ticker_meta.pop("FIFTY", None)
            shutil.rmtree(tmpdir)
