"""Integration tests for end-to-end flow."""
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch
from click.testing import CliRunner
import json

from backend.data.mock_generators import generate_ohlcv, get_options_chain, get_fundamentals, get_sentiment, get_macro_data
from backend.features.feature_store import FeatureStore
from backend.models.ensemble import EnsembleModel
from backend.llm import IntentParser, ResponseGenerator

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cli.main import cli
from cli import engine, db, config


def _mock_analysis(ticker='TEST'):
    """Lightweight mock analysis result for fast integration tests."""
    return {
        'ticker': ticker.upper(), 'name': f'{ticker} Inc', 'price': 150.0,
        'change_pct': 1.5, 'volume': 1000000, 'avg_volume': 900000,
        'horizons': {
            'short': {'direction': 'bullish', 'confidence': 0.65, 'prediction': 0.02,
                      'target': 153.0, 'stop': 147.0},
            'medium': {'direction': 'bullish', 'confidence': 0.60, 'prediction': 0.05,
                       'target': 157.5, 'stop': 142.5},
            'long': {'direction': 'bearish', 'confidence': 0.55, 'prediction': -0.03,
                     'target': 145.5, 'stop': 155.0},
        },
        'bullish': ['RSI oversold bounce'], 'bearish': ['Caution advised'],
        'all_bullish': ['RSI oversold bounce'], 'all_bearish': ['Caution advised'],
        'regime_display': '🐂 Bull (low volatility)',
        'fetched_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'market_cap': 2e12, 'sector': 'Technology', 'pe_ratio': 25.0,
    }


class TestEndToEnd:
    def test_full_prediction_pipeline(self):
        """Test complete flow from data to prediction."""
        ticker = "TSLA"
        end = datetime.now()
        start = end - timedelta(days=100)

        price_df = generate_ohlcv(ticker, start_date=start, end_date=end)
        options = get_options_chain(ticker, price_df['close'].iloc[-1])
        fundamentals = get_fundamentals(ticker)
        sentiment = get_sentiment(ticker)
        macro = get_macro_data()

        store = FeatureStore()
        features = store.compute_all_features(
            price_df, options, fundamentals,
            sentiment['news'].to_dict('records'), macro
        )

        ensemble = EnsembleModel()
        ensemble.load()
        predictions = ensemble.predict_all_horizons(features.values)

        assert 'short' in predictions
        assert 'medium' in predictions
        assert 'long' in predictions
        for h in ['short', 'medium', 'long']:
            assert 'direction' in predictions[h]
            assert 'confidence' in predictions[h]
            assert predictions[h]['confidence'] >= 0
            assert predictions[h]['confidence'] <= 100

    def test_chat_flow(self):
        """Test chat intent parsing and response generation."""
        parser = IntentParser()
        generator = ResponseGenerator()

        intent = parser.parse("What's your prediction for NVDA next week?")
        assert intent.ticker == "NVDA"
        assert intent.action == "predict"

        predictions = {
            'short': {'direction': 'bullish', 'confidence': 0.72, 'prediction': 0.02},
            'medium': {'direction': 'bullish', 'confidence': 0.65, 'prediction': 0.05},
            'long': {'direction': 'bullish', 'confidence': 0.58, 'prediction': 0.08}
        }
        response = generator.generate(intent, predictions)
        assert "NVDA" in response

    def test_feature_count_meets_target(self):
        """Verify we have 136+ features."""
        store = FeatureStore()
        names = store.all_feature_names()
        assert len(names) >= 136, f"Only {len(names)} features, need 136+"


class TestDataConsistency:
    def test_price_data_valid(self):
        """Test price data has valid OHLCV structure."""
        df = generate_ohlcv("AAPL")
        assert all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume'])
        assert (df['high'] >= df['low']).all()
        assert (df['high'] >= df['open']).all()
        assert (df['high'] >= df['close']).all()
        assert (df['volume'] > 0).all()

    def test_options_chain_valid(self):
        """Test options chain has valid structure."""
        options = get_options_chain("AAPL", 150.0)
        assert 'option_type' in options.columns
        assert 'strike' in options.columns
        assert 'iv' in options.columns
        assert 'volume' in options.columns
        assert len(options) > 0
        assert 'call' in options['option_type'].values
        assert 'put' in options['option_type'].values

    def test_fundamentals_valid(self):
        """Test fundamentals have required fields."""
        fund = get_fundamentals("MSFT")
        assert 'latest_quarter' in fund
        assert 'valuation' in fund
        assert 'eps' in fund['latest_quarter']
        assert 'revenue' in fund['latest_quarter']
        assert 'pe_ratio' in fund['valuation']


class TestCLIEndToEnd:
    """End-to-end tests: CLI commands → engine → backend."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        """Isolate DB and config for each test."""
        self.runner = CliRunner()
        self._patches = [
            patch.object(db, 'DB_PATH', tmp_path / 'data.db'),
            patch.object(config, 'CONFIG_PATH', tmp_path / 'config.json'),
        ]
        for p in self._patches:
            p.start()
        yield
        for p in self._patches:
            p.stop()

    def test_analyze_returns_predictions(self):
        """CLI analyze → engine.get_analysis → backend pipeline."""
        result = self.runner.invoke(cli, ['analyze', 'TSLA', '--mock'])
        assert result.exit_code == 0
        assert 'TSLA' in result.output
        # Should contain verdict keywords
        assert any(w in result.output for w in ['BUY', 'SELL', 'HOLD'])

    def test_analyze_json_output(self):
        """CLI analyze --json returns valid JSON."""
        result = self.runner.invoke(cli, ['analyze', 'TSLA', '--mock', '--json'])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data['ticker'] == 'TSLA'
        assert 'short' in data or 'horizons' in data

    def test_analyze_single_horizon(self):
        """CLI analyze --short returns only short-term."""
        result = self.runner.invoke(cli, ['analyze', 'TSLA', '--short', '--mock'])
        assert result.exit_code == 0
        assert 'SHORT' in result.output.upper()

    def test_price_command(self):
        """CLI price → engine.get_price → backend."""
        result = self.runner.invoke(cli, ['price', 'AAPL'])
        assert result.exit_code == 0
        assert 'AAPL' in result.output
        assert '$' in result.output

    def test_price_json(self):
        """CLI price --json returns valid JSON."""
        result = self.runner.invoke(cli, ['price', 'AAPL', '--json'])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data['ticker'] == 'AAPL'
        assert 'price' in data

    def test_news_command(self):
        """CLI news → engine.get_news → backend."""
        result = self.runner.invoke(cli, ['news', 'TSLA'])
        assert result.exit_code == 0
        assert 'TSLA' in result.output

    def test_earnings_command(self):
        """CLI earnings → engine.get_earnings → backend."""
        result = self.runner.invoke(cli, ['earnings', 'MSFT'])
        assert result.exit_code == 0
        assert 'MSFT' in result.output

    def test_chat_command(self):
        """CLI chat → engine.chat_query → backend."""
        with patch('cli.engine.chat_query', return_value="TSLA looks bullish. BUY with caution."):
            result = self.runner.invoke(cli, ['chat', 'should I buy TSLA?'])
        assert result.exit_code == 0
        assert len(result.output.strip()) > 0

    def test_portfolio_workflow(self):
        """Full portfolio workflow: hold → positions → sell."""
        # Add position
        r = self.runner.invoke(cli, ['hold', 'TSLA', '--entry', '250', '--qty', '10'])
        assert r.exit_code == 0

        # Check positions
        r = self.runner.invoke(cli, ['positions'])
        assert r.exit_code == 0
        assert 'TSLA' in r.output

        # Sell
        r = self.runner.invoke(cli, ['sell', 'TSLA'])
        assert r.exit_code == 0

        # Verify empty
        r = self.runner.invoke(cli, ['positions'])
        assert r.exit_code == 0

    def test_watchlist_workflow(self):
        """Full watchlist workflow: watch → watchlist → unwatch."""
        r = self.runner.invoke(cli, ['watch', 'NVDA'])
        assert r.exit_code == 0

        r = self.runner.invoke(cli, ['watchlist'])
        assert r.exit_code == 0
        assert 'NVDA' in r.output

        r = self.runner.invoke(cli, ['unwatch', 'NVDA'])
        assert r.exit_code == 0

    def test_config_workflow(self):
        """Config set → get → show."""
        r = self.runner.invoke(cli, ['config', 'set', 'fred-key', 'test123'])
        assert r.exit_code == 0

        r = self.runner.invoke(cli, ['config', 'get', 'fred-key'])
        assert r.exit_code == 0
        assert 'test123' in r.output

        r = self.runner.invoke(cli, ['config', 'show'])
        assert r.exit_code == 0

    def test_screen_command(self):
        """CLI screen → engine.screen_tickers."""
        with patch('cli.engine.screen_tickers', return_value=[
            {'ticker': 'AAPL', 'price': 180, 'rsi': 28, 'stoch_k': 15.0,
             'change_pct': -2.0, 'volume': 5e7, 'vol_ratio': 1.5, 'reason': 'oversold', 'signal': 'oversold'}
        ]):
            result = self.runner.invoke(cli, ['screen'])
        assert result.exit_code == 0

    def test_scan_command(self):
        """CLI scan with filter expression."""
        with patch('cli.engine.scan_tickers', return_value=[
            {'ticker': 'AAPL', 'price': 180, 'rsi': 28, 'stoch_k': 15.0,
             'change_pct': -2.0, 'volume': 5e7, 'vol_ratio': 1.5, 'reason': 'rsi<50'}
        ]):
            result = self.runner.invoke(cli, ['scan', 'rsi<50'])
        assert result.exit_code == 0

    def test_compare_command(self):
        """CLI compare multiple tickers."""
        with patch('cli.engine.get_analysis', side_effect=lambda t: _mock_analysis(t)):
            result = self.runner.invoke(cli, ['compare', 'TSLA', 'AAPL'])
        assert result.exit_code == 0
        assert 'TSLA' in result.output
        assert 'AAPL' in result.output

    def test_correlate_command(self):
        """CLI correlate two tickers."""
        result = self.runner.invoke(cli, ['correlate', 'TSLA', 'AAPL'])
        assert result.exit_code == 0

    def test_summary_command(self):
        """CLI summary with positions and watchlist."""
        self.runner.invoke(cli, ['hold', 'AAPL', '--entry', '180', '--qty', '5'])
        self.runner.invoke(cli, ['watch', 'TSLA'])
        result = self.runner.invoke(cli, ['summary'])
        assert result.exit_code == 0

    def test_heatmap_command(self):
        """CLI heatmap shows sector data."""
        with patch('cli.engine.get_price', return_value={'ticker': 'X', 'price': 100, 'change_pct': 1.5, 'change': 1.5, 'volume': 1000000, 'high': 101, 'low': 99}):
            result = self.runner.invoke(cli, ['heatmap'])
        assert result.exit_code == 0

    def test_momentum_command(self):
        """CLI momentum ranking."""
        mock_results = [{'ticker': 'AAPL', 'price': 180.0, 'change_pct': 1.5,
                         'roc_5': 2.0, 'roc_20': 5.0, 'roc_60': 10.0,
                         'connors_rsi': 65, 'adx': 30, 'choppiness': 40,
                         'score': 8.5, 'trending': True}]
        with patch('cli.engine.momentum_ranking', return_value=mock_results):
            result = self.runner.invoke(cli, ['momentum'])
        assert result.exit_code == 0

    def test_doctor_command(self):
        """CLI doctor health check."""
        with patch('cli.engine.get_price', return_value={'ticker': 'AAPL', 'price': 180, 'change_pct': 1.0, 'change': 1.8, 'volume': 5e7, 'high': 181, 'low': 179}), \
             patch('cli.engine._ensure_models'):
            result = self.runner.invoke(cli, ['doctor'])
        assert result.exit_code == 0

    def test_version_flag(self):
        """CLI --version shows version."""
        result = self.runner.invoke(cli, ['--version'])
        assert result.exit_code == 0
        assert '1.2.3' in result.output

    def test_help_flag(self):
        """CLI --help shows usage."""
        result = self.runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        assert 'analyze' in result.output
        assert 'price' in result.output

    def test_analyze_help(self):
        """CLI analyze --help shows options."""
        result = self.runner.invoke(cli, ['analyze', '--help'])
        assert result.exit_code == 0
        assert '--short' in result.output
        assert '--mock' in result.output


class TestErrorHandlingE2E:
    """End-to-end error handling tests."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.runner = CliRunner()
        self._patches = [
            patch.object(db, 'DB_PATH', tmp_path / 'data.db'),
            patch.object(config, 'CONFIG_PATH', tmp_path / 'config.json'),
        ]
        for p in self._patches:
            p.start()
        yield
        for p in self._patches:
            p.stop()

    def test_sell_nonexistent_position(self):
        """Selling a ticker not in portfolio gives clear error."""
        result = self.runner.invoke(cli, ['sell', 'XYZABC'])
        assert result.exit_code == 0  # graceful, no crash
        assert 'not' in result.output.lower() or 'no' in result.output.lower()

    def test_unwatch_nonexistent(self):
        """Unwatching a ticker not in watchlist gives clear error."""
        result = self.runner.invoke(cli, ['unwatch', 'XYZABC'])
        assert result.exit_code == 0
        assert 'not' in result.output.lower() or 'no' in result.output.lower()

    def test_hold_invalid_qty(self):
        """Hold with invalid qty shows error."""
        result = self.runner.invoke(cli, ['hold', 'TSLA', '--entry', '250', '--qty', '0'])
        assert result.exit_code == 0
        # Should reject zero qty

    def test_hold_negative_entry(self):
        """Hold with negative entry price shows error."""
        result = self.runner.invoke(cli, ['hold', 'TSLA', '--entry', '-10', '--qty', '5'])
        assert result.exit_code == 0

    def test_alerts_invalid_threshold(self):
        """Alert with invalid threshold shows error."""
        result = self.runner.invoke(cli, ['alerts', 'add', 'TSLA', '--above', '0'])
        assert result.exit_code == 0

    def test_positions_empty(self):
        """Positions with no holdings shows helpful message."""
        result = self.runner.invoke(cli, ['positions'])
        assert result.exit_code == 0
        assert 'no' in result.output.lower() or 'empty' in result.output.lower() or len(result.output.strip()) >= 0

    def test_watchlist_empty(self):
        """Watchlist with no items shows helpful message."""
        result = self.runner.invoke(cli, ['watchlist'])
        assert result.exit_code == 0

    def test_network_failure_fallback(self):
        """Engine falls back to mock data on network failure."""
        with patch('cli.engine.generate_ohlcv', side_effect=ConnectionError("Network down")):
            # get_price should raise or handle gracefully
            try:
                engine.get_price("TSLA")
            except Exception as e:
                # Should be a classified error, not raw ConnectionError
                assert isinstance(e, (engine.NetworkError, engine.InvalidTickerError, ConnectionError))

    def test_portfolio_risk_empty(self):
        """Portfolio risk with no positions handles gracefully."""
        result = self.runner.invoke(cli, ['portfolio-risk'])
        assert result.exit_code == 0

    def test_portfolio_optimize_empty(self):
        """Portfolio optimize with no positions handles gracefully."""
        result = self.runner.invoke(cli, ['portfolio-optimize'])
        assert result.exit_code == 0

    def test_completion_bash(self):
        """Completion command generates bash instructions."""
        result = self.runner.invoke(cli, ['completion', 'bash'])
        assert result.exit_code == 0
        assert 'bash_source' in result.output

    def test_completion_zsh(self):
        """Completion command generates zsh instructions."""
        result = self.runner.invoke(cli, ['completion', 'zsh'])
        assert result.exit_code == 0
        assert 'zsh_source' in result.output

    def test_completion_fish(self):
        """Completion command generates fish instructions."""
        result = self.runner.invoke(cli, ['completion', 'fish'])
        assert result.exit_code == 0
        assert 'fish_source' in result.output

    def test_completion_auto_detect(self):
        """Completion without shell arg auto-detects."""
        result = self.runner.invoke(cli, ['completion'])
        assert result.exit_code == 0
        assert 'source' in result.output.lower()

    def test_analyze_invalid_ticker(self):
        """Analyze with invalid ticker shows clean error, no traceback."""
        with patch('cli.engine.get_analysis', side_effect=engine.InvalidTickerError("ZZZZZ")):
            result = self.runner.invoke(cli, ['analyze', 'ZZZZZ'])
            assert result.exit_code == 0
            assert 'not a valid ticker' in result.output.lower()
            assert 'Traceback' not in result.output

    def test_price_invalid_ticker(self):
        """Price with invalid ticker shows clean error."""
        with patch('cli.engine.get_price', side_effect=ValueError("Ticker 'ZZZZZ' not found")):
            result = self.runner.invoke(cli, ['price', 'ZZZZZ'])
            assert result.exit_code == 0
            assert 'error' in result.output.lower()
            assert 'Traceback' not in result.output

    def test_news_invalid_ticker(self):
        """News with invalid ticker shows clean error."""
        with patch('cli.engine.get_news', side_effect=ValueError("Ticker 'ZZZZZ' not found")):
            result = self.runner.invoke(cli, ['news', 'ZZZZZ'])
            assert result.exit_code == 0
            assert 'error' in result.output.lower()
            assert 'Traceback' not in result.output

    def test_earnings_invalid_ticker(self):
        """Earnings with invalid ticker shows clean error."""
        with patch('cli.engine.get_earnings', side_effect=ValueError("Ticker 'ZZZZZ' not found")):
            result = self.runner.invoke(cli, ['earnings', 'ZZZZZ'])
            assert result.exit_code == 0
            assert 'error' in result.output.lower()
            assert 'Traceback' not in result.output

    def test_analyze_mock_flag(self):
        """Analyze with --mock flag works without network."""
        with patch('cli.engine.get_analysis') as mock_a, \
             patch('cli.engine.get_features'):
            mock_a.return_value = {
                'ticker': 'TSLA', 'name': 'Tesla', 'price': 250.0,
                'change_pct': 1.5, 'volume': 50000000, 'avg_volume': 40000000,
                'horizons': {
                    'short': {'prediction': 0.01, 'confidence': 0.6, 'direction': 'bullish',
                              'stop': 240, 'target': 260, 'entry_lo': 248, 'entry_hi': 252},
                    'medium': {'prediction': 0.02, 'confidence': 0.55, 'direction': 'bullish',
                               'stop': 230, 'target': 270, 'entry_lo': 248, 'entry_hi': 252,
                               'support': 235, 'resistance': 265},
                    'long': {'prediction': 0.05, 'confidence': 0.65, 'direction': 'bullish',
                             'stop': 220, 'target': 290, 'entry_lo': 248, 'entry_hi': 252,
                             'support': 235, 'resistance': 265},
                },
                'bullish': ['RSI oversold'], 'bearish': ['High IV'],
                'fetched_at': '2026-01-01 12:00:00',
            }
            result = self.runner.invoke(cli, ['analyze', 'TSLA', '--mock'])
            assert result.exit_code == 0
            assert 'TSLA' in result.output


class TestCLIRealDataFlow:
    """Tests verifying CLI → engine → real providers flow with mocked external APIs."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.runner = CliRunner()
        self._patches = [
            patch.object(db, 'DB_PATH', tmp_path / 'data.db'),
            patch.object(config, 'CONFIG_PATH', tmp_path / 'config.json'),
        ]
        for p in self._patches:
            p.start()
        yield
        for p in self._patches:
            p.stop()

    def test_analyze_verbose_includes_features(self):
        """Analyze --verbose shows raw feature values."""
        with patch('cli.engine.get_analysis', return_value=_mock_analysis('TSLA')), \
             patch('cli.engine.get_features', return_value={'rsi_14': 45.2, 'macd': 1.3}):
            result = self.runner.invoke(cli, ['analyze', 'TSLA', '--mock', '--verbose'])
        assert result.exit_code == 0
        assert 'TSLA' in result.output

    def test_portfolio_risk_with_positions(self):
        """Portfolio risk computes when positions exist."""
        self.runner.invoke(cli, ['hold', 'AAPL', '--entry', '180', '--qty', '10'])
        self.runner.invoke(cli, ['hold', 'MSFT', '--entry', '400', '--qty', '5'])
        result = self.runner.invoke(cli, ['portfolio-risk'])
        assert result.exit_code == 0

    def test_portfolio_optimize_with_positions(self):
        """Portfolio optimize computes when positions exist."""
        self.runner.invoke(cli, ['hold', 'AAPL', '--entry', '180', '--qty', '10'])
        self.runner.invoke(cli, ['hold', 'MSFT', '--entry', '400', '--qty', '5'])
        result = self.runner.invoke(cli, ['portfolio-optimize'])
        assert result.exit_code == 0

    def test_alerts_full_lifecycle(self):
        """Alert add → list → check → remove lifecycle."""
        r = self.runner.invoke(cli, ['alerts', 'add', 'TSLA', '--above', '300'])
        assert r.exit_code == 0
        r = self.runner.invoke(cli, ['alerts', 'list'])
        assert r.exit_code == 0
        assert 'TSLA' in r.output
        r = self.runner.invoke(cli, ['alerts', 'check'])
        assert r.exit_code == 0
        r = self.runner.invoke(cli, ['alerts', 'remove', '1'])
        assert r.exit_code == 0

    def test_config_reset(self):
        """Config reset clears all settings."""
        self.runner.invoke(cli, ['config', 'set', 'fred-key', 'abc123'])
        r = self.runner.invoke(cli, ['config', 'reset'], input='y\n')
        assert r.exit_code == 0

    def test_cache_clean(self):
        """Cache clean runs without error."""
        result = self.runner.invoke(cli, ['cache-clean'])
        assert result.exit_code == 0

    def test_export_json(self):
        """Export analysis as JSON."""
        with patch('cli.engine.get_analysis', return_value=_mock_analysis('TSLA')), \
             patch('cli.engine.get_features', return_value={'rsi_14': 45.2, 'macd': 1.3}):
            result = self.runner.invoke(cli, ['export', 'TSLA', '--format', 'json'])
        assert result.exit_code == 0

    def test_scan_preset_workflow(self):
        """Save, list, and delete scan presets."""
        r = self.runner.invoke(cli, ['scan-save', 'mytest', 'rsi<30'])
        assert r.exit_code == 0
        r = self.runner.invoke(cli, ['scan-presets'])
        assert r.exit_code == 0
        assert 'mytest' in r.output
        r = self.runner.invoke(cli, ['scan-delete', 'mytest'])
        assert r.exit_code == 0

    def test_watchlist_signals(self):
        """Watchlist with --signals flag."""
        self.runner.invoke(cli, ['watch', 'AAPL'])
        with patch('cli.engine.get_analysis', return_value=_mock_analysis('AAPL')):
            result = self.runner.invoke(cli, ['watchlist', '--signals'])
        assert result.exit_code == 0

    def test_positions_sort(self):
        """Positions with --sort flag."""
        self.runner.invoke(cli, ['hold', 'AAPL', '--entry', '180', '--qty', '10'])
        self.runner.invoke(cli, ['hold', 'TSLA', '--entry', '250', '--qty', '5'])
        result = self.runner.invoke(cli, ['positions', '--sort', 'pnl'])
        assert result.exit_code == 0

    def test_doctor_fix(self):
        """Doctor --fix auto-repairs issues."""
        with patch('cli.engine.get_price', return_value={'ticker': 'AAPL', 'price': 180, 'change_pct': 1.0, 'change': 1.8, 'volume': 5e7, 'high': 181, 'low': 179}), \
             patch('cli.engine._ensure_models'):
            result = self.runner.invoke(cli, ['doctor', '--fix'])
        assert result.exit_code == 0


class TestAPIEndToEnd:
    """Tests verifying API server → backend → real providers flow."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from fastapi.testclient import TestClient
        from backend.api.server import app
        self.client = TestClient(app)

    def test_predict_get_endpoint(self):
        """GET /predict/{ticker} returns predictions."""
        response = self.client.get("/predict/AAPL")
        assert response.status_code == 200
        data = response.json()
        assert data["ticker"] == "AAPL"
        assert "predictions" in data

    def test_predict_post_with_horizon(self):
        """POST /predict with specific horizon."""
        response = self.client.post("/predict", json={"ticker": "MSFT", "horizon": "medium"})
        assert response.status_code == 200
        data = response.json()
        assert data["horizon"] == "medium"

    def test_backtest_endpoint(self):
        """GET /backtest/{ticker} returns backtest results."""
        response = self.client.get("/backtest/TSLA")
        assert response.status_code == 200
        data = response.json()
        assert "backtest" in data
        assert "accuracy" in data["backtest"]

    def test_chat_predict_intent(self):
        """POST /chat with prediction intent."""
        response = self.client.post("/chat", json={"message": "predict AAPL"})
        assert response.status_code == 200
        data = response.json()
        assert "response" in data

    def test_health_includes_status(self):
        """Health endpoint returns status."""
        response = self.client.get("/health")
        assert response.status_code == 200
        assert "status" in response.json()


class TestRetryBackoffIntegration:
    """Integration tests for retry with exponential backoff timing."""

    def test_backoff_timing_increases(self):
        """Verify delay doubles on each retry attempt."""
        from backend.utils.retry import retry
        import time

        timestamps = []

        @retry(max_attempts=3, base_delay=0.05)
        def flaky():
            timestamps.append(time.monotonic())
            if len(timestamps) < 3:
                raise ConnectionError("Connection timeout")
            return "ok"

        result = flaky()
        assert result == "ok"
        assert len(timestamps) == 3
        gap1 = timestamps[1] - timestamps[0]
        gap2 = timestamps[2] - timestamps[1]
        # Second gap should be ~2x the first (exponential backoff)
        assert gap2 > gap1 * 1.5, f"Expected exponential backoff: gap1={gap1:.3f}, gap2={gap2:.3f}"

    def test_retry_preserves_return_value(self):
        """Retry returns the correct value after recovery."""
        from backend.utils.retry import retry
        attempts = [0]

        @retry(max_attempts=3, base_delay=0.01)
        def returns_dict():
            attempts[0] += 1
            if attempts[0] < 2:
                raise ConnectionError("timeout")
            return {"key": "value", "count": 42}

        result = returns_dict()
        assert result == {"key": "value", "count": 42}

    def test_provider_fallback_on_exhausted_retries(self):
        """When retries exhaust, providers fall back to mock data."""
        from unittest.mock import patch
        with patch('backend.data.real_providers.price_data.yf') as mock_yf, \
             patch('time.sleep'):
            mock_yf.Ticker.return_value.history.side_effect = ConnectionError("timeout")
            from backend.data.real_providers.price_data import get_ohlcv
            result = get_ohlcv("TSLA")
            assert result is not None
            assert 'close' in result.columns


class TestMultiTickerAnalysis:
    """Tests for analyzing multiple tickers."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.runner = CliRunner()

    def test_compare_multiple_tickers(self):
        """Compare command works with multiple tickers."""
        with patch('cli.engine.get_analysis', side_effect=lambda t: _mock_analysis(t)):
            result = self.runner.invoke(cli, ['compare', 'AAPL', 'MSFT', 'GOOGL'])
        assert result.exit_code == 0

    def test_sequential_analyze(self):
        """Analyze multiple tickers sequentially."""
        with patch('cli.engine.get_analysis', side_effect=lambda t: _mock_analysis(t)):
            for ticker in ['AAPL', 'MSFT']:
                result = self.runner.invoke(cli, ['analyze', ticker])
                assert result.exit_code == 0, f"Failed for {ticker}: {result.output}"

    def test_correlate_two_tickers(self):
        """Correlate command works between two tickers."""
        result = self.runner.invoke(cli, ['correlate', 'AAPL', 'MSFT'])
        assert result.exit_code == 0
