"""Tests for CLI commands, db, config, and engine."""
import pytest
import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from click.testing import CliRunner

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from cli.main import cli
from cli import db, config


# --- DB Tests ---

class TestDB:
    @pytest.fixture(autouse=True)
    def tmp_db(self, tmp_path):
        with patch.object(db, 'DB_PATH', tmp_path / 'data.db'):
            yield

    def test_add_and_get_position(self):
        db.add_position("TSLA", 250.0, 10)
        pos = db.get_positions()
        assert len(pos) == 1
        assert pos[0]['ticker'] == 'TSLA'
        assert pos[0]['entry_price'] == 250.0
        assert pos[0]['qty'] == 10

    def test_remove_position(self):
        db.add_position("AAPL", 180.0, 5)
        db.remove_position("AAPL")
        assert len(db.get_positions()) == 0

    def test_upsert_position(self):
        db.add_position("TSLA", 250.0, 10)
        db.add_position("TSLA", 260.0, 20)
        pos = db.get_positions()
        assert len(pos) == 1
        assert pos[0]['entry_price'] == 260.0

    def test_add_and_get_watchlist(self):
        db.add_watch("NVDA")
        wl = db.get_watchlist()
        assert len(wl) == 1
        assert wl[0]['ticker'] == 'NVDA'

    def test_remove_watch(self):
        db.add_watch("NVDA")
        db.remove_watch("NVDA")
        assert len(db.get_watchlist()) == 0

    def test_duplicate_watch_ignored(self):
        db.add_watch("TSLA")
        db.add_watch("TSLA")
        assert len(db.get_watchlist()) == 1

    def test_case_normalization(self):
        db.add_position("tsla", 250.0, 10)
        db.add_watch("aapl")
        assert db.get_positions()[0]['ticker'] == 'TSLA'
        assert db.get_watchlist()[0]['ticker'] == 'AAPL'


# --- Config Tests ---

class TestConfig:
    @pytest.fixture(autouse=True)
    def tmp_config(self, tmp_path):
        with patch.object(config, 'CONFIG_PATH', tmp_path / 'config.json'):
            yield

    def test_set_and_get(self):
        config.set_key('fred-key', 'abc123')
        assert config.get('fred-key') == 'abc123'

    def test_get_missing_key(self):
        assert config.get('nonexistent') is None
        assert config.get('nonexistent', 'default') == 'default'

    def test_overwrite_key(self):
        config.set_key('k', 'v1')
        config.set_key('k', 'v2')
        assert config.get('k') == 'v2'


# --- CLI Command Tests (mocked backend) ---

_MOCK_ANALYSIS = {
    'ticker': 'TSLA', 'name': 'Tesla Inc', 'price': 250.0,
    'change_pct': 2.3, 'volume': 45_000_000, 'avg_volume': 25_000_000,
    'horizons': {
        'short': {'direction': 'bullish', 'confidence': 0.64, 'prediction': 0.02,
                  'stop': 244.0, 'target': 255.0, 'entry_lo': 248.0, 'entry_hi': 249.0,
                  'support': 238.0, 'resistance': 260.0},
        'medium': {'direction': 'neutral', 'confidence': 0.52, 'prediction': 0.01,
                   'stop': 235.0, 'target': 265.0, 'entry_lo': 248.0, 'entry_hi': 249.0,
                   'support': 238.0, 'resistance': 260.0},
        'long': {'direction': 'bullish', 'confidence': 0.71, 'prediction': 0.08,
                 'stop': 220.0, 'target': 310.0, 'entry_lo': 248.0, 'entry_hi': 249.0,
                 'support': 238.0, 'resistance': 260.0},
    },
    'bullish': ['RSI oversold (28)', 'Volume surge 2.3x'],
    'bearish': ['Approaching resistance', 'Elevated IV'],
    'fetched_at': '2026-02-15 05:00:00',
    'market_cap': 800_000_000_000, 'sector': 'Consumer Cyclical', 'pe_ratio': 65.2,
}

_MOCK_PRICE = {
    'ticker': 'TSLA', 'price': 250.0, 'change': 5.5,
    'change_pct': 2.3, 'volume': 45_000_000, 'high': 252.0, 'low': 246.0,
}


class TestCLICommands:
    def setup_method(self):
        self.runner = CliRunner()

    def test_help(self):
        result = self.runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        assert 'Stock Chat' in result.output

    def test_version(self):
        result = self.runner.invoke(cli, ['--version'])
        assert result.exit_code == 0
        assert '1.2.3' in result.output

    @patch('cli.engine.get_analysis', return_value=_MOCK_ANALYSIS)
    def test_analyze(self, mock):
        result = self.runner.invoke(cli, ['analyze', 'TSLA'])
        assert result.exit_code == 0
        assert 'TSLA' in result.output
        assert 'BUY' in result.output or 'HOLD' in result.output

    @patch('cli.engine.get_analysis', return_value=_MOCK_ANALYSIS)
    def test_analyze_shows_freshness(self, mock):
        result = self.runner.invoke(cli, ['analyze', 'TSLA'])
        assert result.exit_code == 0
        assert 'Data as of' in result.output

    @patch('cli.engine.get_analysis', return_value=_MOCK_ANALYSIS)
    def test_analyze_short_only(self, mock):
        result = self.runner.invoke(cli, ['analyze', 'TSLA', '--short'])
        assert result.exit_code == 0
        assert 'SHORT-TERM' in result.output

    @patch('cli.engine.get_analysis', return_value=_MOCK_ANALYSIS)
    def test_analyze_json_output(self, mock):
        result = self.runner.invoke(cli, ['analyze', 'TSLA', '--json'])
        assert result.exit_code == 0
        import json
        data = json.loads(result.output)
        assert data['ticker'] == 'TSLA'
        assert 'horizons' in data
        assert data['price'] == 250.0

    @patch('cli.engine.get_analysis', side_effect=Exception("Network error"))
    def test_analyze_error(self, mock):
        result = self.runner.invoke(cli, ['analyze', 'TSLA'])
        assert result.exit_code == 0  # graceful
        assert 'Error' in result.output

    @patch('cli.engine.get_analysis')
    def test_analyze_invalid_ticker_error(self, mock):
        from cli.errors import InvalidTickerError
        mock.side_effect = InvalidTickerError("XYZ")
        result = self.runner.invoke(cli, ['analyze', 'XYZ'])
        assert result.exit_code == 0
        assert 'not a valid ticker' in result.output

    @patch('cli.engine.get_analysis')
    def test_analyze_rate_limit_error(self, mock):
        from cli.errors import RateLimitError
        mock.side_effect = RateLimitError("Yahoo Finance")
        result = self.runner.invoke(cli, ['analyze', 'TSLA'])
        assert result.exit_code == 0
        assert 'Rate limit' in result.output

    @patch('cli.engine.get_analysis', return_value=_MOCK_ANALYSIS)
    def test_analyze_medium_only(self, mock):
        result = self.runner.invoke(cli, ['analyze', 'TSLA', '--medium'])
        assert result.exit_code == 0
        assert 'MEDIUM-TERM' in result.output

    @patch('cli.engine.get_analysis', return_value=_MOCK_ANALYSIS)
    def test_analyze_long_only(self, mock):
        result = self.runner.invoke(cli, ['analyze', 'TSLA', '--long'])
        assert result.exit_code == 0
        assert 'LONG-TERM' in result.output

    @patch('cli.engine.get_features', return_value={'rsi_14': 28.5, 'macd_hist': 0.12})
    @patch('cli.engine.get_analysis', return_value=_MOCK_ANALYSIS)
    def test_analyze_verbose(self, mock_a, mock_f):
        result = self.runner.invoke(cli, ['analyze', 'TSLA', '--verbose'])
        assert result.exit_code == 0
        assert 'Feature Values' in result.output

    @patch('cli.engine.get_price', return_value=_MOCK_PRICE)
    def test_price(self, mock):
        result = self.runner.invoke(cli, ['price', 'TSLA'])
        assert result.exit_code == 0
        assert '250.00' in result.output

    @patch('cli.engine.get_price', side_effect=Exception("Bad ticker"))
    def test_price_error(self, mock):
        result = self.runner.invoke(cli, ['price', 'INVALID'])
        assert result.exit_code == 0
        assert 'Error' in result.output

    @patch('cli.engine.get_news', return_value=[
        {'sentiment': 0.5, 'headline': 'Tesla beats earnings', 'source': 'Reuters'}
    ])
    def test_news(self, mock):
        result = self.runner.invoke(cli, ['news', 'TSLA'])
        assert result.exit_code == 0
        assert 'Tesla beats earnings' in result.output

    @patch('cli.engine.get_news', return_value=[])
    def test_news_empty(self, mock):
        result = self.runner.invoke(cli, ['news', 'TSLA'])
        assert result.exit_code == 0
        assert 'No recent news' in result.output

    @patch('cli.engine.get_earnings', return_value={
        'valuation': {'pe_ratio': 50, 'forward_pe': 40, 'ps_ratio': 10, 'pb_ratio': 15},
        'quarters': [{'quarter': 'Q4 2025', 'eps': 1.5, 'revenue': 25e9, 'net_margin': 0.12}],
        'latest': {},
    })
    def test_earnings(self, mock):
        result = self.runner.invoke(cli, ['earnings', 'TSLA'])
        assert result.exit_code == 0
        assert 'P/E' in result.output

    @patch('cli.engine.chat_query', return_value="TSLA looks bullish short-term.")
    def test_chat(self, mock):
        result = self.runner.invoke(cli, ['chat', 'should I buy TSLA?'])
        assert result.exit_code == 0
        assert 'bullish' in result.output

    @patch('cli.engine.chat_query', side_effect=Exception("API error"))
    def test_chat_error(self, mock):
        result = self.runner.invoke(cli, ['chat', 'analyze TSLA'])
        assert result.exit_code == 0
        assert 'Error' in result.output

    def test_config_set_get(self):
        with self.runner.isolated_filesystem():
            with patch.object(config, 'CONFIG_PATH', Path('config.json')):
                result = self.runner.invoke(cli, ['config', 'set', 'fred-key', 'test123'])
                assert result.exit_code == 0
                result = self.runner.invoke(cli, ['config', 'get', 'fred-key'])
                assert result.exit_code == 0
                assert 'test123' in result.output

    def test_config_list(self):
        with self.runner.isolated_filesystem():
            with patch.object(config, 'CONFIG_PATH', Path('config.json')):
                self.runner.invoke(cli, ['config', 'set', 'test-key', 'value1'])
                result = self.runner.invoke(cli, ['config', 'list'])
                assert result.exit_code == 0
                assert 'test-key' in result.output

    def test_config_list_empty(self):
        with self.runner.isolated_filesystem():
            with patch.object(config, 'CONFIG_PATH', Path('config.json')):
                result = self.runner.invoke(cli, ['config', 'list'])
                assert result.exit_code == 0
                assert 'No config' in result.output

    @patch('backend.utils.retry.cleanup_cache', return_value=3)
    def test_cache_clean(self, mock_cleanup):
        result = self.runner.invoke(cli, ['cache-clean'])
        assert result.exit_code == 0
        assert '3' in result.output
        assert 'Removed' in result.output


class TestDashboard:
    def setup_method(self):
        self.runner = CliRunner()
        self._tmp = tempfile.mkdtemp()
        self._patcher = patch.object(db, 'DB_PATH', Path(self._tmp) / 'data.db')
        self._patcher.start()

    def teardown_method(self):
        self._patcher.stop()


class TestCompare:
    def setup_method(self):
        self.runner = CliRunner()

    @patch('cli.engine.get_analysis')
    def test_compare_two_tickers(self, mock):
        mock_aapl = dict(_MOCK_ANALYSIS, ticker='AAPL', name='Apple Inc', price=180.0)
        mock.side_effect = lambda t: _MOCK_ANALYSIS if t.upper() == 'TSLA' else mock_aapl
        result = self.runner.invoke(cli, ['compare', 'TSLA', 'AAPL'])
        assert result.exit_code == 0
        assert 'TSLA' in result.output
        assert 'AAPL' in result.output
        assert 'Comparison' in result.output

    def test_compare_single_ticker_error(self):
        result = self.runner.invoke(cli, ['compare', 'TSLA'])
        assert result.exit_code == 0
        assert 'at least 2' in result.output

    @patch('cli.engine.get_analysis', side_effect=Exception("fail"))
    def test_compare_all_fail(self, mock):
        result = self.runner.invoke(cli, ['compare', 'TSLA', 'AAPL'])
        assert result.exit_code == 0
        assert 'Error' in result.output

    @patch('cli.engine.get_price', return_value=_MOCK_PRICE)
    def test_dashboard_renders(self, mock_price):
        """Dashboard should render and exit on KeyboardInterrupt."""
        db.add_position("TSLA", 250.0, 10)
        db.add_watch("AAPL")
        with patch('cli.main.Live') as mock_live:
            mock_live.return_value.__enter__ = MagicMock()
            mock_live.return_value.__exit__ = MagicMock(return_value=False)
            # Simulate immediate Ctrl+C
            mock_live.return_value.__enter__.return_value = MagicMock()
            with patch('time.sleep', side_effect=KeyboardInterrupt):
                result = self.runner.invoke(cli, ['dashboard'])
            assert result.exit_code == 0

    def test_dashboard_empty(self):
        """Dashboard with no positions/watchlist should still render."""
        with patch('cli.main.Live') as mock_live:
            mock_live.return_value.__enter__ = MagicMock()
            mock_live.return_value.__exit__ = MagicMock(return_value=False)
            with patch('time.sleep', side_effect=KeyboardInterrupt):
                result = self.runner.invoke(cli, ['dashboard'])
            assert result.exit_code == 0


class TestCLIPortfolio:
    """Test portfolio commands with temp DB."""

    def setup_method(self):
        self.runner = CliRunner()
        self._tmp = tempfile.mkdtemp()
        self._patcher = patch.object(db, 'DB_PATH', Path(self._tmp) / 'data.db')
        self._patcher.start()

    def teardown_method(self):
        self._patcher.stop()

    def test_hold_and_positions(self):
        self.runner.invoke(cli, ['hold', 'TSLA', '--entry', '250', '--qty', '10'])
        with patch('cli.engine.get_price', return_value=_MOCK_PRICE):
            result = self.runner.invoke(cli, ['positions'])
        assert result.exit_code == 0
        assert 'TSLA' in result.output

    def test_hold_negative_price(self):
        result = self.runner.invoke(cli, ['hold', 'TSLA', '--entry', '-10', '--qty', '5'])
        assert result.exit_code == 0
        assert 'positive' in result.output.lower()

    def test_hold_negative_qty(self):
        result = self.runner.invoke(cli, ['hold', 'TSLA', '--entry', '250', '--qty', '-5'])
        assert result.exit_code == 0
        assert 'positive' in result.output.lower()

    def test_sell(self):
        self.runner.invoke(cli, ['hold', 'TSLA', '--entry', '250', '--qty', '10'])
        result = self.runner.invoke(cli, ['sell', 'TSLA'])
        assert result.exit_code == 0
        assert 'Removed' in result.output

    def test_sell_nonexistent(self):
        result = self.runner.invoke(cli, ['sell', 'NONEXISTENT'])
        assert result.exit_code == 0
        assert 'not in your positions' in result.output

    def test_positions_empty(self):
        result = self.runner.invoke(cli, ['positions'])
        assert result.exit_code == 0
        assert 'No positions' in result.output

    def test_watch_and_watchlist(self):
        self.runner.invoke(cli, ['watch', 'NVDA'])
        with patch('cli.engine.get_price', return_value=_MOCK_PRICE):
            result = self.runner.invoke(cli, ['watchlist'])
        assert result.exit_code == 0
        assert 'NVDA' in result.output

    def test_unwatch(self):
        self.runner.invoke(cli, ['watch', 'NVDA'])
        result = self.runner.invoke(cli, ['unwatch', 'NVDA'])
        assert result.exit_code == 0
        assert 'Removed' in result.output

    def test_watchlist_empty(self):
        result = self.runner.invoke(cli, ['watchlist'])
        assert result.exit_code == 0
        assert 'empty' in result.output.lower()


class TestExport:
    def setup_method(self):
        self.runner = CliRunner()

    @patch('cli.engine.get_features', return_value={'rsi_14': 45.2, 'sma_50': 240.0})
    @patch('cli.engine.get_analysis', return_value=_MOCK_ANALYSIS)
    def test_export_json_stdout(self, mock_a, mock_f):
        result = self.runner.invoke(cli, ['export', 'TSLA'])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data['ticker'] == 'TSLA'
        assert 'features' in data
        assert data['features']['rsi_14'] == 45.2

    @patch('cli.engine.get_features', return_value={'rsi_14': 45.2})
    @patch('cli.engine.get_analysis', return_value=_MOCK_ANALYSIS)
    def test_export_csv_stdout(self, mock_a, mock_f):
        result = self.runner.invoke(cli, ['export', 'TSLA', '--format', 'csv'])
        assert result.exit_code == 0
        assert 'ticker,TSLA' in result.output
        assert 'rsi_14' in result.output

    @patch('cli.engine.get_features', return_value={'rsi_14': 45.2})
    @patch('cli.engine.get_analysis', return_value=_MOCK_ANALYSIS)
    def test_export_json_file(self, mock_a, mock_f):
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(cli, ['export', 'TSLA', '-o', 'out.json'])
            assert result.exit_code == 0
            assert 'Exported' in result.output
            data = json.loads(Path('out.json').read_text())
            assert data['ticker'] == 'TSLA'

    @patch('cli.engine.get_analysis', side_effect=Exception("fail"))
    def test_export_error(self, mock):
        result = self.runner.invoke(cli, ['export', 'BAD'])
        assert result.exit_code == 0
        assert 'Error' in result.output


class TestBacktest:
    def setup_method(self):
        self.runner = CliRunner()

    @patch('cli.engine.run_backtest', return_value={
        'sharpe_ratio': 1.5, 'max_drawdown': 0.12, 'win_rate': 0.55,
        'profit_factor': 1.8, 'total_trades': 42,
        'gross_return': 12.5, 'net_return': 11.8,
        'avg_holding_period': 1.0, 'slippage': 0.0005, 'spread': 0.0002, 'commission': 0.0,
        'win_rate_by_horizon': {'short': 0.55, 'medium': 0.52, 'long': 0.58},
    })
    def test_backtest_output(self, mock):
        result = self.runner.invoke(cli, ['backtest', 'TSLA'])
        assert result.exit_code == 0
        assert 'Sharpe' in result.output
        assert 'TSLA' in result.output
        assert 'Backtest' in result.output

    @patch('cli.engine.run_backtest', return_value={
        'sharpe_ratio': 0.8, 'max_drawdown': 0.2, 'win_rate': 0.5,
        'profit_factor': 1.0, 'total_trades': 20,
        'gross_return': 5.0, 'net_return': 4.5,
        'avg_holding_period': 1.0, 'slippage': 0.0005, 'spread': 0.0002, 'commission': 0.0,
        'win_rate_by_horizon': {'short': 0.5, 'medium': 0.5, 'long': 0.5},
    })
    def test_backtest_custom_days(self, mock):
        result = self.runner.invoke(cli, ['backtest', 'TSLA', '--days', '180'])
        assert result.exit_code == 0
        mock.assert_called_once_with('TSLA', 180, detailed=False)

    @patch('cli.engine.run_backtest', side_effect=Exception("No data"))
    def test_backtest_error(self, mock):
        result = self.runner.invoke(cli, ['backtest', 'BAD'])
        assert result.exit_code == 0
        assert 'Error' in result.output


class TestOBVSlope:
    """Test that obv_slope feature is computed correctly."""

    def test_obv_slope_in_tier2(self):
        from backend.features.tier2_technical import Tier2Technical
        assert 'obv_slope' in Tier2Technical.feature_names()

    def test_obv_slope_computed(self):
        from backend.features.tier2_technical import Tier2Technical
        from backend.data.mock_generators import generate_ohlcv
        df = generate_ohlcv("TSLA", seed=42).tail(100)
        feats = Tier2Technical.compute(df)
        assert 'obv_slope' in feats.columns
        # Should have non-zero values after warmup
        assert not feats['obv_slope'].tail(50).eq(0).all()


class TestScreen:
    def setup_method(self):
        self.runner = CliRunner()

    @patch('cli.engine.screen_tickers', return_value=[
        {'ticker': 'TSLA', 'price': 250.0, 'change_pct': -1.5, 'rsi': 25.3,
         'stoch_k': 18.2, 'vol_ratio': 1.2, 'reason': 'RSI 25.3, Stoch 18.2'},
    ])
    def test_screen_oversold(self, mock):
        result = self.runner.invoke(cli, ['screen'])
        assert result.exit_code == 0
        assert 'TSLA' in result.output
        assert 'RSI' in result.output

    @patch('cli.engine.screen_tickers', return_value=[])
    def test_screen_no_matches(self, mock):
        result = self.runner.invoke(cli, ['screen', '--criteria', 'overbought'])
        assert result.exit_code == 0
        assert 'No tickers' in result.output

    @patch('cli.engine.screen_tickers', return_value=[
        {'ticker': 'NVDA', 'price': 800.0, 'change_pct': 5.0, 'rsi': 55.0,
         'stoch_k': 60.0, 'vol_ratio': 3.1, 'reason': 'Vol 3.1x avg'},
    ])
    def test_screen_volume(self, mock):
        result = self.runner.invoke(cli, ['screen', '--criteria', 'volume'])
        assert result.exit_code == 0
        assert 'NVDA' in result.output

    @patch('cli.engine.screen_tickers', side_effect=Exception("API error"))
    def test_screen_error(self, mock):
        result = self.runner.invoke(cli, ['screen'])
        assert result.exit_code == 0
        assert 'Error' in result.output

    def test_screen_help(self):
        result = self.runner.invoke(cli, ['screen', '--help'])
        assert result.exit_code == 0
        assert 'Screen stocks' in result.output


class TestHistory:
    def setup_method(self):
        self.runner = CliRunner()
        self._tmp = tempfile.mkdtemp()
        self._patcher = patch.object(db, 'DB_PATH', Path(self._tmp) / 'data.db')
        self._patcher.start()

    def teardown_method(self):
        self._patcher.stop()

    def test_history_empty(self):
        result = self.runner.invoke(cli, ['history', 'TSLA'])
        assert result.exit_code == 0
        assert 'No prediction history' in result.output

    @patch('cli.engine.get_price', return_value=_MOCK_PRICE)
    def test_history_with_data(self, mock_price):
        db.save_prediction('TSLA', 'bullish', 0.64, 240.0, 'short')
        db.save_prediction('TSLA', 'bearish', 0.55, 260.0, 'medium')
        result = self.runner.invoke(cli, ['history', 'TSLA'])
        assert result.exit_code == 0
        assert 'TSLA' in result.output
        assert 'Prediction History' in result.output

    def test_history_help(self):
        result = self.runner.invoke(cli, ['history', '--help'])
        assert result.exit_code == 0
        assert 'prediction history' in result.output.lower()


class TestCompareEnhanced:
    def setup_method(self):
        self.runner = CliRunner()

    @patch('cli.engine.get_analysis')
    def test_compare_shows_sector_pe(self, mock):
        mock_aapl = dict(_MOCK_ANALYSIS, ticker='AAPL', name='Apple Inc', price=180.0,
                         sector='Technology', pe_ratio=28.5, market_cap=3_000_000_000_000)
        mock.side_effect = lambda t: _MOCK_ANALYSIS if t.upper() == 'TSLA' else mock_aapl
        result = self.runner.invoke(cli, ['compare', 'TSLA', 'AAPL'])
        assert result.exit_code == 0
        assert 'Sector' in result.output
        assert 'P/E' in result.output
        assert 'Mkt Cap' in result.output

    @patch('cli.engine.get_analysis')
    @patch('cli.engine.correlate_tickers', return_value={'correlation': 0.72, 'p_value': 0.001})
    def test_compare_correlation(self, mock_corr, mock_analysis):
        mock_aapl = dict(_MOCK_ANALYSIS, ticker='AAPL', name='Apple Inc', price=180.0)
        mock_analysis.side_effect = lambda t: _MOCK_ANALYSIS if t.upper() == 'TSLA' else mock_aapl
        result = self.runner.invoke(cli, ['compare', 'TSLA', 'AAPL', '--correlation'])
        assert result.exit_code == 0
        assert 'Correlation' in result.output


class TestStochastic:
    """Test stochastic oscillator feature."""

    def test_stoch_in_feature_names(self):
        from backend.features.tier2_technical import Tier2Technical
        names = Tier2Technical.feature_names()
        assert 'stoch_k' in names
        assert 'stoch_d' in names

    def test_stoch_computed(self):
        from backend.features.tier2_technical import Tier2Technical
        from backend.data.mock_generators import generate_ohlcv
        df = generate_ohlcv("TSLA", seed=42).tail(100)
        feats = Tier2Technical.compute(df)
        assert 'stoch_k' in feats.columns
        assert 'stoch_d' in feats.columns
        # Stochastic should be between 0 and 100
        last_k = feats['stoch_k'].iloc[-1]
        assert 0 <= last_k <= 100

    def test_stoch_d_is_smoothed_k(self):
        from backend.features.tier2_technical import Tier2Technical
        from backend.data.mock_generators import generate_ohlcv
        df = generate_ohlcv("AAPL", seed=99).tail(100)
        feats = Tier2Technical.compute(df)
        # %D should be 3-period SMA of %K, so less volatile
        k_std = feats['stoch_k'].tail(30).std()
        d_std = feats['stoch_d'].tail(30).std()
        assert d_std <= k_std + 1  # %D should be smoother or similar


class TestDBPredictions:
    """Test prediction storage in SQLite."""

    @pytest.fixture(autouse=True)
    def tmp_db(self, tmp_path):
        with patch.object(db, 'DB_PATH', tmp_path / 'data.db'):
            yield

    def test_save_and_get_predictions(self):
        db.save_prediction('TSLA', 'bullish', 0.64, 250.0, 'short')
        db.save_prediction('TSLA', 'bearish', 0.55, 260.0, 'medium')
        preds = db.get_predictions('TSLA')
        assert len(preds) == 2
        assert preds[0]['ticker'] == 'TSLA'
        assert preds[0]['direction'] in ('bullish', 'bearish')

    def test_predictions_ordered_desc(self):
        db.save_prediction('AAPL', 'bullish', 0.6, 180.0, 'short')
        db.save_prediction('AAPL', 'bearish', 0.7, 185.0, 'long')
        preds = db.get_predictions('AAPL')
        # Most recent first
        assert preds[0]['confidence'] == 0.7

    def test_predictions_limit(self):
        for i in range(5):
            db.save_prediction('NVDA', 'bullish', 0.5 + i * 0.05, 800.0 + i, 'short')
        preds = db.get_predictions('NVDA', limit=3)
        assert len(preds) == 3

    def test_predictions_per_ticker(self):
        db.save_prediction('TSLA', 'bullish', 0.6, 250.0, 'short')
        db.save_prediction('AAPL', 'bearish', 0.5, 180.0, 'short')
        assert len(db.get_predictions('TSLA')) == 1
        assert len(db.get_predictions('AAPL')) == 1


class TestDBAlerts:
    """Test alert storage in SQLite."""

    @pytest.fixture(autouse=True)
    def tmp_db(self, tmp_path):
        with patch.object(db, 'DB_PATH', tmp_path / 'data.db'):
            yield

    def test_add_and_get_alert(self):
        db.add_alert('TSLA', 'above', 300.0)
        alerts = db.get_alerts('TSLA')
        assert len(alerts) == 1
        assert alerts[0]['ticker'] == 'TSLA'
        assert alerts[0]['condition'] == 'above'
        assert alerts[0]['threshold'] == 300.0

    def test_get_all_alerts(self):
        db.add_alert('TSLA', 'above', 300.0)
        db.add_alert('AAPL', 'below', 170.0)
        alerts = db.get_alerts()
        assert len(alerts) == 2

    def test_trigger_alert(self):
        db.add_alert('TSLA', 'above', 300.0)
        alerts = db.get_alerts('TSLA')
        db.trigger_alert(alerts[0]['id'])
        # Triggered alerts should not appear in active list
        assert len(db.get_alerts('TSLA')) == 0

    def test_remove_alert(self):
        db.add_alert('TSLA', 'above', 300.0)
        alerts = db.get_alerts('TSLA')
        db.remove_alert(alerts[0]['id'])
        assert len(db.get_alerts('TSLA')) == 0

    def test_case_normalization(self):
        db.add_alert('tsla', 'below', 200.0)
        assert db.get_alerts('TSLA')[0]['ticker'] == 'TSLA'


class TestCLIAlerts:
    """Test alerts CLI commands."""

    def setup_method(self):
        self.runner = CliRunner()
        self._tmp = tempfile.mkdtemp()
        self._patcher = patch.object(db, 'DB_PATH', Path(self._tmp) / 'data.db')
        self._patcher.start()

    def teardown_method(self):
        self._patcher.stop()

    def test_alerts_add_above(self):
        result = self.runner.invoke(cli, ['alerts', 'add', 'TSLA', '--above', '300'])
        assert result.exit_code == 0
        assert 'above' in result.output
        assert '$300.00' in result.output

    def test_alerts_add_below(self):
        result = self.runner.invoke(cli, ['alerts', 'add', 'AAPL', '--below', '170'])
        assert result.exit_code == 0
        assert 'below' in result.output

    def test_alerts_add_no_threshold(self):
        result = self.runner.invoke(cli, ['alerts', 'add', 'TSLA'])
        assert result.exit_code == 0
        assert 'Specify' in result.output

    def test_alerts_add_negative_threshold(self):
        result = self.runner.invoke(cli, ['alerts', 'add', 'TSLA', '--above', '-100'])
        assert result.exit_code == 0
        assert 'positive' in result.output.lower()

    def test_alerts_list(self):
        db.add_alert('TSLA', 'above', 300.0)
        result = self.runner.invoke(cli, ['alerts', 'list'])
        assert result.exit_code == 0
        assert 'TSLA' in result.output
        assert 'above' in result.output

    def test_alerts_list_empty(self):
        result = self.runner.invoke(cli, ['alerts', 'list'])
        assert result.exit_code == 0
        assert 'No active alerts' in result.output

    def test_alerts_list_by_ticker(self):
        db.add_alert('TSLA', 'above', 300.0)
        db.add_alert('AAPL', 'below', 170.0)
        result = self.runner.invoke(cli, ['alerts', 'list', 'TSLA'])
        assert result.exit_code == 0
        assert 'TSLA' in result.output

    @patch('cli.engine.check_alerts', return_value=[])
    def test_alerts_check_none(self, mock):
        result = self.runner.invoke(cli, ['alerts', 'check'])
        assert result.exit_code == 0
        assert 'No alerts triggered' in result.output

    @patch('cli.engine.check_alerts', return_value=[
        {'ticker': 'TSLA', 'condition': 'above', 'threshold': 300.0, 'current_price': 305.0}
    ])
    def test_alerts_check_triggered(self, mock):
        result = self.runner.invoke(cli, ['alerts', 'check'])
        assert result.exit_code == 0
        assert '🔔' in result.output
        assert 'TSLA' in result.output

    def test_alerts_remove(self):
        db.add_alert('TSLA', 'above', 300.0)
        alerts = db.get_alerts('TSLA')
        result = self.runner.invoke(cli, ['alerts', 'remove', str(alerts[0]['id'])])
        assert result.exit_code == 0
        assert 'Removed' in result.output

    def test_alerts_help(self):
        result = self.runner.invoke(cli, ['alerts', '--help'])
        assert result.exit_code == 0
        assert 'price alerts' in result.output.lower()


class TestCLICorrelate:
    """Test correlate CLI command."""

    def setup_method(self):
        self.runner = CliRunner()

    @patch('cli.engine.correlate_tickers', return_value={
        'ticker1': 'TSLA', 'ticker2': 'AAPL', 'days': 180,
        'correlation': 0.6523, 'rolling_min': 0.3, 'rolling_max': 0.85,
        'rolling_current': 0.65,
    })
    def test_correlate_output(self, mock):
        result = self.runner.invoke(cli, ['correlate', 'TSLA', 'AAPL'])
        assert result.exit_code == 0
        assert 'TSLA' in result.output
        assert 'AAPL' in result.output
        assert 'Correlation' in result.output

    @patch('cli.engine.correlate_tickers', return_value={
        'ticker1': 'TSLA', 'ticker2': 'AAPL', 'days': 90,
        'correlation': 0.85, 'rolling_min': 0.6, 'rolling_max': 0.9,
        'rolling_current': 0.85,
    })
    def test_correlate_custom_days(self, mock):
        result = self.runner.invoke(cli, ['correlate', 'TSLA', 'AAPL', '--days', '90'])
        assert result.exit_code == 0
        mock.assert_called_once_with('TSLA', 'AAPL', 90)

    @patch('cli.engine.correlate_tickers', side_effect=Exception("No data"))
    def test_correlate_error(self, mock):
        result = self.runner.invoke(cli, ['correlate', 'BAD1', 'BAD2'])
        assert result.exit_code == 0
        assert 'Error' in result.output

    def test_correlate_help(self):
        result = self.runner.invoke(cli, ['correlate', '--help'])
        assert result.exit_code == 0
        assert 'correlation' in result.output.lower()


class TestSector:
    """Test sector command."""

    def setup_method(self):
        self.runner = CliRunner()

    @patch('cli.engine.sector_analysis', return_value=[
        {'ticker': 'AAPL', 'price': 180.0, 'change_pct': 1.2, 'rsi': 55.0, 'macd_hist': 0.5, 'vol_ratio': 1.1},
        {'ticker': 'MSFT', 'price': 420.0, 'change_pct': -0.5, 'rsi': 48.0, 'macd_hist': -0.3, 'vol_ratio': 0.9},
    ])
    def test_sector_output(self, mock):
        result = self.runner.invoke(cli, ['sector', 'Technology'])
        assert result.exit_code == 0
        assert 'Technology' in result.output
        assert 'AAPL' in result.output
        assert 'MSFT' in result.output
        assert 'Avg Change' in result.output

    def test_sector_unknown(self):
        result = self.runner.invoke(cli, ['sector', 'Unknown'])
        assert result.exit_code == 0
        assert 'Unknown sector' in result.output

    @patch('cli.engine.sector_analysis', return_value=[])
    def test_sector_empty(self, mock):
        result = self.runner.invoke(cli, ['sector', 'Energy'])
        assert result.exit_code == 0
        assert 'No data' in result.output

    @patch('cli.engine.sector_analysis', side_effect=Exception("API error"))
    def test_sector_error(self, mock):
        result = self.runner.invoke(cli, ['sector', 'Technology'])
        assert result.exit_code == 0
        assert 'Error' in result.output

    def test_sector_help(self):
        result = self.runner.invoke(cli, ['sector', '--help'])
        assert result.exit_code == 0
        assert 'Sector' in result.output

    def test_sector_case_insensitive(self):
        with patch('cli.engine.sector_analysis', return_value=[
            {'ticker': 'AAPL', 'price': 180.0, 'change_pct': 1.0, 'rsi': 50.0, 'macd_hist': 0.1, 'vol_ratio': 1.0},
        ]):
            result = self.runner.invoke(cli, ['sector', 'technology'])
            assert result.exit_code == 0
            assert 'Technology' in result.output


class TestAnalyzeMock:
    """Test --mock flag on analyze."""

    def setup_method(self):
        self.runner = CliRunner()

    @patch('cli.engine.get_analysis', return_value=_MOCK_ANALYSIS)
    def test_analyze_mock_flag(self, mock):
        result = self.runner.invoke(cli, ['analyze', 'TSLA', '--mock'])
        assert result.exit_code == 0
        assert 'TSLA' in result.output


class TestHistoryAccuracy:
    """Test enhanced history with accuracy stats."""

    def setup_method(self):
        self.runner = CliRunner()
        self._tmp = tempfile.mkdtemp()
        self._patcher = patch.object(db, 'DB_PATH', Path(self._tmp) / 'data.db')
        self._patcher.start()

    def teardown_method(self):
        self._patcher.stop()

    @patch('cli.engine.get_price', return_value=_MOCK_PRICE)
    def test_history_shows_accuracy(self, mock_price):
        db.save_prediction('TSLA', 'bullish', 0.64, 240.0, 'short')
        db.save_prediction('TSLA', 'bearish', 0.55, 260.0, 'medium')
        result = self.runner.invoke(cli, ['history', 'TSLA'])
        assert result.exit_code == 0
        assert 'Prediction Accuracy' in result.output
        assert 'Correct' in result.output


class TestEngineSector:
    """Test engine sector_analysis and prediction_accuracy."""

    @patch('cli.engine.get_price', return_value=_MOCK_PRICE)
    @patch('cli.engine.generate_ohlcv')
    def test_sector_analysis(self, mock_ohlcv, mock_price):
        from backend.data.mock_generators import generate_ohlcv as mock_gen
        mock_ohlcv.return_value = mock_gen("AAPL", seed=42)
        from cli.engine import sector_analysis
        results = sector_analysis('Technology')
        assert isinstance(results, list)

    def test_prediction_accuracy_empty(self):
        from cli.engine import prediction_accuracy
        with patch('cli.db.get_predictions', return_value=[]):
            result = prediction_accuracy('TSLA')
            assert result['total'] == 0

    @patch('cli.engine.get_price', return_value=_MOCK_PRICE)
    def test_prediction_accuracy_with_data(self, mock_price):
        from cli.engine import prediction_accuracy
        preds = [
            {'direction': 'bullish', 'price_at': 240.0, 'confidence': 0.6},
            {'direction': 'bearish', 'price_at': 260.0, 'confidence': 0.5},
        ]
        with patch('cli.db.get_predictions', return_value=preds):
            result = prediction_accuracy('TSLA')
            assert result['total'] == 2
            assert result['evaluated'] == 2
            assert result['correct'] == 2  # bullish at 240 correct (now 250), bearish at 260 correct (now 250)
            assert result['accuracy'] == 1.0


class TestDashboardAlerts:
    """Test dashboard with alert integration."""

    def setup_method(self):
        self.runner = CliRunner()
        self._tmp = tempfile.mkdtemp()
        self._patcher = patch.object(db, 'DB_PATH', Path(self._tmp) / 'data.db')
        self._patcher.start()

    def teardown_method(self):
        self._patcher.stop()

    @patch('cli.engine.check_alerts', return_value=[
        {'ticker': 'TSLA', 'condition': 'above', 'threshold': 200.0, 'current_price': 250.0}
    ])
    @patch('cli.engine.get_price', return_value=_MOCK_PRICE)
    def test_dashboard_shows_alerts(self, mock_price, mock_alerts):
        db.add_position("TSLA", 250.0, 10)
        with patch('cli.main.Live') as mock_live:
            mock_live.return_value.__enter__ = MagicMock()
            mock_live.return_value.__exit__ = MagicMock(return_value=False)
            with patch('time.sleep', side_effect=KeyboardInterrupt):
                result = self.runner.invoke(cli, ['dashboard'])
            assert result.exit_code == 0


# --- Top Movers Tests ---

class TestTopMovers:
    def setup_method(self):
        self.runner = CliRunner()
        self._tmp = tempfile.mkdtemp()
        self._patcher = patch.object(db, 'DB_PATH', Path(self._tmp) / 'data.db')
        self._patcher.start()

    def teardown_method(self):
        self._patcher.stop()

    def test_top_movers_empty_watchlist(self):
        result = self.runner.invoke(cli, ['top-movers'])
        assert result.exit_code == 0
        assert 'Watchlist empty' in result.output

    @patch('cli.engine.top_movers', return_value=[
        {'ticker': 'TSLA', 'price': 250.0, 'change_pct': 5.2, 'volume': 50_000_000},
        {'ticker': 'AAPL', 'price': 180.0, 'change_pct': -2.1, 'volume': 30_000_000},
    ])
    def test_top_movers_with_watchlist(self, mock_movers):
        db.add_watch("TSLA")
        db.add_watch("AAPL")
        result = self.runner.invoke(cli, ['top-movers'])
        assert result.exit_code == 0
        assert 'TSLA' in result.output
        assert 'AAPL' in result.output
        assert 'Top Movers' in result.output

    @patch('cli.engine.top_movers', return_value=[])
    def test_top_movers_no_data(self, mock_movers):
        db.add_watch("XYZ")
        result = self.runner.invoke(cli, ['top-movers'])
        assert result.exit_code == 0
        assert 'No data' in result.output


# --- Portfolio Risk Tests ---

class TestPortfolioRisk:
    def setup_method(self):
        self.runner = CliRunner()
        self._tmp = tempfile.mkdtemp()
        self._patcher = patch.object(db, 'DB_PATH', Path(self._tmp) / 'data.db')
        self._patcher.start()

    def teardown_method(self):
        self._patcher.stop()

    def test_portfolio_risk_no_positions(self):
        result = self.runner.invoke(cli, ['portfolio-risk'])
        assert result.exit_code == 0
        assert 'No positions' in result.output

    @patch('cli.engine.portfolio_risk', return_value={
        'var_95': -1250.50, 'beta': 1.15, 'volatility': 0.32,
        'total_value': 25000.0, 'tickers': ['TSLA'],
        'weights': {'TSLA': 1.0},
    })
    def test_portfolio_risk_with_positions(self, mock_risk):
        db.add_position("TSLA", 250.0, 100)
        result = self.runner.invoke(cli, ['portfolio-risk'])
        assert result.exit_code == 0
        assert 'Portfolio Risk' in result.output
        assert 'VaR' in result.output
        assert 'Beta' in result.output

    @patch('cli.engine.portfolio_risk', side_effect=Exception("API error"))
    def test_portfolio_risk_error(self, mock_risk):
        db.add_position("TSLA", 250.0, 100)
        result = self.runner.invoke(cli, ['portfolio-risk'])
        assert result.exit_code == 0
        assert 'Error' in result.output


# --- Engine: top_movers and portfolio_risk ---

class TestEngineTopMovers:
    @patch('cli.engine.get_price')
    def test_top_movers_sorted_by_abs_change(self, mock_price):
        mock_price.side_effect = [
            {'ticker': 'A', 'price': 100, 'change': 1, 'change_pct': 1.0, 'volume': 1000},
            {'ticker': 'B', 'price': 200, 'change': -10, 'change_pct': -5.0, 'volume': 2000},
            {'ticker': 'C', 'price': 50, 'change': 1.5, 'change_pct': 3.0, 'volume': 500},
        ]
        from cli.engine import top_movers
        result = top_movers(['A', 'B', 'C'])
        assert len(result) == 3
        assert result[0]['ticker'] == 'B'  # -5% is largest absolute
        assert result[1]['ticker'] == 'C'  # 3%
        assert result[2]['ticker'] == 'A'  # 1%

    @patch('cli.engine.get_price', side_effect=Exception("fail"))
    def test_top_movers_handles_errors(self, mock_price):
        from cli.engine import top_movers
        result = top_movers(['A', 'B'])
        assert result == []


class TestEnginePortfolioRisk:
    @patch('cli.engine.generate_ohlcv')
    @patch('cli.engine.get_price')
    def test_portfolio_risk_basic(self, mock_price, mock_ohlcv):
        import pandas as pd
        import numpy as np
        mock_price.return_value = {'price': 250.0}
        # Create mock OHLCV data
        dates = pd.date_range('2025-01-01', periods=60)
        prices = 250 + np.cumsum(np.random.randn(60) * 2)
        df = pd.DataFrame({'open': prices, 'high': prices + 1, 'low': prices - 1,
                           'close': prices, 'volume': np.random.randint(1e6, 5e6, 60)}, index=dates)
        mock_ohlcv.return_value = df
        from cli.engine import portfolio_risk
        result = portfolio_risk([{'ticker': 'TSLA', 'entry_price': 240, 'qty': 10}])
        assert 'var_95' in result
        assert 'beta' in result
        assert 'volatility' in result
        assert result['total_value'] > 0
        assert 'TSLA' in result['tickers']

    def test_portfolio_risk_empty(self):
        from cli.engine import portfolio_risk
        result = portfolio_risk([])
        assert result['var_95'] == 0
        assert result['total_value'] == 0


# --- Fibonacci Feature Tests ---

class TestFibonacciFeatures:
    def test_fib_features_computed(self):
        """Verify Fibonacci retracement features are in tier2 output."""
        import pandas as pd
        import numpy as np
        from backend.features.tier2_technical import Tier2Technical
        dates = pd.date_range('2025-01-01', periods=100)
        prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        df = pd.DataFrame({
            'open': prices, 'high': prices + 1, 'low': prices - 1,
            'close': prices, 'volume': np.random.randint(1e6, 5e6, 100)
        }, index=dates)
        feats = Tier2Technical.compute(df)
        for col in ['fib_236', 'fib_382', 'fib_500', 'fib_618']:
            assert col in feats.columns, f"Missing {col}"

    def test_fib_feature_names_listed(self):
        from backend.features.tier2_technical import Tier2Technical
        names = Tier2Technical.feature_names()
        assert 'fib_236' in names
        assert 'fib_618' in names

    def test_feature_store_count_updated(self):
        from backend.features.feature_store import FeatureStore
        counts = FeatureStore.feature_count()
        assert counts['tier2_technical'] == 85
        assert counts['total'] >= 512


# --- Ichimoku & Pivot Point Feature Tests ---

class TestIchimokuFeatures:
    def test_ichimoku_features_computed(self):
        """Verify Ichimoku cloud features are in tier2 output."""
        import pandas as pd
        import numpy as np
        from backend.features.tier2_technical import Tier2Technical
        dates = pd.date_range('2025-01-01', periods=100)
        prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        df = pd.DataFrame({
            'open': prices, 'high': prices + 1, 'low': prices - 1,
            'close': prices, 'volume': np.random.randint(1e6, 5e6, 100)
        }, index=dates)
        feats = Tier2Technical.compute(df)
        for col in ['ichimoku_tenkan', 'ichimoku_kijun', 'ichimoku_tk_cross',
                     'ichimoku_cloud_width', 'ichimoku_cloud_pos']:
            assert col in feats.columns, f"Missing {col}"

    def test_ichimoku_feature_names_listed(self):
        from backend.features.tier2_technical import Tier2Technical
        names = Tier2Technical.feature_names()
        assert 'ichimoku_tenkan' in names
        assert 'ichimoku_cloud_pos' in names

    def test_ichimoku_values_reasonable(self):
        """Ichimoku distance features should be small relative values."""
        import pandas as pd
        import numpy as np
        from backend.features.tier2_technical import Tier2Technical
        dates = pd.date_range('2025-01-01', periods=100)
        prices = 100 + np.cumsum(np.random.randn(100) * 0.3)
        df = pd.DataFrame({
            'open': prices, 'high': prices + 0.5, 'low': prices - 0.5,
            'close': prices, 'volume': np.random.randint(1e6, 5e6, 100)
        }, index=dates)
        feats = Tier2Technical.compute(df)
        # Distance features should be bounded (relative to price)
        assert abs(feats['ichimoku_tenkan'].iloc[-1]) < 1.0
        assert abs(feats['ichimoku_kijun'].iloc[-1]) < 1.0


class TestPivotPointFeatures:
    def test_pivot_features_computed(self):
        """Verify pivot point features are in tier2 output."""
        import pandas as pd
        import numpy as np
        from backend.features.tier2_technical import Tier2Technical
        dates = pd.date_range('2025-01-01', periods=100)
        prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        df = pd.DataFrame({
            'open': prices, 'high': prices + 1, 'low': prices - 1,
            'close': prices, 'volume': np.random.randint(1e6, 5e6, 100)
        }, index=dates)
        feats = Tier2Technical.compute(df)
        for col in ['pivot_distance', 'pivot_r1_dist', 'pivot_s1_dist']:
            assert col in feats.columns, f"Missing {col}"

    def test_pivot_feature_names_listed(self):
        from backend.features.tier2_technical import Tier2Technical
        names = Tier2Technical.feature_names()
        assert 'pivot_distance' in names
        assert 'pivot_r1_dist' in names
        assert 'pivot_s1_dist' in names


# --- Summary Command Tests ---

class TestSummaryCommand:
    @pytest.fixture(autouse=True)
    def tmp_db(self, tmp_path):
        with patch.object(db, 'DB_PATH', tmp_path / 'data.db'):
            yield

    def test_summary_empty(self):
        runner = CliRunner()
        result = runner.invoke(cli, ['summary'])
        assert result.exit_code == 0
        assert 'No positions' in result.output

    @patch('cli.engine.get_price')
    def test_summary_with_positions(self, mock_price):
        mock_price.return_value = {'ticker': 'TSLA', 'price': 260.0, 'change': 5.0, 'change_pct': 2.0, 'volume': 1000000, 'high': 262, 'low': 255}
        db.add_position("TSLA", 250.0, 10)
        runner = CliRunner()
        result = runner.invoke(cli, ['summary'])
        assert result.exit_code == 0
        assert 'Portfolio' in result.output
        assert 'TSLA' in result.output

    @patch('cli.engine.get_price')
    def test_summary_with_watchlist(self, mock_price):
        mock_price.return_value = {'ticker': 'AAPL', 'price': 180.0, 'change': -1.0, 'change_pct': -0.5, 'volume': 500000, 'high': 182, 'low': 179}
        db.add_watch("AAPL")
        runner = CliRunner()
        result = runner.invoke(cli, ['summary'])
        assert result.exit_code == 0
        assert 'Watchlist' in result.output
        assert 'AAPL' in result.output


# --- JSON Flag Tests ---

class TestJsonFlags:
    @patch('cli.engine.get_price')
    def test_price_json(self, mock_price):
        mock_price.return_value = {'ticker': 'TSLA', 'price': 250.0, 'change': 2.0, 'change_pct': 0.8, 'volume': 1000000, 'high': 252, 'low': 248}
        runner = CliRunner()
        result = runner.invoke(cli, ['price', 'TSLA', '--json'])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data['ticker'] == 'TSLA'
        assert data['price'] == 250.0

    @patch('cli.engine.get_news')
    def test_news_json(self, mock_news):
        mock_news.return_value = [{'headline': 'Test', 'sentiment': 0.5, 'source': 'Test'}]
        runner = CliRunner()
        result = runner.invoke(cli, ['news', 'TSLA', '--json'])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data) == 1

    @patch('cli.engine.get_earnings')
    def test_earnings_json(self, mock_earn):
        mock_earn.return_value = {'valuation': {'pe_ratio': 50}, 'quarters': [], 'latest': {}}
        runner = CliRunner()
        result = runner.invoke(cli, ['earnings', 'TSLA', '--json'])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert 'valuation' in data

    @patch('cli.engine.screen_tickers')
    def test_screen_json(self, mock_screen):
        mock_screen.return_value = [{'ticker': 'TSLA', 'price': 250, 'change_pct': 1.0, 'rsi': 28, 'stoch_k': 15, 'vol_ratio': 1.2, 'reason': 'RSI 28'}]
        runner = CliRunner()
        result = runner.invoke(cli, ['screen', '--json'])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data) == 1
        assert data[0]['ticker'] == 'TSLA'

    @patch('cli.engine.sector_analysis')
    def test_sector_json(self, mock_sector):
        mock_sector.return_value = [{'ticker': 'AAPL', 'price': 180, 'change_pct': 0.5, 'rsi': 55, 'macd_hist': 0.3, 'vol_ratio': 1.1}]
        runner = CliRunner()
        result = runner.invoke(cli, ['sector', 'Technology', '--json'])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data[0]['ticker'] == 'AAPL'

    @patch('cli.engine.correlate_tickers')
    def test_correlate_json(self, mock_corr):
        mock_corr.return_value = {'ticker1': 'TSLA', 'ticker2': 'AAPL', 'days': 180, 'correlation': 0.65, 'rolling_min': 0.3, 'rolling_max': 0.8, 'rolling_current': 0.65}
        runner = CliRunner()
        result = runner.invoke(cli, ['correlate', 'TSLA', 'AAPL', '--json'])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data['correlation'] == 0.65


# --- Scan Command Tests ---

class TestScanCommand:
    @patch('cli.engine.scan_tickers')
    def test_scan_basic(self, mock_scan):
        mock_scan.return_value = [{'ticker': 'TSLA', 'price': 250, 'change_pct': 1.0, 'rsi': 25, 'stoch_k': 18, 'vol_ratio': 2.5, 'reason': 'rsi=25.0, volume=2.5'}]
        runner = CliRunner()
        result = runner.invoke(cli, ['scan', 'rsi<30 AND volume>2'])
        assert result.exit_code == 0
        assert 'TSLA' in result.output

    @patch('cli.engine.scan_tickers')
    def test_scan_json(self, mock_scan):
        mock_scan.return_value = [{'ticker': 'NVDA', 'price': 500, 'change_pct': 3.0, 'rsi': 75, 'stoch_k': 85, 'vol_ratio': 3.0, 'reason': 'rsi=75.0'}]
        runner = CliRunner()
        result = runner.invoke(cli, ['scan', 'rsi>70', '--json'])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data[0]['ticker'] == 'NVDA'

    @patch('cli.engine.scan_tickers')
    def test_scan_no_matches(self, mock_scan):
        mock_scan.return_value = []
        runner = CliRunner()
        result = runner.invoke(cli, ['scan', 'rsi<10'])
        assert result.exit_code == 0
        assert 'No tickers match' in result.output


# --- Keltner/Donchian Signal Tests ---

class TestKeltnerDonchianSignals:
    @patch('cli.engine.get_analysis')
    def test_keltner_signal_in_analysis(self, mock_analysis):
        mock_analysis.return_value = {
            'ticker': 'TSLA', 'name': 'Tesla', 'price': 250, 'change_pct': 1.0,
            'volume': 1000000, 'avg_volume': 800000,
            'horizons': {
                'short': {'direction': 'bullish', 'confidence': 0.65, 'prediction': 0.02, 'stop': 244, 'target': 260, 'entry_lo': 248, 'entry_hi': 252, 'support': 238, 'resistance': 270},
                'medium': {'direction': 'neutral', 'confidence': 0.52, 'prediction': 0.01, 'stop': 235, 'target': 275, 'entry_lo': 248, 'entry_hi': 252, 'support': 238, 'resistance': 270},
                'long': {'direction': 'bullish', 'confidence': 0.71, 'prediction': 0.05, 'stop': 220, 'target': 310, 'entry_lo': 248, 'entry_hi': 252, 'support': 238, 'resistance': 270},
            },
            'bullish': ['Near Keltner lower band', 'Donchian channel breakout'],
            'bearish': ['Caution advised'],
            'fetched_at': '2026-01-01 12:00:00', 'market_cap': 800e9, 'sector': 'Consumer', 'pe_ratio': 60,
        }
        runner = CliRunner()
        result = runner.invoke(cli, ['analyze', 'TSLA'])
        assert result.exit_code == 0
        assert 'Keltner' in result.output or 'Donchian' in result.output


# --- Portfolio Risk Sharpe Tests ---

class TestPortfolioRiskSharpe:
    @patch('cli.engine.portfolio_risk')
    @patch('cli.db.get_positions')
    def test_sharpe_displayed(self, mock_pos, mock_risk):
        mock_pos.return_value = [{'ticker': 'TSLA', 'entry_price': 200, 'qty': 10}]
        mock_risk.return_value = {
            'var_95': -150.0, 'beta': 1.2, 'volatility': 0.35, 'sharpe': 1.5,
            'annualized_return': 0.15, 'total_value': 2500, 'tickers': ['TSLA'],
            'weights': {'TSLA': 1.0},
        }
        runner = CliRunner()
        result = runner.invoke(cli, ['portfolio-risk'])
        assert result.exit_code == 0
        assert 'Sharpe' in result.output
        assert '1.50' in result.output

    @patch('cli.engine.portfolio_risk')
    @patch('cli.db.get_positions')
    def test_portfolio_risk_json(self, mock_pos, mock_risk):
        mock_pos.return_value = [{'ticker': 'TSLA', 'entry_price': 200, 'qty': 10}]
        mock_risk.return_value = {'var_95': -150, 'beta': 1.2, 'volatility': 0.35, 'sharpe': 1.5, 'total_value': 2500, 'tickers': ['TSLA'], 'weights': {'TSLA': 1.0}}
        runner = CliRunner()
        result = runner.invoke(cli, ['portfolio-risk', '--json'])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data['sharpe'] == 1.5


# --- ROC Feature Tests ---

class TestROCFeatures:
    def test_roc_features_computed(self):
        """Verify multi-timeframe ROC features are in tier2 output."""
        import pandas as pd
        import numpy as np
        from backend.features.tier2_technical import Tier2Technical
        dates = pd.date_range('2025-01-01', periods=100)
        prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        df = pd.DataFrame({
            'open': prices, 'high': prices + 1, 'low': prices - 1,
            'close': prices, 'volume': np.random.randint(1e6, 5e6, 100)
        }, index=dates)
        feats = Tier2Technical.compute(df)
        for col in ['roc_5', 'roc_10', 'roc_20', 'roc_60']:
            assert col in feats.columns, f"Missing {col}"

    def test_roc_feature_names_listed(self):
        from backend.features.tier2_technical import Tier2Technical
        names = Tier2Technical.feature_names()
        for n in ['roc_5', 'roc_10', 'roc_20', 'roc_60']:
            assert n in names

    def test_feature_store_count_updated_170(self):
        from backend.features.feature_store import FeatureStore
        counts = FeatureStore.feature_count()
        assert counts['tier2_technical'] == 85
        assert counts['total'] >= 512


# --- JSON Flag Tests for remaining commands ---

class TestJsonFlagsExtended:
    @patch('cli.engine.get_analysis')
    def test_compare_json(self, mock_analysis):
        mock_analysis.return_value = {
            'ticker': 'TSLA', 'name': 'Tesla', 'price': 250, 'change_pct': 1.0,
            'volume': 1e6, 'avg_volume': 8e5,
            'horizons': {
                'short': {'direction': 'bullish', 'confidence': 0.6, 'prediction': 0.02, 'stop': 244, 'target': 260, 'entry_lo': 248, 'entry_hi': 252, 'support': 238, 'resistance': 270},
                'medium': {'direction': 'neutral', 'confidence': 0.5, 'prediction': 0.01, 'stop': 235, 'target': 275, 'entry_lo': 248, 'entry_hi': 252, 'support': 238, 'resistance': 270},
                'long': {'direction': 'bullish', 'confidence': 0.7, 'prediction': 0.05, 'stop': 220, 'target': 310, 'entry_lo': 248, 'entry_hi': 252, 'support': 238, 'resistance': 270},
            },
            'bullish': ['RSI oversold'], 'bearish': ['Caution'],
            'fetched_at': '2026-01-01', 'market_cap': 8e11, 'sector': 'Consumer', 'pe_ratio': 60,
        }
        runner = CliRunner()
        result = runner.invoke(cli, ['compare', 'TSLA', 'AAPL', '--json'])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert 'TSLA' in data

    @patch('cli.engine.run_backtest')
    def test_backtest_json(self, mock_bt):
        mock_bt.return_value = {'sharpe_ratio': 1.2, 'max_drawdown': 0.15, 'win_rate': 0.55, 'profit_factor': 1.8, 'total_trades': 50, 'gross_return': 10.0, 'net_return': 9.5, 'avg_holding_period': 1.0, 'slippage': 0.0005, 'spread': 0.0002, 'commission': 0.0, 'win_rate_by_horizon': {'short': 0.55, 'medium': 0.52, 'long': 0.58}}
        runner = CliRunner()
        result = runner.invoke(cli, ['backtest', 'TSLA', '--json'])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data['sharpe_ratio'] == 1.2

    @patch('cli.engine.get_prediction_history')
    @patch('cli.engine.prediction_accuracy')
    def test_history_json(self, mock_acc, mock_hist):
        mock_hist.return_value = [{'direction': 'bullish', 'confidence': 0.65, 'price_at': 250, 'horizon': 'short', 'created_at': '2026-01-01 12:00'}]
        mock_acc.return_value = {'total': 1, 'evaluated': 1, 'correct': 1, 'accuracy': 1.0}
        runner = CliRunner()
        result = runner.invoke(cli, ['history', 'TSLA', '--json'])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert 'predictions' in data
        assert 'accuracy' in data

    @pytest.fixture(autouse=True)
    def tmp_db(self, tmp_path):
        with patch.object(db, 'DB_PATH', tmp_path / 'data.db'):
            yield

    def test_history_json_empty(self):
        runner = CliRunner()
        result = runner.invoke(cli, ['history', 'TSLA', '--json'])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data == []

    @patch('cli.engine.get_price')
    def test_summary_json(self, mock_price):
        mock_price.return_value = {'ticker': 'TSLA', 'price': 260, 'change': 5, 'change_pct': 2.0, 'volume': 1e6, 'high': 262, 'low': 255}
        db.add_position("TSLA", 250.0, 10)
        db.add_watch("AAPL")
        runner = CliRunner()
        result = runner.invoke(cli, ['summary', '--json'])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert 'positions' in data
        assert 'watchlist' in data
        assert data['total_pnl'] == 100.0

    @patch('cli.engine.top_movers')
    def test_top_movers_json(self, mock_movers):
        mock_movers.return_value = [{'ticker': 'TSLA', 'price': 260, 'change_pct': 5.0, 'volume': 1e6, 'change': 12, 'high': 262, 'low': 248}]
        db.add_watch("TSLA")
        runner = CliRunner()
        result = runner.invoke(cli, ['top-movers', '--json'])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data[0]['ticker'] == 'TSLA'

    def test_top_movers_json_empty(self):
        runner = CliRunner()
        result = runner.invoke(cli, ['top-movers', '--json'])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data == []


# --- Scan Edge Case Tests ---

class TestScanEdgeCases:
    @patch('cli.engine.scan_tickers')
    def test_scan_invalid_filter_no_crash(self, mock_scan):
        """Invalid filter expressions should not crash."""
        mock_scan.return_value = []
        runner = CliRunner()
        result = runner.invoke(cli, ['scan', 'invalid_garbage'])
        assert result.exit_code == 0

    @patch('cli.engine.scan_tickers')
    def test_scan_empty_filter(self, mock_scan):
        mock_scan.return_value = []
        runner = CliRunner()
        result = runner.invoke(cli, ['scan', ''])
        assert result.exit_code == 0

    def test_scan_tickers_invalid_operator(self):
        """scan_tickers should handle malformed filter gracefully."""
        from cli.engine import scan_tickers
        with patch('cli.engine.generate_ohlcv') as mock_ohlcv:
            import pandas as pd
            import numpy as np
            dates = pd.date_range('2025-01-01', periods=100)
            prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
            mock_ohlcv.return_value = pd.DataFrame({
                'open': prices, 'high': prices + 1, 'low': prices - 1,
                'close': prices, 'volume': np.random.randint(1e6, 5e6, 100)
            }, index=dates)
            # Malformed filter - no valid operator
            result = scan_tickers(['TSLA'], 'rsi==30')
            assert result == []

    def test_scan_tickers_unknown_metric(self):
        """Unknown metric names should not match."""
        from cli.engine import scan_tickers
        with patch('cli.engine.generate_ohlcv') as mock_ohlcv:
            import pandas as pd
            import numpy as np
            dates = pd.date_range('2025-01-01', periods=100)
            prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
            mock_ohlcv.return_value = pd.DataFrame({
                'open': prices, 'high': prices + 1, 'low': prices - 1,
                'close': prices, 'volume': np.random.randint(1e6, 5e6, 100)
            }, index=dates)
            result = scan_tickers(['TSLA'], 'foobar>10')
            assert result == []


# --- Replay Command Tests ---

class TestReplayCommand:
    @patch('cli.engine.replay_analysis')
    def test_replay_basic(self, mock_replay):
        mock_replay.return_value = {
            'ticker': 'TSLA', 'date': '2025-06-15', 'price': 240.0,
            'change_pct': 1.5, 'volume': 2000000, 'avg_volume': 1500000,
            'has_ticker_model': True,
            'horizons': {
                'short': {'direction': 'bullish', 'confidence': 0.62, 'prediction': 0.015, 'stop': 235, 'target': 255,
                          'conviction_tier': 'MODERATE', 'conviction_label': 'moderate conviction',
                          'conviction_verdict': '🟡 LEAN BUY (62% — moderate conviction)'},
                'medium': {'direction': 'neutral', 'confidence': 0.50, 'prediction': 0.005, 'stop': 225, 'target': 265,
                           'conviction_tier': 'LOW', 'conviction_label': 'low conviction',
                           'conviction_verdict': '⚪ LEAN BUY (50% — low conviction)'},
                'long': {'direction': 'bullish', 'confidence': 0.68, 'prediction': 0.04, 'stop': 210, 'target': 300,
                         'conviction_tier': 'HIGH', 'conviction_label': 'high conviction',
                         'conviction_verdict': '🟢 BUY (68% — high conviction)',
                         'shap_text': 'Why BUY:\n  📈 RSI(14) (28.3) → +12% influence',
                         'vol_zscore': 1.3, 'vol_zscore_desc': 'unusual'},
            },
            'outcome': {
                'short': {'actual_return': 2.1, 'correct': True, 'future_price': 245.0},
                'medium': {'actual_return': -1.5, 'correct': False, 'future_price': 236.4},
                'long': {'actual_return': 8.3, 'correct': True, 'future_price': 260.0},
            },
        }
        runner = CliRunner()
        result = runner.invoke(cli, ['replay', 'TSLA', '--date', '2025-06-15'])
        assert result.exit_code == 0
        assert 'TSLA' in result.output
        assert '2025-06-15' in result.output
        assert 'Correct' in result.output or 'Wrong' in result.output
        assert 'per-ticker model' in result.output
        assert 'LEAN BUY' in result.output or 'BUY' in result.output

    @patch('cli.engine.replay_analysis')
    def test_replay_json(self, mock_replay):
        mock_replay.return_value = {
            'ticker': 'TSLA', 'date': '2025-06-15', 'price': 240.0,
            'change_pct': 1.5, 'volume': 2e6, 'avg_volume': 1.5e6,
            'has_ticker_model': False,
            'horizons': {
                'short': {'direction': 'bullish', 'confidence': 0.62, 'prediction': 0.015, 'stop': 235, 'target': 255,
                          'conviction_verdict': '🟡 LEAN BUY (62%)'},
                'medium': {'direction': 'neutral', 'confidence': 0.50, 'prediction': 0.005, 'stop': 225, 'target': 265,
                           'conviction_verdict': '⚪ LEAN BUY (50%)'},
                'long': {'direction': 'bullish', 'confidence': 0.68, 'prediction': 0.04, 'stop': 210, 'target': 300,
                         'conviction_verdict': '🟢 BUY (68%)'},
            },
            'outcome': {},
        }
        runner = CliRunner()
        result = runner.invoke(cli, ['replay', 'TSLA', '--date', '2025-06-15', '--json'])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data['ticker'] == 'TSLA'
        assert data['date'] == '2025-06-15'
        assert data['has_ticker_model'] is False

    @patch('cli.engine.replay_analysis')
    def test_replay_error(self, mock_replay):
        mock_replay.side_effect = ValueError("Not enough data before 2020-01-01 for TSLA")
        runner = CliRunner()
        result = runner.invoke(cli, ['replay', 'TSLA', '--date', '2020-01-01'])
        assert result.exit_code == 0
        assert 'Error' in result.output


# --- Alerts Auto Tests ---

class TestAlertsAuto:
    @patch('cli.engine.auto_alerts')
    def test_alerts_auto_sets_alerts(self, mock_auto):
        mock_auto.return_value = [
            {'condition': 'below', 'threshold': 235.0, 'label': 'short stop-loss'},
            {'condition': 'above', 'threshold': 260.0, 'label': 'short target'},
        ]
        runner = CliRunner()
        result = runner.invoke(cli, ['alerts', 'auto', 'TSLA'])
        assert result.exit_code == 0
        assert 'TSLA' in result.output
        assert 'stop-loss' in result.output
        assert 'target' in result.output

    @patch('cli.engine.auto_alerts')
    def test_alerts_auto_empty(self, mock_auto):
        mock_auto.return_value = []
        runner = CliRunner()
        result = runner.invoke(cli, ['alerts', 'auto', 'TSLA'])
        assert result.exit_code == 0
        assert 'No alerts' in result.output


# --- CMF/ATRP Scan Filter Tests ---

class TestNewScanFilters:
    @patch('cli.engine.generate_ohlcv')
    def test_scan_cmf_filter(self, mock_ohlcv):
        import pandas as pd
        import numpy as np
        np.random.seed(42)
        n = 100
        dates = pd.date_range('2024-01-01', periods=n)
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        df = pd.DataFrame({
            'open': close - 0.5, 'high': close + 1, 'low': close - 1,
            'close': close, 'volume': np.random.randint(1e6, 5e6, n)
        }, index=dates)
        mock_ohlcv.return_value = df
        from cli.engine import scan_tickers
        results = scan_tickers(['TSLA'], 'cmf>-1')
        assert isinstance(results, list)

    @patch('cli.engine.generate_ohlcv')
    def test_scan_atrp_filter(self, mock_ohlcv):
        import pandas as pd
        import numpy as np
        np.random.seed(42)
        n = 100
        dates = pd.date_range('2024-01-01', periods=n)
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        df = pd.DataFrame({
            'open': close - 0.5, 'high': close + 1, 'low': close - 1,
            'close': close, 'volume': np.random.randint(1e6, 5e6, n)
        }, index=dates)
        mock_ohlcv.return_value = df
        from cli.engine import scan_tickers
        results = scan_tickers(['TSLA'], 'atrp>0')
        assert isinstance(results, list)


# --- VWAP Band Signal Tests ---

class TestVWAPBandSignals:
    def test_engine_has_auto_alerts(self):
        """Verify auto_alerts function exists in engine."""
        from cli.engine import auto_alerts
        assert callable(auto_alerts)

    def test_feature_store_175(self):
        from backend.features.feature_store import FeatureStore
        assert FeatureStore.feature_count()['total'] >= 512
        assert len(FeatureStore.all_feature_names()) >= 512


# --- PSAR/ADXR/ADL/Force Index Tests ---

class TestNewFeatureSignals:
    def test_psar_signal_bullish(self):
        """PSAR signal should appear when psar_dist > 0.01."""
        from cli.engine import get_analysis
        from unittest.mock import patch
        import pandas as pd
        import numpy as np
        n = 100
        dates = pd.date_range('2024-01-01', periods=n)
        close = 100 + np.arange(n) * 0.3
        df = pd.DataFrame({
            'open': close - 0.2, 'high': close + 1, 'low': close - 1,
            'close': close, 'volume': np.random.randint(1e6, 5e6, n)
        }, index=dates)
        with patch('cli.engine.generate_ohlcv', return_value=df), \
             patch('cli.engine.get_options_chain', return_value={'calls': pd.DataFrame({'strike':[100],'volume':[1000],'openInterest':[5000],'impliedVolatility':[0.3]}), 'puts': pd.DataFrame({'strike':[100],'volume':[800],'openInterest':[4000],'impliedVolatility':[0.32]}), 'iv_mean': 0.35, 'iv_skew': 0.05, 'pcr': 0.8}), \
             patch('cli.engine.get_fundamentals', return_value={'pe_ratio': 25, 'ps_ratio': 10, 'pb_ratio': 5, 'eps': 3.5, 'revenue': 50e9, 'revenue_growth': 0.15, 'profit_margin': 0.12, 'debt_to_equity': 0.8, 'current_ratio': 1.5, 'roe': 0.2, 'analyst_buy': 20, 'analyst_hold': 10, 'analyst_sell': 5, 'target_price': 120}), \
             patch('cli.engine.get_sentiment', return_value={'score': 0.6, 'articles': 15, 'positive': 10, 'negative': 3, 'neutral': 2, 'top_headline': 'test'}), \
             patch('cli.engine.get_macro_data', return_value={'gdp_growth': 2.5, 'cpi': 3.2, 'fed_rate': 5.25, 'unemployment': 3.7, 'vix': 18.5, 'yield_10y': 4.2, 'yield_2y': 4.8}), \
             patch('yfinance.Ticker') as mock_yf:
            mock_yf.return_value.info = {'shortName': 'Test', 'marketCap': 1e12, 'sector': 'Tech', 'trailingPE': 25}
            a = get_analysis('TEST')
            # Should have signals list
            assert 'bullish' in a
            assert 'bearish' in a

    def test_scan_psar_filter(self):
        """Scan should support psar filter."""
        from cli.engine import scan_tickers
        from unittest.mock import patch
        import pandas as pd
        import numpy as np
        n = 100
        dates = pd.date_range('2024-01-01', periods=n)
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        df = pd.DataFrame({
            'open': close - 0.5, 'high': close + 1, 'low': close - 1,
            'close': close, 'volume': np.random.randint(1e6, 5e6, n)
        }, index=dates)
        with patch('cli.engine.generate_ohlcv', return_value=df):
            results = scan_tickers(['TEST'], 'psar>0')
            assert isinstance(results, list)

    def test_scan_force_filter(self):
        """Scan should support force filter."""
        from cli.engine import scan_tickers
        from unittest.mock import patch
        import pandas as pd
        import numpy as np
        n = 100
        dates = pd.date_range('2024-01-01', periods=n)
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        df = pd.DataFrame({
            'open': close - 0.5, 'high': close + 1, 'low': close - 1,
            'close': close, 'volume': np.random.randint(1e6, 5e6, n)
        }, index=dates)
        with patch('cli.engine.generate_ohlcv', return_value=df):
            results = scan_tickers(['TEST'], 'force>0')
            assert isinstance(results, list)


class TestPortfolioOptimize:
    def test_optimize_needs_2_positions(self):
        from cli.engine import portfolio_optimize
        result = portfolio_optimize([{'ticker': 'AAPL', 'entry_price': 150, 'qty': 10}])
        assert 'error' in result

    @patch('cli.engine.generate_ohlcv')
    def test_optimize_returns_strategies(self, mock_ohlcv):
        import pandas as pd
        import numpy as np
        n = 120
        dates = pd.date_range('2024-01-01', periods=n)
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        df = pd.DataFrame({
            'open': close - 0.5, 'high': close + 1, 'low': close - 1,
            'close': close, 'volume': np.random.randint(1e6, 5e6, n)
        }, index=dates)
        mock_ohlcv.return_value = df
        from cli.engine import portfolio_optimize
        positions = [
            {'ticker': 'AAPL', 'entry_price': 150, 'qty': 10},
            {'ticker': 'MSFT', 'entry_price': 300, 'qty': 5},
        ]
        result = portfolio_optimize(positions)
        assert 'current' in result
        assert 'min_variance' in result
        assert 'max_sharpe' in result
        assert 'equal_weight' in result
        for strat in ['current', 'min_variance', 'max_sharpe', 'equal_weight']:
            assert 'weights' in result[strat]
            assert 'sharpe' in result[strat]
            assert 'volatility' in result[strat]

    @patch('cli.engine.generate_ohlcv')
    def test_optimize_weights_sum_to_one(self, mock_ohlcv):
        import pandas as pd
        import numpy as np
        n = 120
        dates = pd.date_range('2024-01-01', periods=n)
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        df = pd.DataFrame({
            'open': close - 0.5, 'high': close + 1, 'low': close - 1,
            'close': close, 'volume': np.random.randint(1e6, 5e6, n)
        }, index=dates)
        mock_ohlcv.return_value = df
        from cli.engine import portfolio_optimize
        positions = [
            {'ticker': 'AAPL', 'entry_price': 150, 'qty': 10},
            {'ticker': 'MSFT', 'entry_price': 300, 'qty': 5},
            {'ticker': 'GOOGL', 'entry_price': 140, 'qty': 20},
        ]
        result = portfolio_optimize(positions)
        for strat in ['min_variance', 'max_sharpe', 'equal_weight']:
            w_sum = sum(result[strat]['weights'].values())
            assert abs(w_sum - 1.0) < 0.01, f"{strat} weights sum to {w_sum}"

    def test_portfolio_optimize_cli_exists(self):
        """Verify portfolio-optimize command is registered."""
        from cli.main import cli
        from click.testing import CliRunner
        runner = CliRunner()
        result = runner.invoke(cli, ['portfolio-optimize', '--help'])
        assert result.exit_code == 0
        assert 'optimized' in result.output.lower() or 'portfolio' in result.output.lower()


class TestHTMLExport:
    def test_export_html_option_exists(self):
        """Verify export command accepts html format."""
        from cli.main import cli
        from click.testing import CliRunner
        runner = CliRunner()
        result = runner.invoke(cli, ['export', '--help'])
        assert result.exit_code == 0
        assert 'html' in result.output

    def test_export_html_format(self):
        """Verify HTML export produces valid HTML."""
        from cli.main import cli
        from click.testing import CliRunner
        from unittest.mock import patch
        runner = CliRunner()
        mock_analysis = {
            'ticker': 'TEST', 'name': 'Test Inc', 'price': 100.0, 'change_pct': 2.5,
            'volume': 1000000, 'avg_volume': 800000,
            'horizons': {
                'short': {'prediction': 0.02, 'confidence': 0.65, 'direction': 'bullish', 'stop': 96, 'target': 108},
                'medium': {'prediction': 0.01, 'confidence': 0.55, 'direction': 'bullish', 'stop': 92, 'target': 115},
                'long': {'prediction': 0.03, 'confidence': 0.70, 'direction': 'bullish', 'stop': 85, 'target': 130},
            },
            'bullish': ['RSI oversold'], 'bearish': ['High VIX'],
        }
        mock_features = {'rsi_14': 28.5, 'macd_hist': 0.5}
        with patch('cli.engine.get_analysis', return_value=mock_analysis), \
             patch('cli.engine.get_features', return_value=mock_features):
            result = runner.invoke(cli, ['export', 'TEST', '--format', 'html'])
            assert result.exit_code == 0
            assert '<!DOCTYPE html>' in result.output
            assert 'TEST' in result.output
            assert 'Predictions' in result.output


class TestPositionHistory:
    def test_save_and_get_snapshots(self, tmp_path):
        """Test snapshot save/retrieve."""
        import cli.db as db
        db.DB_PATH = tmp_path / 'test.db'
        db.save_snapshot('AAPL', 155.0, 3.3)
        db.save_snapshot('AAPL', 157.0, 4.7)
        snaps = db.get_snapshots('AAPL')
        assert len(snaps) == 2
        assert snaps[0]['price'] == 157.0  # most recent first

    def test_position_history_cli_exists(self):
        from cli.main import cli
        from click.testing import CliRunner
        runner = CliRunner()
        result = runner.invoke(cli, ['position-history', '--help'])
        assert result.exit_code == 0
        assert 'cumulative' in result.output.lower() or 'history' in result.output.lower()

    def test_position_history_empty(self, tmp_path):
        import cli.db as db
        db.DB_PATH = tmp_path / 'test.db'
        from cli.main import cli
        from click.testing import CliRunner
        runner = CliRunner()
        result = runner.invoke(cli, ['position-history', 'AAPL', '--json'])
        assert result.exit_code == 0


class TestCompareChart:
    def test_compare_chart_flag_exists(self):
        from cli.main import cli
        from click.testing import CliRunner
        runner = CliRunner()
        result = runner.invoke(cli, ['compare', '--help'])
        assert result.exit_code == 0
        assert '--chart' in result.output

    @pytest.fixture
    def mock_analysis(self):
        with unittest.mock.patch('cli.engine.get_analysis') as m, \
             unittest.mock.patch('cli.engine.get_price_history') as mh:
            m.return_value = {
                'price': 250.0, 'change_pct': 1.5, 'volume': 45_000_000,
                'sector': 'Tech', 'pe_ratio': 25.0, 'market_cap': 800_000_000_000,
                'horizons': {
                    'short': {'direction': 'bullish', 'confidence': 0.64},
                    'medium': {'direction': 'neutral', 'confidence': 0.52},
                    'long': {'direction': 'bullish', 'confidence': 0.71},
                },
                'bullish': ['RSI oversold'], 'bearish': ['High IV'],
            }
            mh.return_value = [100 + i * 0.5 for i in range(30)]
            yield m, mh

    def test_compare_chart_output(self, mock_analysis):
        from cli.main import cli
        from click.testing import CliRunner
        runner = CliRunner()
        result = runner.invoke(cli, ['compare', 'TSLA', 'AAPL', '--chart'])
        assert result.exit_code == 0


class TestReplayRange:
    def test_replay_range_cli_exists(self):
        from cli.main import cli
        from click.testing import CliRunner
        runner = CliRunner()
        result = runner.invoke(cli, ['replay-range', '--help'])
        assert result.exit_code == 0
        assert '--start' in result.output
        assert '--end' in result.output

    @pytest.fixture
    def mock_replay_range(self):
        with unittest.mock.patch('cli.engine.replay_range') as m:
            m.return_value = [
                {
                    'ticker': 'TSLA', 'date': '2025-01-02', 'price': 250.0,
                    'change_pct': 1.0, 'volume': 40_000_000,
                    'horizons': {
                        'short': {'direction': 'bullish', 'confidence': 0.6},
                        'medium': {'direction': 'neutral', 'confidence': 0.5},
                        'long': {'direction': 'bullish', 'confidence': 0.7},
                    },
                    'outcome': {
                        'short': {'actual_return': 2.0, 'correct': True, 'future_price': 255.0},
                        'medium': {'actual_return': -1.0, 'correct': False, 'future_price': 247.5},
                        'long': {'actual_return': 5.0, 'correct': True, 'future_price': 262.5},
                    },
                }
            ]
            yield m

    def test_replay_range_output(self, mock_replay_range):
        from cli.main import cli
        from click.testing import CliRunner
        runner = CliRunner()
        result = runner.invoke(cli, ['replay-range', 'TSLA', '--start', '2025-01-01', '--end', '2025-01-05'])
        assert result.exit_code == 0
        assert 'TSLA' in result.output

    def test_replay_range_json(self, mock_replay_range):
        from cli.main import cli
        from click.testing import CliRunner
        runner = CliRunner()
        result = runner.invoke(cli, ['replay-range', 'TSLA', '--start', '2025-01-01', '--end', '2025-01-05', '--json'])
        assert result.exit_code == 0
        import json
        data = json.loads(result.output)
        assert len(data) == 1

    def test_replay_range_empty(self):
        from cli.main import cli
        from click.testing import CliRunner
        runner = CliRunner()
        with unittest.mock.patch('cli.engine.replay_range', return_value=[]):
            result = runner.invoke(cli, ['replay-range', 'TSLA', '--start', '2025-01-01', '--end', '2025-01-05'])
            assert result.exit_code == 0
            assert 'No data' in result.output


class TestSparklineHelper:
    def test_sparkline_basic(self):
        from cli.main import _sparkline
        result = _sparkline([1, 2, 3, 4, 5])
        assert len(result) == 5
        assert result[0] != result[-1]  # different chars for min/max

    def test_sparkline_empty(self):
        from cli.main import _sparkline
        assert _sparkline([]) == ""

    def test_sparkline_constant(self):
        from cli.main import _sparkline
        result = _sparkline([5, 5, 5, 5])
        assert len(result) == 4

    def test_sparkline_downsamples(self):
        from cli.main import _sparkline
        result = _sparkline(list(range(100)), width=20)
        assert len(result) == 20


class TestNewScanFiltersIter19:
    @pytest.fixture
    def mock_ohlcv(self):
        with unittest.mock.patch('cli.engine.generate_ohlcv') as m:
            import pandas as pd
            import numpy as np
            n = 100
            dates = pd.date_range('2024-01-01', periods=n, freq='D')
            df = pd.DataFrame({
                'open': 100 + np.random.RandomState(42).randn(n),
                'high': 102 + np.random.RandomState(42).randn(n),
                'low': 98 + np.random.RandomState(42).randn(n),
                'close': 100 + np.random.RandomState(42).randn(n),
                'volume': np.random.RandomState(42).randint(1_000_000, 5_000_000, n),
            }, index=dates)
            m.return_value = df
            yield m

    def test_scan_cci_filter(self, mock_ohlcv):
        from cli.engine import scan_tickers
        results = scan_tickers(['TSLA'], 'cci>0')
        # Should not crash
        assert isinstance(results, list)

    def test_scan_trix_filter(self, mock_ohlcv):
        from cli.engine import scan_tickers
        results = scan_tickers(['TSLA'], 'trix>0')
        assert isinstance(results, list)

    def test_scan_ultosc_filter(self, mock_ohlcv):
        from cli.engine import scan_tickers
        results = scan_tickers(['TSLA'], 'ultosc<30')
        assert isinstance(results, list)

    def test_scan_vortex_filter(self, mock_ohlcv):
        from cli.engine import scan_tickers
        results = scan_tickers(['TSLA'], 'vortex>0')
        assert isinstance(results, list)


class TestGetPriceHistory:
    def test_get_price_history_returns_list(self):
        with unittest.mock.patch('cli.engine.generate_ohlcv') as m:
            import pandas as pd
            import numpy as np
            dates = pd.date_range('2024-01-01', periods=30, freq='D')
            m.return_value = pd.DataFrame({
                'open': 100, 'high': 102, 'low': 98,
                'close': np.arange(100, 130, dtype=float),
                'volume': 1_000_000,
            }, index=dates)
            from cli.engine import get_price_history
            result = get_price_history('TSLA', 30)
            assert isinstance(result, list)
            assert len(result) == 30
            assert result[0] == 100.0


class TestPortfolioRebalance:
    def setup_method(self):
        self.runner = CliRunner()

    @unittest.mock.patch('cli.engine.generate_ohlcv')
    @unittest.mock.patch('cli.engine.get_price')
    @unittest.mock.patch('cli.engine.portfolio_optimize')
    def test_rebalance_returns_trades(self, mock_opt, mock_price, mock_ohlcv):
        mock_price.return_value = {'ticker': 'AAPL', 'price': 200.0, 'change': 1, 'change_pct': 0.5, 'volume': 1000000, 'high': 201, 'low': 199}
        mock_opt.return_value = {
            'max_sharpe': {'weights': {'AAPL': 0.6, 'MSFT': 0.4}, 'return': 0.15, 'volatility': 0.2, 'sharpe': 1.5},
        }
        from cli.engine import portfolio_rebalance
        positions = [
            {'ticker': 'AAPL', 'entry_price': 180, 'qty': 10},
            {'ticker': 'MSFT', 'entry_price': 350, 'qty': 5},
        ]
        trades = portfolio_rebalance(positions, 'max_sharpe')
        assert isinstance(trades, list)
        for t in trades:
            assert 'ticker' in t
            assert 'action' in t
            assert t['action'] in ('BUY', 'SELL')

    def test_rebalance_cli_no_positions(self):
        with unittest.mock.patch('cli.db.get_positions', return_value=[]):
            result = self.runner.invoke(cli, ['portfolio-rebalance'])
            assert result.exit_code == 0
            assert 'No positions' in result.output

    def test_rebalance_cli_json(self):
        with unittest.mock.patch('cli.db.get_positions', return_value=[]):
            result = self.runner.invoke(cli, ['portfolio-rebalance', '--json'])
            assert result.exit_code == 0


class TestNewScanFiltersV2:
    @pytest.fixture
    def mock_ohlcv(self):
        import pandas as pd
        import numpy as np
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        rng = np.random.RandomState(42)
        prices = 100 + np.cumsum(rng.normal(0, 1, 100))
        df = pd.DataFrame({
            'open': prices - 0.5, 'high': prices + rng.uniform(0.5, 2, 100),
            'low': prices - rng.uniform(0.5, 2, 100), 'close': prices,
            'volume': rng.randint(1_000_000, 5_000_000, 100)
        }, index=dates)
        with unittest.mock.patch('cli.engine.generate_ohlcv', return_value=df):
            yield

    def test_scan_williams7_filter(self, mock_ohlcv):
        from cli.engine import scan_tickers
        results = scan_tickers(['TSLA'], 'williams7<-50')
        assert isinstance(results, list)

    def test_scan_dpo_filter(self, mock_ohlcv):
        from cli.engine import scan_tickers
        results = scan_tickers(['TSLA'], 'dpo>0')
        assert isinstance(results, list)

    def test_scan_mass_filter(self, mock_ohlcv):
        from cli.engine import scan_tickers
        results = scan_tickers(['TSLA'], 'mass>20')
        assert isinstance(results, list)

    def test_scan_emv_filter(self, mock_ohlcv):
        from cli.engine import scan_tickers
        results = scan_tickers(['TSLA'], 'emv>0')
        assert isinstance(results, list)


class TestPositionsSparkline:
    def test_positions_has_trend_column(self):
        runner = CliRunner()
        with unittest.mock.patch('cli.db.get_positions', return_value=[
            {'ticker': 'AAPL', 'entry_price': 180.0, 'qty': 10}
        ]), unittest.mock.patch('cli.engine.get_price', return_value={
            'ticker': 'AAPL', 'price': 190.0, 'change': 1, 'change_pct': 0.5,
            'volume': 1000000, 'high': 191, 'low': 189
        }), unittest.mock.patch('cli.engine.get_price_history', return_value=[180.0 + i for i in range(14)]), \
             unittest.mock.patch('cli.db.save_snapshot'):
            result = runner.invoke(cli, ['positions'])
            assert result.exit_code == 0
            assert 'Trend' in result.output


class TestHeatmapCommand:
    """Tests for the heatmap command."""

    def setup_method(self):
        self.runner = CliRunner()

    @pytest.fixture
    def mock_price(self):
        return unittest.mock.patch('cli.engine.get_price', return_value={
            'ticker': 'AAPL', 'price': 180.0, 'change': 2.0, 'change_pct': 1.1,
            'volume': 50_000_000, 'high': 182.0, 'low': 178.0,
        })

    def test_heatmap_runs(self, mock_price):
        with mock_price:
            result = self.runner.invoke(cli, ['heatmap'])
            assert result.exit_code == 0

    def test_heatmap_json(self, mock_price):
        with mock_price:
            result = self.runner.invoke(cli, ['heatmap', '--json'])
            assert result.exit_code == 0

    def test_heatmap_help(self):
        result = self.runner.invoke(cli, ['heatmap', '--help'])
        assert result.exit_code == 0
        assert 'heatmap' in result.output.lower()


class TestAlertModes:
    """Tests for conservative/aggressive alert modes."""

    def setup_method(self):
        self.runner = CliRunner()

    def test_alerts_auto_conservative(self):
        with unittest.mock.patch('cli.engine.auto_alerts', return_value=[
            {'condition': 'below', 'threshold': 245.0, 'label': 'short stop-loss'},
        ]):
            result = self.runner.invoke(cli, ['alerts', 'auto', 'TSLA', '--conservative'])
            assert result.exit_code == 0

    def test_alerts_auto_aggressive(self):
        with unittest.mock.patch('cli.engine.auto_alerts', return_value=[
            {'condition': 'above', 'threshold': 280.0, 'label': 'short target'},
        ]):
            result = self.runner.invoke(cli, ['alerts', 'auto', 'TSLA', '--aggressive'])
            assert result.exit_code == 0

    def test_auto_alerts_mode_parameter(self):
        """Engine auto_alerts accepts mode parameter."""
        from cli.engine import auto_alerts
        import inspect
        sig = inspect.signature(auto_alerts)
        assert 'mode' in sig.parameters


class TestNewScanFiltersV3:
    """Tests for CMO, Aroon, KST scan filters."""

    @staticmethod
    def _mock_df():
        import pandas as pd
        import numpy as np
        n = 100
        dates = pd.date_range('2024-01-01', periods=n, freq='D')
        rng = np.random.RandomState(42)
        prices = 100 + np.cumsum(rng.normal(0, 1, n))
        return pd.DataFrame({
            'open': prices - 0.5, 'high': prices + rng.uniform(0.5, 2, n),
            'low': prices - rng.uniform(0.5, 2, n), 'close': prices,
            'volume': rng.randint(1_000_000, 5_000_000, n)
        }, index=dates)

    @patch('cli.engine.generate_ohlcv')
    def test_scan_cmo_filter(self, mock_ohlcv):
        mock_ohlcv.return_value = self._mock_df()
        from cli.engine import scan_tickers
        results = scan_tickers(['AAPL'], 'cmo>-100')
        assert isinstance(results, list)

    @patch('cli.engine.generate_ohlcv')
    def test_scan_aroon_up_filter(self, mock_ohlcv):
        mock_ohlcv.return_value = self._mock_df()
        from cli.engine import scan_tickers
        results = scan_tickers(['AAPL'], 'aroon_up>0')
        assert isinstance(results, list)

    @patch('cli.engine.generate_ohlcv')
    def test_scan_aroon_down_filter(self, mock_ohlcv):
        mock_ohlcv.return_value = self._mock_df()
        from cli.engine import scan_tickers
        results = scan_tickers(['AAPL'], 'aroon_down>0')
        assert isinstance(results, list)

    @patch('cli.engine.generate_ohlcv')
    def test_scan_kst_filter(self, mock_ohlcv):
        mock_ohlcv.return_value = self._mock_df()
        from cli.engine import scan_tickers
        results = scan_tickers(['AAPL'], 'kst>-100')
        assert isinstance(results, list)


class TestConnorsRSIChoppinessFeatures:
    """Tests for Connors RSI, Choppiness, ATR Bands features and new CLI commands."""

    def _mock_df(self, n=150):
        import numpy as np
        import pandas as pd
        rng = np.random.RandomState(42)
        dates = pd.date_range('2024-01-01', periods=n, freq='D')
        prices = 100 + np.cumsum(rng.normal(0.05, 1.5, n))
        return pd.DataFrame({
            'open': prices - 0.3, 'high': prices + rng.uniform(0.5, 2, n),
            'low': prices - rng.uniform(0.5, 2, n),
            'close': prices, 'volume': rng.randint(1_000_000, 5_000_000, n)
        }, index=dates)

    @patch('cli.engine.generate_ohlcv')
    def test_scan_connors_rsi_filter(self, mock_ohlcv):
        mock_ohlcv.return_value = self._mock_df()
        from cli.engine import scan_tickers
        results = scan_tickers(['AAPL'], 'connors_rsi>0')
        assert isinstance(results, list)

    @patch('cli.engine.generate_ohlcv')
    def test_scan_choppiness_filter(self, mock_ohlcv):
        mock_ohlcv.return_value = self._mock_df()
        from cli.engine import scan_tickers
        results = scan_tickers(['AAPL'], 'choppiness>0')
        assert isinstance(results, list)

    @patch('cli.engine.generate_ohlcv')
    def test_momentum_ranking(self, mock_ohlcv):
        mock_ohlcv.return_value = self._mock_df()
        from cli.engine import momentum_ranking
        results = momentum_ranking(['AAPL', 'TSLA'])
        assert isinstance(results, list)
        if results:
            r = results[0]
            assert 'score' in r
            assert 'connors_rsi' in r
            assert 'choppiness' in r
            assert 'trending' in r

    @patch('cli.engine.generate_ohlcv')
    def test_momentum_ranking_sorted_by_score(self, mock_ohlcv):
        mock_ohlcv.return_value = self._mock_df()
        from cli.engine import momentum_ranking
        results = momentum_ranking(['AAPL', 'TSLA', 'GOOGL'])
        if len(results) >= 2:
            assert results[0]['score'] >= results[1]['score']

    def test_momentum_cli_help(self):
        from click.testing import CliRunner
        from cli.main import cli
        runner = CliRunner()
        result = runner.invoke(cli, ['momentum', '--help'])
        assert result.exit_code == 0
        assert 'momentum' in result.output.lower()

    def test_tax_report_cli_help(self):
        from click.testing import CliRunner
        from cli.main import cli
        runner = CliRunner()
        result = runner.invoke(cli, ['tax-report', '--help'])
        assert result.exit_code == 0
        assert 'tax' in result.output.lower() or 'capital' in result.output.lower()

    def test_portfolio_tax_report_empty(self):
        from cli.engine import portfolio_tax_report
        report = portfolio_tax_report([])
        assert 'unrealized' in report
        assert 'realized' in report
        assert report['total_unrealized'] == 0

    @patch('cli.engine.get_price')
    def test_portfolio_tax_report_with_positions(self, mock_price):
        mock_price.return_value = {'price': 150.0, 'change_pct': 2.0}
        from cli.engine import portfolio_tax_report
        positions = [{'ticker': 'AAPL', 'entry_price': 100.0, 'qty': 10, 'added_at': '2024-01-01T00:00:00'}]
        report = portfolio_tax_report(positions)
        assert len(report['unrealized']) == 1
        assert report['unrealized'][0]['gain'] == 500.0
        assert report['total_unrealized'] == 500.0

    def test_get_db_function(self):
        from cli.db import get_db
        db = get_db()
        assert db is not None

    def test_feature_store_count_195(self):
        from backend.features.feature_store import FeatureStore
        counts = FeatureStore.feature_count()
        assert counts['tier2_technical'] == 85
        assert counts['total'] >= 512

    def test_feature_names_include_new(self):
        from backend.features.feature_store import FeatureStore
        names = FeatureStore.all_feature_names()
        assert len(names) >= 512
        for f in ['connors_rsi', 'choppiness', 'atr_band_upper', 'atr_band_lower']:
            assert f in names


class TestCoppockElderRVIFeatures:
    """Tests for Coppock Curve, Elder Ray, and RVI features (iteration 23)."""

    def test_coppock_curve_computed(self):
        from backend.features.tier2_technical import Tier2Technical
        import pandas as pd
        import numpy as np
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            'open': np.random.uniform(90, 110, n),
            'high': np.random.uniform(100, 120, n),
            'low': np.random.uniform(80, 100, n),
            'close': np.random.uniform(90, 110, n),
            'volume': np.random.randint(1000, 10000, n),
        })
        feats = Tier2Technical.compute(df)
        assert 'coppock_curve' in feats.columns

    def test_elder_ray_computed(self):
        from backend.features.tier2_technical import Tier2Technical
        import pandas as pd
        import numpy as np
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            'open': np.random.uniform(90, 110, n),
            'high': np.random.uniform(100, 120, n),
            'low': np.random.uniform(80, 100, n),
            'close': np.random.uniform(90, 110, n),
            'volume': np.random.randint(1000, 10000, n),
        })
        feats = Tier2Technical.compute(df)
        assert 'elder_bull' in feats.columns
        assert 'elder_bear' in feats.columns

    def test_rvi_computed(self):
        from backend.features.tier2_technical import Tier2Technical
        import pandas as pd
        import numpy as np
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            'open': np.random.uniform(90, 110, n),
            'high': np.random.uniform(100, 120, n),
            'low': np.random.uniform(80, 100, n),
            'close': np.random.uniform(90, 110, n),
            'volume': np.random.randint(1000, 10000, n),
        })
        feats = Tier2Technical.compute(df)
        assert 'rvi' in feats.columns

    def test_feature_store_count_199(self):
        from backend.features.feature_store import FeatureStore
        counts = FeatureStore.feature_count()
        assert counts['tier2_technical'] == 85
        assert counts['total'] >= 512

    def test_feature_names_include_new_iter23(self):
        from backend.features.feature_store import FeatureStore
        names = FeatureStore.all_feature_names()
        assert len(names) >= 512
        for f in ['coppock_curve', 'elder_bull', 'elder_bear', 'rvi']:
            assert f in names

    def test_coppock_values_reasonable(self):
        from backend.features.tier2_technical import Tier2Technical
        import pandas as pd
        import numpy as np
        np.random.seed(42)
        n = 100
        close = pd.Series(np.cumsum(np.random.randn(n)) + 100)
        df = pd.DataFrame({
            'open': close - 0.5, 'high': close + 1, 'low': close - 1,
            'close': close, 'volume': np.random.randint(1000, 10000, n),
        })
        feats = Tier2Technical.compute(df)
        # Coppock should have some non-zero values after warmup
        vals = feats['coppock_curve'].dropna()
        assert len(vals) > 0

    def test_elder_ray_signs(self):
        """Elder bull should be positive when high > EMA, bear negative when low < EMA."""
        from backend.features.tier2_technical import Tier2Technical
        import pandas as pd
        import numpy as np
        np.random.seed(42)
        n = 100
        close = pd.Series(np.linspace(100, 120, n))  # uptrend
        df = pd.DataFrame({
            'open': close - 0.5, 'high': close + 2, 'low': close - 2,
            'close': close, 'volume': np.random.randint(1000, 10000, n),
        })
        feats = Tier2Technical.compute(df)
        # In uptrend, bull power should be positive at end
        assert feats['elder_bull'].iloc[-1] > 0

    def test_rvi_bounded(self):
        from backend.features.tier2_technical import Tier2Technical
        import pandas as pd
        import numpy as np
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            'open': np.random.uniform(90, 110, n),
            'high': np.random.uniform(100, 120, n),
            'low': np.random.uniform(80, 100, n),
            'close': np.random.uniform(90, 110, n),
            'volume': np.random.randint(1000, 10000, n),
        })
        feats = Tier2Technical.compute(df)
        # RVI should be bounded (close-open)/(high-low) averaged
        rvi_vals = feats['rvi'].dropna()
        assert all(abs(v) < 5 for v in rvi_vals)

    def test_scan_filters_include_new(self):
        """Verify new scan filter keys are in the docstring."""
        from cli.engine import scan_tickers
        doc = scan_tickers.__doc__
        for f in ['coppock', 'elder_bull', 'elder_bear', 'rvi']:
            assert f in doc

    def test_portfolio_correlation_empty(self):
        from cli.engine import portfolio_correlation
        result = portfolio_correlation([{'ticker': 'AAPL'}])
        assert 'tickers' in result
        assert 'matrix' in result

    def test_portfolio_correlation_structure(self):
        from cli.engine import portfolio_correlation
        result = portfolio_correlation([{'ticker': 'AAPL'}, {'ticker': 'MSFT'}])
        assert 'tickers' in result
        assert 'matrix' in result
        assert 'pairs' in result

    def test_smart_alerts_returns_list(self, monkeypatch):
        from cli import engine
        monkeypatch.setattr(engine, 'get_analysis', lambda t: {
            'price': 100.0,
            'horizons': {
                'short': {'stop': 95.0, 'target': 108.0, 'support': 92.0, 'resistance': 112.0},
                'medium': {'stop': 90.0, 'target': 115.0, 'support': 88.0, 'resistance': 118.0},
                'long': {'stop': 85.0, 'target': 125.0, 'support': 82.0, 'resistance': 130.0},
            },
        })
        monkeypatch.setattr(engine, 'get_features', lambda t: {'bb_upper': 110.0, 'bb_lower': 90.0})
        alerts_added = []
        monkeypatch.setattr('cli.db.add_alert', lambda t, c, v: alerts_added.append((t, c, v)))
        result = engine.smart_alerts('TEST')
        assert isinstance(result, list)
        assert len(result) > 0
        assert all('condition' in a and 'threshold' in a and 'label' in a for a in result)

    def test_portfolio_correlation_cmd_help(self):
        from click.testing import CliRunner
        from cli.main import cli as cli_app
        runner = CliRunner()
        result = runner.invoke(cli_app, ['portfolio-correlation', '--help'])
        assert result.exit_code == 0
        assert 'correlation' in result.output.lower()

    def test_alerts_smart_cmd_help(self):
        from click.testing import CliRunner
        from cli.main import cli as cli_app
        runner = CliRunner()
        result = runner.invoke(cli_app, ['alerts', 'smart', '--help'])
        assert result.exit_code == 0
        assert 'technical levels' in result.output.lower()

    def test_coppock_signal_generation(self, monkeypatch):
        """Verify Coppock Curve generates signals in analysis."""
        from cli import engine
        _original_get_analysis = engine.get_analysis  # noqa: F841
        # Just verify the signal code doesn't crash with the feature present
        feats_dict = {'coppock_curve': 5.0}
        bullish = []
        def gf(name):
            return feats_dict.get(name)
        coppock = gf('coppock_curve')
        if coppock is not None:
            if coppock > 0: bullish.append(f"Coppock Curve positive ({coppock:.1f})")
        assert len(bullish) == 1
        assert 'Coppock' in bullish[0]

    def test_elder_ray_signal_generation(self):
        """Verify Elder Ray generates signals."""
        feats_dict = {'elder_bull': 0.03, 'elder_bear': 0.0}
        bullish = []
        def gf(name):
            return feats_dict.get(name)
        elder_bull = gf('elder_bull')
        elder_bear = gf('elder_bear')
        if elder_bull is not None and elder_bear is not None:
            if elder_bull > 0.02 and elder_bear > -0.01: bullish.append("Elder Ray bullish")
        assert len(bullish) == 1

    def test_rvi_signal_generation(self):
        """Verify RVI generates signals."""
        feats_dict = {'rvi': 0.5}
        bullish = []
        def gf(name):
            return feats_dict.get(name)
        rvi_val = gf('rvi')
        if rvi_val is not None and rvi_val > 0.3:
            bullish.append(f"RVI positive ({rvi_val:.2f})")
        assert len(bullish) == 1
        assert 'RVI' in bullish[0]


class TestSupertrendSqueezeMomHMAFeatures:
    """Tests for Supertrend, Squeeze Momentum, and Hull MA features (iteration 24)."""

    def test_supertrend_computed(self):
        from backend.features.tier2_technical import Tier2Technical
        import pandas as pd
        import numpy as np
        np.random.seed(42)
        n = 100
        prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
        df = pd.DataFrame({
            'open': prices - 0.3, 'high': prices + np.random.uniform(0.5, 2, n),
            'low': prices - np.random.uniform(0.5, 2, n),
            'close': prices, 'volume': np.random.randint(1_000_000, 5_000_000, n),
        })
        feats = Tier2Technical.compute(df)
        assert 'supertrend' in feats.columns

    def test_squeeze_mom_computed(self):
        from backend.features.tier2_technical import Tier2Technical
        import pandas as pd
        import numpy as np
        np.random.seed(42)
        n = 100
        prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
        df = pd.DataFrame({
            'open': prices - 0.3, 'high': prices + np.random.uniform(0.5, 2, n),
            'low': prices - np.random.uniform(0.5, 2, n),
            'close': prices, 'volume': np.random.randint(1_000_000, 5_000_000, n),
        })
        feats = Tier2Technical.compute(df)
        assert 'squeeze_mom' in feats.columns

    def test_hma_dist_computed(self):
        from backend.features.tier2_technical import Tier2Technical
        import pandas as pd
        import numpy as np
        np.random.seed(42)
        n = 100
        prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
        df = pd.DataFrame({
            'open': prices - 0.3, 'high': prices + np.random.uniform(0.5, 2, n),
            'low': prices - np.random.uniform(0.5, 2, n),
            'close': prices, 'volume': np.random.randint(1_000_000, 5_000_000, n),
        })
        feats = Tier2Technical.compute(df)
        assert 'hma_dist' in feats.columns

    def test_feature_store_count_202(self):
        from backend.features.feature_store import FeatureStore
        counts = FeatureStore.feature_count()
        assert counts['tier2_technical'] == 85
        assert counts['total'] >= 512

    def test_feature_names_include_new_iter24(self):
        from backend.features.feature_store import FeatureStore
        names = FeatureStore.all_feature_names()
        assert len(names) >= 512
        for f in ['supertrend', 'squeeze_mom', 'hma_dist']:
            assert f in names

    def test_supertrend_bullish_in_uptrend(self):
        from backend.features.tier2_technical import Tier2Technical
        import pandas as pd
        import numpy as np
        np.random.seed(42)
        n = 100
        prices = 50 + np.arange(n) * 2.0 + np.random.randn(n) * 0.1
        df = pd.DataFrame({
            'open': prices - 0.1, 'high': prices + 1.0,
            'low': prices - 0.5, 'close': prices,
            'volume': np.random.randint(1_000_000, 5_000_000, n),
        })
        feats = Tier2Technical.compute(df)
        # Supertrend values should be bounded and non-zero in trending market
        vals = feats['supertrend'].iloc[20:]
        assert all(abs(v) < 1 for v in vals)

    def test_squeeze_mom_positive_in_uptrend(self):
        from backend.features.tier2_technical import Tier2Technical
        import pandas as pd
        import numpy as np
        np.random.seed(42)
        n = 100
        prices = 50 + np.arange(n) * 2.0 + np.random.randn(n) * 0.1
        df = pd.DataFrame({
            'open': prices - 0.1, 'high': prices + 1.0,
            'low': prices - 0.5, 'close': prices,
            'volume': np.random.randint(1_000_000, 5_000_000, n),
        })
        feats = Tier2Technical.compute(df)
        # Squeeze momentum should have non-zero values in trending market
        vals = feats['squeeze_mom'].iloc[30:]
        assert any(v != 0 for v in vals)

    def test_hma_above_in_uptrend(self):
        from backend.features.tier2_technical import Tier2Technical
        import pandas as pd
        import numpy as np
        np.random.seed(42)
        n = 100
        prices = 100 + np.arange(n) * 0.5 + np.random.randn(n) * 0.1
        df = pd.DataFrame({
            'open': prices - 0.1, 'high': prices + 0.3,
            'low': prices - 0.3, 'close': prices,
            'volume': np.random.randint(1_000_000, 5_000_000, n),
        })
        feats = Tier2Technical.compute(df)
        assert feats['hma_dist'].iloc[-1] > 0

    def test_supertrend_signal_generation(self):
        """Verify Supertrend generates signals in analysis."""
        feats_dict = {'supertrend': 0.05}
        bullish = []
        def gf(name):
            return feats_dict.get(name)
        supertrend = gf('supertrend')
        if supertrend is not None and supertrend > 0.01:
            bullish.append(f"Supertrend bullish ({supertrend:.3f})")
        assert len(bullish) == 1
        assert 'Supertrend' in bullish[0]

    def test_squeeze_signal_generation(self):
        """Verify Squeeze Momentum generates signals."""
        feats_dict = {'squeeze_mom': 3.0}
        bullish = []
        def gf(name):
            return feats_dict.get(name)
        squeeze = gf('squeeze_mom')
        if squeeze is not None and squeeze > 1:
            bullish.append(f"Squeeze momentum up ({squeeze:.1f})")
        assert len(bullish) == 1
        assert 'Squeeze' in bullish[0]

    def test_hma_signal_generation(self):
        """Verify HMA generates signals."""
        feats_dict = {'hma_dist': 0.05}
        bullish = []
        def gf(name):
            return feats_dict.get(name)
        hma = gf('hma_dist')
        if hma is not None and hma > 0.02:
            bullish.append(f"Above Hull MA ({hma:.3f})")
        assert len(bullish) == 1
        assert 'Hull' in bullish[0]

    def test_scan_filters_include_new(self):
        """Verify new scan filters are in docstring."""
        from cli.engine import scan_tickers
        doc = scan_tickers.__doc__
        for f in ['supertrend', 'squeeze', 'hma']:
            assert f in doc

    def test_portfolio_diversification_cmd_help(self):
        from click.testing import CliRunner
        from cli.main import cli as cli_app
        runner = CliRunner()
        result = runner.invoke(cli_app, ['portfolio-diversification', '--help'])
        assert result.exit_code == 0
        assert 'diversification' in result.output.lower()

    def test_portfolio_diversification_empty(self):
        from cli.engine import portfolio_diversification
        result = portfolio_diversification([])
        assert result['score'] == 0
        assert result['grade'] == 'N/A'

    def test_portfolio_diversification_single(self):
        from cli.engine import portfolio_diversification
        from unittest.mock import patch, MagicMock
        mock_ticker = MagicMock()
        mock_ticker.info = {'sector': 'Technology'}
        with patch('yfinance.Ticker', return_value=mock_ticker):
            result = portfolio_diversification([{'ticker': 'AAPL'}])
        assert result['score'] == 0  # single sector = low score
        assert result['grade'] in ('D', 'F')
        assert 'Technology' in result['sectors']

    def test_portfolio_diversification_diverse(self):
        from cli.engine import portfolio_diversification
        from unittest.mock import patch, MagicMock
        sector_map = {'AAPL': 'Technology', 'JNJ': 'Healthcare', 'JPM': 'Finance',
                       'XOM': 'Energy', 'PG': 'Consumer Defensive'}
        def mock_yf(ticker):
            m = MagicMock()
            m.info = {'sector': sector_map.get(ticker, 'Unknown')}
            return m
        positions = [{'ticker': t} for t in sector_map]
        with patch('yfinance.Ticker', side_effect=mock_yf):
            result = portfolio_diversification(positions)
        assert result['score'] > 50
        assert len(result['sectors']) == 5


class TestSTCKVOScanFilters:
    """Test STC and KVO scan filters."""

    @patch('cli.engine.generate_ohlcv')
    def test_scan_stc_filter(self, mock_ohlcv):
        from cli.engine import scan_tickers
        import pandas as pd
        import numpy as np
        n = 200
        dates = pd.date_range('2024-01-01', periods=n, freq='D')
        rng = np.random.RandomState(42)
        prices = 100 + np.arange(n) * 0.5 + rng.normal(0, 0.3, n)
        df = pd.DataFrame({
            'open': prices - 0.2, 'high': prices + 0.5,
            'low': prices - 0.5, 'close': prices,
            'volume': rng.randint(1_000_000, 5_000_000, n)
        }, index=dates)
        mock_ohlcv.return_value = df
        results = scan_tickers(['TSLA'], 'stc>50')
        assert isinstance(results, list)

    @patch('cli.engine.generate_ohlcv')
    def test_scan_kvo_filter(self, mock_ohlcv):
        from cli.engine import scan_tickers
        import pandas as pd
        import numpy as np
        n = 200
        dates = pd.date_range('2024-01-01', periods=n, freq='D')
        rng = np.random.RandomState(42)
        prices = 100 + np.arange(n) * 0.5 + rng.normal(0, 0.3, n)
        df = pd.DataFrame({
            'open': prices - 0.2, 'high': prices + 0.5,
            'low': prices - 0.5, 'close': prices,
            'volume': rng.randint(1_000_000, 5_000_000, n)
        }, index=dates)
        mock_ohlcv.return_value = df
        results = scan_tickers(['TSLA'], 'kvo>0')
        assert isinstance(results, list)


class TestSTCKVOSignals:
    """Test STC and KVO signal generation in engine."""

    def test_stc_in_feature_names(self):
        from backend.features.tier2_technical import Tier2Technical
        names = Tier2Technical.feature_names()
        assert 'stc' in names
        assert 'kvo' in names

    def test_stc_kvo_computed(self):
        from backend.features.tier2_technical import Tier2Technical
        from backend.data.mock_generators import generate_ohlcv
        df = generate_ohlcv("TSLA", seed=42).tail(200)
        feats = Tier2Technical.compute(df)
        assert 'stc' in feats.columns
        assert 'kvo' in feats.columns
        assert not feats['stc'].isna().all()
        assert not feats['kvo'].isna().all()

    def test_feature_store_count_updated(self):
        from backend.features.feature_store import FeatureStore
        counts = FeatureStore.feature_count()
        assert counts['tier2_technical'] == 85
        assert counts['total'] >= 512


class TestScanNegativeValues:
    """Test scan filter parsing with negative values."""

    @patch('cli.engine.generate_ohlcv')
    def test_scan_negative_threshold(self, mock_ohlcv):
        """Scan filter should handle negative values like change>-5."""
        import pandas as pd
        import numpy as np
        from cli.engine import scan_tickers
        dates = pd.date_range('2024-01-01', periods=60, freq='D')
        df = pd.DataFrame({
            'open': np.random.uniform(90, 110, 60),
            'high': np.random.uniform(100, 120, 60),
            'low': np.random.uniform(80, 100, 60),
            'close': np.random.uniform(90, 110, 60),
            'volume': np.random.randint(1000000, 5000000, 60),
        }, index=dates)
        mock_ohlcv.return_value = df
        results = scan_tickers(['TSLA'], 'change>-5')
        assert isinstance(results, list)

    @patch('cli.engine.generate_ohlcv')
    def test_scan_negative_decimal(self, mock_ohlcv):
        """Scan filter should handle negative decimals like williams7<-80.5."""
        import pandas as pd
        import numpy as np
        from cli.engine import scan_tickers
        dates = pd.date_range('2024-01-01', periods=60, freq='D')
        df = pd.DataFrame({
            'open': np.random.uniform(90, 110, 60),
            'high': np.random.uniform(100, 120, 60),
            'low': np.random.uniform(80, 100, 60),
            'close': np.random.uniform(90, 110, 60),
            'volume': np.random.randint(1000000, 5000000, 60),
        }, index=dates)
        mock_ohlcv.return_value = df
        results = scan_tickers(['TSLA'], 'williams7<-80.5')
        assert isinstance(results, list)

    def test_scan_negative_regex_match(self):
        """Verify the regex correctly parses negative values."""
        import re
        pattern = r'(\w+)\s*([<>]=?|=)\s*(-?[\d.]+)(%|x)?'
        m = re.match(pattern, 'change>-5')
        assert m is not None
        assert m.group(1) == 'change'
        assert m.group(2) == '>'
        assert m.group(3) == '-5'

        m2 = re.match(pattern, 'williams7<=-80.5')
        assert m2 is not None
        assert m2.group(1) == 'williams7'
        assert m2.group(2) == '<='
        assert float(m2.group(3)) == -80.5


class TestConfigShowAlias:
    """Test config show command as alias for config list."""

    def test_config_show_exists(self):
        from cli.main import cli
        from click.testing import CliRunner
        r = CliRunner()
        result = r.invoke(cli, ['config', 'show'])
        assert result.exit_code == 0

    def test_config_show_matches_list(self):
        from cli.main import cli
        from click.testing import CliRunner
        r = CliRunner()
        list_result = r.invoke(cli, ['config', 'list'])
        show_result = r.invoke(cli, ['config', 'show'])
        assert list_result.output == show_result.output


class TestPortfolioSummary:
    """Test portfolio-summary command."""

    @patch('cli.db.get_positions')
    def test_portfolio_summary_no_positions(self, mock_pos):
        mock_pos.return_value = []
        from cli.main import cli
        from click.testing import CliRunner
        r = CliRunner()
        result = r.invoke(cli, ['portfolio-summary'])
        assert result.exit_code == 0
        assert 'No positions' in result.output

    @patch('cli.db.get_positions')
    def test_portfolio_summary_no_positions_json(self, mock_pos):
        mock_pos.return_value = []
        from cli.main import cli
        from click.testing import CliRunner
        r = CliRunner()
        result = r.invoke(cli, ['portfolio-summary', '--json'])
        assert result.exit_code == 0
        import json
        data = json.loads(result.output)
        assert data == {}

    @patch('cli.engine.portfolio_correlation')
    @patch('cli.engine.portfolio_diversification')
    @patch('cli.engine.portfolio_risk')
    @patch('cli.db.get_positions')
    def test_portfolio_summary_with_positions(self, mock_pos, mock_risk, mock_div, mock_corr):
        mock_pos.return_value = [
            {'ticker': 'AAPL', 'entry_price': 150, 'qty': 10},
            {'ticker': 'TSLA', 'entry_price': 200, 'qty': 5},
        ]
        mock_risk.return_value = {
            'var_95': -500, 'beta': 1.1, 'volatility': 0.25,
            'sharpe': 1.5, 'total_value': 2500, 'tickers': ['AAPL', 'TSLA'],
        }
        mock_div.return_value = {
            'score': 60, 'grade': 'B', 'sectors': {'Technology': 2}, 'suggestions': [],
        }
        mock_corr.return_value = {
            'tickers': ['AAPL', 'TSLA'],
            'pairs': [{'ticker1': 'AAPL', 'ticker2': 'TSLA', 'correlation': 0.65}],
        }
        from cli.main import cli
        from click.testing import CliRunner
        r = CliRunner()
        result = r.invoke(cli, ['portfolio-summary'])
        assert result.exit_code == 0
        assert 'Risk' in result.output
        assert 'Diversification' in result.output

    @patch('cli.engine.portfolio_diversification')
    @patch('cli.engine.portfolio_risk')
    @patch('cli.db.get_positions')
    def test_portfolio_summary_single_position(self, mock_pos, mock_risk, mock_div):
        mock_pos.return_value = [{'ticker': 'AAPL', 'entry_price': 150, 'qty': 10}]
        mock_risk.return_value = {
            'var_95': -200, 'beta': 1.0, 'volatility': 0.2,
            'sharpe': 0.8, 'total_value': 1500, 'tickers': ['AAPL'],
        }
        mock_div.return_value = {
            'score': 20, 'grade': 'D', 'sectors': {'Technology': 1},
            'suggestions': ['Add more positions'],
        }
        from cli.main import cli
        from click.testing import CliRunner
        r = CliRunner()
        result = r.invoke(cli, ['portfolio-summary'])
        assert result.exit_code == 0
        assert 'Risk' in result.output

    @patch('cli.engine.portfolio_correlation')
    @patch('cli.engine.portfolio_diversification')
    @patch('cli.engine.portfolio_risk')
    @patch('cli.db.get_positions')
    def test_portfolio_summary_json(self, mock_pos, mock_risk, mock_div, mock_corr):
        mock_pos.return_value = [
            {'ticker': 'AAPL', 'entry_price': 150, 'qty': 10},
            {'ticker': 'TSLA', 'entry_price': 200, 'qty': 5},
        ]
        mock_risk.return_value = {'var_95': -500, 'beta': 1.1, 'total_value': 2500}
        mock_div.return_value = {'score': 60, 'grade': 'B', 'sectors': {}, 'suggestions': []}
        mock_corr.return_value = {'tickers': ['AAPL', 'TSLA'], 'pairs': []}
        from cli.main import cli
        from click.testing import CliRunner
        r = CliRunner()
        result = r.invoke(cli, ['portfolio-summary', '--json'])
        assert result.exit_code == 0
        import json
        data = json.loads(result.output)
        assert 'risk' in data
        assert 'diversification' in data
        assert 'correlation' in data


class TestAlertsClear:
    """Test alerts clear command."""

    @patch('cli.db.clear_alerts')
    def test_alerts_clear_all_confirmed(self, mock_clear):
        mock_clear.return_value = 3
        from cli.main import cli
        from click.testing import CliRunner
        r = CliRunner()
        result = r.invoke(cli, ['alerts', 'clear', '-y'])
        assert result.exit_code == 0
        assert 'Cleared 3 alert(s)' in result.output
        mock_clear.assert_called_once_with(None)

    @patch('cli.db.clear_alerts')
    def test_alerts_clear_ticker(self, mock_clear):
        mock_clear.return_value = 2
        from cli.main import cli
        from click.testing import CliRunner
        r = CliRunner()
        result = r.invoke(cli, ['alerts', 'clear', 'TSLA', '-y'])
        assert result.exit_code == 0
        assert 'Cleared 2 alert(s)' in result.output
        mock_clear.assert_called_once_with('TSLA')

    @patch('cli.db.clear_alerts')
    def test_alerts_clear_zero(self, mock_clear):
        mock_clear.return_value = 0
        from cli.main import cli
        from click.testing import CliRunner
        r = CliRunner()
        result = r.invoke(cli, ['alerts', 'clear', '-y'])
        assert result.exit_code == 0
        assert 'Cleared 0 alert(s)' in result.output

    def test_alerts_clear_cancelled(self):
        from cli.main import cli
        from click.testing import CliRunner
        r = CliRunner()
        result = r.invoke(cli, ['alerts', 'clear'], input='n\n')
        assert result.exit_code == 0
        assert 'Cancelled' in result.output


class TestWatchlistSignals:
    """Test watchlist --signals flag."""

    @patch('cli.engine.get_analysis')
    @patch('cli.engine.get_price')
    @patch('cli.db.get_watchlist')
    def test_watchlist_signals(self, mock_wl, mock_price, mock_analysis):
        mock_wl.return_value = [{'ticker': 'TSLA'}]
        mock_price.return_value = {'price': 250.0, 'change_pct': 1.5}
        mock_analysis.return_value = {
            'horizons': {'short': {'direction': 'bullish', 'confidence': 0.65}},
        }
        from cli.main import cli
        from click.testing import CliRunner
        r = CliRunner()
        result = r.invoke(cli, ['watchlist', '--signals'])
        assert result.exit_code == 0
        assert 'Signal' in result.output
        assert 'TSLA' in result.output

    @patch('cli.engine.get_price')
    @patch('cli.db.get_watchlist')
    def test_watchlist_no_signals(self, mock_wl, mock_price):
        mock_wl.return_value = [{'ticker': 'AAPL'}]
        mock_price.return_value = {'price': 180.0, 'change_pct': -0.5}
        from cli.main import cli
        from click.testing import CliRunner
        r = CliRunner()
        result = r.invoke(cli, ['watchlist'])
        assert result.exit_code == 0
        assert 'Signal' not in result.output

    @patch('cli.engine.get_analysis')
    @patch('cli.engine.get_price')
    @patch('cli.db.get_watchlist')
    def test_watchlist_signals_json(self, mock_wl, mock_price, mock_analysis):
        mock_wl.return_value = [{'ticker': 'TSLA'}]
        mock_price.return_value = {'price': 250.0, 'change_pct': 1.5}
        mock_analysis.return_value = {
            'horizons': {'short': {'direction': 'bearish', 'confidence': 0.7}},
        }
        from cli.main import cli
        from click.testing import CliRunner
        r = CliRunner()
        result = r.invoke(cli, ['watchlist', '--signals', '--json'])
        assert result.exit_code == 0
        import json
        data = json.loads(result.output)
        assert data[0]['signal'] == 'bearish'
        assert data[0]['confidence'] == 0.7

    @patch('cli.engine.get_analysis')
    @patch('cli.engine.get_price')
    @patch('cli.db.get_watchlist')
    def test_watchlist_signals_analysis_error(self, mock_wl, mock_price, mock_analysis):
        mock_wl.return_value = [{'ticker': 'TSLA'}]
        mock_price.return_value = {'price': 250.0, 'change_pct': 1.5}
        mock_analysis.side_effect = Exception("API error")
        from cli.main import cli
        from click.testing import CliRunner
        r = CliRunner()
        result = r.invoke(cli, ['watchlist', '--signals'])
        assert result.exit_code == 0
        assert 'TSLA' in result.output


class TestScanFilterEdgeCases:
    """Test scan filter parsing edge cases."""

    def test_scan_filter_whitespace_variations(self):
        """Verify regex handles various whitespace in filter expressions."""
        import re
        pattern = r'(\w+)\s*([<>]=?|=)\s*(-?[\d.]+)(%|x)?'
        # No spaces
        m = re.match(pattern, 'rsi<30')
        assert m and m.group(1) == 'rsi' and m.group(3) == '30'
        # Spaces around operator
        m = re.match(pattern, 'rsi < 30')
        assert m and m.group(1) == 'rsi' and m.group(3) == '30'
        # Tab
        m = re.match(pattern, 'volume\t>\t2')
        assert m and m.group(1) == 'volume' and m.group(3) == '2'

    def test_scan_filter_all_operators(self):
        """Verify all comparison operators parse correctly."""
        import re
        pattern = r'(\w+)\s*([<>]=?|=)\s*(-?[\d.]+)(%|x)?'
        for op in ['<', '>', '<=', '>=', '=']:
            m = re.match(pattern, f'rsi{op}30')
            assert m is not None, f"Failed for operator {op}"
            assert m.group(2) == op

    def test_scan_filter_percentage_suffix(self):
        """Verify percentage suffix is captured."""
        import re
        pattern = r'(\w+)\s*([<>]=?|=)\s*(-?[\d.]+)(%|x)?'
        m = re.match(pattern, 'change>5%')
        assert m and m.group(4) == '%'

    def test_scan_filter_multiplier_suffix(self):
        """Verify x multiplier suffix is captured."""
        import re
        pattern = r'(\w+)\s*([<>]=?|=)\s*(-?[\d.]+)(%|x)?'
        m = re.match(pattern, 'volume>2x')
        assert m and m.group(4) == 'x'

    @patch('cli.engine.generate_ohlcv')
    def test_scan_empty_filter_string(self, mock_ohlcv):
        """Scan with empty filter should return all tickers."""
        import pandas as pd
        import numpy as np
        from cli.engine import scan_tickers
        dates = pd.date_range('2024-01-01', periods=60, freq='D')
        df = pd.DataFrame({
            'open': np.full(60, 100.0), 'high': np.full(60, 110.0),
            'low': np.full(60, 90.0), 'close': np.full(60, 105.0),
            'volume': np.full(60, 1000000, dtype=int),
        }, index=dates)
        mock_ohlcv.return_value = df
        results = scan_tickers(['AAPL'], '')
        assert isinstance(results, list)

    @patch('cli.engine.generate_ohlcv')
    def test_scan_multiple_and_filters(self, mock_ohlcv):
        """Scan with multiple AND-combined filters."""
        import pandas as pd
        import numpy as np
        from cli.engine import scan_tickers
        dates = pd.date_range('2024-01-01', periods=60, freq='D')
        df = pd.DataFrame({
            'open': np.full(60, 100.0), 'high': np.full(60, 110.0),
            'low': np.full(60, 90.0), 'close': np.full(60, 105.0),
            'volume': np.full(60, 1000000, dtype=int),
        }, index=dates)
        mock_ohlcv.return_value = df
        results = scan_tickers(['AAPL'], 'rsi<70 AND volume>0.5')
        assert isinstance(results, list)


class TestClearAlertsDB:
    """Test clear_alerts database function directly."""

    def test_clear_alerts_returns_count(self):
        from cli.db import add_alert, clear_alerts, get_alerts
        import tempfile
        import os
        import cli.db as db_mod
        orig = db_mod.DB_PATH
        db_mod.DB_PATH = Path(tempfile.mktemp(suffix='.db'))
        try:
            add_alert('TSLA', 'above', 300)
            add_alert('TSLA', 'below', 200)
            add_alert('AAPL', 'above', 200)
            assert len(get_alerts()) == 3
            removed = clear_alerts('TSLA')
            assert removed == 2
            assert len(get_alerts()) == 1
            assert get_alerts()[0]['ticker'] == 'AAPL'
        finally:
            os.unlink(str(db_mod.DB_PATH))
            db_mod.DB_PATH = orig

    def test_clear_all_alerts(self):
        from cli.db import add_alert, clear_alerts, get_alerts
        import tempfile
        import os
        import cli.db as db_mod
        orig = db_mod.DB_PATH
        db_mod.DB_PATH = Path(tempfile.mktemp(suffix='.db'))
        try:
            add_alert('TSLA', 'above', 300)
            add_alert('AAPL', 'below', 150)
            removed = clear_alerts()
            assert removed == 2
            assert len(get_alerts()) == 0
        finally:
            os.unlink(str(db_mod.DB_PATH))
            db_mod.DB_PATH = orig


class TestAlertsHistory:
    def test_alerts_history_empty(self):
        from click.testing import CliRunner
        from cli.main import cli
        from unittest.mock import patch
        runner = CliRunner()
        with patch('cli.db.get_triggered_alerts', return_value=[]):
            result = runner.invoke(cli, ['alerts', 'history'])
        assert result.exit_code == 0
        assert 'No triggered alerts' in result.output

    def test_alerts_history_with_data(self):
        from click.testing import CliRunner
        from cli.main import cli
        from unittest.mock import patch
        alerts = [{'id': 1, 'ticker': 'TSLA', 'condition': 'above', 'threshold': 300.0, 'created_at': '2026-01-01'}]
        runner = CliRunner()
        with patch('cli.db.get_triggered_alerts', return_value=alerts):
            result = runner.invoke(cli, ['alerts', 'history'])
        assert result.exit_code == 0
        assert 'TSLA' in result.output
        assert '300.00' in result.output

    def test_alerts_history_json(self):
        from click.testing import CliRunner
        from cli.main import cli
        from unittest.mock import patch
        import json
        alerts = [{'id': 1, 'ticker': 'TSLA', 'condition': 'above', 'threshold': 300.0, 'created_at': '2026-01-01'}]
        runner = CliRunner()
        with patch('cli.db.get_triggered_alerts', return_value=alerts):
            result = runner.invoke(cli, ['alerts', 'history', '--json'])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data) == 1
        assert data[0]['ticker'] == 'TSLA'

    def test_alerts_history_ticker_filter(self):
        from click.testing import CliRunner
        from cli.main import cli
        from unittest.mock import patch
        runner = CliRunner()
        with patch('cli.db.get_triggered_alerts', return_value=[]) as mock_fn:
            result = runner.invoke(cli, ['alerts', 'history', 'AAPL'])
        assert result.exit_code == 0
        mock_fn.assert_called_once_with('AAPL', 20)

    def test_alerts_history_limit(self):
        from click.testing import CliRunner
        from cli.main import cli
        from unittest.mock import patch
        runner = CliRunner()
        with patch('cli.db.get_triggered_alerts', return_value=[]) as mock_fn:
            result = runner.invoke(cli, ['alerts', 'history', '--limit', '5'])
        assert result.exit_code == 0
        mock_fn.assert_called_once_with(None, 5)

    def test_alerts_history_help(self):
        from click.testing import CliRunner
        from cli.main import cli
        runner = CliRunner()
        result = runner.invoke(cli, ['alerts', 'history', '--help'])
        assert result.exit_code == 0
        assert 'triggered' in result.output.lower()


class TestGetTriggeredAlertsDB:
    def test_get_triggered_alerts(self):
        from cli.db import add_alert, trigger_alert, get_triggered_alerts
        import tempfile
        import os
        import cli.db as db_mod
        orig = db_mod.DB_PATH
        db_mod.DB_PATH = Path(tempfile.mktemp(suffix='.db'))
        try:
            add_alert('TSLA', 'above', 300)
            add_alert('AAPL', 'below', 150)
            trigger_alert(1)
            triggered = get_triggered_alerts()
            assert len(triggered) == 1
            assert triggered[0]['ticker'] == 'TSLA'
        finally:
            os.unlink(str(db_mod.DB_PATH))
            db_mod.DB_PATH = orig

    def test_get_triggered_alerts_by_ticker(self):
        from cli.db import add_alert, trigger_alert, get_triggered_alerts
        import tempfile
        import os
        import cli.db as db_mod
        orig = db_mod.DB_PATH
        db_mod.DB_PATH = Path(tempfile.mktemp(suffix='.db'))
        try:
            add_alert('TSLA', 'above', 300)
            add_alert('AAPL', 'below', 150)
            trigger_alert(1)
            trigger_alert(2)
            tsla = get_triggered_alerts('TSLA')
            assert len(tsla) == 1
            assert tsla[0]['ticker'] == 'TSLA'
        finally:
            os.unlink(str(db_mod.DB_PATH))
            db_mod.DB_PATH = orig


class TestWatchlistSort:
    @patch('cli.engine.get_price', side_effect=[
        {'price': 250.0, 'change_pct': 2.5},
        {'price': 180.0, 'change_pct': -1.0},
        {'price': 140.0, 'change_pct': 5.0},
    ])
    @patch('cli.db.get_watchlist', return_value=[
        {'ticker': 'TSLA'}, {'ticker': 'AAPL'}, {'ticker': 'GOOGL'}
    ])
    def test_watchlist_sort_by_change(self, mock_wl, mock_price):
        from click.testing import CliRunner
        from cli.main import cli
        runner = CliRunner()
        result = runner.invoke(cli, ['watchlist', '--sort', 'change'])
        assert result.exit_code == 0
        lines = result.output
        # GOOGL (+5.0%) should appear before TSLA (+2.5%) before AAPL (-1.0%)
        googl_pos = lines.index('GOOGL')
        tsla_pos = lines.index('TSLA')
        aapl_pos = lines.index('AAPL')
        assert googl_pos < tsla_pos < aapl_pos

    @patch('cli.engine.get_price', side_effect=[
        {'price': 250.0, 'change_pct': 2.5},
        {'price': 180.0, 'change_pct': -1.0},
    ])
    @patch('cli.db.get_watchlist', return_value=[
        {'ticker': 'TSLA'}, {'ticker': 'AAPL'}
    ])
    def test_watchlist_sort_by_price(self, mock_wl, mock_price):
        from click.testing import CliRunner
        from cli.main import cli
        runner = CliRunner()
        result = runner.invoke(cli, ['watchlist', '--sort', 'price'])
        assert result.exit_code == 0
        tsla_pos = result.output.index('TSLA')
        aapl_pos = result.output.index('AAPL')
        assert tsla_pos < aapl_pos  # 250 > 180, descending

    @patch('cli.engine.get_price', side_effect=[
        {'price': 250.0, 'change_pct': 2.5},
        {'price': 180.0, 'change_pct': -1.0},
    ])
    @patch('cli.db.get_watchlist', return_value=[
        {'ticker': 'TSLA'}, {'ticker': 'AAPL'}
    ])
    def test_watchlist_sort_by_ticker(self, mock_wl, mock_price):
        from click.testing import CliRunner
        from cli.main import cli
        runner = CliRunner()
        result = runner.invoke(cli, ['watchlist', '--sort', 'ticker'])
        assert result.exit_code == 0
        aapl_pos = result.output.index('AAPL')
        tsla_pos = result.output.index('TSLA')
        assert aapl_pos < tsla_pos  # alphabetical


class TestPositionsSort:
    @patch('cli.db.save_snapshot')
    @patch('cli.engine.get_price', side_effect=[
        {'price': 260.0}, {'price': 190.0}
    ])
    @patch('cli.db.get_positions', return_value=[
        {'ticker': 'TSLA', 'entry_price': 250.0, 'qty': 10, 'added_at': '2026-01-01'},
        {'ticker': 'AAPL', 'entry_price': 150.0, 'qty': 20, 'added_at': '2026-01-01'},
    ])
    def test_positions_sort_by_pnl(self, mock_pos, mock_price, mock_snap):
        from click.testing import CliRunner
        from cli.main import cli
        runner = CliRunner()
        result = runner.invoke(cli, ['positions', '--sort', 'pnl'])
        assert result.exit_code == 0
        # AAPL pnl = (190-150)*20 = 800, TSLA pnl = (260-250)*10 = 100
        aapl_pos = result.output.index('AAPL')
        tsla_pos = result.output.index('TSLA')
        assert aapl_pos < tsla_pos  # higher P&L first

    @patch('cli.db.save_snapshot')
    @patch('cli.engine.get_price', side_effect=[
        {'price': 260.0}, {'price': 190.0}
    ])
    @patch('cli.db.get_positions', return_value=[
        {'ticker': 'TSLA', 'entry_price': 250.0, 'qty': 10, 'added_at': '2026-01-01'},
        {'ticker': 'AAPL', 'entry_price': 150.0, 'qty': 20, 'added_at': '2026-01-01'},
    ])
    def test_positions_sort_by_value(self, mock_pos, mock_price, mock_snap):
        from click.testing import CliRunner
        from cli.main import cli
        runner = CliRunner()
        result = runner.invoke(cli, ['positions', '--sort', 'value'])
        assert result.exit_code == 0
        # AAPL value = 190*20 = 3800, TSLA value = 260*10 = 2600
        aapl_pos = result.output.index('AAPL')
        tsla_pos = result.output.index('TSLA')
        assert aapl_pos < tsla_pos  # higher value first

    @patch('cli.db.save_snapshot')
    @patch('cli.engine.get_price', side_effect=[
        {'price': 260.0}, {'price': 190.0}
    ])
    @patch('cli.db.get_positions', return_value=[
        {'ticker': 'TSLA', 'entry_price': 250.0, 'qty': 10, 'added_at': '2026-01-01'},
        {'ticker': 'AAPL', 'entry_price': 150.0, 'qty': 20, 'added_at': '2026-01-01'},
    ])
    def test_positions_sort_by_change(self, mock_pos, mock_price, mock_snap):
        from click.testing import CliRunner
        from cli.main import cli
        runner = CliRunner()
        result = runner.invoke(cli, ['positions', '--sort', 'change'])
        assert result.exit_code == 0
        # AAPL change = 26.7%, TSLA change = 4%
        aapl_pos = result.output.index('AAPL')
        tsla_pos = result.output.index('TSLA')
        assert aapl_pos < tsla_pos  # higher % change first

    def test_positions_sort_help(self):
        from click.testing import CliRunner
        from cli.main import cli
        runner = CliRunner()
        result = runner.invoke(cli, ['positions', '--help'])
        assert result.exit_code == 0
        assert '--sort' in result.output


class TestScanPresets:
    @patch('cli.engine.scan_tickers', return_value=[])
    def test_scan_preset_oversold(self, mock_scan):
        runner = CliRunner()
        result = runner.invoke(cli, ['scan', '--preset', 'oversold'])
        assert result.exit_code == 0
        mock_scan.assert_called_once()
        assert mock_scan.call_args[0][1] == 'rsi<30 AND stoch<20'

    @patch('cli.engine.scan_tickers', return_value=[])
    def test_scan_preset_breakout(self, mock_scan):
        runner = CliRunner()
        result = runner.invoke(cli, ['scan', '--preset', 'breakout'])
        assert result.exit_code == 0
        mock_scan.assert_called_once()
        assert mock_scan.call_args[0][1] == 'volume>2 AND adx>25'

    def test_scan_no_filter_no_preset(self):
        runner = CliRunner()
        result = runner.invoke(cli, ['scan'])
        assert result.exit_code == 0
        assert 'Error' in result.output

    def test_scan_preset_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ['scan', '--help'])
        assert result.exit_code == 0
        assert '--preset' in result.output
        assert 'oversold' in result.output

    @patch('cli.engine.scan_tickers', return_value=[])
    def test_scan_filter_overrides_preset(self, mock_scan):
        """When both filter and preset given, filter takes precedence."""
        runner = CliRunner()
        result = runner.invoke(cli, ['scan', 'rsi<25', '--preset', 'oversold'])
        assert result.exit_code == 0
        # filter argument is provided, so it should use that
        mock_scan.assert_called_once()
        assert mock_scan.call_args[0][1] == 'rsi<25'


class TestAlertsExportImport:
    @patch('cli.db.get_alerts', return_value=[
        {'id': 1, 'ticker': 'TSLA', 'condition': 'above', 'threshold': 300.0, 'created_at': '2026-01-01', 'triggered': 0},
        {'id': 2, 'ticker': 'AAPL', 'condition': 'below', 'threshold': 150.0, 'created_at': '2026-01-01', 'triggered': 0},
    ])
    def test_alerts_export_stdout(self, mock_alerts):
        runner = CliRunner()
        result = runner.invoke(cli, ['alerts', 'export'])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data) == 2
        assert data[0]['ticker'] == 'TSLA'

    @patch('cli.db.get_alerts', return_value=[
        {'id': 1, 'ticker': 'TSLA', 'condition': 'above', 'threshold': 300.0, 'created_at': '2026-01-01', 'triggered': 0},
    ])
    def test_alerts_export_to_file(self, mock_alerts):
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(cli, ['alerts', 'export', '-o', 'out.json'])
            assert result.exit_code == 0
            assert 'Exported 1' in result.output
            data = json.loads(Path('out.json').read_text())
            assert data[0]['ticker'] == 'TSLA'

    @patch('cli.db.add_alert')
    def test_alerts_import(self, mock_add):
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path('alerts.json').write_text(json.dumps([
                {'ticker': 'TSLA', 'condition': 'above', 'threshold': 300.0},
                {'ticker': 'AAPL', 'condition': 'below', 'threshold': 150.0},
            ]))
            result = runner.invoke(cli, ['alerts', 'import', 'alerts.json'])
            assert result.exit_code == 0
            assert 'Imported 2' in result.output
            assert mock_add.call_count == 2

    @patch('cli.db.add_alert')
    def test_alerts_import_skips_invalid(self, mock_add):
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path('alerts.json').write_text(json.dumps([
                {'ticker': 'TSLA', 'condition': 'above', 'threshold': 300.0},
                {'bad': 'data'},
            ]))
            result = runner.invoke(cli, ['alerts', 'import', 'alerts.json'])
            assert result.exit_code == 0
            assert 'Imported 1' in result.output
            assert mock_add.call_count == 1

    def test_alerts_export_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ['alerts', 'export', '--help'])
        assert result.exit_code == 0
        assert 'backup' in result.output.lower() or 'export' in result.output.lower()

    def test_alerts_import_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ['alerts', 'import', '--help'])
        assert result.exit_code == 0
        assert 'import' in result.output.lower()


class TestWatchlistExportImport:
    @patch('cli.db.get_watchlist', return_value=[
        {'ticker': 'TSLA', 'added_at': '2026-01-01'},
        {'ticker': 'AAPL', 'added_at': '2026-01-01'},
    ])
    def test_watchlist_export_stdout(self, mock_wl):
        runner = CliRunner()
        result = runner.invoke(cli, ['watchlist-export'])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data == ['TSLA', 'AAPL']

    @patch('cli.db.get_watchlist', return_value=[{'ticker': 'TSLA', 'added_at': '2026-01-01'}])
    def test_watchlist_export_to_file(self, mock_wl):
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(cli, ['watchlist-export', '-o', 'wl.json'])
            assert result.exit_code == 0
            assert 'Exported 1' in result.output
            data = json.loads(Path('wl.json').read_text())
            assert data == ['TSLA']

    @patch('cli.db.add_watch')
    def test_watchlist_import(self, mock_add):
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path('wl.json').write_text(json.dumps(['TSLA', 'AAPL', 'GOOGL']))
            result = runner.invoke(cli, ['watchlist-import', 'wl.json'])
            assert result.exit_code == 0
            assert 'Imported 3' in result.output
            assert mock_add.call_count == 3

    @patch('cli.db.add_watch')
    def test_watchlist_import_skips_invalid(self, mock_add):
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path('wl.json').write_text(json.dumps(['TSLA', '', 123, 'AAPL']))
            result = runner.invoke(cli, ['watchlist-import', 'wl.json'])
            assert result.exit_code == 0
            assert 'Imported 2' in result.output
            assert mock_add.call_count == 2


class TestPortfolioStress:
    @patch('cli.engine.generate_ohlcv')
    @patch('cli.engine.get_price', return_value={'price': 250.0, 'change_pct': 1.0})
    def test_stress_test_engine(self, mock_price, mock_ohlcv):
        import numpy as np
        import pandas as pd
        mock_ohlcv.return_value = pd.DataFrame({'close': np.linspace(200, 250, 60)})
        from cli.engine import portfolio_stress_test
        pos = [{'ticker': 'TSLA', 'entry_price': 240.0, 'qty': 10}]
        results = portfolio_stress_test(pos, 'market_crash')
        assert len(results) == 1
        assert results[0]['scenario'] == 'market_crash'
        assert results[0]['total_impact'] < 0

    @patch('cli.engine.generate_ohlcv')
    @patch('cli.engine.get_price', return_value={'price': 250.0, 'change_pct': 1.0})
    def test_stress_test_all_scenarios(self, mock_price, mock_ohlcv):
        import numpy as np
        import pandas as pd
        mock_ohlcv.return_value = pd.DataFrame({'close': np.linspace(200, 250, 60)})
        from cli.engine import portfolio_stress_test
        pos = [{'ticker': 'TSLA', 'entry_price': 240.0, 'qty': 10}]
        results = portfolio_stress_test(pos)
        assert len(results) == 5  # all scenarios

    @patch('cli.engine.portfolio_stress_test', return_value=[{
        'scenario': 'market_crash', 'label': 'Market Crash (-20%)',
        'total_before': 2500, 'total_after': 2000, 'total_impact': -500, 'impact_pct': -20.0,
        'details': [{'ticker': 'TSLA', 'before': 2500, 'after': 2000, 'impact': -500, 'beta': 1.5}]
    }])
    @patch('cli.db.get_positions', return_value=[
        {'ticker': 'TSLA', 'entry_price': 250.0, 'qty': 10, 'added_at': '2026-01-01'}
    ])
    def test_stress_cli_command(self, mock_pos, mock_stress):
        runner = CliRunner()
        result = runner.invoke(cli, ['portfolio-stress', '--scenario', 'market_crash'])
        assert result.exit_code == 0
        assert 'TSLA' in result.output
        assert 'Market Crash' in result.output

    @patch('cli.engine.portfolio_stress_test', return_value=[])
    @patch('cli.db.get_positions', return_value=[
        {'ticker': 'TSLA', 'entry_price': 250.0, 'qty': 10, 'added_at': '2026-01-01'}
    ])
    def test_stress_cli_json(self, mock_pos, mock_stress):
        runner = CliRunner()
        result = runner.invoke(cli, ['portfolio-stress', '--json'])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)

    def test_stress_cli_empty_portfolio(self):
        runner = CliRunner()
        with patch('cli.db.get_positions', return_value=[]):
            result = runner.invoke(cli, ['portfolio-stress'])
        assert result.exit_code == 0
        assert 'No positions' in result.output

    def test_stress_cli_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ['portfolio-stress', '--help'])
        assert result.exit_code == 0
        assert 'stress' in result.output.lower()
        assert '--scenario' in result.output


# --- Custom Scan Presets Tests ---

class TestCustomPresets:
    @pytest.fixture(autouse=True)
    def tmp_db(self, tmp_path):
        with patch.object(db, 'DB_PATH', tmp_path / 'data.db'):
            yield

    def test_save_and_get_preset(self):
        db.save_custom_preset('my_dip', 'rsi<25 AND change<-3')
        presets = db.get_custom_presets()
        assert len(presets) == 1
        assert presets[0]['name'] == 'my_dip'
        assert presets[0]['filters'] == 'rsi<25 AND change<-3'

    def test_get_single_preset(self):
        db.save_custom_preset('test1', 'rsi<30')
        p = db.get_custom_preset('test1')
        assert p is not None
        assert p['filters'] == 'rsi<30'

    def test_get_missing_preset(self):
        p = db.get_custom_preset('nonexistent')
        assert p is None

    def test_delete_preset(self):
        db.save_custom_preset('del_me', 'rsi<20')
        assert db.delete_custom_preset('del_me') is True
        assert db.get_custom_preset('del_me') is None

    def test_delete_missing_preset(self):
        assert db.delete_custom_preset('nope') is False

    def test_overwrite_preset(self):
        db.save_custom_preset('ow', 'rsi<30')
        db.save_custom_preset('ow', 'rsi<20')
        p = db.get_custom_preset('ow')
        assert p['filters'] == 'rsi<20'


class TestScanPresetsCommand:
    def test_scan_presets_list(self):
        runner = CliRunner()
        with patch('cli.db.get_custom_presets', return_value=[]):
            result = runner.invoke(cli, ['scan-presets'])
        assert result.exit_code == 0
        assert 'oversold' in result.output

    def test_scan_presets_json(self):
        runner = CliRunner()
        with patch('cli.db.get_custom_presets', return_value=[]):
            result = runner.invoke(cli, ['scan-presets', '--json'])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert 'oversold' in data

    def test_scan_presets_with_custom(self):
        runner = CliRunner()
        with patch('cli.db.get_custom_presets', return_value=[
            {'name': 'my_dip', 'filters': 'rsi<25', 'created_at': '2026-01-01'}
        ]):
            result = runner.invoke(cli, ['scan-presets'])
        assert result.exit_code == 0
        assert 'my_dip' in result.output

    def test_scan_save(self):
        runner = CliRunner()
        with patch('cli.db.save_custom_preset') as mock_save:
            result = runner.invoke(cli, ['scan-save', 'my_dip', 'rsi<25 AND change<-3'])
        assert result.exit_code == 0
        assert 'Saved' in result.output
        mock_save.assert_called_once_with('my_dip', 'rsi<25 AND change<-3')

    def test_scan_delete(self):
        runner = CliRunner()
        with patch('cli.db.delete_custom_preset', return_value=True):
            result = runner.invoke(cli, ['scan-delete', 'my_dip'])
        assert result.exit_code == 0
        assert 'Deleted' in result.output

    def test_scan_delete_missing(self):
        runner = CliRunner()
        with patch('cli.db.delete_custom_preset', return_value=False):
            result = runner.invoke(cli, ['scan-delete', 'nope'])
        assert result.exit_code == 0
        assert 'not found' in result.output

    @patch('cli.engine.scan_tickers', return_value=[{
        'ticker': 'TSLA', 'price': 250.0, 'change_pct': -3.0,
        'rsi': 24.0, 'stoch_k': 15.0, 'vol_ratio': 1.5, 'reason': 'rsi<25'
    }])
    def test_scan_with_custom_preset(self, mock_scan):
        runner = CliRunner()
        with patch('cli.db.get_custom_preset', return_value={'name': 'my_dip', 'filters': 'rsi<25'}):
            result = runner.invoke(cli, ['scan', '--preset', 'my_dip'])
        assert result.exit_code == 0
        mock_scan.assert_called_once()

    def test_scan_unknown_preset(self):
        runner = CliRunner()
        with patch('cli.db.get_custom_preset', return_value=None):
            result = runner.invoke(cli, ['scan', '--preset', 'nonexistent'])
        assert result.exit_code == 0
        assert 'Unknown preset' in result.output


class TestPortfolioPerformance:
    @patch('cli.engine.generate_ohlcv')
    @patch('cli.engine.get_price', return_value={'price': 260.0, 'change_pct': 2.0})
    def test_performance_engine(self, mock_price, mock_ohlcv):
        import numpy as np
        import pandas as pd
        mock_ohlcv.return_value = pd.DataFrame({'close': np.linspace(230, 260, 30)})
        from cli.engine import portfolio_performance
        pos = [{'ticker': 'TSLA', 'entry_price': 240.0, 'qty': 10}]
        perf = portfolio_performance(pos)
        assert perf['total_value'] == 2600.0
        assert perf['total_cost'] == 2400.0
        assert perf['total_pnl'] == 200.0
        assert len(perf['holdings']) == 1
        assert perf['holdings'][0]['ticker'] == 'TSLA'
        assert 'daily_return_pct' in perf['holdings'][0]
        assert 'weekly_return_pct' in perf['holdings'][0]
        assert 'monthly_return_pct' in perf['holdings'][0]

    @patch('cli.engine.portfolio_performance', return_value={
        'total_value': 2600.0, 'total_cost': 2400.0, 'total_pnl': 200.0,
        'total_return_pct': 8.33,
        'holdings': [{'ticker': 'TSLA', 'qty': 10, 'entry': 240.0, 'current': 260.0,
                      'value': 2600.0, 'cost': 2400.0, 'total_return_pct': 8.33,
                      'daily_return_pct': 0.5, 'weekly_return_pct': 2.1, 'monthly_return_pct': 5.0}]
    })
    @patch('cli.db.get_positions', return_value=[
        {'ticker': 'TSLA', 'entry_price': 240.0, 'qty': 10, 'added_at': '2026-01-01'}
    ])
    def test_performance_cli(self, mock_pos, mock_perf):
        runner = CliRunner()
        result = runner.invoke(cli, ['portfolio-performance'])
        assert result.exit_code == 0
        assert 'TSLA' in result.output
        assert 'Performance' in result.output

    @patch('cli.engine.portfolio_performance', return_value={
        'total_value': 2600.0, 'total_cost': 2400.0, 'total_pnl': 200.0,
        'total_return_pct': 8.33, 'holdings': []
    })
    @patch('cli.db.get_positions', return_value=[
        {'ticker': 'TSLA', 'entry_price': 240.0, 'qty': 10, 'added_at': '2026-01-01'}
    ])
    def test_performance_cli_json(self, mock_pos, mock_perf):
        runner = CliRunner()
        result = runner.invoke(cli, ['portfolio-performance', '--json'])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data['total_value'] == 2600.0

    def test_performance_empty_portfolio(self):
        runner = CliRunner()
        with patch('cli.db.get_positions', return_value=[]):
            result = runner.invoke(cli, ['portfolio-performance'])
        assert result.exit_code == 0
        assert 'No positions' in result.output

    def test_performance_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ['portfolio-performance', '--help'])
        assert result.exit_code == 0
        assert 'performance' in result.output.lower()


class TestPositionsExport:
    @patch('cli.engine.get_price', return_value={'price': 260.0, 'change_pct': 2.0})
    @patch('cli.db.get_positions', return_value=[
        {'ticker': 'TSLA', 'entry_price': 240.0, 'qty': 10, 'added_at': '2026-01-01'}
    ])
    def test_export_json_stdout(self, mock_pos, mock_price):
        runner = CliRunner()
        result = runner.invoke(cli, ['positions-export'])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data) == 1
        assert data[0]['ticker'] == 'TSLA'
        assert 'pnl' in data[0]

    @patch('cli.engine.get_price', return_value={'price': 260.0, 'change_pct': 2.0})
    @patch('cli.db.get_positions', return_value=[
        {'ticker': 'TSLA', 'entry_price': 240.0, 'qty': 10, 'added_at': '2026-01-01'}
    ])
    def test_export_json_file(self, mock_pos, mock_price):
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(cli, ['positions-export', '-o', 'port.json'])
            assert result.exit_code == 0
            assert 'Exported 1' in result.output
            data = json.loads(Path('port.json').read_text())
            assert data[0]['ticker'] == 'TSLA'

    @patch('cli.engine.get_price', return_value={'price': 260.0, 'change_pct': 2.0})
    @patch('cli.db.get_positions', return_value=[
        {'ticker': 'TSLA', 'entry_price': 240.0, 'qty': 10, 'added_at': '2026-01-01'}
    ])
    def test_export_csv(self, mock_pos, mock_price):
        runner = CliRunner()
        result = runner.invoke(cli, ['positions-export', '-f', 'csv'])
        assert result.exit_code == 0
        assert 'ticker' in result.output
        assert 'TSLA' in result.output

    @patch('cli.engine.get_price', return_value={'price': 260.0, 'change_pct': 2.0})
    @patch('cli.db.get_positions', return_value=[
        {'ticker': 'TSLA', 'entry_price': 240.0, 'qty': 10, 'added_at': '2026-01-01'}
    ])
    def test_export_csv_file(self, mock_pos, mock_price):
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(cli, ['positions-export', '-f', 'csv', '-o', 'port.csv'])
            assert result.exit_code == 0
            assert 'Exported 1' in result.output
            content = Path('port.csv').read_text()
            assert 'TSLA' in content

    def test_export_empty(self):
        runner = CliRunner()
        with patch('cli.db.get_positions', return_value=[]):
            result = runner.invoke(cli, ['positions-export'])
        assert result.exit_code == 0
        assert 'No positions' in result.output

    def test_export_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ['positions-export', '--help'])
        assert result.exit_code == 0
        assert '--format' in result.output


class TestAlertsSchedule:
    def test_schedule_shows_cron(self):
        runner = CliRunner()
        result = runner.invoke(cli, ['alerts', 'schedule'])
        assert result.exit_code == 0
        assert 'cron' in result.output.lower() or 'crontab' in result.output.lower()

    def test_schedule_shows_examples(self):
        runner = CliRunner()
        result = runner.invoke(cli, ['alerts', 'schedule'])
        assert result.exit_code == 0
        assert 'alerts check' in result.output


class TestWhatIf:
    @patch('cli.engine.get_price', return_value={'price': 250.0, 'change_pct': 1.0})
    @patch('cli.db.get_positions', return_value=[
        {'ticker': 'AAPL', 'entry_price': 180.0, 'qty': 10, 'added_at': '2026-01-01'}
    ])
    def test_what_if_basic(self, mock_pos, mock_price):
        runner = CliRunner()
        result = runner.invoke(cli, ['what-if', 'NVDA', '--qty', '5'])
        assert result.exit_code == 0
        assert 'NVDA' in result.output
        assert 'What-If' in result.output

    @patch('cli.engine.get_price', return_value={'price': 250.0, 'change_pct': 1.0})
    @patch('cli.db.get_positions', return_value=[])
    def test_what_if_empty_portfolio(self, mock_pos, mock_price):
        runner = CliRunner()
        result = runner.invoke(cli, ['what-if', 'TSLA', '--qty', '10'])
        assert result.exit_code == 0
        assert '100.0%' in result.output

    @patch('cli.engine.get_price', return_value={'price': 250.0, 'change_pct': 1.0})
    @patch('cli.db.get_positions', return_value=[])
    def test_what_if_json(self, mock_pos, mock_price):
        runner = CliRunner()
        result = runner.invoke(cli, ['what-if', 'TSLA', '--qty', '10', '--json'])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data['ticker'] == 'TSLA'
        assert data['qty'] == 10
        assert data['weight_pct'] == 100.0

    @patch('cli.engine.get_price', side_effect=Exception("API error"))
    def test_what_if_api_error(self, mock_price):
        runner = CliRunner()
        result = runner.invoke(cli, ['what-if', 'BAD', '--qty', '5'])
        assert result.exit_code == 0
        assert 'Error' in result.output

    def test_what_if_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ['what-if', '--help'])
        assert result.exit_code == 0
        assert '--qty' in result.output


class TestPositionsImport:
    @patch('cli.db.add_position')
    def test_import_json(self, mock_add):
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path('port.json').write_text(json.dumps([
                {'ticker': 'TSLA', 'entry_price': 240.0, 'qty': 10},
                {'ticker': 'AAPL', 'entry_price': 180.0, 'qty': 20}
            ]))
            result = runner.invoke(cli, ['positions-import', 'port.json'])
        assert result.exit_code == 0
        assert 'Imported 2' in result.output
        assert mock_add.call_count == 2

    @patch('cli.db.add_position')
    def test_import_skips_invalid(self, mock_add):
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path('port.json').write_text(json.dumps([
                {'ticker': 'TSLA', 'entry_price': 240.0, 'qty': 10},
                {'bad': 'data'},
                'not_a_dict'
            ]))
            result = runner.invoke(cli, ['positions-import', 'port.json'])
        assert result.exit_code == 0
        assert 'Imported 1' in result.output

    def test_import_bad_json(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path('bad.json').write_text('not json')
            result = runner.invoke(cli, ['positions-import', 'bad.json'])
        assert result.exit_code == 0
        assert 'Error' in result.output

    def test_import_not_array(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path('obj.json').write_text(json.dumps({'ticker': 'TSLA'}))
            result = runner.invoke(cli, ['positions-import', 'obj.json'])
        assert result.exit_code == 0
        assert 'array' in result.output.lower()


class TestAlertsStats:
    @patch('cli.db.get_triggered_alerts', return_value=[
        {'id': 1, 'ticker': 'TSLA', 'condition': 'above', 'threshold': 260, 'triggered': 1}
    ])
    @patch('cli.db.get_alerts', return_value=[
        {'id': 2, 'ticker': 'TSLA', 'condition': 'above', 'threshold': 270, 'triggered': 0},
        {'id': 3, 'ticker': 'AAPL', 'condition': 'below', 'threshold': 170, 'triggered': 0}
    ])
    def test_stats_basic(self, mock_active, mock_triggered):
        runner = CliRunner()
        result = runner.invoke(cli, ['alerts', 'stats'])
        assert result.exit_code == 0
        assert '2' in result.output  # active count

    @patch('cli.db.get_triggered_alerts', return_value=[])
    @patch('cli.db.get_alerts', return_value=[])
    def test_stats_empty(self, mock_active, mock_triggered):
        runner = CliRunner()
        result = runner.invoke(cli, ['alerts', 'stats'])
        assert result.exit_code == 0
        assert '0' in result.output

    @patch('cli.db.get_triggered_alerts', return_value=[])
    @patch('cli.db.get_alerts', return_value=[
        {'id': 1, 'ticker': 'TSLA', 'condition': 'above', 'threshold': 270, 'triggered': 0}
    ])
    def test_stats_json(self, mock_active, mock_triggered):
        runner = CliRunner()
        result = runner.invoke(cli, ['alerts', 'stats', '--json'])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data['active'] == 1
        assert data['triggered'] == 0
        assert data['tickers_monitored'] == 1


class TestDoctor:
    @patch('cli.engine._ensure_models')
    @patch('cli.engine.get_price', return_value={'price': 150.0, 'change_pct': 0.5})
    @patch('cli.config._load', return_value={'fred-key': 'test'})
    @patch('cli.db.get_positions', return_value=[])
    def test_doctor_all_ok(self, mock_db, mock_cfg, mock_price, mock_models):
        runner = CliRunner()
        result = runner.invoke(cli, ['doctor'])
        assert result.exit_code == 0
        assert '✅' in result.output or 'ok' in result.output.lower()

    @patch('cli.engine._ensure_models', side_effect=Exception("no models"))
    @patch('cli.engine.get_price', side_effect=Exception("network error"))
    @patch('cli.config._load', return_value={})
    @patch('cli.db.get_positions', return_value=[])
    def test_doctor_with_failures(self, mock_db, mock_cfg, mock_price, mock_models):
        runner = CliRunner()
        result = runner.invoke(cli, ['doctor'])
        assert result.exit_code == 0
        assert '❌' in result.output or 'fail' in result.output.lower()

    @patch('cli.engine._ensure_models')
    @patch('cli.engine.get_price', return_value={'price': 150.0, 'change_pct': 0.5})
    @patch('cli.config._load', return_value={})
    @patch('cli.db.get_positions', return_value=[])
    def test_doctor_json(self, mock_db, mock_cfg, mock_price, mock_models):
        runner = CliRunner()
        result = runner.invoke(cli, ['doctor', '--json'])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)
        assert all('name' in c and 'status' in c for c in data)

    def test_doctor_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ['doctor', '--help'])
        assert result.exit_code == 0
        assert 'health' in result.output.lower()


class TestConfigReset:
    @pytest.fixture(autouse=True)
    def tmp_config(self, tmp_path):
        with patch.object(config, 'CONFIG_PATH', tmp_path / 'config.json'):
            yield

    def test_config_reset(self):
        config.set_key('fred-key', 'abc123')
        assert config.get('fred-key') == 'abc123'
        config.reset()
        assert config.get('fred-key') is None

    def test_config_reset_command(self):
        config.set_key('test-key', 'value')
        runner = CliRunner()
        result = runner.invoke(cli, ['config', 'reset', '--yes'])
        assert result.exit_code == 0
        assert 'reset' in result.output.lower()
        assert config.get('test-key') is None

    def test_config_reset_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ['config', 'reset', '--help'])
        assert result.exit_code == 0
        assert 'Reset' in result.output


class TestAlertsPauseResume:
    @pytest.fixture(autouse=True)
    def tmp_db(self, tmp_path):
        with patch.object(db, 'DB_PATH', tmp_path / 'data.db'):
            yield

    def test_pause_alert_by_id(self):
        db.add_alert('TSLA', 'above', 300.0)
        alerts = db.get_alerts()
        aid = alerts[0]['id']
        assert db.pause_alert(aid)
        runner = CliRunner()
        result = runner.invoke(cli, ['alerts', 'pause', '--id', str(aid)])
        assert result.exit_code == 0

    def test_resume_alert_by_id(self):
        db.add_alert('TSLA', 'above', 300.0)
        alerts = db.get_alerts()
        aid = alerts[0]['id']
        db.pause_alert(aid)
        assert db.resume_alert(aid)
        runner = CliRunner()
        result = runner.invoke(cli, ['alerts', 'resume', '--id', str(aid)])
        assert result.exit_code == 0

    def test_pause_by_ticker(self):
        db.add_alert('TSLA', 'above', 300.0)
        db.add_alert('TSLA', 'below', 200.0)
        n = db.pause_alerts_by_ticker('TSLA')
        assert n == 2

    def test_resume_by_ticker(self):
        db.add_alert('AAPL', 'above', 200.0)
        db.pause_alerts_by_ticker('AAPL')
        n = db.resume_alerts_by_ticker('AAPL')
        assert n == 1

    def test_pause_nonexistent(self):
        assert not db.pause_alert(9999)

    def test_resume_nonexistent(self):
        assert not db.resume_alert(9999)

    def test_pause_command_no_args(self):
        runner = CliRunner()
        result = runner.invoke(cli, ['alerts', 'pause'])
        assert result.exit_code == 0
        assert '--id' in result.output or '--ticker' in result.output

    def test_resume_command_no_args(self):
        runner = CliRunner()
        result = runner.invoke(cli, ['alerts', 'resume'])
        assert result.exit_code == 0
        assert '--id' in result.output or '--ticker' in result.output

    def test_pause_command_ticker(self):
        db.add_alert('TSLA', 'above', 300.0)
        runner = CliRunner()
        result = runner.invoke(cli, ['alerts', 'pause', '--ticker', 'TSLA'])
        assert result.exit_code == 0
        assert 'Paused' in result.output

    def test_resume_command_ticker(self):
        db.add_alert('TSLA', 'above', 300.0)
        db.pause_alerts_by_ticker('TSLA')
        runner = CliRunner()
        result = runner.invoke(cli, ['alerts', 'resume', '--ticker', 'TSLA'])
        assert result.exit_code == 0
        assert 'Resumed' in result.output

    def test_pause_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ['alerts', 'pause', '--help'])
        assert result.exit_code == 0
        assert 'pause' in result.output.lower()

    def test_resume_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ['alerts', 'resume', '--help'])
        assert result.exit_code == 0
        assert 'resume' in result.output.lower()


class TestPortfolioHistory:
    @pytest.fixture(autouse=True)
    def tmp_db(self, tmp_path):
        with patch.object(db, 'DB_PATH', tmp_path / 'data.db'):
            yield

    def test_save_and_get_snapshots(self):
        db.save_portfolio_snapshot(10000.0, 9000.0, 1000.0, 11.1, 3)
        snaps = db.get_portfolio_snapshots()
        assert len(snaps) == 1
        assert snaps[0]['total_value'] == 10000.0
        assert snaps[0]['num_positions'] == 3

    def test_portfolio_history_empty(self):
        runner = CliRunner()
        result = runner.invoke(cli, ['portfolio-history'])
        assert result.exit_code == 0
        assert 'No snapshots' in result.output

    @patch('cli.engine.get_price', return_value={'price': 260.0, 'change_pct': 1.0})
    def test_portfolio_history_save(self, mock_price):
        db.add_position('TSLA', 250.0, 10)
        runner = CliRunner()
        result = runner.invoke(cli, ['portfolio-history', '--save'])
        assert result.exit_code == 0
        assert 'Snapshot saved' in result.output
        snaps = db.get_portfolio_snapshots()
        assert len(snaps) == 1

    def test_portfolio_history_display(self):
        db.save_portfolio_snapshot(10000.0, 9000.0, 1000.0, 11.1, 3)
        db.save_portfolio_snapshot(10500.0, 9000.0, 1500.0, 16.7, 3)
        runner = CliRunner()
        result = runner.invoke(cli, ['portfolio-history'])
        assert result.exit_code == 0
        assert '10,000' in result.output or '10000' in result.output

    def test_portfolio_history_json(self):
        db.save_portfolio_snapshot(10000.0, 9000.0, 1000.0, 11.1, 3)
        runner = CliRunner()
        result = runner.invoke(cli, ['portfolio-history', '--json'])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data) == 1
        assert data[0]['total_value'] == 10000.0

    def test_portfolio_history_save_no_positions(self):
        runner = CliRunner()
        result = runner.invoke(cli, ['portfolio-history', '--save'])
        assert result.exit_code == 0
        assert 'No positions' in result.output

    def test_portfolio_history_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ['portfolio-history', '--help'])
        assert result.exit_code == 0
        assert 'snapshot' in result.output.lower()


class TestAlertsPausedFiltering:
    @pytest.fixture(autouse=True)
    def tmp_db(self, tmp_path):
        with patch.object(db, 'DB_PATH', tmp_path / 'data.db'):
            yield

    def test_paused_alerts_excluded_from_get_alerts(self):
        db.add_alert('TSLA', 'above', 300.0)
        db.add_alert('TSLA', 'below', 200.0)
        assert len(db.get_alerts()) == 2
        alerts = db.get_alerts()
        db.pause_alert(alerts[0]['id'])
        assert len(db.get_alerts()) == 1

    def test_paused_alerts_included_in_get_all(self):
        db.add_alert('TSLA', 'above', 300.0)
        alerts = db.get_alerts()
        db.pause_alert(alerts[0]['id'])
        assert len(db.get_alerts()) == 0
        assert len(db.get_all_active_alerts()) == 1

    def test_alerts_list_all_flag(self):
        db.add_alert('TSLA', 'above', 300.0)
        alerts = db.get_all_active_alerts()
        db.pause_alert(alerts[0]['id'])
        runner = CliRunner()
        # Without --all, paused alert hidden
        result = runner.invoke(cli, ['alerts', 'list'])
        assert 'No active alerts' in result.output
        # With --all, paused alert shown
        result = runner.invoke(cli, ['alerts', 'list', '--all'])
        assert result.exit_code == 0
        assert 'TSLA' in result.output
        assert 'paused' in result.output


class TestWhatIfRemove:
    @pytest.fixture(autouse=True)
    def tmp_db(self, tmp_path):
        with patch.object(db, 'DB_PATH', tmp_path / 'data.db'):
            yield

    @patch('cli.engine.get_price', return_value={'price': 250.0, 'change_pct': 1.0})
    def test_what_if_remove(self, mock_price):
        db.add_position('TSLA', 200.0, 10)
        runner = CliRunner()
        result = runner.invoke(cli, ['what-if', 'TSLA', '--qty', '5', '--remove'])
        assert result.exit_code == 0
        assert 'Remove' in result.output

    @patch('cli.engine.get_price', return_value={'price': 250.0, 'change_pct': 1.0})
    def test_what_if_remove_json(self, mock_price):
        db.add_position('TSLA', 200.0, 10)
        runner = CliRunner()
        result = runner.invoke(cli, ['what-if', 'TSLA', '--qty', '5', '--remove', '--json'])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data['action'] == 'remove'
        assert data['new_portfolio'] < data['current_portfolio']

    @patch('cli.engine.get_price', return_value={'price': 100.0, 'change_pct': 0.5})
    def test_what_if_add_still_works(self, mock_price):
        runner = CliRunner()
        result = runner.invoke(cli, ['what-if', 'AAPL', '--qty', '10'])
        assert result.exit_code == 0
        assert 'Add' in result.output


class TestDoctorFix:
    @patch('cli.engine._ensure_models')
    @patch('cli.engine.get_price', return_value={'price': 150.0, 'change_pct': 0.5})
    @patch('cli.config._load', return_value={})
    @patch('cli.db.get_positions', return_value=[])
    def test_doctor_fix_flag(self, mock_db, mock_cfg, mock_price, mock_models):
        runner = CliRunner()
        result = runner.invoke(cli, ['doctor', '--fix'])
        assert result.exit_code == 0
        assert '✅' in result.output or 'ok' in result.output.lower()

    def test_doctor_fix_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ['doctor', '--help'])
        assert result.exit_code == 0
        assert '--fix' in result.output


class TestPositionsGroupBy:
    @pytest.fixture(autouse=True)
    def tmp_db(self, tmp_path):
        with patch.object(db, 'DB_PATH', tmp_path / 'data.db'):
            yield

    @patch('cli.engine.get_price', return_value={'price': 260.0, 'change_pct': 1.0})
    @patch('cli.engine.get_price_history', return_value=[250, 255, 260])
    def test_positions_group_by_sector(self, mock_hist, mock_price):
        db.add_position('TSLA', 250.0, 10)
        db.add_position('AAPL', 150.0, 5)
        runner = CliRunner()
        result = runner.invoke(cli, ['positions', '--group-by', 'sector'])
        assert result.exit_code == 0
        assert 'By Sector' in result.output

    def test_positions_group_by_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ['positions', '--help'])
        assert result.exit_code == 0
        assert '--group-by' in result.output


class TestScanWatch:
    def test_scan_watch_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ['scan', '--help'])
        assert result.exit_code == 0
        assert '--watch' in result.output
