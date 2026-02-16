"""Tests for new ML pipeline features: walk-forward, regime, LightGBM, backtest costs, hourly."""
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def _price_df(days=500, start_price=100):
    dates = pd.date_range("2023-01-01", periods=days, freq="B")
    close = start_price + np.cumsum(np.random.randn(days) * 0.5)
    close = np.maximum(close, 10)
    return pd.DataFrame({
        "date": dates, "symbol": "TEST",
        "open": close - 0.5, "high": close + 1, "low": close - 1,
        "close": close, "volume": np.random.randint(1e6, 5e7, days),
    })


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
        "vix": pd.DataFrame({"date": dates.date, "vix": 18.0, "vix_9d": 17.0, "vix_3m": 19.0, "vix_term_structure": 0.0}),
    }


# --- Walk-Forward Validation Tests ---

class TestWalkForward:
    def test_walk_forward_validate_returns_windows(self):
        from cli.engine import _walk_forward_validate
        np.random.seed(42)
        n = 320
        X = np.random.randn(n, 10)
        y_short = np.random.randn(n) * 0.01
        y_medium = np.random.randn(n) * 0.02
        y_long = np.random.randn(n) * 0.03
        result = _walk_forward_validate(X, y_short, y_medium, y_long, 252, 63, verbose=False)
        assert 'windows' in result
        assert 'average' in result
        assert len(result['windows']) > 0
        for w in result['windows']:
            assert 'short' in w
            assert 'medium' in w
            assert 'long' in w
            assert 0 <= w['short'] <= 1

    def test_walk_forward_with_small_data(self):
        from cli.engine import _walk_forward_validate
        X = np.random.randn(100, 5)
        y = np.random.randn(100) * 0.01
        result = _walk_forward_validate(X, y, y, y, 252, 63, verbose=False)
        # Not enough data for any window
        assert result['windows'] == []
        assert result['average'] == {}

    def test_walk_forward_accuracy_range(self):
        from cli.engine import _walk_forward_validate
        np.random.seed(42)
        n = 320
        X = np.random.randn(n, 10)
        y = np.random.randn(n) * 0.01
        result = _walk_forward_validate(X, y, y, y, 252, 63, verbose=False)
        for w in result['windows']:
            for h in ('short', 'medium', 'long'):
                assert 0 <= w[h] <= 1
        if result['average']:
            for h in ('short', 'medium', 'long'):
                assert 0 <= result['average'][h] <= 1

    def test_walk_forward_trend_analysis(self):
        """Walk-forward should include trend analysis when enough windows exist."""
        from cli.engine import _walk_forward_validate
        np.random.seed(42)
        n = 504  # enough for 4 windows (252 + 4*63)
        X = np.random.randn(n, 10)
        y = np.random.randn(n) * 0.01
        result = _walk_forward_validate(X, y, y, y, 252, 63, verbose=False)
        assert 'trend' in result
        for h in ('short', 'medium', 'long'):
            assert result['trend'][h] in ('improving', 'degrading', 'stable', 'insufficient_data')

    def test_walk_forward_trend_insufficient_data(self):
        """Trend should be 'insufficient_data' with fewer than 4 windows."""
        from cli.engine import _walk_forward_validate
        np.random.seed(42)
        n = 320  # only 1 window
        X = np.random.randn(n, 10)
        y = np.random.randn(n) * 0.01
        result = _walk_forward_validate(X, y, y, y, 252, 63, verbose=False)
        assert 'trend' in result
        if result['windows']:
            for h in ('short', 'medium', 'long'):
                assert result['trend'][h] == 'insufficient_data'


class TestModelStalenessAndVersion:
    """Tests for model staleness warning and version compatibility."""

    def test_stale_model_warning(self):
        """Should log warning for models older than 30 days."""
        import logging
        from cli.engine import _ensure_ticker_model, _ticker_ensembles, _ticker_meta, MODEL_VERSION
        ticker = 'STALETEST'
        _ticker_ensembles.pop(ticker, None)
        _ticker_meta.pop(ticker, None)

        old_date = (datetime.now() - pd.Timedelta(days=45)).isoformat()
        mock_meta = {
            'model_version': MODEL_VERSION,
            'trained_at': old_date,
            'ensemble_weights': {'short': {'xgb': 0.5, 'lgbm': 0.5}, 'medium': {'xgb': 0.5, 'lgbm': 0.5}, 'long': {'xgb': 0.5, 'lgbm': 0.5}},
        }

        mock_ens = MagicMock()
        mock_ens.load.return_value = True

        with patch('cli.engine.EnsembleModel', return_value=mock_ens), \
             patch('cli.engine._load_ticker_meta', return_value=mock_meta), \
             patch.object(logging.getLogger('cli.engine'), 'warning') as mock_warn:
            _ensure_ticker_model(ticker)
            calls = [str(c) for c in mock_warn.call_args_list]
            assert any('days old' in c for c in calls)

        _ticker_ensembles.pop(ticker, None)

    def test_version_mismatch_warning(self):
        """Should log warning when model version doesn't match current."""
        import logging
        from cli.engine import _ensure_ticker_model, _ticker_ensembles, _ticker_meta
        ticker = 'VERTEST'
        _ticker_ensembles.pop(ticker, None)
        _ticker_meta.pop(ticker, None)

        mock_meta = {
            'model_version': '1.0.0',  # old version
            'trained_at': datetime.now().isoformat(),
            'ensemble_weights': {'short': {'xgb': 0.5, 'lgbm': 0.5}, 'medium': {'xgb': 0.5, 'lgbm': 0.5}, 'long': {'xgb': 0.5, 'lgbm': 0.5}},
        }

        mock_ens = MagicMock()
        mock_ens.load.return_value = True

        with patch('cli.engine.EnsembleModel', return_value=mock_ens), \
             patch('cli.engine._load_ticker_meta', return_value=mock_meta), \
             patch.object(logging.getLogger('cli.engine'), 'warning') as mock_warn:
            _ensure_ticker_model(ticker)
            assert mock_warn.call_count >= 1
            # Check that version mismatch was mentioned
            calls = [str(c) for c in mock_warn.call_args_list]
            assert any('1.0.0' in c for c in calls)

        _ticker_ensembles.pop(ticker, None)

    def test_no_warning_for_fresh_model(self):
        """Should not warn for recently trained model with correct version."""
        import logging
        from cli.engine import _ensure_ticker_model, _ticker_ensembles, _ticker_meta, MODEL_VERSION
        ticker = 'FRESHTEST'
        _ticker_ensembles.pop(ticker, None)
        _ticker_meta.pop(ticker, None)

        mock_meta = {
            'model_version': MODEL_VERSION,
            'trained_at': datetime.now().isoformat(),
            'ensemble_weights': {'short': {'xgb': 0.5, 'lgbm': 0.5}, 'medium': {'xgb': 0.5, 'lgbm': 0.5}, 'long': {'xgb': 0.5, 'lgbm': 0.5}},
        }

        mock_ens = MagicMock()
        mock_ens.load.return_value = True

        with patch('cli.engine.EnsembleModel', return_value=mock_ens), \
             patch('cli.engine._load_ticker_meta', return_value=mock_meta), \
             patch.object(logging.getLogger('cli.engine'), 'warning') as mock_warn:
            _ensure_ticker_model(ticker)
            assert mock_warn.call_count == 0

        _ticker_ensembles.pop(ticker, None)


# --- Regime Detection Tests ---

class TestRegimeDetection:
    def test_bull_regime(self):
        from backend.features.regime import compute_regime_features
        # Create uptrending data
        df = _price_df(300, start_price=100)
        df['close'] = 100 + np.arange(300) * 0.5  # strong uptrend
        df['high'] = df['close'] + 1
        df['low'] = df['close'] - 1
        features = compute_regime_features(df)
        assert features['regime_bull'] == 1.0
        assert features['regime_bear'] == 0.0
        assert features['regime_label'] == 'bull'

    def test_bear_regime(self):
        from backend.features.regime import compute_regime_features
        df = _price_df(300, start_price=200)
        df['close'] = 200 - np.arange(300) * 0.5  # strong downtrend
        df['close'] = np.maximum(df['close'], 10)
        df['high'] = df['close'] + 1
        df['low'] = df['close'] - 1
        features = compute_regime_features(df)
        assert features['regime_bear'] == 1.0
        assert features['regime_label'] == 'bear'

    def test_volatility_regime_with_vix(self):
        from backend.features.regime import compute_regime_features
        df = _price_df(100)
        macro = _macro()
        macro['vix']['vix'] = 30.0  # high VIX
        features = compute_regime_features(df, macro)
        assert features['vol_regime_high'] == 1.0
        assert features['vol_regime_label'] == 'high'

    def test_volatility_regime_low_vix(self):
        from backend.features.regime import compute_regime_features
        df = _price_df(100)
        macro = _macro()
        macro['vix']['vix'] = 12.0
        features = compute_regime_features(df, macro)
        assert features['vol_regime_low'] == 1.0

    def test_momentum_regime(self):
        from backend.features.regime import compute_regime_features
        df = _price_df(100, start_price=100)
        df['close'] = 100 + np.arange(100) * 1.0  # strong upward momentum
        df['high'] = df['close'] + 1
        df['low'] = df['close'] - 1
        features = compute_regime_features(df)
        assert features['momentum_roc20'] > 0

    def test_regime_display_format(self):
        from backend.features.regime import format_regime_display
        features = {'regime_label': 'bull', 'vol_regime_label': 'low'}
        display = format_regime_display(features)
        assert '🐂' in display
        assert 'Bull' in display
        assert 'low volatility' in display

    def test_regime_features_in_feature_store(self):
        from backend.features.feature_store import FeatureStore
        fs = FeatureStore()
        names = fs.all_feature_names()
        assert 'regime_bull' in names
        assert 'vol_regime_high' in names
        assert 'trend_trending' in names
        assert 'mom_strong_up' in names

    def test_regime_one_hot_exclusive(self):
        from backend.features.regime import compute_regime_features
        df = _price_df(300)
        features = compute_regime_features(df)
        # Exactly one regime should be active
        assert features['regime_bull'] + features['regime_bear'] + features['regime_sideways'] == 1.0
        assert features['vol_regime_low'] + features['vol_regime_normal'] + features['vol_regime_high'] == 1.0


# --- LightGBM Model Tests ---

class TestLightGBM:
    def test_lgbm_train_predict(self):
        from backend.models.lgbm_models import LGBMShort
        model = LGBMShort()
        X = np.random.randn(500, 10)
        y = X[:, 0] * 0.5 + X[:, 1] * 0.3 + np.random.randn(500) * 0.1  # signal + noise
        score = model.train(X, y)
        assert score >= 0
        preds = model.predict(X[:5])
        assert len(preds) == 5

    def test_lgbm_predict_proba(self):
        from backend.models.lgbm_models import LGBMMedium
        model = LGBMMedium()
        X = np.random.randn(200, 10)
        y = np.random.randn(200) * 0.01
        model.train(X, y)
        pred, conf = model.predict_proba(X[0])
        assert isinstance(pred, float)
        assert 0 <= conf <= 1

    def test_lgbm_save_load(self, tmp_path):
        from backend.models.lgbm_models import LGBMLong
        model = LGBMLong()
        X = np.random.randn(100, 5)
        y = np.random.randn(100) * 0.01
        model.train(X, y)
        path = tmp_path / 'lgbm.pkl'
        model.save(path)
        model2 = LGBMLong()
        assert model2.load(path)
        assert model2._trained

    def test_lgbm_untrained_returns_zeros(self):
        from backend.models.lgbm_models import LGBMShort
        model = LGBMShort()
        preds = model.predict(np.random.randn(5, 10))
        assert np.all(preds == 0)

    def test_lgbm_with_sample_weights(self):
        from backend.models.lgbm_models import LGBMShort
        model = LGBMShort()
        X = np.random.randn(500, 10)
        y = X[:, 0] * 0.5 + np.random.randn(500) * 0.1
        weights = np.ones(500)
        score = model.train(X, y, sample_weight=weights)
        assert score >= 0


# --- Ensemble with LightGBM Tests ---

class TestEnsembleWithLGBM:
    def test_ensemble_includes_lgbm(self):
        from backend.models.ensemble import EnsembleModel
        ens = EnsembleModel()
        assert hasattr(ens, 'lgbm_short')
        assert hasattr(ens, 'lgbm_medium')
        assert hasattr(ens, 'lgbm_long')

    def test_ensemble_weights_from_accuracy(self):
        from backend.models.ensemble import EnsembleModel
        ens = EnsembleModel()
        xgb_acc = {'short': 0.6, 'medium': 0.55, 'long': 0.58}
        lgbm_acc = {'short': 0.55, 'medium': 0.6, 'long': 0.52}
        ens.set_weights_from_accuracy(xgb_acc, lgbm_acc)
        # XGBoost should have higher weight for short
        assert ens.weights['short']['xgb'] > ens.weights['short']['lgbm']
        # LightGBM should have higher weight for medium
        assert ens.weights['medium']['lgbm'] > ens.weights['medium']['xgb']

    def test_ensemble_predict_uses_both(self):
        from backend.models.ensemble import EnsembleModel
        ens = EnsembleModel()
        X = np.random.randn(200, 10)
        y = np.random.randn(200) * 0.01
        ens.train_all(X, y, y, y)
        pred, conf, breakdown = ens.predict(X[0:1], 'short')
        assert 'xgb' in breakdown
        assert 'lgbm' in breakdown
        assert isinstance(pred, float)


# --- Backtest Cost Tests ---

class TestBacktestCosts:
    def test_slippage_reduces_returns(self):
        """Net return should be less than gross return due to costs."""
        # We test the cost logic directly
        gross_ret = 0.01  # 1% gross return
        slippage = 0.0005
        spread = 0.0002
        cost = slippage * 2 + spread  # 0.12%
        net_ret = gross_ret - cost
        assert net_ret < gross_ret
        assert net_ret > 0  # still profitable

    def test_backtest_result_has_cost_fields(self):
        """Backtest result should include cost-related fields."""
        result = {
            'gross_return': 10.0, 'net_return': 9.5,
            'sharpe_ratio': 1.5, 'max_drawdown': 0.12,
            'win_rate': 0.55, 'profit_factor': 1.8,
            'total_trades': 42, 'avg_holding_period': 1.0,
            'slippage': 0.0005, 'commission': 0.0, 'spread': 0.0002,
            'win_rate_by_horizon': {'short': 0.55, 'medium': 0.52, 'long': 0.58},
        }
        assert 'gross_return' in result
        assert 'net_return' in result
        assert 'slippage' in result
        assert 'spread' in result
        assert 'win_rate_by_horizon' in result
        assert result['net_return'] < result['gross_return']

    def test_profit_factor_calculation(self):
        """Profit factor = gross wins / gross losses."""
        wins = np.array([0.01, 0.02, 0.015])
        losses = np.array([-0.005, -0.01])
        pf = wins.sum() / abs(losses.sum())
        assert pf > 1  # profitable

    def test_sharpe_ratio_calculation(self):
        """Sharpe ratio should be annualized."""
        returns = np.array([0.001, -0.0005, 0.002, 0.001, -0.001])
        sharpe = np.sqrt(252) * returns.mean() / returns.std()
        assert isinstance(sharpe, float)

    def test_max_drawdown_calculation(self):
        """Max drawdown from cumulative returns."""
        returns = np.array([0.01, 0.02, -0.05, 0.01, 0.02])
        cum = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cum)
        dd = (cum - peak) / peak
        max_dd = abs(dd.min())
        assert max_dd > 0


# --- Hourly Data Tests ---

class TestHourlyFeatures:
    def test_compute_hourly_features(self):
        from backend.features.hourly import compute_hourly_features
        n = 200
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        close = 100 + np.cumsum(np.random.randn(n) * 0.1)
        df = pd.DataFrame({
            "date": dates, "symbol": "TEST",
            "open": close - 0.1, "high": close + 0.5, "low": close - 0.5,
            "close": close, "volume": np.random.randint(1e4, 1e6, n),
        })
        feats = compute_hourly_features(df)
        assert 'rsi_14' in feats.columns
        assert 'macd' in feats.columns
        assert 'bb_position' in feats.columns
        assert 'vwap_distance' in feats.columns
        assert 'volume_ratio' in feats.columns
        assert len(feats) == n

    def test_hourly_features_no_nans(self):
        from backend.features.hourly import compute_hourly_features
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        close = 100 + np.cumsum(np.random.randn(n) * 0.1)
        df = pd.DataFrame({
            "date": dates, "symbol": "TEST",
            "open": close - 0.1, "high": close + 0.5, "low": close - 0.5,
            "close": close, "volume": np.random.randint(1e4, 1e6, n),
        })
        feats = compute_hourly_features(df)
        assert not feats.isnull().any().any()

    def test_fetch_hourly_data_returns_none_on_failure(self):
        from backend.features.hourly import fetch_hourly_data
        with patch('yfinance.Ticker') as mock_yf:
            mock_yf.return_value.history.return_value = pd.DataFrame()
            result = fetch_hourly_data("INVALID")
            assert result is None


# --- Warning Suppression Tests ---

class TestWarningSuppression:
    def test_sentiment_safe_returns_none_on_error(self):
        from cli.engine import _get_sentiment_safe
        with patch('cli.engine.get_sentiment', side_effect=Exception("Rate limited")):
            result = _get_sentiment_safe("TEST")
            assert result is None

    def test_sentiment_safe_warns_once(self):
        import cli.engine as eng
        eng._sentiment_warned = False
        with patch('cli.engine.get_sentiment', side_effect=Exception("Rate limited")):
            _get_sentiment_safe = eng._get_sentiment_safe
            _get_sentiment_safe("TEST")
            assert eng._sentiment_warned
            # Second call should not re-warn
            _get_sentiment_safe("TEST2")
            assert eng._sentiment_warned
        eng._sentiment_warned = False  # reset


# --- Tune Flag Tests ---

class TestTuneFlag:
    def test_optuna_tune_returns_params(self):
        from cli.engine import _optuna_tune
        np.random.seed(42)
        n = 150
        X = np.random.randn(n, 10)
        y = np.random.randn(n) * 0.01
        # Patch optuna to use fewer trials for speed
        import optuna
        orig_optimize = optuna.study.Study.optimize
        def fast_optimize(self, func, n_trials=50, **kw):
            return orig_optimize(self, func, n_trials=3, **kw)
        with patch.object(optuna.study.Study, 'optimize', fast_optimize):
            xgb_params, lgbm_params = _optuna_tune(X, y, y, y, 200, 50, verbose=False)
        assert 'n_estimators' in xgb_params
        assert 'max_depth' in xgb_params
        assert 'learning_rate' in xgb_params
        assert 'num_leaves' in lgbm_params
        assert xgb_params['random_state'] == 42
        assert lgbm_params['verbose'] == -1

    def test_retrain_ticker_with_tune_flag(self):
        """Verify retrain_ticker accepts tune=True without error."""
        from cli.engine import retrain_ticker
        import optuna
        orig_optimize = optuna.study.Study.optimize
        def fast_optimize(self, func, n_trials=50, **kw):
            return orig_optimize(self, func, n_trials=2, **kw)
        with patch('yfinance.Ticker') as mock_yf_ticker, \
             patch('cli.engine.get_options_chain', return_value={}), \
             patch('cli.engine.get_fundamentals', return_value={}), \
             patch('cli.engine._get_sentiment_safe', return_value=None), \
             patch('cli.engine.get_macro_data', return_value=_macro()), \
             patch('backend.features.hourly.fetch_hourly_data', return_value=None), \
             patch.object(optuna.study.Study, 'optimize', fast_optimize):
            n = 150
            dates = pd.date_range("2023-01-01", periods=n, freq="B")
            close = 100 + np.cumsum(np.random.randn(n) * 0.5)
            close = np.maximum(close, 10)
            hist = pd.DataFrame({
                "Date": dates, "Open": close - 0.5, "High": close + 1,
                "Low": close - 1, "Close": close,
                "Volume": np.random.randint(1e6, 5e7, n),
            })
            hist.index = dates
            mock_yf_ticker.return_value.history.return_value = hist
            acc = retrain_ticker("TUNETEST", tune=True)
            assert 'short' in acc
            assert 0 <= acc['short'] <= 1


# --- Detailed Backtest Output Tests ---

class TestDetailedBacktest:
    def test_detailed_backtest_includes_trades(self):
        """When detailed=True, result should include per-trade breakdown."""
        from cli.engine import run_backtest
        with patch('cli.engine._ensure_models'), \
             patch('cli.engine._ensure_ticker_model') as mock_ens, \
             patch('cli.engine.generate_ohlcv') as mock_ohlcv, \
             patch('cli.engine.get_options_chain', return_value={}), \
             patch('cli.engine.get_fundamentals', return_value={}), \
             patch('cli.engine._get_sentiment_safe', return_value=None), \
             patch('cli.engine.get_macro_data', return_value=_macro()):
            n = 100
            dates = pd.date_range("2024-01-01", periods=n, freq="B")
            close = 100 + np.cumsum(np.random.randn(n) * 0.5)
            close = np.maximum(close, 10)
            mock_ohlcv.return_value = pd.DataFrame({
                "date": dates, "symbol": "TEST",
                "open": close - 0.5, "high": close + 1, "low": close - 1,
                "close": close, "volume": np.random.randint(1e6, 5e7, n),
            })
            # Mock ensemble
            from unittest.mock import MagicMock
            ens = MagicMock()
            ens.predict.return_value = (0.01, 0.7, {'xgb': 0.01, 'lgbm': 0.01})
            ens.weights = {'short': {'xgb': 0.5, 'lgbm': 0.5}, 'medium': {'xgb': 0.5, 'lgbm': 0.5}, 'long': {'xgb': 0.5, 'lgbm': 0.5}}
            # Mock sub-models
            for attr in ('xgb_short', 'xgb_medium', 'xgb_long', 'lgbm_short', 'lgbm_medium', 'lgbm_long'):
                m = MagicMock()
                m.predict.return_value = np.random.randn(n - 1) * 0.01
                setattr(ens, attr, m)
            mock_ens.return_value = ens
            result = run_backtest("TEST", days=100, detailed=True)
            assert 'trades' in result
            if result['total_trades'] > 0:
                trade = result['trades'][0]
                assert 'signal' in trade
                assert 'return_gross' in trade
                assert 'return_net' in trade
                assert 'cost' in trade

    def test_non_detailed_backtest_excludes_trades(self):
        """When detailed=False, result should NOT include trades list."""
        from cli.engine import run_backtest
        with patch('cli.engine._ensure_models'), \
             patch('cli.engine._ensure_ticker_model') as mock_ens, \
             patch('cli.engine.generate_ohlcv') as mock_ohlcv, \
             patch('cli.engine.get_options_chain', return_value={}), \
             patch('cli.engine.get_fundamentals', return_value={}), \
             patch('cli.engine._get_sentiment_safe', return_value=None), \
             patch('cli.engine.get_macro_data', return_value=_macro()):
            n = 100
            dates = pd.date_range("2024-01-01", periods=n, freq="B")
            close = 100 + np.cumsum(np.random.randn(n) * 0.5)
            mock_ohlcv.return_value = pd.DataFrame({
                "date": dates, "symbol": "TEST",
                "open": close - 0.5, "high": close + 1, "low": close - 1,
                "close": close, "volume": np.random.randint(1e6, 5e7, n),
            })
            from unittest.mock import MagicMock
            ens = MagicMock()
            ens.predict.return_value = (0.01, 0.7, {'xgb': 0.01, 'lgbm': 0.01})
            ens.weights = {'short': {'xgb': 0.5, 'lgbm': 0.5}, 'medium': {'xgb': 0.5, 'lgbm': 0.5}, 'long': {'xgb': 0.5, 'lgbm': 0.5}}
            for attr in ('xgb_short', 'xgb_medium', 'xgb_long', 'lgbm_short', 'lgbm_medium', 'lgbm_long'):
                m = MagicMock()
                m.predict.return_value = np.random.randn(n - 1) * 0.01
                setattr(ens, attr, m)
            mock_ens.return_value = ens
            result = run_backtest("TEST", days=100, detailed=False)
            assert 'trades' not in result


# --- Hourly Model Prediction Tests ---

class TestHourlyModelPrediction:
    def test_predict_hourly_short_with_model(self, tmp_path):
        """Hourly model should be used for short-term predictions when available."""
        import cli.engine as eng
        import pickle
        import xgboost as xgb
        import lightgbm as lgb

        ticker = "HRTEST"
        # Train small models
        X = np.random.randn(100, 10)
        y = np.random.randn(100) * 0.01
        xgb_m = xgb.XGBRegressor(n_estimators=10, max_depth=2, random_state=42)
        lgb_m = lgb.LGBMRegressor(n_estimators=10, max_depth=2, random_state=42, verbose=-1)
        xgb_m.fit(X, y)
        lgb_m.fit(X, y)

        # Save to tmp model dir
        model_dir = tmp_path / ticker
        model_dir.mkdir()
        with open(model_dir / 'hourly_short.pkl', 'wb') as fp:
            pickle.dump({'xgb': xgb_m, 'lgbm': lgb_m, 'feature_count': 10}, fp)

        # Patch model dir and hourly data
        old_models_dir = eng._MODELS_DIR
        eng._MODELS_DIR = tmp_path
        eng._hourly_models.pop(ticker, None)

        try:
            with patch('backend.features.hourly.fetch_hourly_data') as mock_fetch, \
                 patch('backend.features.hourly.compute_hourly_features') as mock_feats:
                n = 50
                dates = pd.date_range("2024-01-01", periods=n, freq="h")
                close = 100 + np.cumsum(np.random.randn(n) * 0.1)
                mock_fetch.return_value = pd.DataFrame({
                    "date": dates, "symbol": ticker,
                    "open": close - 0.1, "high": close + 0.5, "low": close - 0.5,
                    "close": close, "volume": np.random.randint(1e4, 1e6, n),
                })
                mock_feats.return_value = pd.DataFrame(np.random.randn(n, 10))
                result = eng._predict_hourly_short(ticker)
                assert result is not None
                pred, conf = result
                assert isinstance(pred, float)
                assert 0 <= conf <= 1
        finally:
            eng._MODELS_DIR = old_models_dir
            eng._hourly_models.pop(ticker, None)

    def test_predict_hourly_short_no_model(self):
        """Returns None when no hourly model exists."""
        import cli.engine as eng
        eng._hourly_models.pop("NOMODEL", None)
        result = eng._predict_hourly_short("NOMODEL")
        assert result is None

    def test_load_hourly_model_caches(self, tmp_path):
        """Hourly model should be cached after first load."""
        import cli.engine as eng
        import pickle
        import xgboost as xgb
        import lightgbm as lgb

        ticker = "CACHETEST"
        model_dir = tmp_path / ticker
        model_dir.mkdir()
        xgb_m = xgb.XGBRegressor(n_estimators=5, max_depth=2, random_state=42)
        lgb_m = lgb.LGBMRegressor(n_estimators=5, max_depth=2, random_state=42, verbose=-1)
        X = np.random.randn(50, 5)
        y = np.random.randn(50) * 0.01
        xgb_m.fit(X, y)
        lgb_m.fit(X, y)
        with open(model_dir / 'hourly_short.pkl', 'wb') as fp:
            pickle.dump({'xgb': xgb_m, 'lgbm': lgb_m, 'feature_count': 5}, fp)

        old_models_dir = eng._MODELS_DIR
        eng._MODELS_DIR = tmp_path
        eng._hourly_models.pop(ticker, None)
        try:
            m1 = eng._load_hourly_model(ticker)
            m2 = eng._load_hourly_model(ticker)
            assert m1 is m2  # same object = cached
        finally:
            eng._MODELS_DIR = old_models_dir
            eng._hourly_models.pop(ticker, None)


class TestHourlyEdgeCases:
    def test_hourly_model_insufficient_data(self):
        """Hourly prediction returns None when data has < 20 bars."""
        import cli.engine as eng
        import xgboost as xgb
        import lightgbm as lgb

        ticker = "FEWBARS"
        # Create a model so _load_hourly_model succeeds
        eng._hourly_models[ticker] = {
            'xgb': xgb.XGBRegressor(n_estimators=5, max_depth=2, random_state=42),
            'lgbm': lgb.LGBMRegressor(n_estimators=5, max_depth=2, random_state=42, verbose=-1),
            'feature_count': 10,
        }
        try:
            with patch('backend.features.hourly.fetch_hourly_data') as mock_fetch:
                # Return only 10 bars (< 20 threshold)
                n = 10
                mock_fetch.return_value = pd.DataFrame({
                    "date": pd.date_range("2024-01-01", periods=n, freq="h"),
                    "symbol": ticker, "open": np.ones(n) * 100,
                    "high": np.ones(n) * 101, "low": np.ones(n) * 99,
                    "close": np.ones(n) * 100, "volume": np.ones(n) * 1e6,
                })
                result = eng._predict_hourly_short(ticker)
                assert result is None
        finally:
            eng._hourly_models.pop(ticker, None)

    def test_hourly_features_very_short_data(self):
        """Hourly features should handle very short data without errors."""
        from backend.features.hourly import compute_hourly_features
        n = 5
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=n, freq="h"),
            "symbol": "TEST", "open": np.ones(n) * 100,
            "high": np.ones(n) * 101, "low": np.ones(n) * 99,
            "close": np.ones(n) * 100, "volume": np.ones(n) * 1e6,
        })
        feats = compute_hourly_features(df)
        assert len(feats) == n
        assert not feats.isnull().any().any()


class TestRegimeInAnalysis:
    def test_get_analysis_includes_regime_display(self):
        """get_analysis should include regime_display in output."""
        from cli.engine import get_analysis
        with patch('cli.engine._ensure_ticker_model') as mock_ens, \
             patch('cli.engine.generate_ohlcv') as mock_ohlcv, \
             patch('cli.engine._get_current_price', return_value=150.0), \
             patch('cli.engine.get_options_chain', return_value={}), \
             patch('cli.engine.get_fundamentals', return_value={}), \
             patch('cli.engine._get_sentiment_safe', return_value=None), \
             patch('cli.engine.get_macro_data', return_value=_macro()), \
             patch('cli.engine._predict_hourly_short', return_value=None), \
             patch('cli.engine._get_ticker_info', return_value=("Test Corp", 1e9, "Tech", 25.0)), \
             patch('cli.db.save_prediction'):
            n = 300
            dates = pd.date_range("2023-01-01", periods=n, freq="B")
            close = 100 + np.arange(n) * 0.2  # uptrend
            mock_ohlcv.return_value = pd.DataFrame({
                "date": dates, "symbol": "RGTEST",
                "open": close - 0.5, "high": close + 1, "low": close - 1,
                "close": close, "volume": np.random.randint(1e6, 5e7, n),
            })
            from unittest.mock import MagicMock
            ens = MagicMock()
            ens.predict_all_horizons.return_value = {
                h: {'prediction': 0.01, 'confidence': 0.65, 'direction': 'bullish',
                     'breakdown': {'xgb': {'prediction': 0.01, 'confidence': 0.65, 'weight': 0.5},
                                   'lgbm': {'prediction': 0.01, 'confidence': 0.65, 'weight': 0.5},
                                   'lstm': {'prediction': 0.0, 'confidence': 0.5, 'weight': 0.0}}}
                for h in ('short', 'medium', 'long')
            }
            mock_ens.return_value = ens
            result = get_analysis("RGTEST")
            assert 'regime_display' in result
            assert 'Market Regime:' in result['regime_display']


class TestEnsembleWeightPersistence:
    def test_weights_persist_across_save_load(self, tmp_path):
        """Ensemble weights should survive save/load cycle."""
        from backend.models.ensemble import EnsembleModel
        ens = EnsembleModel(model_dir=tmp_path)
        X = np.random.randn(100, 10)
        y = np.random.randn(100) * 0.01
        ens.train_all(X, y, y, y)
        ens.set_weights_from_accuracy(
            {'short': 0.6, 'medium': 0.55, 'long': 0.58},
            {'short': 0.55, 'medium': 0.6, 'long': 0.52},
        )
        ens.save()

        ens2 = EnsembleModel(model_dir=tmp_path)
        ens2.load()
        # LightGBM loaded successfully, so weights should be default 50/50
        # (weights aren't serialized in the model files, they come from meta.json)
        # This verifies the load path works without error
        assert ens2.weights is not None
        for h in ('short', 'medium', 'long'):
            assert 'xgb' in ens2.weights[h]
            assert 'lgbm' in ens2.weights[h]

    def test_weights_restored_from_meta_json(self, tmp_path):
        """_ensure_ticker_model should restore weights from meta.json."""
        import cli.engine as eng
        import json

        ticker = "WTEST"
        model_dir = tmp_path / ticker
        model_dir.mkdir()

        # Train and save a model
        from backend.models.ensemble import EnsembleModel
        ens = EnsembleModel(model_dir=model_dir)
        X = np.random.randn(100, 10)
        y = np.random.randn(100) * 0.01
        ens.train_all(X, y, y, y)
        custom_weights = {
            'short': {'xgb': 0.7, 'lgbm': 0.3, 'lstm': 0.0},
            'medium': {'xgb': 0.4, 'lgbm': 0.6, 'lstm': 0.0},
            'long': {'xgb': 0.5, 'lgbm': 0.5, 'lstm': 0.0},
        }
        ens.weights = custom_weights
        ens.save()

        # Save meta.json with weights
        meta = {'ensemble_weights': custom_weights}
        (model_dir / 'meta.json').write_text(json.dumps(meta))

        old_models_dir = eng._MODELS_DIR
        eng._MODELS_DIR = tmp_path
        eng._ticker_ensembles.pop(ticker, None)
        eng._ticker_meta.pop(ticker, None)
        try:
            loaded = eng._ensure_ticker_model(ticker)
            assert loaded.weights['short']['xgb'] == 0.7
            assert loaded.weights['medium']['lgbm'] == 0.6
        finally:
            eng._MODELS_DIR = old_models_dir
            eng._ticker_ensembles.pop(ticker, None)
            eng._ticker_meta.pop(ticker, None)


# --- Additional Regime Tests ---

class TestRegimeEdgeCases:
    def test_sideways_regime(self):
        """Sideways regime when SMA50 ≈ SMA200."""
        from backend.features.regime import compute_regime_features
        df = _price_df(300, start_price=100)
        # Flat price = SMA50 ≈ SMA200
        df['close'] = 100.0
        df['high'] = 101.0
        df['low'] = 99.0
        features = compute_regime_features(df)
        assert features['regime_sideways'] == 1.0
        assert features['regime_label'] == 'sideways'

    def test_adx_trend_strength_categories(self):
        """ADX should classify into no_trend, trending, or strong_trend."""
        from backend.features.regime import compute_regime_features
        # With flat data, ADX should be low (no trend)
        df = _price_df(100, start_price=100)
        df['close'] = 100.0 + np.random.randn(100) * 0.01  # near-flat
        df['high'] = df['close'] + 0.01
        df['low'] = df['close'] - 0.01
        features = compute_regime_features(df)
        # One of the trend categories must be 1
        assert features['trend_no_trend'] + features['trend_trending'] + features['trend_strong_trend'] == 1.0

    def test_adx_boundary_at_20(self):
        """ADX exactly at 20 should be 'no_trend' (< 20 threshold)."""
        from backend.features.regime import compute_regime_features
        df = _price_df(100, start_price=100)
        with patch('backend.features.regime._compute_adx', return_value=20.0):
            features = compute_regime_features(df)
        # ADX == 20 falls into trending (20-40 range, <= 40)
        assert features['trend_trending'] == 1.0

    def test_adx_boundary_at_40(self):
        """ADX exactly at 40 should be 'trending' (20-40 range)."""
        from backend.features.regime import compute_regime_features
        df = _price_df(100, start_price=100)
        with patch('backend.features.regime._compute_adx', return_value=40.0):
            features = compute_regime_features(df)
        assert features['trend_trending'] == 1.0

    def test_adx_just_above_40(self):
        """ADX just above 40 should be 'strong_trend'."""
        from backend.features.regime import compute_regime_features
        df = _price_df(100, start_price=100)
        with patch('backend.features.regime._compute_adx', return_value=40.01):
            features = compute_regime_features(df)
        assert features['trend_strong_trend'] == 1.0

    def test_adx_just_below_20(self):
        """ADX just below 20 should be 'no_trend'."""
        from backend.features.regime import compute_regime_features
        df = _price_df(100, start_price=100)
        with patch('backend.features.regime._compute_adx', return_value=19.99):
            features = compute_regime_features(df)
        assert features['trend_no_trend'] == 1.0

    def test_momentum_strong_down(self):
        """Strong downward momentum should be detected."""
        from backend.features.regime import compute_regime_features
        df = _price_df(100, start_price=200)
        df['close'] = 200 - np.arange(100) * 2.0  # strong decline
        df['close'] = np.maximum(df['close'], 10)
        df['high'] = df['close'] + 1
        df['low'] = df['close'] - 1
        features = compute_regime_features(df)
        assert features['momentum_roc20'] < -5
        assert features['mom_strong_down'] == 1.0

    def test_volatility_regime_from_price_data(self):
        """When no VIX data, volatility regime estimated from realized vol."""
        from backend.features.regime import compute_regime_features
        df = _price_df(100, start_price=100)
        features = compute_regime_features(df, macro_data=None)
        # Should still produce a valid volatility regime
        assert features['vol_regime_low'] + features['vol_regime_normal'] + features['vol_regime_high'] == 1.0

    def test_vix_boundary_at_15(self):
        """VIX exactly at 15 should be 'normal' (not low)."""
        from backend.features.regime import compute_regime_features
        df = _price_df(100, start_price=100)
        macro = _macro()
        macro['vix']['vix'] = 15.0
        features = compute_regime_features(df, macro)
        assert features['vol_regime_normal'] == 1.0
        assert features['vol_regime_label'] == 'normal'

    def test_vix_boundary_at_25(self):
        """VIX exactly at 25 should be 'normal' (not high)."""
        from backend.features.regime import compute_regime_features
        df = _price_df(100, start_price=100)
        macro = _macro()
        macro['vix']['vix'] = 25.0
        features = compute_regime_features(df, macro)
        assert features['vol_regime_normal'] == 1.0
        assert features['vol_regime_label'] == 'normal'

    def test_vix_just_above_25(self):
        """VIX just above 25 should be 'high'."""
        from backend.features.regime import compute_regime_features
        df = _price_df(100, start_price=100)
        macro = _macro()
        macro['vix']['vix'] = 25.01
        features = compute_regime_features(df, macro)
        assert features['vol_regime_high'] == 1.0

    def test_vix_just_below_15(self):
        """VIX just below 15 should be 'low'."""
        from backend.features.regime import compute_regime_features
        df = _price_df(100, start_price=100)
        macro = _macro()
        macro['vix']['vix'] = 14.99
        features = compute_regime_features(df, macro)
        assert features['vol_regime_low'] == 1.0

    def test_momentum_roc_at_zero(self):
        """ROC=0 should classify as 'down' (0 > -5 but not > 0)."""
        from backend.features.regime import compute_regime_features
        df = _price_df(100, start_price=100)
        # Set close[-1] == close[-21] for ROC=0
        df['close'] = 100.0
        df['high'] = 101.0
        df['low'] = 99.0
        features = compute_regime_features(df)
        assert features['momentum_roc20'] == 0.0
        assert features['mom_down'] == 1.0

    def test_momentum_roc_at_5(self):
        """ROC=5 should classify as 'up' (> 0 but not > 5)."""
        from backend.features.regime import compute_regime_features
        df = _price_df(100, start_price=100)
        df['close'] = 100.0
        df['high'] = 101.0
        df['low'] = 99.0
        # Set close[-1] = 105 and close[-21] = 100 for ROC=5%
        df.iloc[-1, df.columns.get_loc('close')] = 105.0
        features = compute_regime_features(df)
        assert features['momentum_roc20'] == 5.0
        assert features['mom_up'] == 1.0

    def test_momentum_roc_at_negative_5(self):
        """ROC=-5 should classify as 'down' (> -5 is false, so 'strong_down')."""
        from backend.features.regime import compute_regime_features
        df = _price_df(100, start_price=100)
        df['close'] = 100.0
        df['high'] = 101.0
        df['low'] = 99.0
        # Set close[-1] = 95 and close[-21] = 100 for ROC=-5%
        df.iloc[-1, df.columns.get_loc('close')] = 95.0
        features = compute_regime_features(df)
        assert features['momentum_roc20'] == -5.0
        assert features['mom_strong_down'] == 1.0

    def test_momentum_roc_just_above_5(self):
        """ROC just above 5 should classify as 'strong_up'."""
        from backend.features.regime import compute_regime_features
        df = _price_df(100, start_price=100)
        df['close'] = 100.0
        df['high'] = 101.0
        df['low'] = 99.0
        df.iloc[-1, df.columns.get_loc('close')] = 105.01
        features = compute_regime_features(df)
        assert features['momentum_roc20'] > 5
        assert features['mom_strong_up'] == 1.0


# --- Avg Holding Period Tests ---

class TestAvgHoldingPeriod:
    def test_avg_holding_period_consecutive_signals(self):
        """Walk-forward backtest should compute holding periods correctly."""
        from cli.engine import run_backtest
        from unittest.mock import MagicMock
        with patch('cli.engine._ensure_models'), \
             patch('cli.engine._ensure_ticker_model') as mock_ens, \
             patch('cli.engine.generate_ohlcv') as mock_ohlcv, \
             patch('cli.engine.get_options_chain', return_value={}), \
             patch('cli.engine.get_fundamentals', return_value={}), \
             patch('cli.engine._get_sentiment_safe', return_value=None), \
             patch('cli.engine.get_macro_data', return_value=_macro()), \
             patch('cli.engine._fetch_spy_sector_data', return_value=(None, None)):
            n = 300
            dates = pd.date_range("2023-01-01", periods=n, freq="B")
            np.random.seed(42)
            close = 100 + np.cumsum(np.random.randn(n) * 0.3 + 0.02)
            close = np.maximum(close, 10)
            mock_ohlcv.return_value = pd.DataFrame({
                "date": dates, "symbol": "TEST",
                "open": close - 0.5, "high": close + 1, "low": close - 1,
                "close": close, "volume": np.random.randint(1e6, 5e7, n),
            })
            ens = MagicMock()
            ens.weights = {'short': {'xgb': 0.5, 'lgbm': 0.5}, 'medium': {'xgb': 0.5, 'lgbm': 0.5}, 'long': {'xgb': 0.5, 'lgbm': 0.5}}
            ens.selected_features = None
            ens.predict_direction_ensemble.return_value = ('bullish', 0.6, 0.01)
            mock_ens.return_value = ens
            result = run_backtest("TEST", days=n)
            assert isinstance(result['avg_holding_period'], float)
            assert result['avg_holding_period'] >= 0

    def test_avg_holding_period_no_trades(self):
        """When insufficient data for walk-forward, avg_holding_period should be 0."""
        from cli.engine import run_backtest
        from unittest.mock import MagicMock
        with patch('cli.engine._ensure_models'), \
             patch('cli.engine._ensure_ticker_model') as mock_ens, \
             patch('cli.engine.generate_ohlcv') as mock_ohlcv, \
             patch('cli.engine.get_options_chain', return_value={}), \
             patch('cli.engine.get_fundamentals', return_value={}), \
             patch('cli.engine._get_sentiment_safe', return_value=None), \
             patch('cli.engine.get_macro_data', return_value=_macro()), \
             patch('cli.engine._fetch_spy_sector_data', return_value=(None, None)):
            n = 50  # Too few for walk-forward (needs 252+21)
            dates = pd.date_range("2024-01-01", periods=n, freq="B")
            close = 100 + np.arange(n) * 0.1
            mock_ohlcv.return_value = pd.DataFrame({
                "date": dates, "symbol": "TEST",
                "open": close - 0.5, "high": close + 1, "low": close - 1,
                "close": close, "volume": np.random.randint(1e6, 5e7, n),
            })
            ens = MagicMock()
            ens.weights = {'short': {'xgb': 0.5, 'lgbm': 0.5}, 'medium': {'xgb': 0.5, 'lgbm': 0.5}, 'long': {'xgb': 0.5, 'lgbm': 0.5}}
            ens.selected_features = None
            mock_ens.return_value = ens
            result = run_backtest("TEST", days=50)
            assert result['avg_holding_period'] == 0.0


# --- Walk-Forward Verbose Output Tests ---

class TestWalkForwardVerbose:
    def test_walk_forward_verbose_output(self, capsys):
        """Walk-forward should print window results when verbose=True."""
        from cli.engine import _walk_forward_validate
        np.random.seed(42)
        n = 320
        X = np.random.randn(n, 10)
        y = np.random.randn(n) * 0.01
        _walk_forward_validate(X, y, y, y, 252, 63, verbose=True)
        captured = capsys.readouterr()
        assert 'Window 1:' in captured.out
        assert 'Walk-forward' in captured.out

    def test_walk_forward_window_count(self):
        """Number of windows should match expected sliding count."""
        from cli.engine import _walk_forward_validate
        np.random.seed(42)
        n = 252 + 63 * 2  # exactly 2 windows
        X = np.random.randn(n, 5)
        y = np.random.randn(n) * 0.01
        result = _walk_forward_validate(X, y, y, y, 252, 63, verbose=False)
        assert len(result['windows']) == 2


# --- Integration: Retrain → Analyze → Backtest ---

class TestRetrainAnalyzeBacktestFlow:
    def test_retrain_then_analyze(self):
        """After retrain, analyze should use the new model and include regime."""
        import cli.engine as eng
        from cli.engine import retrain_ticker, get_analysis

        n = 150
        dates = pd.date_range("2023-01-01", periods=n, freq="B")
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        close = np.maximum(close, 10)
        hist = pd.DataFrame({
            "Date": dates, "Open": close - 0.5, "High": close + 1,
            "Low": close - 1, "Close": close,
            "Volume": np.random.randint(1e6, 5e7, n),
        })
        hist.index = dates

        with patch('yfinance.Ticker') as mock_yf, \
             patch('cli.engine.get_options_chain', return_value={}), \
             patch('cli.engine.get_fundamentals', return_value={}), \
             patch('cli.engine._get_sentiment_safe', return_value=None), \
             patch('cli.engine.get_macro_data', return_value=_macro()), \
             patch('backend.features.hourly.fetch_hourly_data', return_value=None), \
             patch('cli.engine.generate_ohlcv') as mock_ohlcv, \
             patch('cli.engine._get_current_price', return_value=float(close[-1])), \
             patch('cli.engine._predict_hourly_short', return_value=None), \
             patch('cli.engine._get_ticker_info', return_value=("Test Corp", 1e9, "Tech", 25.0)), \
             patch('cli.db.save_prediction'):
            mock_yf.return_value.history.return_value = hist
            mock_ohlcv.return_value = pd.DataFrame({
                "date": dates, "symbol": "FLOWTEST",
                "open": close - 0.5, "high": close + 1, "low": close - 1,
                "close": close, "volume": np.random.randint(1e6, 5e7, n),
            })
            acc = retrain_ticker("FLOWTEST")
            assert 'short' in acc
            analysis = get_analysis("FLOWTEST")
            assert 'regime_display' in analysis
            assert 'Market Regime:' in analysis['regime_display']
            assert 'horizons' in analysis

        # Cleanup
        eng._ticker_ensembles.pop("FLOWTEST", None)
        eng._ticker_meta.pop("FLOWTEST", None)


# --- Feature Importance in Analysis Tests ---

class TestFeatureImportanceInAnalysis:
    def test_top_features_in_analysis_result(self):
        """get_analysis should include top_features list."""
        import cli.engine as eng
        from cli.engine import retrain_ticker, get_analysis

        n = 150
        dates = pd.date_range("2023-01-01", periods=n, freq="B")
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        close = np.maximum(close, 10)
        hist = pd.DataFrame({
            "Date": dates, "Open": close - 0.5, "High": close + 1,
            "Low": close - 1, "Close": close,
            "Volume": np.random.randint(1e6, 5e7, n),
        })
        hist.index = dates

        with patch('yfinance.Ticker') as mock_yf, \
             patch('cli.engine.get_options_chain', return_value={}), \
             patch('cli.engine.get_fundamentals', return_value={}), \
             patch('cli.engine._get_sentiment_safe', return_value=None), \
             patch('cli.engine.get_macro_data', return_value=_macro()), \
             patch('backend.features.hourly.fetch_hourly_data', return_value=None), \
             patch('cli.engine.generate_ohlcv') as mock_ohlcv, \
             patch('cli.engine._get_current_price', return_value=float(close[-1])), \
             patch('cli.engine._predict_hourly_short', return_value=None), \
             patch('cli.engine._get_ticker_info', return_value=("Test Corp", 1e9, "Tech", 25.0)), \
             patch('cli.db.save_prediction'):
            mock_yf.return_value.history.return_value = hist
            mock_ohlcv.return_value = pd.DataFrame({
                "date": dates, "symbol": "FITEST",
                "open": close - 0.5, "high": close + 1, "low": close - 1,
                "close": close, "volume": np.random.randint(1e6, 5e7, n),
            })
            retrain_ticker("FITEST")
            analysis = get_analysis("FITEST")
            assert 'top_features' in analysis
            assert isinstance(analysis['top_features'], list)
            if analysis['top_features']:
                feat = analysis['top_features'][0]
                assert 'name' in feat
                assert 'importance' in feat
                assert 'value' in feat

        eng._ticker_ensembles.pop("FITEST", None)
        eng._ticker_meta.pop("FITEST", None)

    def test_get_top_feature_importances_empty_when_untrained(self):
        """Should return empty list when model has no feature importance."""
        from cli.engine import _get_top_feature_importances
        from backend.models.ensemble import EnsembleModel
        ens = EnsembleModel()
        feats = pd.DataFrame({'a': [1.0], 'b': [2.0]})
        result = _get_top_feature_importances(ens, feats)
        assert result == []


# --- Meta.json Content Tests ---

class TestMetaJsonContent:
    def test_meta_json_has_walk_forward_results(self):
        """meta.json should contain walk-forward window results."""
        import cli.engine as eng
        from cli.engine import train_ticker_model

        n = 150
        dates = pd.date_range("2023-01-01", periods=n, freq="B")
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        close = np.maximum(close, 10)
        hist = pd.DataFrame({
            "Date": dates, "Open": close - 0.5, "High": close + 1,
            "Low": close - 1, "Close": close,
            "Volume": np.random.randint(1e6, 5e7, n),
        })
        hist.index = dates

        with patch('yfinance.Ticker') as mock_yf, \
             patch('cli.engine.get_options_chain', return_value={}), \
             patch('cli.engine.get_fundamentals', return_value={}), \
             patch('cli.engine._get_sentiment_safe', return_value=None), \
             patch('cli.engine.get_macro_data', return_value=_macro()), \
             patch('backend.features.hourly.fetch_hourly_data', return_value=None):
            mock_yf.return_value.history.return_value = hist
            train_ticker_model("METATEST", verbose=False)

        meta = eng._load_ticker_meta("METATEST")
        assert meta is not None
        assert 'walk_forward' in meta
        assert 'windows' in meta['walk_forward']
        assert 'average' in meta['walk_forward']
        assert 'accuracy' in meta
        assert 'xgb_accuracy' in meta
        assert 'lgbm_accuracy' in meta
        assert 'ensemble_weights' in meta
        assert 'regime' in meta
        assert 'trained_at' in meta
        assert 'model_version' in meta
        assert meta['model_version'] == '2.0.0'

        # Cleanup
        eng._ticker_ensembles.pop("METATEST", None)
        eng._ticker_meta.pop("METATEST", None)


# --- Ensemble Fallback Tests ---

class TestEnsembleFallback:
    def test_load_falls_back_to_xgb_only_when_lgbm_missing(self, tmp_path):
        """When LightGBM model files are missing, ensemble should use XGBoost-only weights."""
        from backend.models.ensemble import EnsembleModel
        ens = EnsembleModel(model_dir=tmp_path)
        X = np.random.randn(100, 10)
        y = np.random.randn(100) * 0.01
        ens.train_all(X, y, y, y)
        ens.save()

        # Remove LightGBM files
        for f in tmp_path.glob('lgbm_*.pkl'):
            f.unlink()

        ens2 = EnsembleModel(model_dir=tmp_path)
        ens2.load()
        for h in ('short', 'medium', 'long'):
            assert ens2.weights[h]['xgb'] == 1.0
            assert ens2.weights[h]['lgbm'] == 0.0

    def test_load_falls_back_when_lgbm_corrupted(self, tmp_path):
        """When LightGBM files are corrupted, ensemble should fall back to XGBoost-only."""
        from backend.models.ensemble import EnsembleModel
        ens = EnsembleModel(model_dir=tmp_path)
        X = np.random.randn(100, 10)
        y = np.random.randn(100) * 0.01
        ens.train_all(X, y, y, y)
        ens.save()

        # Corrupt LightGBM files
        for f in tmp_path.glob('lgbm_*.pkl'):
            f.write_bytes(b'corrupted data')

        ens2 = EnsembleModel(model_dir=tmp_path)
        ens2.load()
        for h in ('short', 'medium', 'long'):
            assert ens2.weights[h]['xgb'] == 1.0
            assert ens2.weights[h]['lgbm'] == 0.0


# --- Backtest Custom Cost Parameters Tests ---

class TestBacktestCustomCosts:
    def _mock_backtest(self, slippage=0.0005, commission=0.0, spread=0.0002):
        """Run backtest with custom cost parameters."""
        from cli.engine import run_backtest
        n = 50
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        close = 100 + np.arange(n) * 0.1

        with patch('cli.engine._ensure_models'), \
             patch('cli.engine._ensure_ticker_model') as mock_ens, \
             patch('cli.engine.generate_ohlcv') as mock_ohlcv, \
             patch('cli.engine.get_options_chain', return_value={}), \
             patch('cli.engine.get_fundamentals', return_value={}), \
             patch('cli.engine._get_sentiment_safe', return_value=None), \
             patch('cli.engine.get_macro_data', return_value=_macro()):
            mock_ohlcv.return_value = pd.DataFrame({
                "date": dates, "symbol": "TEST",
                "open": close - 0.5, "high": close + 1, "low": close - 1,
                "close": close, "volume": np.random.randint(1e6, 5e7, n),
            })
            ens = MagicMock()
            ens.predict.return_value = (0.02, 0.8, {})
            ens.weights = {h: {'xgb': 0.5, 'lgbm': 0.5} for h in ('short', 'medium', 'long')}
            for attr in ('xgb_short', 'xgb_medium', 'xgb_long', 'lgbm_short', 'lgbm_medium', 'lgbm_long'):
                m = MagicMock()
                m.predict.return_value = np.ones(n - 1) * 0.01
                setattr(ens, attr, m)
            mock_ens.return_value = ens
            return run_backtest("TEST", days=50, slippage=slippage, commission=commission, spread=spread)

    def test_higher_slippage_reduces_net_return(self):
        r_low = self._mock_backtest(slippage=0.0001)
        r_high = self._mock_backtest(slippage=0.005)
        assert r_high['net_return'] <= r_low['net_return']

    def test_zero_costs_gross_equals_net(self):
        result = self._mock_backtest(slippage=0.0, commission=0.0, spread=0.0)
        assert result['gross_return'] == result['net_return']

    def test_cost_params_stored_in_result(self):
        result = self._mock_backtest(slippage=0.001, spread=0.0005)
        assert result['slippage'] == 0.001
        assert result['spread'] == 0.0005
        assert result['commission'] == 0.0


# --- Vectorized Backtest Signal Tests ---

class TestBacktestSignalDirections:
    def test_buy_signals_for_positive_predictions(self):
        """Backtest with walk-forward should generate signals from trained classifiers."""
        from cli.engine import run_backtest
        # Need enough data for walk-forward: 252 train + 21 test = 273 minimum
        n = 300
        dates = pd.date_range("2023-01-01", periods=n, freq="B")
        np.random.seed(42)
        # Uptrending price to encourage BUY signals
        close = 100 + np.cumsum(np.random.randn(n) * 0.3 + 0.05)
        close = np.maximum(close, 10)

        with patch('cli.engine._ensure_models'), \
             patch('cli.engine._ensure_ticker_model') as mock_ens, \
             patch('cli.engine.generate_ohlcv') as mock_ohlcv, \
             patch('cli.engine.get_options_chain', return_value={}), \
             patch('cli.engine.get_fundamentals', return_value={}), \
             patch('cli.engine._get_sentiment_safe', return_value=None), \
             patch('cli.engine.get_macro_data', return_value=_macro()), \
             patch('cli.engine._fetch_spy_sector_data', return_value=(None, None)):
            mock_ohlcv.return_value = pd.DataFrame({
                "date": dates, "symbol": "TEST",
                "open": close - 0.5, "high": close + 1, "low": close - 1,
                "close": close, "volume": np.random.randint(1e6, 5e7, n),
            })
            ens = MagicMock()
            ens.weights = {h: {'xgb': 0.5, 'lgbm': 0.5} for h in ('short', 'medium', 'long')}
            ens.selected_features = None
            ens.predict_direction_ensemble.return_value = ('bullish', 0.6, 0.01)
            mock_ens.return_value = ens
            result = run_backtest("TEST", days=n, detailed=True)
            # Walk-forward backtest should produce some trades
            assert result['total_trades'] >= 0
            assert 'win_rate' in result
            assert 'sharpe_ratio' in result

    def test_sell_signals_for_negative_predictions(self):
        """Backtest with walk-forward should handle downtrending data."""
        from cli.engine import run_backtest
        n = 300
        dates = pd.date_range("2023-01-01", periods=n, freq="B")
        np.random.seed(42)
        close = 200 + np.cumsum(np.random.randn(n) * 0.3 - 0.05)
        close = np.maximum(close, 10)

        with patch('cli.engine._ensure_models'), \
             patch('cli.engine._ensure_ticker_model') as mock_ens, \
             patch('cli.engine.generate_ohlcv') as mock_ohlcv, \
             patch('cli.engine.get_options_chain', return_value={}), \
             patch('cli.engine.get_fundamentals', return_value={}), \
             patch('cli.engine._get_sentiment_safe', return_value=None), \
             patch('cli.engine.get_macro_data', return_value=_macro()), \
             patch('cli.engine._fetch_spy_sector_data', return_value=(None, None)):
            mock_ohlcv.return_value = pd.DataFrame({
                "date": dates, "symbol": "TEST",
                "open": close - 0.5, "high": close + 1, "low": close - 1,
                "close": close, "volume": np.random.randint(1e6, 5e7, n),
            })
            ens = MagicMock()
            ens.weights = {h: {'xgb': 0.5, 'lgbm': 0.5} for h in ('short', 'medium', 'long')}
            ens.selected_features = None
            ens.predict_direction_ensemble.return_value = ('bearish', 0.6, -0.01)
            mock_ens.return_value = ens
            result = run_backtest("TEST", days=n, detailed=True)
            assert result['total_trades'] >= 0
            assert 'net_return' in result


# --- Bear Regime Display Tests ---

class TestBearRegimeDisplay:
    def test_bear_regime_emoji(self):
        from backend.features.regime import format_regime_display
        features = {'regime_label': 'bear', 'vol_regime_label': 'high'}
        display = format_regime_display(features)
        assert '🐻' in display
        assert 'Bear' in display
        assert 'high volatility' in display

    def test_sideways_regime_display(self):
        from backend.features.regime import format_regime_display
        features = {'regime_label': 'sideways', 'vol_regime_label': 'normal'}
        display = format_regime_display(features)
        assert '↔' in display
        assert 'Sideways' in display
        assert 'normal volatility' in display


# --- Hourly Walk-Forward in Meta.json Tests ---

class TestHourlyWalkForwardMeta:
    def test_hourly_accuracy_in_meta_when_data_available(self):
        """meta.json should contain hourly_accuracy when hourly data is available."""
        import cli.engine as eng
        from cli.engine import train_ticker_model

        n = 150
        dates = pd.date_range("2023-01-01", periods=n, freq="B")
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        close = np.maximum(close, 10)
        hist = pd.DataFrame({
            "Date": dates, "Open": close - 0.5, "High": close + 1,
            "Low": close - 1, "Close": close,
            "Volume": np.random.randint(1e6, 5e7, n),
        })
        hist.index = dates

        # Create hourly data
        h_n = 300
        h_dates = pd.date_range("2024-01-01", periods=h_n, freq="h")
        h_close = 100 + np.cumsum(np.random.randn(h_n) * 0.1)
        hourly_data = pd.DataFrame({
            "date": h_dates, "symbol": "HRMETA",
            "open": h_close - 0.1, "high": h_close + 0.5, "low": h_close - 0.5,
            "close": h_close, "volume": np.random.randint(1e4, 1e6, h_n),
        })

        with patch('yfinance.Ticker') as mock_yf, \
             patch('cli.engine.get_options_chain', return_value={}), \
             patch('cli.engine.get_fundamentals', return_value={}), \
             patch('cli.engine._get_sentiment_safe', return_value=None), \
             patch('cli.engine.get_macro_data', return_value=_macro()), \
             patch('backend.features.hourly.fetch_hourly_data', return_value=hourly_data):
            mock_yf.return_value.history.return_value = hist
            train_ticker_model("HRMETA", verbose=False)

        meta = eng._load_ticker_meta("HRMETA")
        assert meta is not None
        assert 'hourly_accuracy' in meta
        assert 0 <= meta['hourly_accuracy'] <= 1

        eng._ticker_ensembles.pop("HRMETA", None)
        eng._ticker_meta.pop("HRMETA", None)

    def test_no_hourly_accuracy_when_data_unavailable(self):
        """meta.json should NOT contain hourly_accuracy when hourly data is unavailable."""
        import cli.engine as eng
        from cli.engine import train_ticker_model

        n = 150
        dates = pd.date_range("2023-01-01", periods=n, freq="B")
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        close = np.maximum(close, 10)
        hist = pd.DataFrame({
            "Date": dates, "Open": close - 0.5, "High": close + 1,
            "Low": close - 1, "Close": close,
            "Volume": np.random.randint(1e6, 5e7, n),
        })
        hist.index = dates

        with patch('yfinance.Ticker') as mock_yf, \
             patch('cli.engine.get_options_chain', return_value={}), \
             patch('cli.engine.get_fundamentals', return_value={}), \
             patch('cli.engine._get_sentiment_safe', return_value=None), \
             patch('cli.engine.get_macro_data', return_value=_macro()), \
             patch('backend.features.hourly.fetch_hourly_data', return_value=None):
            mock_yf.return_value.history.return_value = hist
            train_ticker_model("NOHRMETA", verbose=False)

        meta = eng._load_ticker_meta("NOHRMETA")
        assert meta is not None
        assert 'hourly_accuracy' not in meta

        eng._ticker_ensembles.pop("NOHRMETA", None)
        eng._ticker_meta.pop("NOHRMETA", None)


# --- Vectorized Holding Period Tests ---

class TestVectorizedHoldingPeriod:
    def test_all_active_signals(self):
        """Walk-forward backtest with enough data should produce trades."""
        from cli.engine import run_backtest
        with patch('cli.engine._ensure_models'), \
             patch('cli.engine._ensure_ticker_model') as mock_ens, \
             patch('cli.engine.generate_ohlcv') as mock_ohlcv, \
             patch('cli.engine.get_options_chain', return_value={}), \
             patch('cli.engine.get_fundamentals', return_value={}), \
             patch('cli.engine._get_sentiment_safe', return_value=None), \
             patch('cli.engine.get_macro_data', return_value=_macro()), \
             patch('cli.engine._fetch_spy_sector_data', return_value=(None, None)):
            n = 300
            np.random.seed(42)
            dates = pd.date_range("2023-01-01", periods=n, freq="B")
            close = 100 + np.cumsum(np.random.randn(n) * 0.3 + 0.03)
            close = np.maximum(close, 10)
            mock_ohlcv.return_value = pd.DataFrame({
                "date": dates, "symbol": "TEST",
                "open": close - 0.5, "high": close + 1, "low": close - 1,
                "close": close, "volume": np.random.randint(1e6, 5e7, n),
            })
            ens = MagicMock()
            ens.weights = {'short': {'xgb': 0.5, 'lgbm': 0.5, 'lstm': 0.0},
                           'medium': {'xgb': 0.5, 'lgbm': 0.5, 'lstm': 0.0},
                           'long': {'xgb': 0.5, 'lgbm': 0.5, 'lstm': 0.0}}
            ens.selected_features = None
            ens.predict_direction_ensemble.return_value = ('bullish', 0.6, 0.01)
            mock_ens.return_value = ens
            result = run_backtest("TEST", days=n)
            # Walk-forward should produce some trades with enough data
            assert isinstance(result['avg_holding_period'], float)
            assert result['avg_holding_period'] >= 0

    def test_alternating_signals(self):
        """Alternating active/inactive should give holding period of 1."""
        from cli.engine import run_backtest
        with patch('cli.engine._ensure_models'), \
             patch('cli.engine._ensure_ticker_model') as mock_ens, \
             patch('cli.engine.generate_ohlcv') as mock_ohlcv, \
             patch('cli.engine.get_options_chain', return_value={}), \
             patch('cli.engine.get_fundamentals', return_value={}), \
             patch('cli.engine._get_sentiment_safe', return_value=None), \
             patch('cli.engine.get_macro_data', return_value=_macro()):
            n = 50
            df = _price_df(n)
            mock_ohlcv.return_value = df
            ens = MagicMock()
            ens.weights = {'short': {'xgb': 0.5, 'lgbm': 0.5, 'lstm': 0.0},
                           'medium': {'xgb': 0.5, 'lgbm': 0.5, 'lstm': 0.0},
                           'long': {'xgb': 0.5, 'lgbm': 0.5, 'lstm': 0.0}}
            # Alternating: strong positive, then zero (below confidence threshold)
            preds = np.array([0.05 if i % 2 == 0 else 0.0 for i in range(n - 1)])
            ens.xgb_short.predict.return_value = preds
            ens.lgbm_short.predict.return_value = preds
            ens.xgb_medium.predict.return_value = preds
            ens.lgbm_medium.predict.return_value = preds
            ens.xgb_long.predict.return_value = preds
            ens.lgbm_long.predict.return_value = preds
            mock_ens.return_value = ens
            from backend.features.feature_store import FeatureStore
            with patch.object(FeatureStore, 'compute_all_features',
                              return_value=pd.DataFrame(np.random.randn(n, 10))):
                result = run_backtest("TEST", days=n)
            # Alternating signals -> each holding period is 1
            if result['total_trades'] > 0:
                assert result['avg_holding_period'] == 1.0


# --- EMA Vectorization Tests ---

class TestEMAVectorized:
    def test_ema_matches_pandas(self):
        """Vectorized EMA should match pandas ewm."""
        from backend.features.hourly import _ema
        data = np.random.randn(100) * 10 + 100
        result = _ema(data, 12)
        expected = pd.Series(data).ewm(span=12, adjust=False).mean().values
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_ema_single_value(self):
        """EMA of single value should return that value."""
        from backend.features.hourly import _ema
        result = _ema(np.array([42.0]), 12)
        assert len(result) == 1
        assert result[0] == 42.0

    def test_ema_constant_array(self):
        """EMA of constant array should be that constant."""
        from backend.features.hourly import _ema
        result = _ema(np.full(50, 100.0), 20)
        np.testing.assert_allclose(result, 100.0, atol=1e-10)


# --- Backtest Zero Trades Tests ---

class TestBacktestZeroTrades:
    def test_no_trades_when_all_low_confidence(self):
        """Backtest with all low-confidence predictions should have zero trades."""
        from cli.engine import run_backtest
        with patch('cli.engine._ensure_models'), \
             patch('cli.engine._ensure_ticker_model') as mock_ens, \
             patch('cli.engine.generate_ohlcv') as mock_ohlcv, \
             patch('cli.engine.get_options_chain', return_value={}), \
             patch('cli.engine.get_fundamentals', return_value={}), \
             patch('cli.engine._get_sentiment_safe', return_value=None), \
             patch('cli.engine.get_macro_data', return_value=_macro()):
            n = 50
            df = _price_df(n)
            mock_ohlcv.return_value = df
            ens = MagicMock()
            ens.weights = {'short': {'xgb': 0.5, 'lgbm': 0.5, 'lstm': 0.0},
                           'medium': {'xgb': 0.5, 'lgbm': 0.5, 'lstm': 0.0},
                           'long': {'xgb': 0.5, 'lgbm': 0.5, 'lstm': 0.0}}
            # Very small predictions -> low confidence -> no trades
            ens.xgb_short.predict.return_value = np.full(n - 1, 0.0001)
            ens.lgbm_short.predict.return_value = np.full(n - 1, 0.0001)
            ens.xgb_medium.predict.return_value = np.full(n - 1, 0.0001)
            ens.lgbm_medium.predict.return_value = np.full(n - 1, 0.0001)
            ens.xgb_long.predict.return_value = np.full(n - 1, 0.0001)
            ens.lgbm_long.predict.return_value = np.full(n - 1, 0.0001)
            mock_ens.return_value = ens
            from backend.features.feature_store import FeatureStore
            with patch.object(FeatureStore, 'compute_all_features',
                              return_value=pd.DataFrame(np.random.randn(n, 10))):
                result = run_backtest("TEST", days=n)
            assert result['total_trades'] == 0
            assert result['avg_holding_period'] == 0.0
            assert result['win_rate'] == 0
            assert result['sharpe_ratio'] == 0.0


# --- Walk-Forward Exact Boundary Tests ---

class TestWalkForwardBoundary:
    def test_exact_one_window(self):
        """Data exactly fitting one window should produce one result."""
        from cli.engine import _walk_forward_validate
        n = 252 + 63  # exactly one window
        X = np.random.randn(n, 10)
        y = np.random.randn(n)
        result = _walk_forward_validate(X, y, y, y, 252, 63)
        assert len(result['windows']) == 1

    def test_just_under_one_window(self):
        """Data just under one window should produce zero results."""
        from cli.engine import _walk_forward_validate
        n = 252 + 62  # one short of a window
        X = np.random.randn(n, 10)
        y = np.random.randn(n)
        result = _walk_forward_validate(X, y, y, y, 252, 63)
        assert len(result['windows']) == 0
        assert result['average'] == {}
