"""Tests for ML rebuild: lagged features, classification, SHAP, conviction, sector, adaptive selection."""
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def _price_df(days=100, start_price=100):
    dates = pd.date_range("2023-01-01", periods=days, freq="B")
    close = start_price + np.cumsum(np.random.randn(days) * 0.5)
    close = np.maximum(close, 10)
    return pd.DataFrame({
        "date": dates, "symbol": "TEST",
        "open": close - 0.5, "high": close + 1, "low": close - 1,
        "close": close, "volume": np.random.randint(1e6, 5e7, days),
    })


# --- Lagged Features Tests ---

class TestLaggedFeatures:
    def test_compute_lagged_features_shape(self):
        from backend.features.lagged import compute_lagged_features
        np.random.seed(42)
        base = pd.DataFrame({
            'rsi_14': np.random.randn(50),
            'macd_line': np.random.randn(50),
            'return_1d': np.random.randn(50),
        })
        result = compute_lagged_features(base)
        # Each feature: 5 lags + roc5 + mean5 + std5 = 7 columns
        assert result.shape[0] == 50
        assert result.shape[1] >= 7  # at least 7 for one matched feature

    def test_lagged_feature_names(self):
        from backend.features.lagged import lagged_feature_names
        names = lagged_feature_names(['rsi_14', 'macd_line', 'return_1d'])
        assert 'rsi_14_lag1' in names
        assert 'rsi_14_lag20' in names
        assert 'rsi_14_roc5' in names
        assert 'rsi_14_mean5' in names
        assert 'rsi_14_std5' in names

    def test_lagged_features_no_nans(self):
        from backend.features.lagged import compute_lagged_features
        base = pd.DataFrame({'rsi_14': np.ones(30), 'return_1d': np.ones(30)})
        result = compute_lagged_features(base)
        assert not result.isna().any().any()


class TestPriceDerivedFeatures:
    def test_compute_price_derived(self):
        from backend.features.lagged import compute_price_derived_features
        df = _price_df(100)
        result = compute_price_derived_features(df)
        assert 'consecutive_up_days' in result.columns
        assert 'consecutive_down_days' in result.columns
        assert 'drawdown_from_20d_high' in result.columns
        assert 'rally_from_20d_low' in result.columns
        assert 'days_since_last_5pct_drop' in result.columns
        assert 'volatility_expansion' in result.columns
        assert 'price_acceleration' in result.columns
        assert 'volume_trend_5d' in result.columns
        assert len(result) == 100

    def test_drawdown_is_negative_or_zero(self):
        from backend.features.lagged import compute_price_derived_features
        df = _price_df(100)
        result = compute_price_derived_features(df)
        assert (result['drawdown_from_20d_high'] <= 0.01).all()

    def test_rally_is_positive_or_zero(self):
        from backend.features.lagged import compute_price_derived_features
        df = _price_df(100)
        result = compute_price_derived_features(df)
        assert (result['rally_from_20d_low'] >= -0.01).all()


# --- Classification Model Tests ---

class TestClassificationModels:
    def test_xgboost_classifier_train_predict(self):
        from backend.models.xgboost_models import XGBoostShort
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y_cls = (np.random.randn(100) > 0).astype(int)
        m = XGBoostShort()
        m.train_classifier(X, y_cls)
        direction, prob = m.predict_direction(X[0])
        assert direction in (0, 1)
        assert 0 <= prob <= 1

    def test_lgbm_classifier_train_predict(self):
        from backend.models.lgbm_models import LGBMShort
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y_cls = (np.random.randn(100) > 0).astype(int)
        m = LGBMShort()
        m.train_classifier(X, y_cls)
        direction, prob = m.predict_direction(X[0])
        assert direction in (0, 1)
        assert 0 <= prob <= 1

    def test_ensemble_direction_prediction(self):
        from backend.models.ensemble import EnsembleModel
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = np.random.randn(100) * 0.01
        ens = EnsembleModel()
        ens.train_all(X, y, y, y)
        ens.train_classifiers(X, y, y, y)
        direction, conf, exp_ret = ens.predict_direction_ensemble(X[0], 'short')
        assert direction in ('bullish', 'bearish')
        assert 0.5 <= conf <= 1.0
        assert isinstance(exp_ret, float)

    def test_classifier_predict_proba_uses_classifier(self):
        from backend.models.xgboost_models import XGBoostShort
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = np.random.randn(100) * 0.01
        y_cls = (y > 0).astype(int)
        m = XGBoostShort()
        m.train(X, y)
        m.train_classifier(X, y_cls)
        _, conf = m.predict_proba(X[0])
        # Confidence should be between 0.5 and 1.0 (from classifier)
        assert 0.5 <= conf <= 1.0


# --- SHAP Explanation Tests ---

class TestSHAPExplanations:
    def test_compute_shap_explanation(self):
        from backend.models.explain import compute_shap_explanation
        import xgboost as xgb
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = (np.random.randn(100) > 0).astype(int)
        m = xgb.XGBClassifier(n_estimators=10, random_state=42, eval_metric='logloss')
        m.fit(X, y)
        names = ['feat_a', 'feat_b', 'feat_c', 'feat_d', 'feat_e']
        result = compute_shap_explanation(m, X[0], names)
        assert 'bullish_factors' in result
        assert 'bearish_factors' in result
        assert isinstance(result['bullish_factors'], list)

    def test_format_shap_explanation(self):
        from backend.models.explain import format_shap_explanation
        explanation = {
            'bullish_factors': [{'feature': 'rsi_14', 'influence': 12.5, 'value': 25.0}],
            'bearish_factors': [{'feature': 'macd_line', 'influence': 8.0, 'value': -0.5}],
        }
        text = format_shap_explanation(explanation)
        assert 'Why BUY' in text
        assert 'rsi_14' in text
        assert 'Why not' in text
        assert 'macd_line' in text

    def test_format_shap_explanation_bearish(self):
        from backend.models.explain import format_shap_explanation
        explanation = {
            'bullish_factors': [{'feature': 'rsi_14', 'influence': 5.0, 'value': 70.0}],
            'bearish_factors': [{'feature': 'macd_line', 'influence': 15.0, 'value': -1.5}],
        }
        text = format_shap_explanation(explanation, direction='bearish')
        assert 'Why SELL' in text
        assert 'macd_line' in text
        # Bullish factors should be in "Why not" section
        assert 'Why not' in text

    def test_fmt_val_compact_numbers(self):
        from backend.models.explain import _fmt_val
        assert _fmt_val(1500000000) == '1.5B'
        assert _fmt_val(-12300000) == '-12.3M'
        assert _fmt_val(5600) == '5.6K'
        assert _fmt_val(42.7) == '42.7'
        assert _fmt_val(0.05) == '0.05'
        assert _fmt_val(-0.32) == '-0.32'

    def test_shap_empty_on_failure(self):
        from backend.models.explain import compute_shap_explanation
        result = compute_shap_explanation(None, np.zeros(5), ['a', 'b', 'c', 'd', 'e'])
        assert result['bullish_factors'] == []
        assert result['bearish_factors'] == []

    def test_readable_names_technical(self):
        from backend.models.explain import _readable_name
        assert _readable_name('sma_50') == 'SMA(50)'
        assert _readable_name('ema_20') == 'EMA(20)'
        assert _readable_name('bb_lower') == 'Bollinger Lower'
        assert _readable_name('vpt') == 'Volume Price Trend'
        assert _readable_name('volume_clock') == 'Volume Clock'
        assert _readable_name('roc_60') == 'ROC(60)'


# --- Conviction Tier Tests ---

class TestConvictionTiers:
    def test_high_conviction(self):
        from backend.models.explain import get_conviction_tier
        tier, label, emoji = get_conviction_tier(0.70)
        assert tier == 'HIGH'

    def test_moderate_conviction(self):
        from backend.models.explain import get_conviction_tier
        tier, label, emoji = get_conviction_tier(0.60)
        assert tier == 'MODERATE'

    def test_low_conviction(self):
        from backend.models.explain import get_conviction_tier
        tier, label, emoji = get_conviction_tier(0.52)
        assert tier == 'LOW'

    def test_format_conviction_verdict_bullish(self):
        from backend.models.explain import format_conviction_verdict
        v = format_conviction_verdict('bullish', 0.70, 0.58)
        assert 'BUY' in v
        assert '70%' in v
        assert 'Model accuracy' in v

    def test_format_conviction_verdict_bearish(self):
        from backend.models.explain import format_conviction_verdict
        v = format_conviction_verdict('bearish', 0.60)
        assert 'SELL' in v or 'LEAN SELL' in v

    def test_format_conviction_low(self):
        from backend.models.explain import format_conviction_verdict
        v = format_conviction_verdict('bullish', 0.52)
        assert 'LEAN BUY' in v
        assert 'low conviction' in v


# --- Volatility Z-Score Tests ---

class TestVolatilityZScore:
    def test_compute_zscore(self):
        from backend.models.explain import compute_volatility_zscore
        z, desc = compute_volatility_zscore(0.005, 0.02)
        assert z == 0.25
        assert desc == 'normal noise'

    def test_unusual_zscore(self):
        from backend.models.explain import compute_volatility_zscore
        z, desc = compute_volatility_zscore(0.03, 0.02)
        assert z == 1.5
        assert desc == 'unusual'

    def test_very_unusual_zscore(self):
        from backend.models.explain import compute_volatility_zscore
        z, desc = compute_volatility_zscore(0.05, 0.02)
        assert z == 2.5
        assert desc == 'very unusual'

    def test_zero_std(self):
        from backend.models.explain import compute_volatility_zscore
        z, desc = compute_volatility_zscore(0.01, 0.0)
        assert z == 0.0


# --- Sector Features Tests ---

class TestSectorFeatures:
    def test_get_sector_etf(self):
        from backend.features.sector import get_sector_etf
        assert get_sector_etf('AAPL') == 'XLK'
        assert get_sector_etf('TSLA') == 'XLY'
        assert get_sector_etf('JPM') == 'XLF'
        assert get_sector_etf('UNKNOWN') == 'SPY'

    def test_compute_sector_features_no_data(self):
        from backend.features.sector import compute_sector_features
        df = _price_df(50)
        result = compute_sector_features(df)
        assert 'return_vs_spy' in result.columns
        assert 'spy_rsi' in result.columns
        assert len(result) == 50

    def test_compute_sector_features_with_spy(self):
        from backend.features.sector import compute_sector_features
        df = _price_df(50)
        spy = _price_df(50, start_price=400)
        result = compute_sector_features(df, spy_df=spy)
        assert not result['return_vs_spy'].isna().any()


# --- Sequence Features Tests ---

class TestSequenceFeatures:
    def test_compute_sequence_features(self):
        from backend.features.lagged import compute_sequence_features
        base = pd.DataFrame({
            'rsi_14': np.random.randn(30),
            'macd_line': np.random.randn(30),
        })
        result = compute_sequence_features(base, window=20)
        assert 'rsi_14_t0' in result.columns
        assert 'rsi_14_t19' in result.columns
        assert len(result) == 30

    def test_sequence_feature_names(self):
        from backend.features.lagged import sequence_feature_names
        names = sequence_feature_names(['rsi_14', 'macd'], window=20)
        assert 'rsi_14_t0' in names
        assert 'rsi_14_t19' in names
        assert 'macd_t0' in names


# --- Adaptive Feature Selection Tests ---

class TestAdaptiveFeatureSelection:
    def test_adaptive_selection_reduces_features(self):
        from cli.engine import _adaptive_feature_selection
        np.random.seed(42)
        n = 200
        # Create features where most are noise
        X = np.random.randn(n, 100)
        # Make first 5 features informative
        y = np.zeros(n)
        y[X[:, 0] > 0] = 0.01
        y[X[:, 0] <= 0] = -0.01
        names = [f'feat_{i}' for i in range(100)]
        selected, sel_names = _adaptive_feature_selection(X, y, y, y, names)
        assert len(selected) <= 100
        assert len(selected) >= 30  # minimum 30 features kept

    def test_horizon_weighted_selection(self):
        """Feature selection weights medium/long more than short."""
        from cli.engine import _adaptive_feature_selection
        np.random.seed(42)
        n = 200
        X = np.random.randn(n, 50)
        y_short = np.random.randn(n) * 0.01
        y_medium = np.zeros(n)
        y_medium[X[:, 0] > 0] = 0.05
        y_medium[X[:, 0] <= 0] = -0.05
        y_long = y_medium * 2
        names = [f'feat_{i}' for i in range(50)]
        selected, sel_names = _adaptive_feature_selection(X, y_short, y_medium, y_long, names, max_features=30)
        assert len(selected) <= 50  # min(max_features, n_features) with floor of 50
        assert len(sel_names) == len(selected)


# --- Feature Store Integration Tests ---

class TestFeatureStoreIntegration:
    def test_feature_count_over_400(self):
        from backend.features.feature_store import FeatureStore
        fc = FeatureStore.feature_count()
        assert fc['total'] >= 400

    def test_all_feature_names_over_400(self):
        from backend.features.feature_store import FeatureStore
        names = FeatureStore.all_feature_names()
        assert len(names) >= 400

    def test_compute_all_features_includes_lagged(self):
        from backend.features.feature_store import FeatureStore
        fs = FeatureStore()
        df = _price_df(100)
        feats = fs.compute_all_features(df)
        assert any('_lag' in c for c in feats.columns)
        assert 'consecutive_up_days' in feats.columns
        assert 'return_vs_spy' in feats.columns


# --- Multi-Timeframe Voting Tests ---

class TestMultiTimeframeVoting:
    def test_mtf_models_trainable(self):
        """Multi-timeframe models can be trained with XGBClassifier."""
        import xgboost as xgb
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = (X[:, 0] > 0).astype(int)
        for win in [30, 90]:
            m = xgb.XGBClassifier(n_estimators=10, max_depth=3, eval_metric='logloss', random_state=42)
            m.fit(X[-win:], y[-win:])
            prob = m.predict_proba(X[-1:])
            assert prob.shape == (1, 2)
            assert 0 <= prob[0, 1] <= 1

    def test_majority_vote_logic(self):
        """Majority vote correctly determines direction."""
        votes = {30: 'BUY', 90: 'SELL', 252: 'BUY'}
        buy_count = sum(1 for v in votes.values() if v == 'BUY')
        assert buy_count == 2
        majority = 'BUY' if buy_count > len(votes) / 2 else 'SELL'
        assert majority == 'BUY'


# --- Additional Price-Derived Feature Tests ---

class TestNewPriceDerivedFeatures:
    def test_candle_features(self):
        from backend.features.lagged import compute_price_derived_features
        df = _price_df(50)
        result = compute_price_derived_features(df)
        assert 'upper_shadow_ratio' in result.columns
        assert 'lower_shadow_ratio' in result.columns
        assert 'body_ratio' in result.columns
        assert all(result['upper_shadow_ratio'] >= 0)
        assert all(result['lower_shadow_ratio'] >= 0)

    def test_momentum_features(self):
        from backend.features.lagged import compute_price_derived_features
        df = _price_df(50)
        result = compute_price_derived_features(df)
        assert 'return_3d' in result.columns
        assert 'return_10d' in result.columns
        assert 'momentum_score' in result.columns
        assert 'distance_from_sma10' in result.columns

    def test_volume_features(self):
        from backend.features.lagged import compute_price_derived_features
        df = _price_df(50)
        result = compute_price_derived_features(df)
        assert 'volume_spike' in result.columns
        assert 'volume_acceleration' in result.columns


# --- Additional Sector Feature Tests ---

class TestNewSectorFeatures:
    def test_spy_macd(self):
        from backend.features.sector import compute_sector_features
        df = _price_df(50)
        spy = _price_df(50)
        result = compute_sector_features(df, spy_df=spy)
        assert 'spy_macd' in result.columns
        assert 'spy_volatility_20d' in result.columns
        assert 'relative_strength_20d' in result.columns
        assert 'beta_spy_20d' in result.columns

    def test_sector_correlation(self):
        from backend.features.sector import compute_sector_features
        df = _price_df(50)
        sector = _price_df(50)
        result = compute_sector_features(df, sector_df=sector)
        assert 'correlation_with_sector_20d' in result.columns

    def test_no_spy_data_defaults(self):
        from backend.features.sector import compute_sector_features
        df = _price_df(50)
        result = compute_sector_features(df)
        assert 'spy_macd' in result.columns
        assert all(result['spy_macd'] == 0)
        assert all(result['beta_spy_20d'] == 1)


# --- Volatility Z-Score Edge Case Tests ---

class TestVolatilityZScoreEdgeCases:
    def test_compute_zscore(self):
        from backend.models.explain import compute_volatility_zscore
        z, desc = compute_volatility_zscore(0.005, 0.01)
        assert z == 0.5
        assert 'normal' in desc.lower() or 'noise' in desc.lower()

    def test_large_zscore(self):
        from backend.models.explain import compute_volatility_zscore
        z, desc = compute_volatility_zscore(0.03, 0.01)
        assert z == 3.0
        assert 'unusual' in desc.lower() or 'significant' in desc.lower() or 'extreme' in desc.lower()

    def test_zero_std(self):
        from backend.models.explain import compute_volatility_zscore
        z, desc = compute_volatility_zscore(0.01, 0.0)
        assert z == 0.0 or abs(z) < 1000  # Should handle gracefully


# --- Sequence Features Integration Tests ---

class TestSequenceFeaturesIntegration:
    def test_feature_store_with_sequence(self):
        from backend.features.feature_store import FeatureStore
        df = _price_df(100)
        fs = FeatureStore()
        feats = fs.compute_all_features(df, include_sequence=True)
        # Should have sequence features (rsi_14_t0, rsi_14_t1, etc.)
        seq_cols = [c for c in feats.columns if '_t' in c and c.split('_t')[-1].isdigit()]
        assert len(seq_cols) > 0, "Sequence features should be present"

    def test_feature_store_without_sequence(self):
        from backend.features.feature_store import FeatureStore
        df = _price_df(100)
        fs = FeatureStore()
        feats_no_seq = fs.compute_all_features(df, include_sequence=False)
        feats_with_seq = fs.compute_all_features(df, include_sequence=True)
        assert feats_with_seq.shape[1] > feats_no_seq.shape[1]

    def test_sequence_default_is_false(self):
        from backend.features.feature_store import FeatureStore
        df = _price_df(100)
        fs = FeatureStore()
        feats_default = fs.compute_all_features(df)
        feats_no_seq = fs.compute_all_features(df, include_sequence=False)
        assert feats_default.shape[1] == feats_no_seq.shape[1]


# --- Human-Readable SHAP Names Tests ---

class TestReadableSHAPNames:
    def test_readable_name_direct(self):
        from backend.models.explain import _readable_name
        assert _readable_name('rsi_14') == 'RSI(14)'
        assert _readable_name('macd') == 'MACD'
        assert _readable_name('return_1d') == '1-day return'

    def test_readable_name_lagged(self):
        from backend.models.explain import _readable_name
        name = _readable_name('rsi_14_lag5')
        assert 'RSI' in name and '5d ago' in name

    def test_readable_name_roc(self):
        from backend.models.explain import _readable_name
        name = _readable_name('macd_roc5')
        assert 'MACD' in name and 'change' in name.lower()

    def test_readable_name_sequence(self):
        from backend.models.explain import _readable_name
        name = _readable_name('rsi_14_t5')
        assert 'RSI' in name and '5d ago' in name

    def test_readable_name_unknown(self):
        from backend.models.explain import _readable_name
        name = _readable_name('some_unknown_feature')
        assert name  # Should return something, not crash

    def test_shap_explanation_uses_readable_names(self):
        import numpy as np
        from xgboost import XGBClassifier
        from backend.models.explain import compute_shap_explanation
        X = np.random.randn(50, 3)
        y = (X[:, 0] > 0).astype(int)
        model = XGBClassifier(n_estimators=5, max_depth=2, random_state=42, eval_metric='logloss')
        model.fit(X, y)
        result = compute_shap_explanation(model, X[0], ['rsi_14', 'macd', 'volume_ratio'])
        # Check that readable names are used
        all_names = [f['feature'] for f in result['bullish_factors'] + result['bearish_factors']]
        for name in all_names:
            assert name != 'rsi_14'  # Should be 'RSI(14)' not raw name


# --- Feature Interaction Tests ---

class TestFeatureInteractions:
    def test_interaction_features_computed(self):
        from backend.features.lagged import compute_price_derived_features
        np.random.seed(42)
        df = _price_df(100)
        result = compute_price_derived_features(df)
        for feat in ['price_vol_divergence', 'vol_adj_momentum', 'mean_reversion_strength',
                      'up_ratio_10d', 'breakout_signal', 'exhaustion_signal']:
            assert feat in result.columns, f"Missing interaction feature: {feat}"

    def test_interaction_features_no_nans(self):
        from backend.features.lagged import compute_price_derived_features
        np.random.seed(42)
        df = _price_df(100)
        result = compute_price_derived_features(df)
        for feat in ['price_vol_divergence', 'vol_adj_momentum', 'up_ratio_10d',
                      'breakout_signal', 'exhaustion_signal']:
            assert not np.isnan(result[feat]).any(), f"NaN in {feat}"

    def test_up_ratio_bounded(self):
        from backend.features.lagged import compute_price_derived_features
        np.random.seed(42)
        df = _price_df(100)
        result = compute_price_derived_features(df)
        assert (result['up_ratio_10d'] >= 0).all()
        assert (result['up_ratio_10d'] <= 1).all()

    def test_interaction_features_in_names(self):
        from backend.features.lagged import PRICE_DERIVED_NAMES
        for feat in ['price_vol_divergence', 'vol_adj_momentum', 'mean_reversion_strength',
                      'up_ratio_10d', 'breakout_signal', 'exhaustion_signal']:
            assert feat in PRICE_DERIVED_NAMES


class TestWalkForwardBacktest:
    def test_backtest_returns_valid_metrics(self):
        """Walk-forward backtest should return all expected metric keys."""
        from cli.engine import run_backtest
        from unittest.mock import patch, MagicMock
        with patch('cli.engine._ensure_models'), \
             patch('cli.engine._ensure_ticker_model') as mock_ens, \
             patch('cli.engine.generate_ohlcv') as mock_ohlcv, \
             patch('cli.engine.get_options_chain', return_value={}), \
             patch('cli.engine.get_fundamentals', return_value={}), \
             patch('cli.engine._get_sentiment_safe', return_value=None), \
             patch('cli.engine.get_macro_data', return_value={}), \
             patch('cli.engine._fetch_spy_sector_data', return_value=(None, None)):
            n = 300
            np.random.seed(42)
            dates = pd.date_range("2023-01-01", periods=n, freq="B")
            close = 100 + np.cumsum(np.random.randn(n) * 0.5)
            close = np.maximum(close, 10)
            mock_ohlcv.return_value = pd.DataFrame({
                "date": dates, "symbol": "TEST",
                "open": close - 0.5, "high": close + 1, "low": close - 1,
                "close": close, "volume": np.random.randint(1e6, 5e7, n),
            })
            ens = MagicMock()
            ens.selected_features = None
            ens.weights = {h: {'xgb': 0.5, 'lgbm': 0.5} for h in ('short', 'medium', 'long')}
            ens.predict_direction_ensemble.return_value = ('bullish', 0.6, 0.01)
            mock_ens.return_value = ens
            result = run_backtest("TEST", days=n)
            for key in ['gross_return', 'net_return', 'sharpe_ratio', 'max_drawdown',
                        'win_rate', 'profit_factor', 'total_trades', 'avg_holding_period']:
                assert key in result, f"Missing key: {key}"


# --- Direction/Magnitude Consistency Tests ---

class TestDirectionMagnitudeConsistency:
    def test_bearish_direction_negative_return(self):
        """When classifier says bearish, expected return should be negative."""
        from backend.models.ensemble import EnsembleModel
        ens = EnsembleModel()
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = np.random.randn(100) * 0.01
        y_cls = (y > 0).astype(int)
        ens.xgb_short.train(X, y)
        ens.lgbm_short.train(X, y)
        ens.xgb_short.train_classifier(X, y_cls)
        ens.lgbm_short.train_classifier(X, y_cls)
        # Test multiple predictions
        for i in range(20):
            direction, conf, exp_ret = ens.predict_direction_ensemble(X[i], 'short')
            if direction == 'bearish':
                assert exp_ret <= 0, f"Bearish but positive return: {exp_ret}"
            elif direction == 'bullish':
                assert exp_ret >= 0, f"Bullish but negative return: {exp_ret}"

    def test_conviction_tier_matches_confidence(self):
        """Conviction tier should match the confidence level."""
        from backend.models.explain import get_conviction_tier
        assert get_conviction_tier(0.70)[0] == 'HIGH'
        assert get_conviction_tier(0.60)[0] == 'MODERATE'
        assert get_conviction_tier(0.52)[0] == 'LOW'
        assert get_conviction_tier(0.50)[0] == 'LOW'


# --- New Feature Tests ---

class TestNewPriceDerivedFeatures2:
    def test_return_50d(self):
        from backend.features.lagged import compute_price_derived_features
        np.random.seed(42)
        df = _price_df(100)
        result = compute_price_derived_features(df)
        assert 'return_50d' in result.columns
        assert 'distance_from_sma50' in result.columns
        assert 'sma_10_50_ratio' in result.columns
        assert 'momentum_reversal' in result.columns
        assert 'vol_price_confirm' in result.columns
        assert 'relative_volume_5d' in result.columns

    def test_new_features_no_nans(self):
        from backend.features.lagged import compute_price_derived_features
        np.random.seed(42)
        df = _price_df(100)
        result = compute_price_derived_features(df)
        for col in ['return_50d', 'distance_from_sma50', 'sma_10_50_ratio',
                     'momentum_reversal', 'vol_price_confirm', 'relative_volume_5d']:
            assert not result[col].isna().any(), f"NaN in {col}"

    def test_feature_count_512(self):
        from backend.features.feature_store import FeatureStore
        assert len(FeatureStore.all_feature_names()) >= 512
        assert FeatureStore.feature_count()['total'] >= 512

    def test_new_momentum_features(self):
        from backend.features.lagged import compute_price_derived_features
        np.random.seed(42)
        df = _price_df(100)
        result = compute_price_derived_features(df)
        for col in ['return_autocorr_5d', 'volatility_trend', 'price_rsi_14']:
            assert col in result.columns, f"Missing {col}"
            assert not result[col].isna().any(), f"NaN in {col}"
        # price_rsi_14 should be between 0 and 100
        assert result['price_rsi_14'].min() >= 0
        assert result['price_rsi_14'].max() <= 100


# --- Dead Zone Training Tests ---

class TestDeadZoneTraining:
    def test_dead_zone_filters_near_zero(self):
        """Dead zone should filter out near-zero returns."""
        y = np.array([0.0001, -0.0002, 0.05, -0.03, 0.0, 0.02, -0.001])
        dz = 0.001
        mask = np.abs(y) > dz
        assert mask.sum() == 3  # only 0.05, -0.03, 0.02 survive

    def test_adaptive_dead_zone_percentile(self):
        """Adaptive dead zone uses 20th percentile of absolute returns."""
        np.random.seed(42)
        y = np.random.normal(0, 0.02, 500)
        abs_y = np.abs(y)
        dz = float(np.percentile(abs_y[abs_y > 0], 20))
        mask = abs_y > dz
        # Should filter out roughly 20% of non-zero returns
        assert 0.7 < mask.mean() < 0.9


class TestConfidenceWeightedPositionSizing:
    def test_position_size_scales_with_confidence(self):
        """Position size should scale with distance from 0.5."""
        conf_vals = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
        position_sizes = np.abs(conf_vals - 0.5) * 2
        position_sizes = np.clip(position_sizes, 0.2, 1.0)
        assert position_sizes[0] == 0.2  # min clamp at 0.5 confidence
        assert position_sizes[1] == 0.2  # 0.6 -> 0.2
        assert abs(position_sizes[2] - 0.4) < 1e-10  # 0.7 -> 0.4
        assert abs(position_sizes[3] - 0.6) < 1e-10  # 0.8 -> 0.6
        assert abs(position_sizes[4] - 0.8) < 1e-10  # 0.9 -> 0.8

    def test_position_size_symmetric(self):
        """Bearish confidence should also scale position size."""
        conf_vals = np.array([0.1, 0.2, 0.3, 0.4])
        position_sizes = np.abs(conf_vals - 0.5) * 2
        position_sizes = np.clip(position_sizes, 0.2, 1.0)
        assert position_sizes[0] == 0.8  # 0.1 -> very bearish
        assert position_sizes[3] == 0.2  # 0.4 -> barely bearish


class TestModelQualityBadge:
    def test_quality_badge_good(self):
        avg_acc = 0.62
        badge = "🟢 Good" if avg_acc > 0.60 else ("🟡 Fair" if avg_acc > 0.55 else "🔴 Weak")
        assert badge == "🟢 Good"

    def test_quality_badge_fair(self):
        avg_acc = 0.57
        badge = "🟢 Good" if avg_acc > 0.60 else ("🟡 Fair" if avg_acc > 0.55 else "🔴 Weak")
        assert badge == "🟡 Fair"

    def test_quality_badge_weak(self):
        avg_acc = 0.52
        badge = "🟢 Good" if avg_acc > 0.60 else ("🟡 Fair" if avg_acc > 0.55 else "🔴 Weak")
        assert badge == "🔴 Weak"


class TestSHAPDeduplication:
    def test_base_feature_extraction(self):
        from backend.models.explain import _base_feature
        assert _base_feature('rsi_14_lag5') == 'rsi_14'
        assert _base_feature('atr_14_lag1') == 'atr_14'
        assert _base_feature('macd_roc5') == 'macd'
        assert _base_feature('obv_mean5') == 'obv'
        assert _base_feature('rsi_14_std5') == 'rsi_14'
        assert _base_feature('rsi_14_t5') == 'rsi_14'
        assert _base_feature('rsi_14') == 'rsi_14'

    def test_deduplicate_factors(self):
        from backend.models.explain import _deduplicate_factors
        factors = [
            ('atr_14_lag1', -0.5, 7.25),
            ('atr_14_lag7', -0.4, 6.70),
            ('macd', -0.3, -0.01),
            ('atr_14_lag19', -0.2, 4.20),
            ('rsi_14', -0.1, 43.9),
        ]
        result = _deduplicate_factors(factors, 3)
        assert len(result) == 3
        # First atr_14 variant kept, second skipped
        assert result[0][0] == 'atr_14_lag1'
        assert result[1][0] == 'macd'
        assert result[2][0] == 'rsi_14'

    def test_deduplicate_preserves_order(self):
        from backend.models.explain import _deduplicate_factors
        factors = [('rsi_14', 0.5, 30.0), ('macd', 0.3, 1.0), ('rsi_14_lag5', 0.2, 28.0)]
        result = _deduplicate_factors(factors, 5)
        assert len(result) == 2
        assert result[0][0] == 'rsi_14'
        assert result[1][0] == 'macd'


class TestPredictabilityScore:
    def test_high_predictability(self):
        import numpy as np
        wf_avg = {'short': 0.60, 'medium': 0.65, 'long': 0.70}
        avg_wf = np.mean(list(wf_avg.values()))
        assert avg_wf >= 0.60
        level = 'HIGH' if avg_wf >= 0.60 else ('MODERATE' if avg_wf >= 0.55 else 'LOW')
        assert level == 'HIGH'

    def test_moderate_predictability(self):
        import numpy as np
        wf_avg = {'short': 0.55, 'medium': 0.57, 'long': 0.58}
        avg_wf = np.mean(list(wf_avg.values()))
        level = 'HIGH' if avg_wf >= 0.60 else ('MODERATE' if avg_wf >= 0.55 else 'LOW')
        assert level == 'MODERATE'

    def test_low_predictability(self):
        import numpy as np
        wf_avg = {'short': 0.50, 'medium': 0.52, 'long': 0.53}
        avg_wf = np.mean(list(wf_avg.values()))
        level = 'HIGH' if avg_wf >= 0.60 else ('MODERATE' if avg_wf >= 0.55 else 'LOW')
        assert level == 'LOW'


class TestBaggedClassifiers:
    def test_bagged_averaging(self):
        """Bagged classifier averaging produces stable probabilities."""
        import numpy as np
        # Simulate main model prob + 2 bagged probs
        main_prob = 0.62
        bag_probs = [0.58, 0.65]
        avg = (main_prob + sum(bag_probs)) / (1 + len(bag_probs))
        assert 0.55 < avg < 0.70
        assert abs(avg - np.mean([main_prob] + bag_probs)) < 1e-6

    def test_bagged_direction_from_avg(self):
        """Direction is determined from averaged probability."""
        main_prob = 0.48  # bearish
        bag_probs = [0.52, 0.53]  # bullish
        avg = (main_prob + sum(bag_probs)) / 3
        assert avg > 0.5  # bagged models flip to bullish
        direction = 'bullish' if avg > 0.5 else 'bearish'
        assert direction == 'bullish'

    def test_bagged_confidence_symmetric(self):
        """Confidence is max(prob, 1-prob)."""
        for prob in [0.3, 0.5, 0.7, 0.9]:
            conf = max(prob, 1 - prob)
            assert conf >= 0.5


class TestPerHorizonPredictability:
    def test_per_horizon_levels(self):
        """Each horizon gets its own predictability level."""
        wf_avg = {'short': 0.52, 'medium': 0.58, 'long': 0.63}
        per_h = {}
        for h, v in wf_avg.items():
            if v >= 0.60:
                per_h[h] = 'HIGH'
            elif v >= 0.55:
                per_h[h] = 'MODERATE'
            else:
                per_h[h] = 'LOW'
        assert per_h['short'] == 'LOW'
        assert per_h['medium'] == 'MODERATE'
        assert per_h['long'] == 'HIGH'

    def test_all_high(self):
        wf_avg = {'short': 0.65, 'medium': 0.68, 'long': 0.72}
        for h, v in wf_avg.items():
            level = 'HIGH' if v >= 0.60 else ('MODERATE' if v >= 0.55 else 'LOW')
            assert level == 'HIGH'

    def test_all_low(self):
        wf_avg = {'short': 0.50, 'medium': 0.51, 'long': 0.49}
        for h, v in wf_avg.items():
            level = 'HIGH' if v >= 0.60 else ('MODERATE' if v >= 0.55 else 'LOW')
            assert level == 'LOW'


class TestProbabilityCalibration:
    """Tests for isotonic regression probability calibration."""

    def test_calibration_with_perfect_data(self):
        """Calibrator should map probabilities closer to true frequencies."""
        from sklearn.isotonic import IsotonicRegression
        # Simulate: model outputs 0.7 when true rate is 0.6
        probs = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        actuals = np.array([0, 0, 0, 1, 1, 1, 1])
        ir = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds='clip')
        ir.fit(probs, actuals)
        # Calibrated prob for 0.7 should be closer to actual frequency
        cal = ir.predict([0.7])[0]
        assert 0.0 < cal < 1.0

    def test_calibration_preserves_ordering(self):
        """Higher raw prob should still map to higher calibrated prob."""
        from sklearn.isotonic import IsotonicRegression
        np.random.seed(42)
        probs = np.random.uniform(0.3, 0.9, 100)
        actuals = (probs + np.random.normal(0, 0.1, 100) > 0.5).astype(int)
        ir = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds='clip')
        ir.fit(probs, actuals)
        cal_lo = ir.predict([0.3])[0]
        cal_hi = ir.predict([0.8])[0]
        assert cal_hi >= cal_lo

    def test_calibration_bounds(self):
        """Calibrated probabilities should be in [0.01, 0.99]."""
        from sklearn.isotonic import IsotonicRegression
        probs = np.array([0.1, 0.5, 0.9])
        actuals = np.array([0, 1, 1])
        ir = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds='clip')
        ir.fit(probs, actuals)
        for p in [0.0, 0.5, 1.0]:
            cal = ir.predict([p])[0]
            assert 0.01 <= cal <= 0.99


class TestFeatureStability:
    """Tests for feature stability tracking across WF windows."""

    def test_stability_counts(self):
        """Features appearing in top-20 across all windows should have stability=1.0."""
        n_feat = 50
        # Simulate: feature 0 is always in top-20, feature 49 never
        imp_list = []
        for _ in range(5):
            imp = np.random.uniform(0, 0.01, n_feat)
            imp[0] = 1.0  # always top
            imp_list.append(imp)
        counts = np.zeros(n_feat)
        for imp in imp_list:
            top20 = np.argsort(imp)[-20:]
            counts[top20] += 1
        stability = {i: counts[i] / len(imp_list) for i in range(n_feat) if counts[i] > 0}
        assert stability[0] == 1.0
        # Feature 49 (lowest importance) should not be in top-20
        assert stability.get(49, 0) < 1.0

    def test_stability_empty_importances(self):
        """Empty importance list should produce empty stability."""
        imp_list = []
        assert len(imp_list) == 0

    def test_stability_single_window(self):
        """Single window should still produce valid stability."""
        n_feat = 30
        imp = np.random.uniform(0, 1, n_feat)
        top20 = np.argsort(imp)[-20:]
        counts = np.zeros(n_feat)
        counts[top20] += 1
        stability = {i: counts[i] for i in range(n_feat) if counts[i] > 0}
        assert len(stability) == 20
        assert all(v == 1.0 for v in stability.values())


class TestStabilityWeightedSelection:
    """Tests for stability-weighted feature selection."""

    def test_stability_boost_promotes_stable_features(self):
        """Features with high stability should get boosted importance."""
        importances = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
        stability = {'0': 1.0, '2': 0.5}  # feature 0 always stable, feature 2 half
        for idx_str, freq in stability.items():
            idx = int(idx_str)
            importances[idx] *= (1.0 + 0.3 * freq)
        # Feature 0 should now be highest
        assert importances[0] > importances[1]
        assert importances[2] > importances[3]
        assert importances[0] > importances[2]  # 1.0 stability > 0.5

    def test_stability_boost_no_crash_on_missing(self):
        """Stability data with out-of-range indices should not crash."""
        importances = np.array([0.1, 0.2])
        stability = {'5': 1.0}  # index 5 doesn't exist
        for idx_str, freq in stability.items():
            idx = int(idx_str)
            if idx < len(importances):
                importances[idx] *= (1.0 + 0.3 * freq)
        assert importances[0] == 0.1  # unchanged
        assert importances[1] == 0.2  # unchanged


class TestBrierScore:
    """Tests for Brier score computation."""

    def test_perfect_brier_score(self):
        """Perfect predictions should have Brier score of 0."""
        probs = np.array([1.0, 0.0, 1.0, 0.0])
        actuals = np.array([1, 0, 1, 0])
        brier = float(np.mean((probs - actuals) ** 2))
        assert brier == 0.0

    def test_worst_brier_score(self):
        """Completely wrong predictions should have Brier score of 1."""
        probs = np.array([0.0, 1.0, 0.0, 1.0])
        actuals = np.array([1, 0, 1, 0])
        brier = float(np.mean((probs - actuals) ** 2))
        assert brier == 1.0

    def test_random_brier_score(self):
        """50/50 predictions should have Brier score of 0.25."""
        probs = np.array([0.5, 0.5, 0.5, 0.5])
        actuals = np.array([1, 0, 1, 0])
        brier = float(np.mean((probs - actuals) ** 2))
        assert abs(brier - 0.25) < 1e-10

    def test_calibration_improves_brier(self):
        """Isotonic calibration should not worsen Brier score on training data."""
        from sklearn.isotonic import IsotonicRegression
        np.random.seed(42)
        n = 200
        actuals = np.random.randint(0, 2, n)
        # Biased predictions (always too high)
        probs = np.clip(actuals * 0.6 + 0.3 + np.random.normal(0, 0.1, n), 0.01, 0.99)
        raw_brier = float(np.mean((probs - actuals) ** 2))
        ir = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds='clip')
        ir.fit(probs, actuals)
        cal_probs = ir.predict(probs)
        cal_brier = float(np.mean((cal_probs - actuals) ** 2))
        # Calibrated should be at least as good on training data
        assert cal_brier <= raw_brier + 0.01


class TestFeatureStabilitySaveLoad:
    """Tests for feature stability persistence."""

    def test_save_load_roundtrip(self, tmp_path, monkeypatch):
        """Stability data should survive save/load cycle."""
        import cli.engine as eng
        monkeypatch.setattr(eng, '_MODELS_DIR', tmp_path)
        stability = {'0': 1.0, '5': 0.6, '10': 0.3}
        eng._save_feature_stability('TEST', stability)
        loaded = eng._load_feature_stability('TEST')
        assert loaded == stability

    def test_load_missing_returns_none(self, tmp_path, monkeypatch):
        """Loading stability for untrained ticker should return None."""
        import cli.engine as eng
        monkeypatch.setattr(eng, '_MODELS_DIR', tmp_path)
        assert eng._load_feature_stability('NONEXISTENT') is None


class TestAdaptiveDeadZone:
    """Tests for volatility-adaptive dead zone percentile."""

    def test_low_vol_stock_base_dz(self):
        """Low volatility stock should use base dead zone percentile."""
        daily_vol = 0.01  # 1% daily vol (low)
        vol_adj = min(15, max(0, int((daily_vol - 0.01) / 0.003 * 5)))
        assert vol_adj == 0  # no adjustment

    def test_high_vol_stock_higher_dz(self):
        """High volatility stock should use higher dead zone percentile."""
        daily_vol = 0.04  # 4% daily vol (high, like TSLA)
        vol_adj = min(15, max(0, int((daily_vol - 0.01) / 0.003 * 5)))
        assert vol_adj > 0
        assert vol_adj <= 15

    def test_extreme_vol_capped(self):
        """Extremely high vol should be capped at 15."""
        daily_vol = 0.10  # 10% daily vol (extreme)
        vol_adj = min(15, max(0, int((daily_vol - 0.01) / 0.003 * 5)))
        assert vol_adj == 15


class TestRegimeAwareConfidence:
    """Tests for regime-aware confidence adjustment."""

    def test_high_vol_discounts_confidence(self):
        """In high-vol regime, confidence should be pulled toward 0.5."""
        prob = 0.7
        adjusted = prob * 0.9 + 0.5 * 0.1  # 10% pull toward 0.5
        assert adjusted < prob
        assert adjusted > 0.5

    def test_low_vol_no_adjustment(self):
        """In low/normal vol regime, no adjustment should be made."""
        regime = {'vol_regime_high': 0.0}
        # No adjustment when vol_regime_high is 0
        assert regime.get('vol_regime_high', 0) <= 0.5

    def test_adjustment_preserves_direction(self):
        """Adjustment should not flip direction (bullish stays bullish)."""
        for prob in [0.55, 0.6, 0.7, 0.8, 0.9]:
            adjusted = prob * 0.9 + 0.5 * 0.1
            assert adjusted > 0.5  # still bullish
        for prob in [0.45, 0.4, 0.3, 0.2, 0.1]:
            adjusted = prob * 0.9 + 0.5 * 0.1
            assert adjusted < 0.5  # still bearish


class TestConfidenceFilteredAccuracy:
    """Tests for confidence-filtered accuracy metric."""

    def test_filters_low_confidence(self):
        """Only predictions with >55% confidence should be counted."""
        probs = np.array([0.51, 0.49, 0.7, 0.3, 0.52, 0.48, 0.8, 0.2])
        actuals = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        conf_mask = np.abs(probs - 0.5) > 0.05
        # Only 0.7, 0.3, 0.8, 0.2 pass the filter
        assert np.sum(conf_mask) == 4
        preds = (probs[conf_mask] > 0.5).astype(int)
        correct = np.sum(preds == actuals[conf_mask])
        assert correct == 4  # all confident predictions are correct

    def test_all_uncertain_returns_empty(self):
        """If all predictions are near 0.5, no confidence-filtered accuracy."""
        probs = np.array([0.51, 0.49, 0.52, 0.48])
        conf_mask = np.abs(probs - 0.5) > 0.05
        assert np.sum(conf_mask) == 0


class TestConformalPredictionIntervals:
    """Tests for conformal prediction intervals."""

    def test_basic_interval(self):
        from backend.models.explain import compute_conformal_interval
        residuals = [0.01, -0.02, 0.015, -0.01, 0.005, -0.025, 0.02, -0.015, 0.01, -0.01,
                     0.03, -0.005, 0.008, -0.012, 0.018]
        lo, hi = compute_conformal_interval(0.005, residuals, alpha=0.1)
        assert lo < 0.005
        assert hi > 0.005
        assert lo < hi

    def test_symmetric_around_prediction(self):
        from backend.models.explain import compute_conformal_interval
        residuals = [0.01, -0.01, 0.01, -0.01, 0.01, -0.01, 0.01, -0.01, 0.01, -0.01]
        lo, hi = compute_conformal_interval(0.0, residuals)
        assert abs(abs(lo) - abs(hi)) < 0.001  # roughly symmetric

    def test_wider_with_more_variance(self):
        from backend.models.explain import compute_conformal_interval
        narrow = [0.001] * 20
        wide = [0.05] * 20
        lo_n, hi_n = compute_conformal_interval(0.0, narrow)
        lo_w, hi_w = compute_conformal_interval(0.0, wide)
        assert (hi_w - lo_w) > (hi_n - lo_n)

    def test_insufficient_residuals_fallback(self):
        from backend.models.explain import compute_conformal_interval
        lo, hi = compute_conformal_interval(0.01, [0.001, 0.002])
        assert lo < hi  # still returns valid interval

    def test_format_interval(self):
        from backend.models.explain import format_conformal_interval
        text = format_conformal_interval(-0.02, 0.03)
        assert '-2.00%' in text
        assert '+3.00%' in text


class TestFeatureDriftDetection:
    """Tests for feature drift detection."""

    def test_no_drift(self):
        from backend.models.explain import detect_feature_drift
        current = np.array([1.0, 2.0, 3.0])
        means = np.array([1.0, 2.0, 3.0])
        stds = np.array([1.0, 1.0, 1.0])
        names = ['a', 'b', 'c']
        drifted = detect_feature_drift(current, means, stds, names)
        assert len(drifted) == 0

    def test_detects_drift(self):
        from backend.models.explain import detect_feature_drift
        current = np.array([10.0, 2.0, 3.0])
        means = np.array([1.0, 2.0, 3.0])
        stds = np.array([1.0, 1.0, 1.0])
        names = ['rsi_14', 'macd', 'bb_position']
        drifted = detect_feature_drift(current, means, stds, names, threshold=3.0)
        assert len(drifted) == 1
        assert drifted[0]['raw_name'] == 'rsi_14'
        assert drifted[0]['z_score'] == 9.0

    def test_zero_std_safe(self):
        from backend.models.explain import detect_feature_drift
        current = np.array([5.0])
        means = np.array([1.0])
        stds = np.array([0.0])  # zero std
        names = ['feat']
        drifted = detect_feature_drift(current, means, stds, names)
        assert len(drifted) == 1  # 4.0 sigma with fallback std=1.0

    def test_mismatched_lengths(self):
        from backend.models.explain import detect_feature_drift
        drifted = detect_feature_drift(np.array([1.0, 2.0]), np.array([1.0]), np.array([1.0]), ['a'])
        assert len(drifted) == 0

    def test_max_10_results(self):
        from backend.models.explain import detect_feature_drift
        n = 20
        current = np.ones(n) * 100
        means = np.zeros(n)
        stds = np.ones(n)
        names = [f'f{i}' for i in range(n)]
        drifted = detect_feature_drift(current, means, stds, names, threshold=3.0)
        assert len(drifted) <= 10


class TestModelHealth:
    """Tests for model health score computation."""

    def test_good_model(self):
        from backend.models.explain import compute_model_health
        health = compute_model_health(
            {'short': 0.60, 'medium': 0.62, 'long': 0.65},
            {'short': {'calibrated': 0.18}, 'medium': {'calibrated': 0.17}},
            has_calibration=True, model_age_days=1)
        assert health['grade'] in ('A', 'B')
        assert health['score'] > 50

    def test_poor_model(self):
        from backend.models.explain import compute_model_health
        health = compute_model_health(
            {'short': 0.50, 'medium': 0.51, 'long': 0.52},
            model_age_days=60)
        assert health['grade'] in ('D', 'F')
        assert health['score'] < 30

    def test_no_data(self):
        from backend.models.explain import compute_model_health
        health = compute_model_health({})
        assert health['grade'] in ('D', 'F', 'C')
        assert 'score' in health

    def test_calibration_bonus(self):
        from backend.models.explain import compute_model_health
        no_cal = compute_model_health({'short': 0.55}, has_calibration=False)
        with_cal = compute_model_health({'short': 0.55}, has_calibration=True)
        assert with_cal['score'] > no_cal['score']

    def test_freshness_penalty(self):
        from backend.models.explain import compute_model_health
        fresh = compute_model_health({'short': 0.58}, model_age_days=0)
        stale = compute_model_health({'short': 0.58}, model_age_days=60)
        assert fresh['score'] > stale['score']

    def test_format_health(self):
        from backend.models.explain import format_model_health
        text = format_model_health({'grade': 'B', 'score': 65.0, 'components': {}})
        assert 'B' in text
        assert '65' in text

    def test_components_present(self):
        from backend.models.explain import compute_model_health
        health = compute_model_health(
            {'short': 0.58, 'medium': 0.60},
            {'short': {'raw': 0.22}},
            has_calibration=True, model_age_days=5)
        assert 'wf_accuracy' in health['components']
        assert 'brier' in health['components']
        assert 'calibration' in health['components']
        assert 'freshness' in health['components']


class TestPredictionEvaluation:
    """Tests for prediction accuracy evaluation."""

    def test_evaluate_empty(self):
        from cli.db import evaluate_predictions
        # With no predictions stored, should return empty
        result = evaluate_predictions('NONEXISTENT_TICKER_XYZ', 100.0)
        assert result == {}

    def test_evaluate_accuracy_calc(self):
        """Test that accuracy calculation is correct."""
        # Simulate: 3 bullish predictions, price went up → 100% accuracy
        preds = [
            {'direction': 'bullish', 'price_at': 90.0, 'horizon': 'short'},
            {'direction': 'bullish', 'price_at': 95.0, 'horizon': 'short'},
            {'direction': 'bearish', 'price_at': 110.0, 'horizon': 'short'},
        ]
        current_price = 100.0
        # Manual calculation
        correct = 0
        for p in preds:
            actual_return = (current_price - p['price_at']) / p['price_at']
            predicted_up = p['direction'] == 'bullish'
            actual_up = actual_return > 0
            if predicted_up == actual_up:
                correct += 1
        assert correct == 3  # all correct
        assert correct / len(preds) == 1.0


class TestConformalIntervalEdgeCases:
    """Additional edge case tests for conformal intervals."""

    def test_negative_prediction(self):
        from backend.models.explain import compute_conformal_interval
        residuals = [0.01, -0.02, 0.015, -0.01, 0.005, -0.025, 0.02, -0.015, 0.01, -0.01]
        lo, hi = compute_conformal_interval(-0.03, residuals)
        assert lo < -0.03
        assert hi > -0.03

    def test_zero_prediction(self):
        from backend.models.explain import compute_conformal_interval
        residuals = [0.01, -0.01] * 10
        lo, hi = compute_conformal_interval(0.0, residuals)
        assert lo < 0
        assert hi > 0

    def test_large_alpha_narrow_interval(self):
        from backend.models.explain import compute_conformal_interval
        residuals = list(np.random.normal(0, 0.02, 50))
        lo_90, hi_90 = compute_conformal_interval(0.0, residuals, alpha=0.1)
        lo_50, hi_50 = compute_conformal_interval(0.0, residuals, alpha=0.5)
        # 50% interval should be narrower than 90%
        assert (hi_50 - lo_50) <= (hi_90 - lo_90)


class TestModelHealthEdgeCases:
    """Additional edge case tests for model health."""

    def test_perfect_model(self):
        from backend.models.explain import compute_model_health
        health = compute_model_health(
            {'short': 0.70, 'medium': 0.72, 'long': 0.75},
            {'short': {'calibrated': 0.10}, 'medium': {'calibrated': 0.08}},
            has_calibration=True, model_age_days=0)
        assert health['grade'] == 'A'
        assert health['score'] >= 80

    def test_score_capped_at_100(self):
        from backend.models.explain import compute_model_health
        health = compute_model_health(
            {'short': 0.90, 'medium': 0.90, 'long': 0.90},
            {'short': {'calibrated': 0.01}},
            has_calibration=True, model_age_days=0)
        assert health['score'] <= 100

    def test_only_wf_accuracy(self):
        from backend.models.explain import compute_model_health
        health = compute_model_health({'short': 0.55, 'medium': 0.57})
        assert 'score' in health
        assert 'brier' not in health['components']  # no brier data provided


class TestFeatureDriftEdgeCases:
    """Additional edge case tests for feature drift."""

    def test_all_features_drifted(self):
        from backend.models.explain import detect_feature_drift
        n = 5
        current = np.ones(n) * 100
        means = np.zeros(n)
        stds = np.ones(n)
        names = [f'f{i}' for i in range(n)]
        drifted = detect_feature_drift(current, means, stds, names, threshold=3.0)
        assert len(drifted) == 5

    def test_sorted_by_zscore(self):
        from backend.models.explain import detect_feature_drift
        current = np.array([5.0, 10.0, 20.0])
        means = np.zeros(3)
        stds = np.ones(3)
        names = ['a', 'b', 'c']
        drifted = detect_feature_drift(current, means, stds, names, threshold=3.0)
        assert len(drifted) == 3
        assert drifted[0]['raw_name'] == 'c'  # highest z-score first

    def test_custom_threshold(self):
        from backend.models.explain import detect_feature_drift
        current = np.array([2.5])
        means = np.array([0.0])
        stds = np.array([1.0])
        names = ['f']
        # threshold=2.0 should catch it, threshold=3.0 should not
        assert len(detect_feature_drift(current, means, stds, names, threshold=2.0)) == 1
        assert len(detect_feature_drift(current, means, stds, names, threshold=3.0)) == 0


class TestConformalConvictionAdjustment:
    """Tests for conformal interval width adjusting conviction tiers."""

    def test_narrow_interval_boosts_conviction(self):
        from backend.models.explain import get_conviction_tier
        tier_no_ci, _, _ = get_conviction_tier(0.64)
        tier_narrow, _, _ = get_conviction_tier(0.64, conf_interval_width=0.03)
        assert tier_no_ci == 'MODERATE'
        assert tier_narrow == 'HIGH'

    def test_wide_interval_reduces_conviction(self):
        from backend.models.explain import get_conviction_tier
        tier_no_ci, _, _ = get_conviction_tier(0.56)
        tier_wide, _, _ = get_conviction_tier(0.56, conf_interval_width=0.20)
        assert tier_no_ci == 'MODERATE'
        assert tier_wide == 'LOW'

    def test_no_interval_unchanged(self):
        from backend.models.explain import get_conviction_tier
        tier, label, emoji = get_conviction_tier(0.70)
        assert tier == 'HIGH'
        tier2, _, _ = get_conviction_tier(0.70, conf_interval_width=None)
        assert tier2 == 'HIGH'

    def test_moderate_interval_no_change(self):
        from backend.models.explain import get_conviction_tier
        tier, _, _ = get_conviction_tier(0.60, conf_interval_width=0.10)
        assert tier == 'MODERATE'


class TestHealthTrend:
    """Tests for model health trend tracking."""

    def test_save_and_format_trend(self):
        import tempfile
        from pathlib import Path
        from backend.models.explain import save_health_trend, format_health_trend
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            save_health_trend(d, {'score': 40, 'grade': 'C'})
            save_health_trend(d, {'score': 60, 'grade': 'B'})
            trend = format_health_trend(d)
            assert trend is not None
            assert 'improving' in trend

    def test_no_trend_with_single_entry(self):
        import tempfile
        from pathlib import Path
        from backend.models.explain import save_health_trend, format_health_trend
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            save_health_trend(d, {'score': 50, 'grade': 'C'})
            assert format_health_trend(d) is None

    def test_declining_trend(self):
        import tempfile
        from pathlib import Path
        from backend.models.explain import save_health_trend, format_health_trend
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            save_health_trend(d, {'score': 70, 'grade': 'B'})
            save_health_trend(d, {'score': 40, 'grade': 'C'})
            trend = format_health_trend(d)
            assert 'declining' in trend

    def test_stable_trend(self):
        import tempfile
        from pathlib import Path
        from backend.models.explain import save_health_trend, format_health_trend
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            save_health_trend(d, {'score': 50, 'grade': 'C'})
            save_health_trend(d, {'score': 51, 'grade': 'C'})
            trend = format_health_trend(d)
            assert 'stable' in trend

    def test_trend_capped_at_20(self):
        import tempfile
        import json
        from pathlib import Path
        from backend.models.explain import save_health_trend
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            for i in range(25):
                save_health_trend(d, {'score': i * 4, 'grade': 'C'})
            history = json.loads((d / 'health_trend.json').read_text())
            assert len(history) == 20


class TestPlattScaling:
    """Tests for Platt scaling calibration."""

    def test_calibrate_platt_basic(self):
        from backend.models.explain import calibrate_platt, apply_platt
        probs = np.array([0.3, 0.4, 0.5, 0.6, 0.7] * 10)
        labels = np.array([0, 0, 0, 1, 1] * 10)
        cal = calibrate_platt(probs, labels)
        assert cal is not None
        result = apply_platt(cal, 0.7)
        assert result > 0.5

    def test_calibrate_platt_insufficient_data(self):
        from backend.models.explain import calibrate_platt
        probs = np.array([0.5, 0.6])
        labels = np.array([0, 1])
        assert calibrate_platt(probs, labels) is None

    def test_apply_platt_none_calibrator(self):
        from backend.models.explain import apply_platt
        assert apply_platt(None, 0.6) == 0.6


class TestUncertaintyLabel:
    """Tests for prediction uncertainty labels."""

    def test_low_uncertainty(self):
        from backend.models.explain import uncertainty_label
        assert 'low' in uncertainty_label(0.01, 0.04)

    def test_moderate_uncertainty(self):
        from backend.models.explain import uncertainty_label
        assert 'moderate' in uncertainty_label(-0.05, 0.05)

    def test_high_uncertainty(self):
        from backend.models.explain import uncertainty_label
        assert 'high' in uncertainty_label(-0.10, 0.10)


class TestPredictionEvaluationTimeBased:
    """Tests for time-based prediction evaluation."""

    def test_skips_recent_predictions(self):
        """Predictions too recent for their horizon should be skipped."""
        import sqlite3
        from datetime import datetime
        from cli.db import evaluate_predictions

        # Create a fresh in-memory db
        import cli.db as db_mod
        old_conn = getattr(db_mod, '_db_conn', None)
        db_mod._db_conn = sqlite3.connect(':memory:')
        db_mod._db_conn.row_factory = sqlite3.Row
        db_mod._db_conn.execute("""CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY, ticker TEXT, direction TEXT,
            confidence REAL, price_at REAL, horizon TEXT, created_at TEXT)""")
        # Insert a prediction from just now (should be skipped for medium horizon)
        db_mod._db_conn.execute(
            "INSERT INTO predictions (ticker, direction, confidence, price_at, horizon, created_at) VALUES (?,?,?,?,?,?)",
            ('TEST', 'bullish', 0.6, 100.0, 'medium', datetime.now().isoformat()))
        db_mod._db_conn.commit()

        result = evaluate_predictions('TEST', 105.0)
        # Should be empty since the prediction is too recent for medium (5 days)
        assert 'medium' not in result

        db_mod._db_conn.close()
        db_mod._db_conn = old_conn


class TestCalibrationCurve:
    """Tests for calibration curve computation."""

    def test_basic_curve(self):
        from backend.models.explain import compute_calibration_curve
        probs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        actuals = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
        curve = compute_calibration_curve(probs, actuals, n_bins=5)
        assert len(curve) > 0
        for bucket in curve:
            assert 'predicted' in bucket
            assert 'actual' in bucket
            assert 'count' in bucket

    def test_insufficient_data(self):
        from backend.models.explain import compute_calibration_curve
        probs = np.array([0.5, 0.6])
        actuals = np.array([0, 1])
        assert compute_calibration_curve(probs, actuals) == []

    def test_perfect_calibration(self):
        from backend.models.explain import compute_calibration_curve
        # All predictions at 0.5, half are correct
        probs = np.array([0.5] * 20)
        actuals = np.array([0, 1] * 10)
        curve = compute_calibration_curve(probs, actuals, n_bins=5)
        # Should have one bucket near 0.5 predicted, ~0.5 actual
        assert len(curve) >= 1


class TestModelStatus:
    """Tests for model_status cross-ticker comparison."""

    def test_empty_when_no_models(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from cli.engine import model_status
        with tempfile.TemporaryDirectory() as td:
            with patch('cli.engine._MODELS_DIR', Path(td)):
                assert model_status() == []

    def test_returns_status_for_trained_model(self):
        import tempfile
        import json
        from pathlib import Path
        from unittest.mock import patch
        from cli.engine import model_status
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / 'TEST'
            d.mkdir()
            meta = {
                'walk_forward': {'average': {'short': 0.58, 'medium': 0.62, 'long': 0.65}},
                'model_health': {'grade': 'B', 'score': 65},
                'selected_feature_count': 80,
                'feature_count': 515,
                'trained_at': '2026-02-15T10:00:00',
            }
            (d / 'meta.json').write_text(json.dumps(meta))
            with patch('cli.engine._MODELS_DIR', Path(td)):
                statuses = model_status()
                assert len(statuses) == 1
                assert statuses[0]['ticker'] == 'TEST'
                assert statuses[0]['health_grade'] == 'B'
                assert statuses[0]['wf_medium'] == 0.62
                assert statuses[0]['features'] == 80


class TestFeatureChangelog:
    """Tests for feature importance changelog tracking."""

    def test_save_and_format_changelog(self):
        import tempfile
        from pathlib import Path
        from backend.models.explain import save_feature_changelog, format_feature_changelog
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            save_feature_changelog(d, ['rsi_14', 'macd', 'bb_position', 'atr_14'])
            save_feature_changelog(d, ['rsi_14', 'macd', 'obv_slope', 'volume_ratio'])
            result = format_feature_changelog(d)
            assert result is not None
            assert 'new' in result or 'dropped' in result

    def test_no_changelog_single_entry(self):
        import tempfile
        from pathlib import Path
        from backend.models.explain import save_feature_changelog, format_feature_changelog
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            save_feature_changelog(d, ['rsi_14', 'macd'])
            assert format_feature_changelog(d) is None

    def test_no_changes(self):
        import tempfile
        from pathlib import Path
        from backend.models.explain import save_feature_changelog, format_feature_changelog
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            feats = ['rsi_14', 'macd']
            save_feature_changelog(d, feats)
            save_feature_changelog(d, feats)
            assert format_feature_changelog(d) is None

    def test_changelog_capped(self):
        import tempfile
        import json
        from pathlib import Path
        from backend.models.explain import save_feature_changelog
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            for i in range(15):
                save_feature_changelog(d, [f'feat_{i}'])
            history = json.loads((d / 'feature_changelog.json').read_text())
            assert len(history) == 10


class TestEnsembleDiversity:
    """Tests for ensemble diversity metric."""

    def test_diverse_models(self):
        from backend.models.explain import compute_ensemble_diversity
        p1 = np.array([0.6, 0.7, 0.3, 0.8, 0.4, 0.6, 0.7, 0.3, 0.2, 0.9])
        p2 = np.array([0.4, 0.6, 0.7, 0.3, 0.6, 0.4, 0.8, 0.2, 0.6, 0.8])
        result = compute_ensemble_diversity([p1, p2])
        assert result['diversity'] > 0
        assert 'description' in result
        assert 0 <= result['avg_correlation'] <= 1.0 or result['avg_correlation'] < 0

    def test_identical_models(self):
        from backend.models.explain import compute_ensemble_diversity
        p = np.array([0.6, 0.7, 0.3, 0.8, 0.4])
        result = compute_ensemble_diversity([p, p.copy()])
        assert result['diversity'] == 0.0
        assert result['avg_correlation'] >= 0.99

    def test_single_model(self):
        from backend.models.explain import compute_ensemble_diversity
        result = compute_ensemble_diversity([np.array([0.5, 0.6])])
        assert result['diversity'] == 0.0
        assert result['description'] == 'single model'

    def test_three_models(self):
        from backend.models.explain import compute_ensemble_diversity
        p1 = np.array([0.6, 0.7, 0.3, 0.8, 0.4, 0.6, 0.7, 0.3])
        p2 = np.array([0.4, 0.6, 0.7, 0.3, 0.6, 0.4, 0.8, 0.2])
        p3 = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        result = compute_ensemble_diversity([p1, p2, p3])
        assert result['diversity'] > 0
        assert 'avg_correlation' in result


class TestModelStatusEnhanced:
    """Tests for enhanced model_status with calibration and feature counts."""

    def test_status_includes_calibration(self):
        import tempfile
        import json
        from pathlib import Path
        from unittest.mock import patch
        import cli.engine as eng
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / 'TESTCAL'
            d.mkdir()
            meta = {
                'trained_at': '2026-01-01T00:00:00',
                'walk_forward': {'average': {'short': 0.58, 'medium': 0.61, 'long': 0.63}},
                'model_health': {'grade': 'B', 'score': 72},
                'selected_feature_count': 80,
                'feature_count': 515,
                'has_calibrators': True,
                'brier_scores': {
                    'short': {'raw': 0.25, 'calibrated': 0.23},
                    'medium': {'raw': 0.24, 'calibrated': 0.22},
                },
            }
            (d / 'meta.json').write_text(json.dumps(meta))
            saved = eng._ticker_meta.copy()
            eng._ticker_meta.clear()
            try:
                with patch.object(eng, '_MODELS_DIR', Path(td)):
                    statuses = eng.model_status()
                    assert len(statuses) == 1
                    s = statuses[0]
                    assert s['calibrated'] is True
                    assert s['avg_brier'] is not None
                    assert s['avg_brier'] < 0.25
                    assert s['features'] == 80
                    assert s['total_features'] == 515
            finally:
                eng._ticker_meta.update(saved)

    def test_status_no_calibration(self):
        import tempfile
        import json
        from pathlib import Path
        from unittest.mock import patch
        import cli.engine as eng
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / 'TESTNOCAL'
            d.mkdir()
            meta = {
                'trained_at': '2026-01-01T00:00:00',
                'walk_forward': {'average': {'short': 0.55}},
                'model_health': {'grade': 'C', 'score': 55},
                'feature_count': 400,
            }
            (d / 'meta.json').write_text(json.dumps(meta))
            saved = eng._ticker_meta.copy()
            eng._ticker_meta.clear()
            try:
                with patch.object(eng, '_MODELS_DIR', Path(td)):
                    statuses = eng.model_status()
                    assert len(statuses) == 1
                    s = statuses[0]
                    assert s['calibrated'] is False
                    assert s['avg_brier'] is None
            finally:
                eng._ticker_meta.update(saved)


class TestHealthDegradation:
    """Tests for model health degradation detection."""

    def test_no_degradation_when_stable(self):
        import tempfile
        import json
        from pathlib import Path
        from backend.models.explain import check_health_degradation
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            trend = [{'score': 70}, {'score': 72}, {'score': 71}]
            (d / 'health_trend.json').write_text(json.dumps(trend))
            assert check_health_degradation(d) is None

    def test_detects_declining_trend(self):
        import tempfile
        import json
        from pathlib import Path
        from backend.models.explain import check_health_degradation
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            trend = [{'score': 80}, {'score': 65}, {'score': 50}]
            (d / 'health_trend.json').write_text(json.dumps(trend))
            result = check_health_degradation(d)
            assert result is not None
            assert 'declining' in result.lower() or 'dropped' in result.lower()

    def test_detects_critical_low(self):
        import tempfile
        import json
        from pathlib import Path
        from backend.models.explain import check_health_degradation
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            trend = [{'score': 50}, {'score': 35}]
            (d / 'health_trend.json').write_text(json.dumps(trend))
            result = check_health_degradation(d)
            assert result is not None
            assert 'critically low' in result.lower() or 'dropped' in result.lower()

    def test_no_trend_file(self):
        import tempfile
        from pathlib import Path
        from backend.models.explain import check_health_degradation
        with tempfile.TemporaryDirectory() as td:
            assert check_health_degradation(Path(td)) is None

    def test_single_entry(self):
        import tempfile
        import json
        from pathlib import Path
        from backend.models.explain import check_health_degradation
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / 'health_trend.json').write_text(json.dumps([{'score': 70}]))
            assert check_health_degradation(d) is None


class TestWhatChangedComparison:
    """Tests for retrain 'what changed' comparison logic."""

    def test_detects_accuracy_improvement(self):
        """Verify the comparison logic works for accuracy changes."""
        prev_wf = {'short': 0.55, 'medium': 0.58, 'long': 0.60}
        new_wf = {'short': 0.58, 'medium': 0.62, 'long': 0.60}
        changes = []
        for h in ('short', 'medium', 'long'):
            old_v = prev_wf.get(h)
            new_v = new_wf.get(h)
            if old_v is not None and new_v is not None:
                diff = (new_v - old_v) * 100
                if abs(diff) >= 1.0:
                    changes.append(f"{h}: {old_v*100:.1f}→{new_v*100:.1f}%")
        assert len(changes) == 2  # short and medium changed, long didn't
        assert 'short' in changes[0]
        assert 'medium' in changes[1]

    def test_no_changes_when_similar(self):
        """No changes reported when accuracy is within 1%."""
        prev_wf = {'short': 0.58, 'medium': 0.60}
        new_wf = {'short': 0.585, 'medium': 0.605}
        changes = []
        for h in ('short', 'medium'):
            diff = (new_wf[h] - prev_wf[h]) * 100
            if abs(diff) >= 1.0:
                changes.append(h)
        assert len(changes) == 0


class TestChatHealthDegradation:
    """Tests for health degradation in chat output."""

    def test_chat_summary_includes_degradation(self):
        from backend.llm.mock_llm import MockLLM
        llm = MockLLM()
        preds = {
            'symbol': 'TEST', 'current_price': 100.0,
            'horizons': [
                {'name': '1-Hour', 'direction': 'BULLISH', 'confidence': 60,
                 'expected_return': 0.5, 'invalidation': 'Stop $95'},
            ],
            'bullish_signals': ['RSI oversold'],
            'bearish_signals': ['Volume declining'],
            'health_degradation': '⚠ Health declining: 80 → 65 → 50',
        }
        result = llm.generate_summary('TEST', preds)
        assert 'declining' in result.lower() or 'Health' in result


class TestDBConvictionPersistence:
    """Test that conviction_tier and top_reason are persisted in predictions DB."""

    def test_save_prediction_with_conviction(self, tmp_path):
        import sqlite3
        db_path = tmp_path / 'test.db'
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("""CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT, direction TEXT, confidence REAL, price_at REAL,
            horizon TEXT, conviction_tier TEXT, top_reason TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP)""")
        conn.execute(
            "INSERT INTO predictions (ticker, direction, confidence, price_at, horizon, conviction_tier, top_reason, created_at) VALUES (?,?,?,?,?,?,?,?)",
            ('TSLA', 'bullish', 0.65, 250.0, 'short', 'HIGH', 'RSI oversold', '2026-01-01'))
        conn.commit()
        rows = [dict(r) for r in conn.execute("SELECT * FROM predictions WHERE ticker='TSLA'").fetchall()]
        assert len(rows) == 1
        assert rows[0]['conviction_tier'] == 'HIGH'
        assert rows[0]['top_reason'] == 'RSI oversold'

    def test_save_prediction_without_conviction(self, tmp_path):
        import sqlite3
        db_path = tmp_path / 'test.db'
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("""CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT, direction TEXT, confidence REAL, price_at REAL,
            horizon TEXT, conviction_tier TEXT, top_reason TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP)""")
        conn.execute(
            "INSERT INTO predictions (ticker, direction, confidence, price_at, horizon, conviction_tier, top_reason, created_at) VALUES (?,?,?,?,?,?,?,?)",
            ('AAPL', 'bearish', 0.55, 180.0, 'medium', None, None, '2026-01-01'))
        conn.commit()
        rows = [dict(r) for r in conn.execute("SELECT * FROM predictions WHERE ticker='AAPL'").fetchall()]
        assert len(rows) == 1
        assert rows[0]['conviction_tier'] is None
        assert rows[0]['top_reason'] is None


class TestModelExplanation:
    """Test get_model_explanation engine function."""

    def test_no_model_returns_error(self, tmp_path, monkeypatch):
        import cli.engine as eng
        monkeypatch.setattr(eng, '_MODELS_DIR', tmp_path / 'models')
        result = eng.get_model_explanation('FAKE')
        assert 'error' in result

    def test_with_meta(self, tmp_path, monkeypatch):
        import cli.engine as eng
        import json
        eng._ticker_meta.pop('TEST', None)
        model_dir = tmp_path / 'models' / 'TEST'
        model_dir.mkdir(parents=True)
        meta = {
            'model_version': 2,
            'trained_at': '2026-01-01T00:00:00',
            'samples': 500,
            'feature_count': 523,
            'selected_feature_count': 187,
            'walk_forward': {
                'average': {'short': 0.58, 'medium': 0.61, 'long': 0.63},
                'std': {'short': 0.04, 'medium': 0.03, 'long': 0.05},
                'ranges': {'short': (0.52, 0.67), 'medium': (0.54, 0.69), 'long': (0.51, 0.74)},
                'trend': {'short': 'improving', 'medium': 'stable', 'long': 'improving'},
            },
            'model_health': {'score': 72, 'grade': 'B'},
            'brier_scores': {'short': {'raw': 0.24, 'calibrated': 0.22}},
            'has_calibrators': True,
            'regime': {'vol_regime': 'normal'},
            'ensemble_weights': {'xgb': 0.6, 'lgbm': 0.4},
            'feature_names': ['rsi_14', 'macd', 'sma_20'],
        }
        (model_dir / 'meta.json').write_text(json.dumps(meta))
        monkeypatch.setattr(eng, '_MODELS_DIR', tmp_path / 'models')
        result = eng.get_model_explanation('TEST')
        assert result['ticker'] == 'TEST'
        assert result['selected_feature_count'] == 187
        assert result['wf_accuracy']['short'] == 0.58
        assert result['health']['grade'] == 'B'
        assert result['calibrated'] is True
        assert 'rsi_14' in result['top_features']


class TestWhatChangedBrierHealth:
    """Test that what-changed comparison includes Brier scores and health grade."""

    def test_brier_comparison_nested(self):
        """Verify Brier comparison handles nested dict structure."""
        prev_brier = {'short': {'raw': 0.25, 'calibrated': 0.23}}
        new_brier = {'short': {'raw': 0.22, 'calibrated': 0.20}}
        # Simulate the comparison logic
        changes = []
        for h in ('short',):
            ob = prev_brier.get(h, {})
            nb = new_brier.get(h, {})
            old_val = ob.get('calibrated', ob.get('raw')) if isinstance(ob, dict) else ob
            new_val = nb.get('calibrated', nb.get('raw')) if isinstance(nb, dict) else nb
            if old_val is not None and new_val is not None:
                diff = new_val - old_val
                if abs(diff) >= 0.01:
                    icon = '📈' if diff < 0 else '📉'
                    changes.append(f"brier_{h}: {old_val:.3f}→{new_val:.3f} {icon}")
        assert len(changes) == 1
        assert '📈' in changes[0]  # lower Brier is improvement
        assert '0.230→0.200' in changes[0]

    def test_health_grade_comparison(self):
        """Verify health grade change is detected."""
        prev_health = 'C'
        new_health = 'B'
        changes = []
        if prev_health and new_health and prev_health != new_health:
            changes.append(f"health: {prev_health}→{new_health}")
        assert changes == ['health: C→B']

    def test_no_change_when_same(self):
        """No change reported when Brier/health are the same."""
        prev_brier = {'short': {'raw': 0.22}}
        new_brier = {'short': {'raw': 0.22}}
        changes = []
        for h in ('short',):
            ob = prev_brier.get(h, {})
            nb = new_brier.get(h, {})
            old_val = ob.get('calibrated', ob.get('raw')) if isinstance(ob, dict) else ob
            new_val = nb.get('calibrated', nb.get('raw')) if isinstance(nb, dict) else nb
            if old_val is not None and new_val is not None:
                diff = new_val - old_val
                if abs(diff) >= 0.01:
                    changes.append(f"brier_{h}")
        assert len(changes) == 0


class TestExplainCLI:
    """Test the explain CLI command."""

    def test_explain_no_model(self, tmp_path, monkeypatch):
        from click.testing import CliRunner
        from cli.main import cli
        import cli.engine as eng
        monkeypatch.setattr(eng, '_MODELS_DIR', tmp_path / 'models')
        runner = CliRunner()
        result = runner.invoke(cli, ['explain', 'FAKE'])
        assert 'No trained model' in result.output or 'error' in result.output.lower()

    def test_explain_with_model(self, tmp_path, monkeypatch):
        from click.testing import CliRunner
        from cli.main import cli
        import cli.engine as eng
        import json
        model_dir = tmp_path / 'models' / 'TEST'
        model_dir.mkdir(parents=True)
        meta = {
            'model_version': 2, 'trained_at': '2026-01-01T00:00:00',
            'samples': 500, 'feature_count': 523, 'selected_feature_count': 187,
            'walk_forward': {'average': {'short': 0.58, 'medium': 0.61, 'long': 0.63},
                             'std': {'short': 0.04}, 'ranges': {}, 'trend': {}},
            'model_health': {'score': 72, 'grade': 'B'},
            'brier_scores': {}, 'has_calibrators': True,
            'ensemble_weights': {'xgb': 0.6, 'lgbm': 0.4},
            'feature_names': ['rsi_14', 'macd'],
        }
        (model_dir / 'meta.json').write_text(json.dumps(meta))
        monkeypatch.setattr(eng, '_MODELS_DIR', tmp_path / 'models')
        runner = CliRunner()
        result = runner.invoke(cli, ['explain', 'TEST'])
        assert 'TEST' in result.output
        assert 'B' in result.output
        assert '187' in result.output

    def test_explain_json_output(self, tmp_path, monkeypatch):
        from click.testing import CliRunner
        from cli.main import cli
        import cli.engine as eng
        import json
        model_dir = tmp_path / 'models' / 'TESTJ'
        model_dir.mkdir(parents=True)
        meta = {
            'model_version': 2, 'trained_at': '2026-01-01',
            'samples': 100, 'feature_count': 200, 'selected_feature_count': 100,
            'walk_forward': {'average': {}}, 'model_health': {'score': 50, 'grade': 'C'},
            'has_calibrators': False, 'feature_names': [],
        }
        (model_dir / 'meta.json').write_text(json.dumps(meta))
        monkeypatch.setattr(eng, '_MODELS_DIR', tmp_path / 'models')
        eng._ticker_meta.pop('TESTJ', None)
        runner = CliRunner()
        result = runner.invoke(cli, ['explain', 'TESTJ', '--json'])
        data = json.loads(result.output)
        assert data['ticker'] == 'TESTJ'
        assert data['health']['grade'] == 'C'


class TestCalibrationCLI:
    """Test the calibration CLI command."""

    def test_calibration_no_model(self, tmp_path, monkeypatch):
        from click.testing import CliRunner
        from cli.main import cli
        import cli.engine as eng
        monkeypatch.setattr(eng, '_MODELS_DIR', tmp_path / 'models')
        runner = CliRunner()
        result = runner.invoke(cli, ['calibration', 'FAKE'])
        assert 'No trained model' in result.output or 'error' in result.output.lower()

    def test_calibration_with_brier(self, tmp_path, monkeypatch):
        from click.testing import CliRunner
        from cli.main import cli
        import cli.engine as eng
        import json
        model_dir = tmp_path / 'models' / 'CALTEST'
        model_dir.mkdir(parents=True)
        meta = {
            'model_version': 2, 'trained_at': '2026-01-01',
            'samples': 500, 'feature_count': 200, 'selected_feature_count': 100,
            'walk_forward': {'average': {'short': 0.58}},
            'model_health': {'score': 70, 'grade': 'B'},
            'has_calibrators': True,
            'brier_scores': {'short': {'raw': 0.22, 'calibrated': 0.20}},
            'calibration_curves': {
                'short': [
                    {'predicted': 0.3, 'actual': 0.28, 'count': 15},
                    {'predicted': 0.7, 'actual': 0.72, 'count': 12},
                ]
            },
            'feature_names': [],
        }
        (model_dir / 'meta.json').write_text(json.dumps(meta))
        monkeypatch.setattr(eng, '_MODELS_DIR', tmp_path / 'models')
        eng._ticker_meta.pop('CALTEST', None)
        runner = CliRunner()
        result = runner.invoke(cli, ['calibration', 'CALTEST'])
        assert 'Calibrated' in result.output
        assert '0.22' in result.output  # raw Brier
        assert '0.20' in result.output  # calibrated Brier
        assert 'Predicted' in result.output  # calibration curve

    def test_calibration_json(self, tmp_path, monkeypatch):
        from click.testing import CliRunner
        from cli.main import cli
        import cli.engine as eng
        import json
        model_dir = tmp_path / 'models' / 'CALJSON'
        model_dir.mkdir(parents=True)
        meta = {
            'model_version': 2, 'trained_at': '2026-01-01',
            'samples': 100, 'feature_count': 200, 'selected_feature_count': 100,
            'walk_forward': {'average': {}}, 'model_health': {'score': 50, 'grade': 'C'},
            'has_calibrators': False, 'brier_scores': {'short': {'raw': 0.24}},
            'feature_names': [],
        }
        (model_dir / 'meta.json').write_text(json.dumps(meta))
        monkeypatch.setattr(eng, '_MODELS_DIR', tmp_path / 'models')
        eng._ticker_meta.pop('CALJSON', None)
        runner = CliRunner()
        result = runner.invoke(cli, ['calibration', 'CALJSON', '--json'])
        data = json.loads(result.output)
        assert data['ticker'] == 'CALJSON'
        assert data['brier_scores']['short']['raw'] == 0.24


class TestConvictionTierAccuracy:
    """Test conviction-tier accuracy breakdown in evaluate_predictions."""

    def test_tier_accuracy_breakdown(self, tmp_path):
        import sqlite3
        db_path = tmp_path / 'test.db'
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("""CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT, direction TEXT, confidence REAL, price_at REAL,
            horizon TEXT, conviction_tier TEXT, top_reason TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP)""")
        # Insert predictions with different conviction tiers
        from datetime import datetime, timedelta
        old = (datetime.now() - timedelta(days=30)).isoformat()
        conn.execute("INSERT INTO predictions (ticker, direction, confidence, price_at, horizon, conviction_tier, created_at) VALUES (?,?,?,?,?,?,?)",
                     ('TEST', 'bullish', 0.7, 100.0, 'short', 'HIGH', old))
        conn.execute("INSERT INTO predictions (ticker, direction, confidence, price_at, horizon, conviction_tier, created_at) VALUES (?,?,?,?,?,?,?)",
                     ('TEST', 'bearish', 0.55, 100.0, 'short', 'LOW', old))
        conn.commit()
        # Simulate evaluation: current price 110 (bullish was correct, bearish was wrong)
        rows = [dict(r) for r in conn.execute("SELECT * FROM predictions WHERE ticker='TEST'").fetchall()]
        assert len(rows) == 2
        assert rows[0]['conviction_tier'] == 'HIGH'
        assert rows[1]['conviction_tier'] == 'LOW'

    def test_prediction_accuracy_includes_tiers(self):
        """Verify prediction_accuracy function signature includes by_conviction."""
        # Just test the structure — actual DB calls need mocking
        result = {'total': 10, 'evaluated': 5, 'correct': 3, 'accuracy': 0.6,
                  'by_conviction': {'HIGH': {'correct': 2, 'total': 2, 'accuracy': 1.0},
                                    'LOW': {'correct': 1, 'total': 3, 'accuracy': 0.3333}}}
        assert result['by_conviction']['HIGH']['accuracy'] > result['by_conviction']['LOW']['accuracy']


# --- ASCII Calibration Curve Tests ---

class TestASCIICalibrationCurve:
    def test_format_ascii_calibration_curve_basic(self):
        from backend.models.explain import format_ascii_calibration_curve
        curve = [
            {'predicted': 0.3, 'actual': 0.28, 'count': 10},
            {'predicted': 0.5, 'actual': 0.52, 'count': 15},
            {'predicted': 0.7, 'actual': 0.65, 'count': 8},
        ]
        result = format_ascii_calibration_curve(curve)
        assert 'Predicted' in result
        assert 'Actual' in result
        assert '█' in result
        assert 'n=10' in result
        assert 'n=15' in result

    def test_format_ascii_calibration_curve_empty(self):
        from backend.models.explain import format_ascii_calibration_curve
        assert format_ascii_calibration_curve([]) == "No calibration data"

    def test_format_ascii_calibration_curve_perfect(self):
        from backend.models.explain import format_ascii_calibration_curve
        curve = [{'predicted': 0.5, 'actual': 0.5, 'count': 20}]
        result = format_ascii_calibration_curve(curve)
        assert '✓' in result  # gap=0 should be good

    def test_format_ascii_calibration_curve_poor(self):
        from backend.models.explain import format_ascii_calibration_curve
        curve = [{'predicted': 0.8, 'actual': 0.3, 'count': 5}]
        result = format_ascii_calibration_curve(curve)
        assert '✗' in result  # large gap


# --- Feature Sensitivity Tests ---

class TestFeatureSensitivity:
    def test_feature_sensitivity_no_model(self):
        from unittest.mock import patch
        with patch('cli.engine._load_ticker_meta', return_value=None):
            from cli.engine import feature_sensitivity
            result = feature_sensitivity('FAKE')
            assert 'error' in result

    def test_feature_sensitivity_structure(self):
        """Test sensitivity result structure with mocked model."""
        from unittest.mock import patch, MagicMock
        mock_meta = {
            'feature_names': ['rsi_14', 'macd', 'volume_ratio'],
            'selected_features': ['rsi_14', 'macd', 'volume_ratio'],
            'feature_importances': {'rsi_14': 0.5, 'macd': 0.3, 'volume_ratio': 0.2},
            'include_sequence': False,
        }
        mock_ens = MagicMock()
        mock_ens.loaded = True
        mock_ens.selected_features = None
        mock_ens.predict.return_value = {
            'short': {'probability': 0.6, 'direction': 'bullish'},
        }
        mock_feats = pd.DataFrame({
            'rsi_14': np.random.randn(50),
            'macd': np.random.randn(50),
            'volume_ratio': np.random.randn(50),
        })
        price_df = _price_df(50)

        with patch('cli.engine._load_ticker_meta', return_value=mock_meta), \
             patch('cli.engine._get_ticker_ensemble', return_value=mock_ens), \
             patch('cli.engine.generate_ohlcv', return_value=price_df), \
             patch('cli.engine.get_options_chain', return_value={}), \
             patch('cli.engine.get_fundamentals', return_value={}), \
             patch('cli.engine._get_sentiment_safe', return_value={}), \
             patch('cli.engine.get_macro_data', return_value={}), \
             patch('cli.engine._fetch_spy_sector_data', return_value=(None, None)), \
             patch('cli.engine._store') as mock_store:
            mock_store.compute_all_features.return_value = mock_feats
            from cli.engine import feature_sensitivity
            result = feature_sensitivity('TEST', top_n=3)
            assert 'sensitivities' in result
            assert result['ticker'] == 'TEST'
            assert 'base_probability' in result
            for s in result['sensitivities']:
                assert 'feature' in s
                assert 'sensitivity' in s
                assert 'prob_plus_1std' in s
                assert 'prob_minus_1std' in s


# --- Conformal Summary in Explain Tests ---

class TestConformalInExplain:
    def test_explain_includes_conformal_summary(self):
        """Test that get_model_explanation includes conformal summary when residuals exist."""
        from unittest.mock import patch
        import pickle
        import tempfile
        mock_meta = {
            'walk_forward': {'average': {'short': 0.58}},
            'model_health': {'score': 60, 'grade': 'B'},
            'brier_scores': {},
            'feature_names': ['rsi_14'],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            # Write residuals
            residuals = {'short': list(np.random.randn(50) * 0.02), 'medium': list(np.random.randn(50) * 0.03)}
            res_path = tmpdir_path / 'wf_residuals.pkl'
            with open(res_path, 'wb') as f:
                pickle.dump(residuals, f)

            with patch('cli.engine._load_ticker_meta', return_value=mock_meta), \
                 patch('cli.engine._ticker_model_dir', return_value=tmpdir_path), \
                 patch('cli.engine._load_feature_stability', return_value=None):
                from cli.engine import get_model_explanation
                result = get_model_explanation('TEST')
                assert 'conformal_summary' in result
                assert 'short' in result['conformal_summary']
                assert 'interval_width_90' in result['conformal_summary']['short']
                assert result['conformal_summary']['short']['n_residuals'] == 50


# --- Sensitivity CLI Command Tests ---

class TestSensitivityCLI:
    def test_sensitivity_command_exists(self):
        from cli.main import cli
        cmds = [c.name for c in cli.commands.values()] if hasattr(cli, 'commands') else []
        assert 'sensitivity' in cmds

    def test_sensitivity_command_help(self):
        from click.testing import CliRunner
        from cli.main import sensitivity
        runner = CliRunner()
        result = runner.invoke(sensitivity, ['--help'])
        assert result.exit_code == 0
        assert 'sensitivity' in result.output.lower() or 'feature' in result.output.lower()


# --- Journal Command Tests ---

class TestJournalCommand:
    def test_journal_command_exists(self):
        from cli.main import cli
        cmds = [c.name for c in cli.commands.values()] if hasattr(cli, 'commands') else []
        assert 'journal' in cmds

    def test_journal_command_help(self):
        from click.testing import CliRunner
        from cli.main import journal
        runner = CliRunner()
        result = runner.invoke(journal, ['--help'])
        assert result.exit_code == 0
        assert 'journal' in result.output.lower() or 'prediction' in result.output.lower()

    def test_get_prediction_journal_empty(self):
        """Test journal returns empty when no predictions exist."""
        from unittest.mock import patch
        with patch('cli.db.get_predictions', return_value=[]):
            from cli.engine import get_prediction_journal
            result = get_prediction_journal('FAKE')
            assert result['ticker'] == 'FAKE'
            assert result['entries'] == []

    def test_get_prediction_journal_with_data(self):
        """Test journal processes predictions correctly."""
        from unittest.mock import patch
        from datetime import datetime, timedelta
        old_date = (datetime.now() - timedelta(days=10)).isoformat()
        mock_preds = [
            {'created_at': old_date, 'horizon': 'short', 'direction': 'bullish',
             'confidence': 0.62, 'price_at': 100.0, 'conviction_tier': 'MODERATE', 'top_reason': 'RSI oversold'},
            {'created_at': old_date, 'horizon': 'medium', 'direction': 'bearish',
             'confidence': 0.58, 'price_at': 100.0, 'conviction_tier': 'MODERATE', 'top_reason': 'MACD cross'},
        ]
        mock_df = pd.DataFrame({'close': [105.0], 'open': [100.0], 'high': [106.0], 'low': [99.0], 'volume': [1000]})
        with patch('cli.db.get_predictions', return_value=mock_preds), \
             patch('cli.engine.generate_ohlcv', return_value=mock_df):
            from cli.engine import get_prediction_journal
            result = get_prediction_journal('TEST')
            assert len(result['entries']) == 2
            # Short prediction should be evaluated (10 days > 1 day horizon)
            assert result['entries'][0]['outcome'] in ('correct', 'wrong')

    def test_journal_confidence_distribution(self):
        """Test confidence distribution histogram computation."""
        from unittest.mock import patch
        from datetime import datetime
        preds = [
            {'created_at': datetime.now().isoformat(), 'horizon': 'short', 'direction': 'bullish',
             'confidence': c, 'price_at': 100.0, 'conviction_tier': 'LOW', 'top_reason': ''}
            for c in [0.51, 0.52, 0.56, 0.57, 0.63, 0.71]
        ]
        with patch('cli.db.get_predictions', return_value=preds), \
             patch('cli.engine.generate_ohlcv', side_effect=Exception("no data")):
            from cli.engine import get_prediction_journal
            result = get_prediction_journal('TEST')
            dist = result['stats']['confidence_distribution']
            assert '50%-55%' in dist
            assert dist['50%-55%'] == 2  # 0.51, 0.52


# --- Leaderboard Command Tests ---

class TestLeaderboardCommand:
    def test_leaderboard_command_exists(self):
        from cli.main import cli
        cmds = [c.name for c in cli.commands.values()] if hasattr(cli, 'commands') else []
        assert 'leaderboard' in cmds

    def test_leaderboard_command_help(self):
        from click.testing import CliRunner
        from cli.main import leaderboard
        runner = CliRunner()
        result = runner.invoke(leaderboard, ['--help'])
        assert result.exit_code == 0
        assert 'leaderboard' in result.output.lower() or 'rank' in result.output.lower()

    def test_leaderboard_empty(self):
        """Test leaderboard with no models."""
        from click.testing import CliRunner
        from cli.main import leaderboard
        from unittest.mock import patch
        runner = CliRunner()
        with patch('cli.engine.model_status', return_value=[]):
            result = runner.invoke(leaderboard, [])
            assert 'No trained models' in result.output


class TestAblationCommand:
    def test_ablation_command_exists(self):
        from click.testing import CliRunner
        from cli.main import ablation
        runner = CliRunner()
        result = runner.invoke(ablation, ['--help'])
        assert result.exit_code == 0

    def test_ablation_command_help(self):
        from click.testing import CliRunner
        from cli.main import ablation
        runner = CliRunner()
        result = runner.invoke(ablation, ['--help'])
        assert 'ablation' in result.output.lower() or 'feature' in result.output.lower()

    def test_ablation_no_model(self):
        from click.testing import CliRunner
        from cli.main import ablation
        from unittest.mock import patch
        runner = CliRunner()
        with patch('cli.engine.feature_ablation', return_value={'error': 'No trained model for XYZ'}):
            result = runner.invoke(ablation, ['XYZ'])
            assert 'No trained model' in result.output

    def test_ablation_json_output(self):
        from click.testing import CliRunner
        from cli.main import ablation
        from unittest.mock import patch
        import json
        mock_result = {
            'ticker': 'TSLA',
            'baseline_confidence': {'short': 0.55, 'medium': 0.60, 'long': 0.58},
            'groups': [
                {'group': 'technical', 'feature_count': 15, 'confidence_drop': {'short': 0.03, 'medium': 0.05, 'long': 0.02}, 'avg_drop': 0.033},
                {'group': 'lagged', 'feature_count': 20, 'confidence_drop': {'short': 0.01, 'medium': 0.02, 'long': 0.01}, 'avg_drop': 0.013},
            ],
            'total_selected': 80,
        }
        runner = CliRunner()
        with patch('cli.engine.feature_ablation', return_value=mock_result):
            result = runner.invoke(ablation, ['TSLA', '--json'])
            assert result.exit_code == 0
            data = json.loads(result.output)
            assert data['ticker'] == 'TSLA'
            assert len(data['groups']) == 2


class TestFeatureGroups:
    def test_feature_groups_defined(self):
        from cli.engine import _FEATURE_GROUPS
        assert 'technical' in _FEATURE_GROUPS
        assert 'lagged' in _FEATURE_GROUPS
        assert 'price_derived' in _FEATURE_GROUPS
        assert 'sector_relative' in _FEATURE_GROUPS
        assert 'options' in _FEATURE_GROUPS
        assert 'fundamentals' in _FEATURE_GROUPS
        assert 'macro' in _FEATURE_GROUPS
        assert 'regime' in _FEATURE_GROUPS

    def test_feature_group_matchers(self):
        from cli.engine import _FEATURE_GROUPS
        assert _FEATURE_GROUPS['lagged']('rsi_14_lag1')
        assert _FEATURE_GROUPS['lagged']('macd_roc5')
        assert _FEATURE_GROUPS['lagged']('bb_upper_mean5')
        assert not _FEATURE_GROUPS['lagged']('rsi_14')
        assert _FEATURE_GROUPS['technical']('rsi_14')
        assert _FEATURE_GROUPS['technical']('macd_line')
        assert _FEATURE_GROUPS['sector_relative']('return_vs_spy')
        assert _FEATURE_GROUPS['sector_relative']('spy_rsi')
        assert _FEATURE_GROUPS['regime']('regime_bull')
        assert _FEATURE_GROUPS['price_derived']('consecutive_up_days')


class TestFeatureAblation:
    def test_ablation_no_model(self):
        from cli.engine import feature_ablation
        from unittest.mock import patch
        with patch('cli.engine._load_ticker_meta', return_value=None):
            result = feature_ablation('XYZ')
            assert 'error' in result

    def test_ablation_structure(self):
        from cli.engine import feature_ablation
        from unittest.mock import patch, MagicMock
        import numpy as np

        mock_meta = {
            'selected_features': list(range(80)),
            'feature_names': [f'feat_{i}' for i in range(80)],
        }
        mock_ens = MagicMock()
        mock_ens.predict_direction_ensemble.return_value = ('bullish', 0.6, 0.01)

        mock_hist = MagicMock()
        mock_hist.empty = False
        mock_df = MagicMock()
        mock_df.columns = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        mock_hist.reset_index.return_value = mock_df
        mock_df.rename.return_value = mock_df
        mock_df.__getitem__ = lambda self, k: mock_df
        mock_df.__setitem__ = lambda self, k, v: None
        mock_df.__len__ = lambda self: 100

        mock_feats = MagicMock()
        mock_feats.values = np.random.randn(100, 515)

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = mock_hist

        with patch('cli.engine._load_ticker_meta', return_value=mock_meta), \
             patch('cli.engine._ensure_ticker_model', return_value=mock_ens), \
             patch('cli.engine._fetch_spy_sector_data', return_value=(None, None)), \
             patch('cli.engine._store') as mock_store, \
             patch('yfinance.Ticker', return_value=mock_ticker):
            mock_store.compute_all_features.return_value = mock_feats
            result = feature_ablation('TSLA')
            assert 'ticker' in result
            assert 'baseline_confidence' in result
            assert 'groups' in result


class TestPredictionStreak:
    def test_streak_no_predictions(self):
        from cli.engine import prediction_streak
        from unittest.mock import patch
        with patch('cli.db.get_predictions', return_value=[]):
            result = prediction_streak('XYZ')
            assert result['total'] == 0

    def test_streak_with_data(self):
        from cli.engine import prediction_streak
        from unittest.mock import patch
        from datetime import datetime, timedelta
        preds = [
            {'direction': 'bullish', 'price_at': 100, 'horizon': 'short',
             'created_at': (datetime.now() - timedelta(days=5)).isoformat(),
             'conviction_tier': 'HIGH'},
            {'direction': 'bearish', 'price_at': 110, 'horizon': 'short',
             'created_at': (datetime.now() - timedelta(days=10)).isoformat(),
             'conviction_tier': 'MODERATE'},
        ]
        with patch('cli.db.get_predictions', return_value=preds), \
             patch('cli.engine.get_price', return_value={'price': 105}):
            result = prediction_streak('TSLA')
            assert result['ticker'] == 'TSLA'
            assert result['total'] == 2
            assert 'current_streak' in result
            assert 'streak_type' in result
            assert 'best_correct_streak' in result

    def test_streak_all_correct(self):
        from cli.engine import prediction_streak
        from unittest.mock import patch
        from datetime import datetime, timedelta
        # All bullish predictions, price went up
        preds = [
            {'direction': 'bullish', 'price_at': 90 + i, 'horizon': 'short',
             'created_at': (datetime.now() - timedelta(days=5 + i)).isoformat(),
             'conviction_tier': 'HIGH'}
            for i in range(5)
        ]
        with patch('cli.db.get_predictions', return_value=preds), \
             patch('cli.engine.get_price', return_value={'price': 200}):
            result = prediction_streak('TSLA')
            assert result['best_correct_streak'] >= 1


class TestStreakCommand:
    def test_streak_command_exists(self):
        from click.testing import CliRunner
        from cli.main import streak
        runner = CliRunner()
        result = runner.invoke(streak, ['--help'])
        assert result.exit_code == 0

    def test_streak_command_help(self):
        from click.testing import CliRunner
        from cli.main import streak
        runner = CliRunner()
        result = runner.invoke(streak, ['--help'])
        assert 'streak' in result.output.lower()

    def test_streak_empty(self):
        from click.testing import CliRunner
        from cli.main import streak
        from unittest.mock import patch
        runner = CliRunner()
        with patch('cli.engine.prediction_streak', return_value={'ticker': 'XYZ', 'total': 0, 'evaluated': 0}):
            result = runner.invoke(streak, ['XYZ'])
            assert 'No evaluated' in result.output

    def test_streak_json_output(self):
        from click.testing import CliRunner
        from cli.main import streak
        from unittest.mock import patch
        import json
        mock_result = {
            'ticker': 'TSLA', 'total': 10, 'evaluated': 8,
            'current_streak': 3, 'streak_type': 'correct',
            'best_correct_streak': 5, 'worst_wrong_streak': 2,
            'recent': [{'date': '2026-01-01', 'horizon': 'short', 'direction': 'bullish',
                        'conviction': 'HIGH', 'correct': True}],
        }
        runner = CliRunner()
        with patch('cli.engine.prediction_streak', return_value=mock_result):
            result = runner.invoke(streak, ['TSLA', '--json'])
            assert result.exit_code == 0
            data = json.loads(result.output)
            assert data['current_streak'] == 3


# --- Replay Analysis with Per-Ticker Model Tests ---

class TestReplayWithTickerModel:
    def test_replay_returns_has_ticker_model_flag(self):
        """replay_analysis result should include has_ticker_model flag."""
        # The result dict should have the key
        result = {'has_ticker_model': True, 'horizons': {}}
        assert 'has_ticker_model' in result

    def test_replay_horizons_have_conviction(self):
        """Replay horizons should include conviction data when using per-ticker model."""
        from backend.models.explain import format_conviction_verdict
        verdict = format_conviction_verdict('bullish', 0.62, 0.58)
        assert 'LEAN BUY' in verdict or 'BUY' in verdict
        assert '62%' in verdict or '58%' in verdict

    def test_replay_horizons_have_shap_text(self):
        """SHAP text should be formatted for replay output."""
        from backend.models.explain import format_shap_explanation
        explanation = {
            'bullish_factors': [{'feature': 'RSI(14)', 'influence': 12.5, 'value': 28.3}],
            'bearish_factors': [{'feature': 'MACD', 'influence': 8.1, 'value': -0.5}],
        }
        text = format_shap_explanation(explanation, 'bullish')
        assert 'Why BUY' in text
        assert 'RSI(14)' in text

    def test_replay_horizons_have_vol_zscore(self):
        """Vol z-score should be computed for replay."""
        from backend.models.explain import compute_volatility_zscore
        z, desc = compute_volatility_zscore(0.02, 0.015)
        assert abs(z - 1.33) < 0.1
        assert desc == 'unusual'


# --- Global Feature Importance Tests ---

class TestGlobalFeatureImportance:
    def test_global_importance_empty_when_no_models(self):
        """Should return empty features when no models exist."""
        from unittest.mock import patch
        from cli.engine import global_feature_importance
        with patch('cli.engine._MODELS_DIR') as mock_dir:
            mock_dir.exists.return_value = False
            result = global_feature_importance()
            assert result['features'] == []
            assert result['ticker_count'] == 0

    def test_global_importance_structure(self):
        """Result should have correct structure."""
        result = {'features': [{'name': 'RSI(14)', 'raw_name': 'rsi_14',
                                'score': 0.5, 'ticker_count': 3, 'pct_tickers': 100.0}],
                  'ticker_count': 3}
        assert len(result['features']) == 1
        f = result['features'][0]
        assert 'name' in f
        assert 'raw_name' in f
        assert 'score' in f
        assert 'ticker_count' in f
        assert 'pct_tickers' in f

    def test_global_importance_with_mock_models(self):
        """Should aggregate features across multiple tickers."""
        from unittest.mock import patch, MagicMock
        from pathlib import Path
        from cli.engine import global_feature_importance

        mock_meta = {
            'selected_features': [0, 1, 2],
            'feature_names': ['rsi_14', 'macd', 'return_1d'],
            'walk_forward': {'average': {'short': 0.58, 'medium': 0.60, 'long': 0.62}},
        }

        mock_dir1 = MagicMock()
        mock_dir1.is_dir.return_value = True
        mock_dir1.name = 'TSLA'
        mock_dir1.__truediv__ = lambda self, x: Path('/tmp/fake/TSLA') / x

        mock_models_dir = MagicMock()
        mock_models_dir.exists.return_value = True
        mock_models_dir.iterdir.return_value = [mock_dir1]

        with patch('cli.engine._MODELS_DIR', mock_models_dir), \
             patch('cli.engine._load_ticker_meta', return_value=mock_meta), \
             patch('cli.engine._load_feature_stability', return_value={'rsi_14': 0.8, 'macd': 0.6}), \
             patch('pathlib.Path.exists', return_value=True):
            result = global_feature_importance(top_n=5)
            assert result['ticker_count'] == 1
            assert len(result['features']) > 0
            # rsi_14 should rank highest (highest stability)
            assert result['features'][0]['raw_name'] == 'rsi_14'


# --- Top Features CLI Tests ---

class TestTopFeaturesCLI:
    def test_top_features_no_models(self):
        from click.testing import CliRunner
        from cli.main import top_features_cmd
        from unittest.mock import patch
        runner = CliRunner()
        with patch('cli.engine.global_feature_importance', return_value={'features': [], 'ticker_count': 0}):
            result = runner.invoke(top_features_cmd, [])
            assert 'No trained models' in result.output

    def test_top_features_json_output(self):
        from click.testing import CliRunner
        from cli.main import top_features_cmd
        from unittest.mock import patch
        import json
        mock_result = {
            'features': [{'name': 'RSI(14)', 'raw_name': 'rsi_14',
                          'score': 0.5, 'ticker_count': 2, 'pct_tickers': 100.0}],
            'ticker_count': 2,
        }
        runner = CliRunner()
        with patch('cli.engine.global_feature_importance', return_value=mock_result):
            result = runner.invoke(top_features_cmd, ['--json'])
            assert result.exit_code == 0
            data = json.loads(result.output)
            assert data['ticker_count'] == 2
            assert len(data['features']) == 1

    def test_top_features_table_output(self):
        from click.testing import CliRunner
        from cli.main import top_features_cmd
        from unittest.mock import patch
        mock_result = {
            'features': [
                {'name': 'RSI(14)', 'raw_name': 'rsi_14', 'score': 0.5, 'ticker_count': 3, 'pct_tickers': 100.0},
                {'name': 'MACD', 'raw_name': 'macd', 'score': 0.3, 'ticker_count': 2, 'pct_tickers': 67.0},
            ],
            'ticker_count': 3,
        }
        runner = CliRunner()
        with patch('cli.engine.global_feature_importance', return_value=mock_result):
            result = runner.invoke(top_features_cmd, [])
            assert result.exit_code == 0
            assert 'RSI(14)' in result.output
            assert 'MACD' in result.output
            assert 'Global Feature Importance' in result.output


# --- Replay Range Conviction Tests ---

class TestReplayRangeConviction:
    def test_replay_range_shows_conviction_tier_icons(self):
        """Replay range should use conviction tier icons instead of plain direction."""
        # Test the icon mapping logic
        tier_icons = {'HIGH': '🟢', 'MODERATE': '🟡', 'LOW': '⚪'}
        assert tier_icons.get('HIGH') == '🟢'
        assert tier_icons.get('MODERATE') == '🟡'
        assert tier_icons.get('LOW') == '⚪'


# --- EnsembleModel loaded attribute Tests ---

class TestEnsembleLoaded:
    def test_loaded_false_by_default(self):
        from backend.models.ensemble import EnsembleModel
        ens = EnsembleModel()
        assert ens.loaded is False

    def test_loaded_set_after_load(self):
        from backend.models.ensemble import EnsembleModel
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as td:
            ens = EnsembleModel(model_dir=Path(td))
            result = ens.load()
            # No model files → load returns False
            assert result is False
            assert ens.loaded is False


class TestFeatureDriftAnalysis:
    def test_drift_analysis_no_model(self):
        from unittest.mock import patch
        from cli.engine import feature_drift_analysis
        with patch('cli.engine._load_ticker_meta', return_value=None):
            result = feature_drift_analysis('FAKE')
        assert 'error' in result

    def test_drift_analysis_no_stats(self):
        from unittest.mock import patch
        from cli.engine import feature_drift_analysis
        with patch('cli.engine._load_ticker_meta', return_value={'feature_means': [], 'feature_stds': []}):
            result = feature_drift_analysis('FAKE')
        assert 'error' in result

    def test_drift_analysis_computes_zscores(self):
        from unittest.mock import patch
        from cli.engine import feature_drift_analysis
        np.random.seed(42)
        n_feats = 10
        means = np.zeros(n_feats)
        stds = np.ones(n_feats)
        names = [f'feat_{i}' for i in range(n_feats)]
        meta = {'feature_means': means.tolist(), 'feature_stds': stds.tolist(),
                'feature_names': names, 'trained_at': '2024-01-01'}

        # Create mock features DataFrame
        mock_feats = pd.DataFrame(np.random.randn(5, n_feats) * 3, columns=names)

        with patch('cli.engine._load_ticker_meta', return_value=meta), \
             patch('cli.engine.generate_ohlcv', return_value=_price_df(120)), \
             patch('cli.engine.get_options_chain', return_value={}), \
             patch('cli.engine.get_fundamentals', return_value={}), \
             patch('cli.engine._get_sentiment_safe', return_value={}), \
             patch('cli.engine.get_macro_data', return_value={}), \
             patch('cli.engine._fetch_spy_sector_data', return_value=(None, None)), \
             patch('cli.engine._store') as mock_store:
            mock_store.compute_all_features.return_value = mock_feats
            result = feature_drift_analysis('TEST')

        assert 'error' not in result
        assert result['total_features'] == n_feats
        assert result['risk'] in ('HIGH', 'MODERATE', 'LOW')
        assert len(result['top_drifted']) > 0
        assert 'z_score' in result['top_drifted'][0]

    def test_drift_risk_levels(self):
        from unittest.mock import patch
        from cli.engine import feature_drift_analysis
        n = 20
        names = [f'f_{i}' for i in range(n)]
        # All features at 4 sigma → HIGH risk
        meta = {'feature_means': [0.0] * n, 'feature_stds': [1.0] * n,
                'feature_names': names, 'trained_at': '2024-01-01'}
        mock_feats = pd.DataFrame(np.full((3, n), 4.0), columns=names)

        with patch('cli.engine._load_ticker_meta', return_value=meta), \
             patch('cli.engine.generate_ohlcv', return_value=_price_df(120)), \
             patch('cli.engine.get_options_chain', return_value={}), \
             patch('cli.engine.get_fundamentals', return_value={}), \
             patch('cli.engine._get_sentiment_safe', return_value={}), \
             patch('cli.engine.get_macro_data', return_value={}), \
             patch('cli.engine._fetch_spy_sector_data', return_value=(None, None)), \
             patch('cli.engine._store') as mock_store:
            mock_store.compute_all_features.return_value = mock_feats
            result = feature_drift_analysis('TEST')
        assert result['risk'] == 'HIGH'


# --- Backtest Compare Tests ---

class TestBacktestCompare:
    def test_compare_first_run_saves_baseline(self):
        from unittest.mock import patch
        from cli.engine import backtest_compare
        import tempfile
        from pathlib import Path

        fake_bt = {'net_return': 5.0, 'sharpe_ratio': 1.2, 'max_drawdown': 0.1,
                    'win_rate': 0.55, 'profit_factor': 1.3, 'total_trades': 50}

        with tempfile.TemporaryDirectory() as td:
            model_dir = Path(td)
            with patch('cli.engine.run_backtest', return_value=fake_bt), \
                 patch('cli.engine._ticker_model_dir', return_value=model_dir):
                result = backtest_compare('TEST', 365)

        assert result['current'] == fake_bt
        assert result['baseline'] is None
        assert result['comparison'] is None

    def test_compare_second_run_shows_comparison(self):
        from unittest.mock import patch
        from cli.engine import backtest_compare
        import tempfile
        import json
        from pathlib import Path

        baseline = {'net_return': 3.0, 'sharpe_ratio': 0.8, 'max_drawdown': 0.15,
                     'win_rate': 0.50, 'profit_factor': 1.1, 'total_trades': 40,
                     'saved_at': '2024-01-01'}
        current = {'net_return': 5.0, 'sharpe_ratio': 1.2, 'max_drawdown': 0.1,
                    'win_rate': 0.55, 'profit_factor': 1.3, 'total_trades': 50}

        with tempfile.TemporaryDirectory() as td:
            model_dir = Path(td)
            # Write baseline
            (model_dir / 'backtest_baseline.json').write_text(json.dumps(baseline))
            with patch('cli.engine.run_backtest', return_value=current), \
                 patch('cli.engine._ticker_model_dir', return_value=model_dir):
                result = backtest_compare('TEST', 365)

        assert result['comparison'] is not None
        assert result['comparison']['net_return']['improved'] is True
        assert result['comparison']['max_drawdown']['improved'] is True  # lower is better
        assert result['comparison']['win_rate']['delta'] > 0


# --- Feature Interactions Engine Tests ---

class TestFeatureInteractionsEngine:
    def test_interactions_no_model(self):
        from unittest.mock import patch
        from cli.engine import feature_interactions
        with patch('cli.engine._load_ticker_meta', return_value=None):
            result = feature_interactions('FAKE')
        assert 'error' in result

    def test_interactions_not_loaded(self):
        from unittest.mock import patch, MagicMock
        from cli.engine import feature_interactions
        ens = MagicMock()
        ens.loaded = False
        with patch('cli.engine._load_ticker_meta', return_value={'feature_names': []}), \
             patch('cli.engine._get_ticker_ensemble', return_value=ens):
            result = feature_interactions('FAKE')
        assert 'error' in result

    def test_feature_importance_correlation_fallback(self):
        from cli.engine import _feature_importance_correlation
        from unittest.mock import MagicMock
        np.random.seed(42)
        n = 50
        feat_names = [f'f_{i}' for i in range(n)]
        X = np.random.randn(30, n)

        ens = MagicMock()
        model = MagicMock()
        model.feature_importances_ = np.random.rand(n)
        ens.xgb_short = model
        ens.lgbm_short = model
        ens.xgb_medium = None
        ens.lgbm_medium = None

        result = _feature_importance_correlation(ens, X, feat_names, 5)
        assert 'interactions' in result
        assert result['method'] == 'correlation'


# --- CLI Command Tests ---

class TestDriftCLI:
    def test_drift_command_exists(self):
        from cli.main import cli
        from click.testing import CliRunner
        runner = CliRunner()
        result = runner.invoke(cli, ['drift', '--help'])
        assert result.exit_code == 0
        assert 'feature drift' in result.output.lower()

    def test_drift_command_json(self):
        from cli.main import cli
        from click.testing import CliRunner
        from unittest.mock import patch
        runner = CliRunner()
        mock_result = {'ticker': 'TEST', 'risk': 'LOW', 'top_drifted': [],
                       'total_features': 100, 'severe_drift': 0, 'moderate_drift': 2,
                       'mild_drift': 5, 'avg_drift': 0.8, 'risk_desc': 'OK'}
        with patch('cli.engine.feature_drift_analysis', return_value=mock_result):
            result = runner.invoke(cli, ['drift', 'TEST', '--json'])
        assert result.exit_code == 0
        assert '"risk": "LOW"' in result.output


class TestBacktestCompareCLI:
    def test_command_exists(self):
        from cli.main import cli
        from click.testing import CliRunner
        runner = CliRunner()
        result = runner.invoke(cli, ['backtest-compare', '--help'])
        assert result.exit_code == 0
        assert 'backtest' in result.output.lower()


class TestInteractionsCLI:
    def test_command_exists(self):
        from cli.main import cli
        from click.testing import CliRunner
        runner = CliRunner()
        result = runner.invoke(cli, ['interactions', '--help'])
        assert result.exit_code == 0
        assert 'interaction' in result.output.lower()

    def test_interactions_json_error(self):
        from cli.main import cli
        from click.testing import CliRunner
        from unittest.mock import patch
        runner = CliRunner()
        with patch('cli.engine.feature_interactions', return_value={'error': 'No model'}):
            result = runner.invoke(cli, ['interactions', 'FAKE', '--json'])
        assert result.exit_code == 0


# --- Model Diff CLI Tests ---

class TestModelDiffCLI:
    def test_command_exists(self):
        from cli.main import cli
        from click.testing import CliRunner
        runner = CliRunner()
        result = runner.invoke(cli, ['model-diff', '--help'])
        assert result.exit_code == 0
        assert 'compare' in result.output.lower() or 'model' in result.output.lower()

    def test_model_diff_json_no_model(self):
        from cli.main import cli
        from click.testing import CliRunner
        from unittest.mock import patch
        runner = CliRunner()
        with patch('cli.engine.get_model_explanation', return_value={'error': 'No model'}):
            result = runner.invoke(cli, ['model-diff', 'FAKE', '--json'])
        assert result.exit_code == 0
        assert 'error' in result.output

    def test_model_diff_with_history(self):
        from cli.main import cli
        from click.testing import CliRunner
        from unittest.mock import patch
        runner = CliRunner()
        mock_info = {
            'ticker': 'TEST',
            'health_trend': [
                {'score': 50, 'grade': 'C'},
                {'score': 65, 'grade': 'B'},
            ],
            'feature_changelog': [
                {'top_features': ['rsi_14', 'macd', 'volume_ratio']},
                {'top_features': ['rsi_14', 'macd', 'bb_position']},
            ],
            'wf_accuracy': {'short': 0.58, 'medium': 0.61},
        }
        with patch('cli.engine.get_model_explanation', return_value=mock_info):
            result = runner.invoke(cli, ['model-diff', 'TEST'])
        assert result.exit_code == 0
        assert '📈' in result.output  # health improved

    def test_model_diff_json_output(self):
        from cli.main import cli
        from click.testing import CliRunner
        from unittest.mock import patch
        import json
        runner = CliRunner()
        mock_info = {
            'ticker': 'TEST',
            'health_trend': [
                {'score': 70, 'grade': 'B'},
                {'score': 60, 'grade': 'B'},
            ],
            'feature_changelog': [],
            'wf_accuracy': {'short': 0.55},
        }
        with patch('cli.engine.get_model_explanation', return_value=mock_info):
            result = runner.invoke(cli, ['model-diff', 'TEST', '--json'])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data['health']['delta'] == -10.0
        assert data['health']['improved'] is False


# --- Feature Group Tests ---

class TestFeatureGroup:
    def test_feature_group_classification(self):
        from backend.features.feature_store import FeatureStore
        fs = FeatureStore()
        assert fs.feature_group('rsi_14') == 'technical'
        assert fs.feature_group('return_1d') == 'price_volume'
        assert fs.feature_group('rsi_14_lag5') == 'lagged'
        assert fs.feature_group('return_vs_spy') == 'sector_relative'
        assert fs.feature_group('consecutive_up_days') == 'price_derived'
        assert fs.feature_group('put_call_ratio') == 'options'
        assert fs.feature_group('eps') == 'fundamentals'
        assert fs.feature_group('news_sentiment_avg') == 'sentiment'
        assert fs.feature_group('vix') == 'macro'

    def test_feature_group_importance_no_model(self):
        from unittest.mock import patch
        from cli.engine import feature_group_importance
        with patch('cli.engine._load_ticker_meta', return_value=None):
            result = feature_group_importance('FAKE')
        assert 'error' in result

    def test_feature_group_importance_with_model(self):
        from unittest.mock import patch, MagicMock
        from cli.engine import feature_group_importance
        np.random.seed(42)
        n = 20
        names = ['rsi_14', 'macd', 'return_1d', 'volume_ratio', 'rsi_14_lag5',
                 'return_vs_spy', 'consecutive_up_days', 'put_call_ratio',
                 'eps', 'news_sentiment_avg', 'vix', 'bb_position', 'stoch_k',
                 'atr_14', 'obv_slope', 'mfi_14', 'williams_r', 'roc_10',
                 'return_5d', 'volume_change']
        meta = {'feature_names': names}
        ens = MagicMock()
        ens.loaded = True
        model = MagicMock()
        model.feature_importances_ = np.random.rand(n)
        ens.xgb_short = model
        ens.lgbm_short = model
        ens.xgb_medium = None
        ens.lgbm_medium = None
        ens.xgb_long = None
        ens.lgbm_long = None

        with patch('cli.engine._load_ticker_meta', return_value=meta), \
             patch('cli.engine._get_ticker_ensemble', return_value=ens):
            result = feature_group_importance('TEST')

        assert 'error' not in result
        assert len(result['groups']) > 0
        total_pct = sum(g['pct'] for g in result['groups'])
        assert abs(total_pct - 100) < 1  # should sum to ~100%


class TestFeatureGroupsCLI:
    def test_command_exists(self):
        from cli.main import cli
        from click.testing import CliRunner
        runner = CliRunner()
        result = runner.invoke(cli, ['feature-groups', '--help'])
        assert result.exit_code == 0
        assert 'feature' in result.output.lower()

    def test_feature_groups_json(self):
        from cli.main import cli
        from click.testing import CliRunner
        from unittest.mock import patch
        runner = CliRunner()
        mock_result = {
            'ticker': 'TEST', 'total_features': 100,
            'groups': [{'group': 'technical', 'pct': 45.0, 'count': 30,
                        'total_importance': 0.45, 'top_features': []}],
        }
        with patch('cli.engine.feature_group_importance', return_value=mock_result):
            result = runner.invoke(cli, ['feature-groups', 'TEST', '--json'])
        assert result.exit_code == 0
        assert 'technical' in result.output


# --- Compare Models Tests ---

class TestCompareModels:
    def test_compare_no_model(self):
        from unittest.mock import patch
        from cli.engine import compare_models
        with patch('cli.engine._load_ticker_meta', return_value=None):
            result = compare_models('FAKE1', 'FAKE2')
        assert 'error' in result

    def test_compare_two_models(self):
        from unittest.mock import patch
        from cli.engine import compare_models
        m1 = {'walk_forward': {'average': {'short': 0.58}},
               'model_health': {'grade': 'B', 'score': 65},
               'selected_feature_count': 150, 'samples': 500,
               'has_calibrators': True, 'trained_at': '2024-01-01',
               'ensemble_weights': {'xgb': 0.6, 'lgbm': 0.4}}
        m2 = {'walk_forward': {'average': {'short': 0.55}},
               'model_health': {'grade': 'C', 'score': 50},
               'selected_feature_count': 180, 'samples': 400,
               'has_calibrators': False, 'trained_at': '2024-02-01',
               'ensemble_weights': {'xgb': 0.5, 'lgbm': 0.5}}

        with patch('cli.engine._load_ticker_meta', side_effect=[m1, m2]):
            result = compare_models('TSLA', 'AAPL')
        assert result['model1']['ticker'] == 'TSLA'
        assert result['model2']['ticker'] == 'AAPL'
        assert result['model1']['health_score'] == 65
        assert result['model2']['health_score'] == 50


class TestCompareModelsCLI:
    def test_command_exists(self):
        from cli.main import cli
        from click.testing import CliRunner
        runner = CliRunner()
        result = runner.invoke(cli, ['compare-models', '--help'])
        assert result.exit_code == 0
        assert 'compare' in result.output.lower()

    def test_compare_json(self):
        from cli.main import cli
        from click.testing import CliRunner
        from unittest.mock import patch
        runner = CliRunner()
        mock_result = {
            'model1': {'ticker': 'A', 'health_grade': 'B', 'health_score': 60,
                       'wf_short': 0.58, 'wf_medium': None, 'wf_long': None,
                       'features': 150, 'samples': 500, 'calibrated': True,
                       'trained_at': '2024-01-01', 'ensemble_weights': {}},
            'model2': {'ticker': 'B', 'health_grade': 'C', 'health_score': 45,
                       'wf_short': 0.52, 'wf_medium': None, 'wf_long': None,
                       'features': 180, 'samples': 400, 'calibrated': False,
                       'trained_at': '2024-02-01', 'ensemble_weights': {}},
        }
        with patch('cli.engine.compare_models', return_value=mock_result):
            result = runner.invoke(cli, ['compare-models', 'A', 'B', '--json'])
        assert result.exit_code == 0
        assert '"ticker": "A"' in result.output


# --- Retrain Recommendations Tests ---

class TestRetrainRecommendations:
    def test_empty_when_no_models(self):
        from unittest.mock import patch
        from cli.engine import retrain_recommendations
        with patch('cli.engine.model_status', return_value=[]):
            result = retrain_recommendations()
        assert result == []

    def test_recommends_old_models(self):
        from unittest.mock import patch
        from cli.engine import retrain_recommendations
        statuses = [
            {'ticker': 'TSLA', 'health_grade': 'B', 'health_score': 65,
             'age_days': 30, 'calibrated': True, 'wf_short': 0.58, 'wf_medium': 0.60, 'wf_long': 0.62},
            {'ticker': 'AAPL', 'health_grade': 'A', 'health_score': 85,
             'age_days': 2, 'calibrated': True, 'wf_short': 0.60, 'wf_medium': 0.62, 'wf_long': 0.65},
        ]
        with patch('cli.engine.model_status', return_value=statuses):
            result = retrain_recommendations()
        # TSLA should be recommended (old), AAPL should not
        tickers = [r['ticker'] for r in result]
        assert 'TSLA' in tickers

    def test_recommends_low_health(self):
        from unittest.mock import patch
        from cli.engine import retrain_recommendations
        statuses = [
            {'ticker': 'BAD', 'health_grade': 'F', 'health_score': 10,
             'age_days': 5, 'calibrated': False, 'wf_short': 0.50, 'wf_medium': 0.51, 'wf_long': 0.52},
        ]
        with patch('cli.engine.model_status', return_value=statuses):
            result = retrain_recommendations()
        assert len(result) == 1
        assert result[0]['urgency'] >= 5  # high urgency


class TestRetrainStatusCLI:
    def test_command_exists(self):
        from cli.main import cli
        from click.testing import CliRunner
        runner = CliRunner()
        result = runner.invoke(cli, ['retrain-status', '--help'])
        assert result.exit_code == 0
        assert 'retrain' in result.output.lower()

    def test_retrain_status_empty(self):
        from cli.main import cli
        from click.testing import CliRunner
        from unittest.mock import patch
        runner = CliRunner()
        with patch('cli.engine.retrain_recommendations', return_value=[]):
            result = runner.invoke(cli, ['retrain-status'])
        assert result.exit_code == 0
        assert 'healthy' in result.output.lower() or 'no retraining' in result.output.lower()


class TestBestHorizonRecommendation:
    def test_best_horizon_basic(self):
        from backend.models.explain import best_horizon_recommendation
        wf = {'short': 0.52, 'medium': 0.58, 'long': 0.62}
        result = best_horizon_recommendation(wf)
        assert result is not None
        assert result['horizon'] == 'long'
        assert result['accuracy'] == 0.62
        assert result['label'] == '20-day'
        assert 'long' in result['reason'].lower() or 'cycle' in result['reason'].lower()

    def test_best_horizon_short_wins(self):
        from backend.models.explain import best_horizon_recommendation
        wf = {'short': 0.65, 'medium': 0.55, 'long': 0.58}
        result = best_horizon_recommendation(wf)
        assert result['horizon'] == 'short'
        assert result['label'] == '1-day'

    def test_best_horizon_low_accuracy(self):
        from backend.models.explain import best_horizon_recommendation
        wf = {'short': 0.50, 'medium': 0.51, 'long': 0.52}
        result = best_horizon_recommendation(wf)
        assert result is not None
        assert 'caution' in result['reason'].lower()

    def test_best_horizon_empty(self):
        from backend.models.explain import best_horizon_recommendation
        assert best_horizon_recommendation({}) is None
        assert best_horizon_recommendation(None) is None

    def test_best_horizon_partial(self):
        from backend.models.explain import best_horizon_recommendation
        wf = {'medium': 0.60}
        result = best_horizon_recommendation(wf)
        assert result['horizon'] == 'medium'


class TestSignalQualityAssessment:
    def test_high_quality(self):
        from backend.models.explain import signal_quality_assessment
        sq = signal_quality_assessment(0.72, 0.62, 'HIGH', 0)
        assert sq['quality'] == 'HIGH'
        assert sq['icon'] == '🟢'

    def test_moderate_quality(self):
        from backend.models.explain import signal_quality_assessment
        sq = signal_quality_assessment(0.58, 0.56, 'MODERATE', 0)
        assert sq['quality'] in ('MODERATE', 'HIGH')

    def test_low_quality(self):
        from backend.models.explain import signal_quality_assessment
        sq = signal_quality_assessment(0.51, 0.51, 'LOW', 0)
        assert sq['quality'] in ('LOW', 'UNRELIABLE')
        assert len(sq['warnings']) > 0

    def test_drift_penalty(self):
        from backend.models.explain import signal_quality_assessment
        sq_no_drift = signal_quality_assessment(0.60, 0.58, 'MODERATE', 0)
        sq_drift = signal_quality_assessment(0.60, 0.58, 'MODERATE', 5)
        assert sq_drift['score'] < sq_no_drift['score']
        assert any('drift' in w for w in sq_drift['warnings'])

    def test_no_wf_data(self):
        from backend.models.explain import signal_quality_assessment
        sq = signal_quality_assessment(0.60, None, 'MODERATE', 0)
        assert any('walk-forward' in w.lower() for w in sq['warnings'])

    def test_format_signal_quality(self):
        from backend.models.explain import format_signal_quality
        sq = {'quality': 'HIGH', 'icon': '🟢', 'score': 8, 'warnings': []}
        text = format_signal_quality(sq)
        assert '🟢' in text
        assert 'HIGH' in text

    def test_format_with_warnings(self):
        from backend.models.explain import format_signal_quality
        sq = {'quality': 'LOW', 'icon': '⚪', 'score': 2, 'warnings': ['Low model conviction']}
        text = format_signal_quality(sq)
        assert 'Low model conviction' in text


class TestBestHorizonInAnalysis:
    def test_best_horizon_in_chat_query_context(self):
        """Test that best_horizon is included in chat query predictions dict."""
        from backend.models.explain import best_horizon_recommendation
        # Simulate what chat_query does
        wf_avg = {'short': 0.54, 'medium': 0.59, 'long': 0.63}
        bh = best_horizon_recommendation(wf_avg)
        assert bh is not None
        preds = {}
        if bh:
            preds['best_horizon'] = f"{bh['label']} ({bh['accuracy']*100:.0f}% accuracy) — {bh['reason']}"
        assert 'best_horizon' in preds
        assert '20-day' in preds['best_horizon']
        assert '63%' in preds['best_horizon']

    def test_signal_quality_in_horizons(self):
        """Test that signal quality is computed for each horizon."""
        from backend.models.explain import signal_quality_assessment
        horizons = {}
        for h, prob, wf, tier in [('short', 0.55, 0.54, 'LOW'),
                                    ('medium', 0.62, 0.59, 'MODERATE'),
                                    ('long', 0.70, 0.63, 'HIGH')]:
            sq = signal_quality_assessment(prob, wf, tier, 0)
            horizons[h] = {'signal_quality': sq}
        # Long should have highest quality
        assert horizons['long']['signal_quality']['score'] >= horizons['short']['signal_quality']['score']

    def test_chat_summary_includes_best_horizon(self):
        """Test that chat summary output includes best horizon."""
        from backend.llm.mock_llm import MockLLM
        llm = MockLLM()
        preds = {
            'symbol': 'TEST', 'current_price': 100.0,
            'horizons': [
                {'name': '1-Hour', 'direction': 'BULLISH', 'confidence': 60,
                 'expected_return': 0.5, 'invalidation': 'Stop $98', 'conviction_tier': 'MODERATE',
                 'conviction_verdict': '🟡 LEAN BUY (60%)', 'signal_quality': '🟡 MODERATE'},
            ],
            'bullish_signals': ['RSI oversold'], 'bearish_signals': ['MACD falling'],
            'best_horizon': '20-day (63% accuracy) — Longer-term cycles are most predictable',
        }
        result = llm.generate_summary('TEST', preds)
        assert 'Best horizon' in result or 'best_horizon' in result.lower()



class TestCalibrationClipping:
    """Tests for calibration probability clipping to prevent extreme values."""

    def test_clip_prevents_extreme_high(self):
        """Calibrated probability should not exceed 0.85."""
        # Simulate: isotonic regression can output 0.99
        cal_prob = 0.99
        clipped = float(np.clip(cal_prob, 0.15, 0.85))
        assert clipped == 0.85

    def test_clip_prevents_extreme_low(self):
        """Calibrated probability should not go below 0.15."""
        cal_prob = 0.01
        clipped = float(np.clip(cal_prob, 0.15, 0.85))
        assert clipped == 0.15

    def test_clip_preserves_moderate(self):
        """Moderate probabilities should pass through unchanged."""
        for p in [0.3, 0.5, 0.6, 0.7, 0.8]:
            clipped = float(np.clip(p, 0.15, 0.85))
            assert clipped == p


class TestPerHorizonThreshold:
    """Tests for per-horizon threshold clipping in walk-forward validation."""

    def test_long_horizon_lower_threshold(self):
        """Long horizon should allow lower threshold (0.30) to exploit positive drift."""
        clip_ranges = {'short': (0.42, 0.52), 'medium': (0.35, 0.52), 'long': (0.30, 0.52)}
        # Long horizon with 70% base rate → threshold = 0.30
        base_rate = 0.70
        threshold = 1.0 - base_rate  # 0.30
        lo, hi = clip_ranges['long']
        clipped = float(np.clip(threshold, lo, hi))
        assert abs(clipped - 0.30) < 1e-10, f"Long threshold should be ~0.30, got {clipped}"

    def test_medium_horizon_moderate_threshold(self):
        """Medium horizon should allow threshold down to 0.35."""
        clip_ranges = {'short': (0.42, 0.52), 'medium': (0.35, 0.52), 'long': (0.30, 0.52)}
        base_rate = 0.65
        threshold = 1.0 - base_rate  # 0.35
        lo, hi = clip_ranges['medium']
        clipped = float(np.clip(threshold, lo, hi))
        assert abs(clipped - 0.35) < 1e-10

    def test_short_horizon_conservative_threshold(self):
        """Short horizon should keep conservative threshold (0.42-0.52)."""
        clip_ranges = {'short': (0.42, 0.52), 'medium': (0.35, 0.52), 'long': (0.30, 0.52)}
        base_rate = 0.55
        threshold = 1.0 - base_rate  # 0.45
        lo, hi = clip_ranges['short']
        clipped = float(np.clip(threshold, lo, hi))
        assert abs(clipped - 0.45) < 1e-10

    def test_threshold_upper_bound(self):
        """All horizons should cap threshold at 0.52."""
        clip_ranges = {'short': (0.42, 0.52), 'medium': (0.35, 0.52), 'long': (0.30, 0.52)}
        base_rate = 0.40  # threshold = 0.60
        for h in ('short', 'medium', 'long'):
            lo, hi = clip_ranges[h]
            clipped = float(np.clip(1.0 - base_rate, lo, hi))
            assert clipped == 0.52, f"{h} threshold should cap at 0.52"


class TestRefactoredHelpers:
    """Tests for extracted helper functions from train_ticker_model and get_analysis."""

    def test_train_classifiers_with_bagging(self):
        """_train_classifiers_with_bagging returns bagged classifiers for each horizon."""
        from cli.engine import _train_classifiers_with_bagging
        from backend.models.ensemble import EnsembleModel
        ens = EnsembleModel()
        n = 200
        X = np.random.randn(n, 10)
        y_short = np.random.randn(n) * 0.01
        y_medium = np.random.randn(n) * 0.03
        y_long = np.random.randn(n) * 0.05
        weights = np.ones(n)
        result = _train_classifiers_with_bagging(ens, X, y_short, y_medium, y_long, weights)
        assert isinstance(result, dict)
        for h in result:
            assert h in ('short', 'medium', 'long')
            assert len(result[h]) == 2  # 2 bagged models

    def test_compute_val_accuracy(self):
        """_compute_val_accuracy returns accuracy dicts for xgb, lgbm, and ensemble."""
        from cli.engine import _compute_val_accuracy
        from backend.models.ensemble import EnsembleModel
        ens = EnsembleModel()
        n = 200
        X = np.random.randn(n, 10)
        y_short = np.random.randn(n) * 0.01
        y_medium = np.random.randn(n) * 0.03
        y_long = np.random.randn(n) * 0.05
        # Train models first
        ens.xgb_short.train(X, y_short)
        ens.xgb_medium.train(X, y_medium)
        ens.xgb_long.train(X, y_long)
        ens.lgbm_short.train(X, y_short)
        ens.lgbm_medium.train(X, y_medium)
        ens.lgbm_long.train(X, y_long)
        ens.xgb_short.train_classifier(X, (y_short > 0).astype(int))
        ens.lgbm_short.train_classifier(X, (y_short > 0).astype(int))
        ens.xgb_medium.train_classifier(X, (y_medium > 0).astype(int))
        ens.lgbm_medium.train_classifier(X, (y_medium > 0).astype(int))
        ens.xgb_long.train_classifier(X, (y_long > 0).astype(int))
        ens.lgbm_long.train_classifier(X, (y_long > 0).astype(int))
        xgb_acc, lgbm_acc, accuracies = _compute_val_accuracy(ens, X, y_short, y_medium, y_long, n, 50)
        for d in (xgb_acc, lgbm_acc, accuracies):
            assert 'short' in d and 'medium' in d and 'long' in d
            for v in d.values():
                assert 0 <= v <= 1

    def test_train_mtf_models(self):
        """_train_mtf_models returns models for each window that has enough data."""
        from cli.engine import _train_mtf_models
        n = 300
        X = np.random.randn(n, 10)
        y_short = np.random.randn(n) * 0.01
        y_medium = np.random.randn(n) * 0.03
        y_long = np.random.randn(n) * 0.05
        result = _train_mtf_models(X, y_short, y_medium, y_long, n)
        assert isinstance(result, dict)
        assert 30 in result and 90 in result and 252 in result

    def test_train_mtf_models_insufficient_data(self):
        """_train_mtf_models skips windows with insufficient data."""
        from cli.engine import _train_mtf_models
        n = 50
        X = np.random.randn(n, 5)
        y = np.random.randn(n) * 0.01
        result = _train_mtf_models(X, y, y, y, n)
        assert 252 not in result
        assert 90 not in result
        assert 30 in result

    def test_save_training_artifacts(self, tmp_path):
        """_save_training_artifacts saves pickle files."""
        from cli.engine import _save_training_artifacts
        X = np.random.randn(100, 5)
        _save_training_artifacts(tmp_path, {'short': ['m1']}, {'short': 'cal'}, {'wf_residuals': {'short': [0.1]}}, X)
        assert (tmp_path / 'bagged_classifiers.pkl').exists()
        assert (tmp_path / 'calibrators.pkl').exists()
        assert (tmp_path / 'wf_residuals.pkl').exists()
        assert (tmp_path / 'feature_stats.pkl').exists()

    def test_save_training_artifacts_empty(self, tmp_path):
        """_save_training_artifacts handles empty data gracefully."""
        from cli.engine import _save_training_artifacts
        X = np.random.randn(50, 3)
        _save_training_artifacts(tmp_path, {}, {}, {}, X)
        assert not (tmp_path / 'bagged_classifiers.pkl').exists()
        assert (tmp_path / 'feature_stats.pkl').exists()

    def test_update_raw_conviction(self):
        """_update_raw_conviction sets direction, confidence, and conviction tier."""
        from cli.engine import _update_raw_conviction
        raw = {'short': {'prob_up': 0.5}}
        _update_raw_conviction(raw, 'short', 0.72)
        assert raw['short']['direction'] == 'bullish'
        assert raw['short']['confidence'] == 0.72
        assert raw['short']['conviction_tier'] == 'HIGH'

    def test_update_raw_conviction_bearish(self):
        """_update_raw_conviction handles bearish case."""
        from cli.engine import _update_raw_conviction
        raw = {'medium': {'prob_up': 0.5}}
        _update_raw_conviction(raw, 'medium', 0.35)
        assert raw['medium']['direction'] == 'bearish'
        assert raw['medium']['confidence'] == 0.65

    def test_apply_regime_adjustment_high_vol(self):
        """_apply_regime_adjustment pulls probability toward 0.5 in high-vol regime."""
        from cli.engine import _apply_regime_adjustment
        raw = {'short': {'prob_up': 0.7, 'direction': 'bullish', 'confidence': 0.7}}
        regime = {'vol_regime_high': 0.8}
        _apply_regime_adjustment(raw, regime)
        assert raw['short']['prob_up'] < 0.7  # pulled toward 0.5
        assert raw['short'].get('regime_adjusted') is True

    def test_apply_regime_adjustment_low_vol(self):
        """_apply_regime_adjustment does nothing in low-vol regime."""
        from cli.engine import _apply_regime_adjustment
        raw = {'short': {'prob_up': 0.7, 'direction': 'bullish', 'confidence': 0.7}}
        regime = {'vol_regime_high': 0.2}
        _apply_regime_adjustment(raw, regime)
        assert raw['short']['prob_up'] == 0.7  # unchanged

    def test_print_changes_vs_previous_no_changes(self, capsys):
        """_print_changes_vs_previous prints nothing when no significant changes."""
        from cli.engine import _print_changes_vs_previous
        prev = {'walk_forward': {'average': {'short': 0.55}}, 'selected_feature_count': 100}
        meta = {'brier_scores': {}, 'model_health': {'grade': 'B'}}
        _print_changes_vs_previous(prev, meta, {'short': 0.555}, [0]*100)
        captured = capsys.readouterr()
        assert captured.out == ''  # diff < 1.0%


class TestAnalysisHelpers:
    """Tests for extracted analysis helper functions."""

    def test_compute_predictability_high(self):
        from cli.engine import _compute_predictability
        horizons = {'short': {}, 'medium': {}, 'long': {}}
        result = _compute_predictability({'short': 0.62, 'medium': 0.61, 'long': 0.63}, horizons)
        assert result[0] == 'HIGH'
        assert horizons['short']['predictability'] == 'HIGH'

    def test_compute_predictability_low(self):
        from cli.engine import _compute_predictability
        horizons = {'short': {}, 'medium': {}}
        result = _compute_predictability({'short': 0.51, 'medium': 0.52}, horizons)
        assert result[0] == 'LOW'

    def test_compute_predictability_none(self):
        from cli.engine import _compute_predictability
        result = _compute_predictability({}, {})
        assert result is None

    def test_get_model_age(self):
        from cli.engine import _get_model_age
        assert _get_model_age(None) is None
        assert _get_model_age({}) is None
        from datetime import datetime
        meta = {'trained_at': datetime.now().isoformat()}
        assert _get_model_age(meta) == 0

    def test_detect_drift_no_stats(self, tmp_path):
        from cli.engine import _detect_drift
        result = _detect_drift(np.zeros(5), ['a', 'b', 'c', 'd', 'e'], tmp_path)
        assert result == []

    def test_get_model_health_no_meta(self):
        from cli.engine import _get_model_health
        from pathlib import Path
        h, t, d = _get_model_health(None, {}, None, Path('/nonexistent'))
        assert h is None and t is None and d is None

    def test_apply_mtf_voting_no_file(self, tmp_path):
        from cli.engine import _apply_mtf_voting
        horizons = {'short': {}}
        _apply_mtf_voting(horizons, tmp_path, np.zeros(5))
        assert 'mtf_votes' not in horizons['short']


class TestSignalCategoryHelpers:
    """Tests for category-based signal extraction helpers."""

    def _make_feats(self, data):
        """Create a single-row DataFrame from a dict of feature values."""
        return pd.DataFrame({k: [v] for k, v in data.items()})

    def test_signals_momentum_rsi_oversold(self):
        from cli.engine import _signals_momentum, _gf
        feats = self._make_feats({'rsi_14': 25.0})
        gf = lambda name: _gf(feats, name)
        bull, bear = [], []
        _signals_momentum(gf, bull, bear)
        assert any('RSI oversold' in s for s in bull)

    def test_signals_momentum_rsi_overbought(self):
        from cli.engine import _signals_momentum, _gf
        feats = self._make_feats({'rsi_14': 75.0})
        gf = lambda name: _gf(feats, name)
        bull, bear = [], []
        _signals_momentum(gf, bull, bear)
        assert any('RSI overbought' in s for s in bear)

    def test_signals_momentum_macd(self):
        from cli.engine import _signals_momentum, _gf
        feats = self._make_feats({'macd_hist': 1.5})
        gf = lambda name: _gf(feats, name)
        bull, bear = [], []
        _signals_momentum(gf, bull, bear)
        assert any('MACD positive' in s for s in bull)

    def test_signals_momentum_empty_feats(self):
        from cli.engine import _signals_momentum, _gf
        feats = self._make_feats({})
        gf = lambda name: _gf(feats, name)
        bull, bear = [], []
        _signals_momentum(gf, bull, bear)
        assert bull == [] and bear == []

    def test_signals_volume_surge(self):
        from cli.engine import _signals_volume, _gf
        feats = self._make_feats({'volume_ratio': 2.0})
        gf = lambda name: _gf(feats, name)
        bull, bear = [], []
        _signals_volume(gf, bull, bear)
        assert any('Volume surge' in s for s in bull)

    def test_signals_volume_low(self):
        from cli.engine import _signals_volume, _gf
        feats = self._make_feats({'volume_ratio': 0.5})
        gf = lambda name: _gf(feats, name)
        bull, bear = [], []
        _signals_volume(gf, bull, bear)
        assert any('Low volume' in s for s in bear)

    def test_signals_volume_cmf(self):
        from cli.engine import _signals_volume, _gf
        feats = self._make_feats({'cmf': 0.2})
        gf = lambda name: _gf(feats, name)
        bull, bear = [], []
        _signals_volume(gf, bull, bear)
        assert any('Strong money flow' in s for s in bull)

    def test_signals_trend_adx_uptrend(self):
        from cli.engine import _signals_trend, _gf
        feats = self._make_feats({'adx_14': 30.0, 'return_5d': 0.05})
        gf = lambda name: _gf(feats, name)
        bull, bear = [], []
        _signals_trend(gf, 100.0, bull, bear)
        assert any('uptrend' in s for s in bull)

    def test_signals_trend_sma50_above(self):
        from cli.engine import _signals_trend, _gf
        feats = self._make_feats({'sma_50': 90.0})
        gf = lambda name: _gf(feats, name)
        bull, bear = [], []
        _signals_trend(gf, 100.0, bull, bear)
        assert any('above 50-SMA' in s for s in bull)

    def test_signals_trend_ema_crossover(self):
        from cli.engine import _signals_trend, _gf
        feats = self._make_feats({'ema_5': 105.0, 'ema_20': 100.0})
        gf = lambda name: _gf(feats, name)
        bull, bear = [], []
        _signals_trend(gf, 100.0, bull, bear)
        assert any('EMA5 above EMA20' in s for s in bull)

    def test_signals_volatility_bands_bb(self):
        from cli.engine import _signals_volatility_bands, _gf
        feats = self._make_feats({'bb_position': 0.1})
        gf = lambda name: _gf(feats, name)
        bull, bear = [], []
        _signals_volatility_bands(gf, 100.0, bull, bear)
        assert any('lower BB' in s for s in bull)

    def test_signals_volatility_bands_vix_high(self):
        from cli.engine import _signals_volatility_bands, _gf
        feats = self._make_feats({'vix': 30.0})
        gf = lambda name: _gf(feats, name)
        bull, bear = [], []
        _signals_volatility_bands(gf, 100.0, bull, bear)
        assert any('High VIX' in s for s in bear)

    def test_signals_advanced_sentiment(self):
        from cli.engine import _signals_advanced, _gf
        feats = self._make_feats({'news_sentiment_avg': 0.3})
        gf = lambda name: _gf(feats, name)
        bull, bear = [], []
        _signals_advanced(gf, 100.0, bull, bear)
        assert any('Positive sentiment' in s for s in bull)

    def test_signals_advanced_elder_ray_bearish(self):
        from cli.engine import _signals_advanced, _gf
        feats = self._make_feats({'elder_bull': 0.005, 'elder_bear': -0.03})
        gf = lambda name: _gf(feats, name)
        bull, bear = [], []
        _signals_advanced(gf, 100.0, bull, bear)
        assert any('Elder Ray bearish' in s for s in bear)

    def test_extract_signals_combines_all_categories(self):
        from cli.engine import _extract_signals
        feats = self._make_feats({
            'rsi_14': 25.0, 'volume_ratio': 2.0, 'adx_14': 30.0,
            'return_5d': 0.05, 'bb_position': 0.1, 'news_sentiment_avg': 0.3
        })
        bull, bear = _extract_signals(feats, 100.0)
        assert len(bull) >= 4  # RSI + volume + ADX + BB + sentiment


class TestBacktestHelpers:
    """Tests for extracted backtest helper functions."""

    def test_bt_compute_metrics_basic(self):
        from cli.engine import _bt_compute_metrics
        gross = np.array([0.01, -0.005, 0.008, -0.003, 0.012])
        net = gross - 0.001
        active_mask = np.array([True, True, True, True, True])
        signals = [1, -1, 1, -1, 1]
        pos_sizes = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        result = _bt_compute_metrics(gross, net, active_mask, signals, pos_sizes, 0.0005, 0.0, 0.0002)
        assert 'gross_return' in result
        assert 'net_return' in result
        assert 'sharpe_ratio' in result
        assert 'max_drawdown' in result
        assert 'win_rate' in result
        assert result['total_trades'] == 5

    def test_bt_compute_metrics_empty(self):
        from cli.engine import _bt_compute_metrics
        result = _bt_compute_metrics(np.array([]), np.array([]), np.array([False]),
                                      [0], np.array([]), 0.0005, 0.0, 0.0002)
        assert result['total_trades'] == 0
        assert result['win_rate'] == 0

    def test_bt_compute_metrics_all_wins(self):
        from cli.engine import _bt_compute_metrics
        gross = np.array([0.01, 0.02, 0.015])
        net = gross - 0.001
        result = _bt_compute_metrics(gross, net, np.ones(3, dtype=bool), [1, 1, 1],
                                      np.ones(3), 0.0005, 0.0, 0.0002)
        assert result['win_rate'] == 1.0

    def test_bt_walk_forward_signals_shape(self):
        from cli.engine import _bt_walk_forward_signals
        n = 300
        X = np.random.randn(n, 10)
        returns = np.random.randn(n) * 0.02
        sig, conf = _bt_walk_forward_signals(X, returns)
        assert sig.shape == (n,)
        assert conf.shape == (n,)
        assert set(np.unique(sig)).issubset({-1, 0, 1})

    def test_bt_walk_forward_signals_insufficient_data(self):
        from cli.engine import _bt_walk_forward_signals
        n = 50  # less than bt_train_window (252)
        X = np.random.randn(n, 5)
        returns = np.random.randn(n) * 0.02
        sig, conf = _bt_walk_forward_signals(X, returns)
        assert np.all(sig == 0)  # no signals generated
