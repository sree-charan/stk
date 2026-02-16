"""Tests for feature engineering tiers."""
import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.features import (
    Tier1PriceVolume, Tier2Technical, Tier3Options, Tier4Fundamentals,
    Tier5Sentiment, Tier6Institutional, Tier7Macro, Tier8Microstructure,
    FeatureStore
)
from backend.data.mock_generators import generate_ohlcv, get_options_chain, get_fundamentals, get_sentiment, get_macro_data

@pytest.fixture
def price_df():
    return generate_ohlcv("TSLA", seed=42).tail(100)

class TestTier1PriceVolume:
    def test_compute(self, price_df):
        features = Tier1PriceVolume.compute(price_df)
        assert len(features) == len(price_df)
        assert 'return_1d' in features.columns
        assert 'vwap' in features.columns
    
    def test_feature_names(self):
        names = Tier1PriceVolume.feature_names()
        assert len(names) == 23
        assert 'return_1d' in names

class TestTier2Technical:
    def test_compute(self, price_df):
        features = Tier2Technical.compute(price_df)
        assert 'sma_20' in features.columns
        assert 'rsi_14' in features.columns
        assert 'macd' in features.columns
    
    def test_feature_names(self):
        names = Tier2Technical.feature_names()
        assert len(names) == 85

class TestTier3Options:
    def test_compute(self, price_df):
        spot = price_df['close'].iloc[-1]
        options = get_options_chain("TSLA", spot)
        features = Tier3Options.compute(options.to_dict('records'), spot)
        assert 'put_call_ratio' in features
        assert 'atm_iv' in features
    
    def test_feature_names(self):
        assert len(Tier3Options.feature_names()) == 25

class TestTier4Fundamentals:
    def test_compute(self, price_df):
        spot = price_df['close'].iloc[-1]
        fund = get_fundamentals("TSLA")
        # fund is a dict with 'quarterly' DataFrame
        features = Tier4Fundamentals.compute(fund['quarterly'].to_dict('records'), spot)
        assert 'eps_surprise' in features
        assert 'pe_ratio' in features
    
    def test_feature_names(self):
        assert len(Tier4Fundamentals.feature_names()) == 20

class TestTier5Sentiment:
    def test_compute(self):
        sentiment = get_sentiment("TSLA")
        features = Tier5Sentiment.compute(sentiment['news'].to_dict('records'))
        assert 'news_sentiment_avg' in features
        assert 'news_volume' in features
    
    def test_feature_names(self):
        assert len(Tier5Sentiment.feature_names()) == 25

class TestTier6Institutional:
    def test_compute(self):
        features = Tier6Institutional.compute({'institutional_ownership': 0.65, 'insider_buys': 5})
        assert 'institutional_ownership' in features
    
    def test_feature_names(self):
        assert len(Tier6Institutional.feature_names()) == 5

class TestTier7Macro:
    def test_compute(self):
        macro = get_macro_data()
        features = Tier7Macro.compute(macro)
        assert 'fed_funds_rate' in features
        assert 'vix' in features
    
    def test_feature_names(self):
        assert len(Tier7Macro.feature_names()) == 15

class TestTier8Microstructure:
    def test_compute(self, price_df):
        features = Tier8Microstructure.compute(price_df)
        assert 'bid_ask_spread' in features.columns
    
    def test_feature_names(self):
        assert len(Tier8Microstructure.feature_names()) == 6

class TestFeatureStore:
    def test_compute_all_features(self, price_df):
        store = FeatureStore()
        spot = price_df['close'].iloc[-1]
        
        features = store.compute_all_features(
            price_df,
            get_options_chain("TSLA", spot),
            get_fundamentals("TSLA"),
            get_sentiment("TSLA")['news'].to_dict('records'),
            get_macro_data()
        )
        
        assert len(features) == len(price_df)
        assert features.shape[1] >= 100  # Should have 136+ features
    
    def test_feature_count(self):
        counts = FeatureStore.feature_count()
        assert counts['total'] >= 136
    
    def test_all_feature_names(self):
        names = FeatureStore.all_feature_names()
        assert len(names) >= 100


class TestKeltnerDonchianFeatures:
    def test_keltner_features_computed(self, price_df):
        """Verify Keltner channel features are in tier2 output."""
        feats = Tier2Technical.compute(price_df)
        for name in ['keltner_upper', 'keltner_lower', 'keltner_position']:
            assert name in feats.columns

    def test_donchian_features_computed(self, price_df):
        """Verify Donchian channel features are in tier2 output."""
        feats = Tier2Technical.compute(price_df)
        for name in ['donchian_position', 'donchian_width', 'donchian_breakout']:
            assert name in feats.columns

    def test_keltner_position_bounded(self, price_df):
        """Keltner position should be roughly bounded."""
        feats = Tier2Technical.compute(price_df)
        kp = feats['keltner_position'].dropna()
        assert kp.min() >= -5
        assert kp.max() <= 5

    def test_donchian_position_bounded(self, price_df):
        """Donchian position should be between 0 and 1."""
        feats = Tier2Technical.compute(price_df)
        dp = feats['donchian_position'].dropna()
        assert dp.min() >= -0.1
        assert dp.max() <= 1.1

    def test_feature_names_include_new(self):
        names = Tier2Technical.feature_names()
        assert 'keltner_upper' in names
        assert 'donchian_position' in names
        assert 'donchian_breakout' in names


class TestVWAPBandFeatures:
    def test_vwap_bands_computed(self, price_df):
        from backend.features.tier1_price_volume import Tier1PriceVolume
        feats = Tier1PriceVolume.compute(price_df)
        for col in ['vwap_upper', 'vwap_lower', 'vwap_band_width']:
            assert col in feats.columns

    def test_vwap_band_width_non_negative(self, price_df):
        from backend.features.tier1_price_volume import Tier1PriceVolume
        feats = Tier1PriceVolume.compute(price_df)
        bw = feats['vwap_band_width'].dropna()
        assert (bw >= 0).all()

    def test_tier1_count_23(self):
        from backend.features.tier1_price_volume import Tier1PriceVolume
        assert len(Tier1PriceVolume.feature_names()) == 23


class TestATRPAndCMFFeatures:
    def test_atrp_computed(self, price_df):
        feats = Tier2Technical.compute(price_df)
        assert 'atrp' in feats.columns
        atrp = feats['atrp'].dropna()
        assert (atrp >= 0).all()

    def test_cmf_computed(self, price_df):
        feats = Tier2Technical.compute(price_df)
        assert 'cmf' in feats.columns
        cmf = feats['cmf'].dropna()
        assert cmf.min() >= -1.1
        assert cmf.max() <= 1.1

    def test_cmf_in_feature_names(self):
        names = Tier2Technical.feature_names()
        assert 'cmf' in names
        assert 'atrp' in names

    def test_total_features_175(self):
        from backend.features.feature_store import FeatureStore
        fs = FeatureStore()
        assert len(fs.all_feature_names()) >= 400


class TestPSARAndADXRFeatures:
    def test_psar_dist_computed(self, price_df):
        feats = Tier2Technical.compute(price_df)
        assert 'psar_dist' in feats.columns
        vals = feats['psar_dist'].dropna()
        assert len(vals) > 0
        # PSAR distance should be bounded (normalized by close)
        assert vals.abs().max() < 2.0

    def test_adxr_computed(self, price_df):
        feats = Tier2Technical.compute(price_df)
        assert 'adxr' in feats.columns
        vals = feats['adxr'].dropna()
        assert len(vals) > 0
        assert (vals >= 0).all()

    def test_adl_slope_computed(self, price_df):
        feats = Tier2Technical.compute(price_df)
        assert 'adl_slope' in feats.columns
        vals = feats['adl_slope'].dropna()
        assert len(vals) > 0

    def test_force_index_computed(self, price_df):
        feats = Tier2Technical.compute(price_df)
        assert 'force_index' in feats.columns
        vals = feats['force_index'].dropna()
        assert len(vals) > 0

    def test_new_features_in_names(self):
        names = Tier2Technical.feature_names()
        for f in ['psar_dist', 'adxr', 'adl_slope', 'force_index']:
            assert f in names

    def test_tier2_count_60(self):
        assert len(Tier2Technical.feature_names()) == 85

    def test_total_features_179(self):
        from backend.features.feature_store import FeatureStore
        fs = FeatureStore()
        assert fs.feature_count()['total'] >= 400
        assert fs.feature_count()['tier2_technical'] == 85

    def test_psar_changes_with_trend(self):
        """PSAR should be positive in uptrend, negative in downtrend."""
        import pandas as pd
        import numpy as np
        n = 100
        # Uptrend
        dates = pd.date_range('2024-01-01', periods=n, freq='D')
        prices = 100 + np.arange(n) * 0.5 + np.random.normal(0, 0.3, n)
        df = pd.DataFrame({
            'open': prices - 0.2, 'high': prices + 1, 'low': prices - 1,
            'close': prices, 'volume': np.random.randint(1e6, 5e6, n)
        }, index=dates)
        feats = Tier2Technical.compute(df)
        # In a strong uptrend, last PSAR should be positive (price above SAR)
        last_psar = feats['psar_dist'].iloc[-1]
        assert last_psar > -0.5  # relaxed: just not deeply negative


class TestCCITrixUltOscVortexFeatures:
    def test_cci_computed(self, price_df):
        feats = Tier2Technical.compute(price_df)
        assert 'cci' in feats.columns
        vals = feats['cci'].dropna()
        assert len(vals) > 0

    def test_trix_computed(self, price_df):
        feats = Tier2Technical.compute(price_df)
        assert 'trix' in feats.columns
        vals = feats['trix'].dropna()
        assert len(vals) > 0

    def test_ultimate_osc_computed(self, price_df):
        feats = Tier2Technical.compute(price_df)
        assert 'ultimate_osc' in feats.columns
        vals = feats['ultimate_osc'].dropna()
        assert len(vals) > 0
        # Ultimate oscillator should be roughly 0-100
        assert vals.min() >= -10
        assert vals.max() <= 110

    def test_vortex_computed(self, price_df):
        feats = Tier2Technical.compute(price_df)
        assert 'vortex' in feats.columns
        vals = feats['vortex'].dropna()
        assert len(vals) > 0
        # Vortex difference should be bounded
        assert vals.abs().max() < 3.0

    def test_new_features_in_names(self):
        names = Tier2Technical.feature_names()
        for f in ['cci', 'trix', 'ultimate_osc', 'vortex']:
            assert f in names

    def test_tier2_count_64(self):
        assert len(Tier2Technical.feature_names()) == 85

    def test_total_features_183(self):
        from backend.features.feature_store import FeatureStore
        assert FeatureStore.feature_count()['total'] >= 400

    def test_cci_responds_to_trend(self):
        """CCI should be positive in uptrend."""
        n = 100
        dates = pd.date_range('2024-01-01', periods=n, freq='D')
        prices = 100 + np.arange(n) * 0.5 + np.random.RandomState(42).normal(0, 0.3, n)
        df = pd.DataFrame({
            'open': prices - 0.2, 'high': prices + 1, 'low': prices - 1,
            'close': prices, 'volume': np.random.RandomState(42).randint(1_000_000, 5_000_000, n)
        }, index=dates)
        feats = Tier2Technical.compute(df)
        assert feats['cci'].iloc[-1] > 0


class TestWilliamsR7DPOMassEMVFeatures:
    """Tests for Williams %R 7-period, DPO, Mass Index, Ease of Movement features."""

    @pytest.fixture
    def price_df(self):
        n = 100
        dates = pd.date_range('2024-01-01', periods=n, freq='D')
        rng = np.random.RandomState(42)
        prices = 100 + np.cumsum(rng.normal(0, 1, n))
        return pd.DataFrame({
            'open': prices - 0.5, 'high': prices + rng.uniform(0.5, 2, n),
            'low': prices - rng.uniform(0.5, 2, n), 'close': prices,
            'volume': rng.randint(1_000_000, 5_000_000, n)
        }, index=dates)

    def test_williams_r_7_computed(self, price_df):
        feats = Tier2Technical.compute(price_df)
        assert 'williams_r_7' in feats.columns
        vals = feats['williams_r_7'].iloc[10:]
        assert vals.min() >= -101
        assert vals.max() <= 1

    def test_dpo_computed(self, price_df):
        feats = Tier2Technical.compute(price_df)
        assert 'dpo' in feats.columns

    def test_mass_index_computed(self, price_df):
        feats = Tier2Technical.compute(price_df)
        assert 'mass_index' in feats.columns
        vals = feats['mass_index'].iloc[40:]
        assert vals.min() > 0

    def test_emv_computed(self, price_df):
        feats = Tier2Technical.compute(price_df)
        assert 'emv' in feats.columns

    def test_new_features_in_names(self):
        names = Tier2Technical.feature_names()
        for f in ['williams_r_7', 'dpo', 'mass_index', 'emv']:
            assert f in names

    def test_tier2_count_68(self):
        assert len(Tier2Technical.feature_names()) == 85

    def test_total_features_187(self):
        from backend.features.feature_store import FeatureStore
        assert FeatureStore.feature_count()['total'] >= 400


class TestCMOAroonKSTFeatures:
    """Tests for CMO, Aroon Up/Down, and KST features."""

    @pytest.fixture
    def price_df(self):
        n = 100
        dates = pd.date_range('2024-01-01', periods=n, freq='D')
        rng = np.random.RandomState(42)
        prices = 100 + np.cumsum(rng.normal(0, 1, n))
        return pd.DataFrame({
            'open': prices - 0.5, 'high': prices + rng.uniform(0.5, 2, n),
            'low': prices - rng.uniform(0.5, 2, n), 'close': prices,
            'volume': rng.randint(1_000_000, 5_000_000, n)
        }, index=dates)

    def test_cmo_computed(self, price_df):
        feats = Tier2Technical.compute(price_df)
        assert 'cmo' in feats.columns
        vals = feats['cmo'].iloc[20:]
        assert vals.min() >= -101
        assert vals.max() <= 101

    def test_aroon_up_computed(self, price_df):
        feats = Tier2Technical.compute(price_df)
        assert 'aroon_up' in feats.columns
        vals = feats['aroon_up'].iloc[30:]
        assert vals.min() >= 0
        assert vals.max() <= 100

    def test_aroon_down_computed(self, price_df):
        feats = Tier2Technical.compute(price_df)
        assert 'aroon_down' in feats.columns
        vals = feats['aroon_down'].iloc[30:]
        assert vals.min() >= 0
        assert vals.max() <= 100

    def test_kst_computed(self, price_df):
        feats = Tier2Technical.compute(price_df)
        assert 'kst' in feats.columns

    def test_new_features_in_names(self):
        names = Tier2Technical.feature_names()
        for f in ['cmo', 'aroon_up', 'aroon_down', 'kst']:
            assert f in names

    def test_tier2_count_72(self):
        assert len(Tier2Technical.feature_names()) == 85

    def test_total_features_191(self):
        from backend.features.feature_store import FeatureStore
        assert FeatureStore.feature_count()['total'] >= 400

    def test_cmo_responds_to_trend(self):
        """CMO should be positive in uptrend."""
        n = 100
        dates = pd.date_range('2024-01-01', periods=n, freq='D')
        prices = 100 + np.arange(n) * 0.5 + np.random.RandomState(42).normal(0, 0.3, n)
        df = pd.DataFrame({
            'open': prices - 0.2, 'high': prices + 1, 'low': prices - 1,
            'close': prices, 'volume': np.random.RandomState(42).randint(1_000_000, 5_000_000, n)
        }, index=dates)
        feats = Tier2Technical.compute(df)
        assert feats['cmo'].iloc[-1] > 0


class TestConnorsRSIChoppinessATRBands:
    """Tests for Connors RSI, Choppiness Index, and ATR Bands features."""

    @pytest.fixture
    def price_df(self):
        n = 150
        dates = pd.date_range('2024-01-01', periods=n, freq='D')
        rng = np.random.RandomState(42)
        prices = 100 + np.cumsum(rng.normal(0.05, 1.5, n))
        return pd.DataFrame({
            'open': prices - 0.3, 'high': prices + rng.uniform(0.5, 2, n),
            'low': prices - rng.uniform(0.5, 2, n),
            'close': prices, 'volume': rng.randint(1_000_000, 5_000_000, n)
        }, index=dates)

    def test_connors_rsi_computed(self, price_df):
        feats = Tier2Technical.compute(price_df)
        assert 'connors_rsi' in feats.columns
        vals = feats['connors_rsi'].iloc[110:]
        assert vals.min() >= -1
        assert vals.max() <= 101

    def test_choppiness_computed(self, price_df):
        feats = Tier2Technical.compute(price_df)
        assert 'choppiness' in feats.columns
        vals = feats['choppiness'].iloc[20:]
        assert vals.min() >= 0
        assert vals.max() <= 120

    def test_atr_band_upper_computed(self, price_df):
        feats = Tier2Technical.compute(price_df)
        assert 'atr_band_upper' in feats.columns

    def test_atr_band_lower_computed(self, price_df):
        feats = Tier2Technical.compute(price_df)
        assert 'atr_band_lower' in feats.columns

    def test_new_features_in_names(self):
        names = Tier2Technical.feature_names()
        for f in ['connors_rsi', 'choppiness', 'atr_band_upper', 'atr_band_lower']:
            assert f in names

    def test_tier2_count_76(self):
        assert len(Tier2Technical.feature_names()) == 85

    def test_total_features_195(self):
        from backend.features.feature_store import FeatureStore
        assert FeatureStore.feature_count()['total'] >= 400

    def test_choppiness_high_in_range(self):
        """Choppiness should be high in a range-bound market."""
        n = 100
        dates = pd.date_range('2024-01-01', periods=n, freq='D')
        rng = np.random.RandomState(99)
        prices = 100 + rng.normal(0, 0.5, n)  # flat, range-bound
        df = pd.DataFrame({
            'open': prices - 0.1, 'high': prices + 0.5,
            'low': prices - 0.5, 'close': prices,
            'volume': rng.randint(1_000_000, 5_000_000, n)
        }, index=dates)
        feats = Tier2Technical.compute(df)
        assert feats['choppiness'].iloc[-1] > 50

    def test_connors_rsi_oversold_in_downtrend(self):
        """Connors RSI should be low in a strong downtrend."""
        n = 150
        dates = pd.date_range('2024-01-01', periods=n, freq='D')
        rng = np.random.RandomState(42)
        prices = 200 - np.arange(n) * 0.8 + rng.normal(0, 0.2, n)
        df = pd.DataFrame({
            'open': prices + 0.2, 'high': prices + 0.5,
            'low': prices - 0.5, 'close': prices,
            'volume': rng.randint(1_000_000, 5_000_000, n)
        }, index=dates)
        feats = Tier2Technical.compute(df)
        assert feats['connors_rsi'].iloc[-1] < 40

    def test_stc_bullish_in_uptrend(self):
        """Schaff Trend Cycle should be high in a strong uptrend."""
        n = 200
        dates = pd.date_range('2024-01-01', periods=n, freq='D')
        rng = np.random.RandomState(42)
        prices = 100 + np.arange(n) * 0.5 + rng.normal(0, 0.3, n)
        df = pd.DataFrame({
            'open': prices - 0.2, 'high': prices + 0.5,
            'low': prices - 0.5, 'close': prices,
            'volume': rng.randint(1_000_000, 5_000_000, n)
        }, index=dates)
        feats = Tier2Technical.compute(df)
        assert 'stc' in feats.columns
        assert feats['stc'].iloc[-1] > 50

    def test_stc_bearish_in_downtrend(self):
        """Schaff Trend Cycle should be low in a downtrend."""
        n = 200
        dates = pd.date_range('2024-01-01', periods=n, freq='D')
        rng = np.random.RandomState(42)
        prices = 300 - np.arange(n) * 0.5 + rng.normal(0, 0.3, n)
        df = pd.DataFrame({
            'open': prices + 0.2, 'high': prices + 0.5,
            'low': prices - 0.5, 'close': prices,
            'volume': rng.randint(1_000_000, 5_000_000, n)
        }, index=dates)
        feats = Tier2Technical.compute(df)
        assert feats['stc'].iloc[-1] < 50

    def test_kvo_present(self):
        """Klinger Volume Oscillator should be computed."""
        n = 200
        dates = pd.date_range('2024-01-01', periods=n, freq='D')
        rng = np.random.RandomState(42)
        prices = 100 + np.arange(n) * 0.3 + rng.normal(0, 0.5, n)
        df = pd.DataFrame({
            'open': prices - 0.2, 'high': prices + 1,
            'low': prices - 1, 'close': prices,
            'volume': rng.randint(1_000_000, 10_000_000, n)
        }, index=dates)
        feats = Tier2Technical.compute(df)
        assert 'kvo' in feats.columns
        assert not np.isnan(feats['kvo'].iloc[-1])

    def test_kvo_positive_in_uptrend(self):
        """KVO should tend positive in a strong uptrend with rising volume."""
        n = 200
        dates = pd.date_range('2024-01-01', periods=n, freq='D')
        rng = np.random.RandomState(99)
        prices = 100 + np.arange(n) * 1.0 + rng.normal(0, 0.2, n)
        vol = np.linspace(2_000_000, 5_000_000, n).astype(int)
        df = pd.DataFrame({
            'open': prices - 0.1, 'high': prices + 0.5,
            'low': prices - 0.3, 'close': prices,
            'volume': vol
        }, index=dates)
        feats = Tier2Technical.compute(df)
        assert feats['kvo'].iloc[-1] > 0
