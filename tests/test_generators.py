"""Tests for mock data generators."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.data.mock_generators import (
    generate_ohlcv, get_ohlcv, get_options_chain, 
    get_fundamentals, get_sentiment, get_macro_data
)

class TestPriceGenerator:
    def test_generate_ohlcv(self):
        df = generate_ohlcv("TSLA", seed=42)
        assert len(df) > 400  # ~2 years of trading days
        assert all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume'])
        assert (df['high'] >= df['low']).all()
        assert (df['high'] >= df['close']).all()
        assert (df['low'] <= df['close']).all()
    
    def test_get_ohlcv_daily(self):
        df = get_ohlcv("AAPL", "daily", 30)
        assert len(df) == 30
    
    def test_different_symbols(self):
        tsla = get_ohlcv("TSLA", "daily", 10)
        aapl = get_ohlcv("AAPL", "daily", 10)
        # Different symbols should have different prices
        assert tsla['close'].iloc[-1] != aapl['close'].iloc[-1]

class TestOptionsGenerator:
    def test_get_options_chain(self):
        df = get_options_chain("TSLA", 250.0)
        assert len(df) > 0
        assert 'strike' in df.columns
        assert 'option_type' in df.columns
        assert 'iv' in df.columns
        assert set(df['option_type'].unique()) == {'call', 'put'}
    
    def test_strikes_around_spot(self):
        spot = 100.0
        df = get_options_chain("TEST", spot)
        # Should have strikes above and below spot
        assert df['strike'].min() < spot
        assert df['strike'].max() > spot

class TestFundamentalsGenerator:
    def test_get_fundamentals(self):
        data = get_fundamentals("TSLA")
        assert 'quarterly' in data
        df = data['quarterly']
        assert len(df) >= 4  # At least 4 quarters
        assert 'eps' in df.columns
        assert 'revenue' in df.columns
        assert 'eps_surprise_pct' in df.columns

class TestSentimentGenerator:
    def test_get_sentiment(self):
        data = get_sentiment("TSLA")
        assert 'news' in data
        assert len(data['news']) > 0
        assert 'sentiment' in data['news'].columns
        assert 'headline' in data['news'].columns

class TestMacroGenerator:
    def test_get_macro_data(self):
        data = get_macro_data()
        # Data is structured with sub-dicts
        assert 'interest_rates' in data or 'vix' in data
        if 'vix' in data:
            assert len(data['vix']) > 0
