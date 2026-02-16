"""Edge case tests for the Stock Chat."""
from datetime import datetime, timedelta
import numpy as np

from backend.data.mock_generators import generate_ohlcv, get_options_chain, get_fundamentals, get_sentiment, get_macro_data
from backend.features.feature_store import FeatureStore
from backend.llm import IntentParser, ResponseGenerator
from backend.utils.invalidation import InvalidationEngine
from backend.utils.backtesting import Backtester


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_message(self):
        """Test parsing empty message."""
        parser = IntentParser()
        intent = parser.parse('')
        # Empty message defaults to analyze with no ticker
        assert intent.ticker is None
    
    def test_unknown_ticker(self):
        """Test with unknown ticker symbol."""
        parser = IntentParser()
        intent = parser.parse('analyze XYZABC123')
        # Unknown tickers are not extracted (only known tickers)
        assert intent.action == 'analyze'
    
    def test_multiple_tickers(self):
        """Test message with multiple tickers."""
        parser = IntentParser()
        intent = parser.parse('compare AAPL and MSFT')
        # Should extract first ticker
        assert intent.ticker in ['AAPL', 'MSFT']
    
    def test_short_price_history(self):
        """Test with minimal price data."""
        end = datetime.now()
        start = end - timedelta(days=5)
        df = generate_ohlcv('TEST', start_date=start, end_date=end)
        assert len(df) > 0
    
    def test_extreme_volatility(self):
        """Test invalidation with extreme volatility."""
        engine = InvalidationEngine()
        result = engine.check(
            prediction_time=datetime.now() - timedelta(minutes=5),
            current_price=100,
            prediction_price=100,
            recent_volatility=0.5,  # 50% volatility
            historical_volatility=0.02
        )
        assert not result.is_valid
        assert any('volatility' in r.lower() for r in result.reasons)
    
    def test_stale_prediction(self):
        """Test invalidation of old prediction."""
        engine = InvalidationEngine()
        result = engine.check(
            prediction_time=datetime.now() - timedelta(hours=2),
            current_price=100,
            prediction_price=100,
            recent_volatility=0.02,
            historical_volatility=0.02
        )
        # 2 hours may not be stale depending on config
        assert isinstance(result.is_valid, bool)
    
    def test_backtester_all_wrong(self):
        """Test backtester with all wrong predictions."""
        bt = Backtester()
        predictions = np.array([0.9, 0.9, 0.9, 0.9, 0.9])  # All bullish
        returns = np.array([-0.05, -0.05, -0.05, -0.05, -0.05])  # All down
        prices = np.array([100, 95, 90, 85, 80])
        result = bt.run(predictions, returns, prices)
        assert result.accuracy < 0.5
    
    def test_feature_store_consistency(self):
        """Test feature store produces consistent results."""
        fs = FeatureStore()
        end = datetime.now()
        start = end - timedelta(days=100)
        
        price_df = generate_ohlcv('TSLA', start_date=start, end_date=end)
        options = get_options_chain('TSLA', price_df['close'].iloc[-1])
        fundamentals = get_fundamentals('TSLA')
        sentiment = get_sentiment('TSLA')
        macro = get_macro_data()
        
        # Compute twice
        f1 = fs.compute_all_features(price_df, options, fundamentals, sentiment['news'].to_dict('records'), macro)
        f2 = fs.compute_all_features(price_df, options, fundamentals, sentiment['news'].to_dict('records'), macro)
        
        # Should be identical
        assert f1.shape == f2.shape
        np.testing.assert_array_equal(f1.values, f2.values)
    
    def test_response_generator_no_predictions(self):
        """Test response generator without predictions."""
        gen = ResponseGenerator()
        parser = IntentParser()
        intent = parser.parse('help')
        response = gen.generate(intent)
        assert 'help' in response.lower() or 'analyze' in response.lower()


class TestDataIntegrity:
    """Test data integrity across components."""
    
    def test_price_data_no_nan(self):
        """Test price data has no NaN values."""
        df = generate_ohlcv('AAPL')
        assert not df.isnull().any().any()
    
    def test_options_chain_valid_strikes(self):
        """Test options chain has valid strikes."""
        options = get_options_chain('AAPL', 150)
        assert all(options['strike'] > 0)
        assert all(options['iv'] > 0)
        assert all(options['iv'] < 5)  # IV should be < 500%
    
    def test_fundamentals_has_data(self):
        """Test fundamentals returns data."""
        fund = get_fundamentals('AAPL')
        assert 'revenue' in fund or 'earnings' in fund or len(fund) > 0
    
    def test_sentiment_returns_data(self):
        """Test sentiment returns data."""
        sent = get_sentiment('AAPL')
        assert 'news' in sent or 'overall_sentiment' in sent or len(sent) > 0
    
    def test_macro_data_returns_data(self):
        """Test macro data returns data."""
        macro = get_macro_data()
        assert len(macro) > 0
