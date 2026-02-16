"""Tests for utility modules."""
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.utils.invalidation import InvalidationEngine
from backend.utils.backtesting import Backtester, BacktestResult
from backend.utils.config import Config, config

class TestInvalidationEngine:
    def test_valid_prediction(self):
        engine = InvalidationEngine()
        result = engine.check(
            prediction_time=datetime.now() - timedelta(hours=1),
            current_price=100,
            prediction_price=100,
            recent_volatility=0.02,
            historical_volatility=0.02
        )
        assert result.is_valid
        assert result.severity == 'none'
    
    def test_stale_prediction(self):
        engine = InvalidationEngine(max_age_hours=2)
        result = engine.check(
            prediction_time=datetime.now() - timedelta(hours=5),
            current_price=100,
            prediction_price=100,
            recent_volatility=0.02,
            historical_volatility=0.02
        )
        assert not result.is_valid
        assert 'old' in result.reasons[0]
    
    def test_volatility_spike(self):
        engine = InvalidationEngine(volatility_threshold=1.5)
        result = engine.check(
            prediction_time=datetime.now(),
            current_price=100,
            prediction_price=100,
            recent_volatility=0.06,
            historical_volatility=0.02
        )
        assert not result.is_valid
        assert result.severity == 'critical'
    
    def test_price_move(self):
        engine = InvalidationEngine()
        result = engine.check(
            prediction_time=datetime.now(),
            current_price=112,
            prediction_price=100,
            recent_volatility=0.02,
            historical_volatility=0.02
        )
        assert not result.is_valid
        assert 'moved' in result.reasons[0]
    
    def test_material_news(self):
        engine = InvalidationEngine()
        result = engine.check(
            prediction_time=datetime.now(),
            current_price=100,
            prediction_price=100,
            recent_volatility=0.02,
            historical_volatility=0.02,
            news=[{'headline': 'Company announces merger with rival'}]
        )
        assert not result.is_valid
        assert result.severity == 'critical'

class TestBacktester:
    def test_perfect_predictions(self):
        bt = Backtester(threshold=0.5)
        predictions = np.array([0.8, 0.8, 0.2, 0.2])
        actuals = np.array([0.05, 0.03, -0.02, -0.04])
        prices = np.array([100, 105, 103, 99])
        
        result = bt.run(predictions, actuals, prices)
        assert result.accuracy == 1.0
        assert result.win_rate == 1.0
    
    def test_random_predictions(self):
        bt = Backtester()
        np.random.seed(42)
        predictions = np.random.random(100)
        actuals = np.random.randn(100) * 0.02
        prices = 100 + np.cumsum(actuals)
        
        result = bt.run(predictions, actuals, prices)
        assert 0 <= result.accuracy <= 1
        assert result.total_trades > 0
    
    def test_result_to_dict(self):
        result = BacktestResult(0.65, 0.7, 0.6, 1.5, 0.1, 50, 0.6, 1.8)
        d = result.to_dict()
        assert d['accuracy'] == 0.65
        assert d['sharpe_ratio'] == 1.5

class TestConfig:
    def test_default_values(self):
        c = Config()
        assert c.SHORT_HORIZON_HOURS == 1
        assert c.MEDIUM_HORIZON_DAYS == 5
        assert len(c.TICKERS) > 0
    
    def test_global_config(self):
        assert config.BULLISH_THRESHOLD == 0.6
        assert 'TSLA' in config.TICKERS
