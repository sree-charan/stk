"""Test ML models."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from backend.models.xgboost_models import XGBoostShort, XGBoostMedium, XGBoostLong
from backend.models.lstm_model import LSTMModel
from backend.models.ensemble import EnsembleModel


class TestXGBoostModels:
    def test_short_train_predict(self):
        model = XGBoostShort()
        X = np.random.randn(100, 50)
        y = np.random.randn(100)
        score = model.train(X, y)
        assert score > -1  # strong regularization may give negative R² on random data
        pred = model.predict(X[0])
        assert len(pred) == 1
    
    def test_medium_train_predict(self):
        model = XGBoostMedium()
        X = np.random.randn(100, 50)
        y = np.random.randn(100)
        score = model.train(X, y)
        assert score > -1
        pred, conf = model.predict_proba(X[0])
        assert 0 <= conf <= 1
    
    def test_long_train_predict(self):
        model = XGBoostLong()
        X = np.random.randn(100, 50)
        y = np.random.randn(100)
        score = model.train(X, y)
        assert score > -1  # strong regularization may give negative R² on random data


class TestLSTMModel:
    def test_train_predict(self):
        model = LSTMModel(seq_length=10)
        X = np.random.randn(50, 20)
        y = np.random.randn(50)
        score = model.train(X, y, epochs=5)
        assert isinstance(score, float)
        pred = model.predict(X)
        assert len(pred) == 1
    
    def test_predict_proba(self):
        model = LSTMModel(seq_length=10)
        X = np.random.randn(50, 20)
        y = np.random.randn(50)
        model.train(X, y, epochs=5)
        pred, conf = model.predict_proba(X)
        assert 0 <= conf <= 1

    def test_predict_untrained(self):
        model = LSTMModel(seq_length=10)
        pred = model.predict(np.random.randn(30, 20))
        assert pred[0] == 0.0

    def test_predict_short_input(self):
        model = LSTMModel(seq_length=10)
        X = np.random.randn(50, 20)
        y = np.random.randn(50)
        model.train(X, y, epochs=5)
        # Input shorter than seq_length — should pad
        pred = model.predict(np.random.randn(5, 20))
        assert len(pred) == 1

    def test_predict_1d_input(self):
        model = LSTMModel(seq_length=5)
        X = np.random.randn(30, 10)
        y = np.random.randn(30)
        model.train(X, y, epochs=5)
        pred = model.predict(np.random.randn(10))
        assert len(pred) == 1

    def test_save_load(self, tmp_path):
        model = LSTMModel(seq_length=10)
        X = np.random.randn(50, 20)
        y = np.random.randn(50)
        model.train(X, y, epochs=5)
        path = tmp_path / "lstm.pkl"
        model.save(path)
        assert path.exists()

        model2 = LSTMModel()
        assert model2.load(path)
        assert model2._trained
        pred = model2.predict(X)
        assert len(pred) == 1

    def test_load_nonexistent(self, tmp_path):
        model = LSTMModel()
        assert not model.load(tmp_path / "nope.pkl")

    def test_train_empty_sequences(self):
        model = LSTMModel(seq_length=100)
        X = np.random.randn(10, 5)  # too short for seq_length=100
        y = np.random.randn(10)
        score = model.train(X, y, epochs=5)
        assert score == 0.0


class TestEnsemble:
    def test_train_all(self):
        ensemble = EnsembleModel()
        X = np.random.randn(100, 50)
        y_short = np.random.randn(100)
        y_medium = np.random.randn(100)
        y_long = np.random.randn(100)
        scores = ensemble.train_all(X, y_short, y_medium, y_long)
        assert 'xgb_short' in scores
        assert 'lstm' in scores
    
    def test_predict_all_horizons(self):
        ensemble = EnsembleModel()
        X = np.random.randn(100, 50)
        y = np.random.randn(100)
        ensemble.train_all(X, y, y, y)
        results = ensemble.predict_all_horizons(X)
        assert 'short' in results
        assert 'medium' in results
        assert 'long' in results
        assert results['short']['direction'] in ['bullish', 'bearish', 'neutral']


class TestModelTrainer:
    def test_trainer_init(self):
        from backend.models.train import ModelTrainer
        trainer = ModelTrainer('AAPL')
        assert trainer.ticker == 'AAPL'
        assert trainer.feature_store is not None
        assert trainer.ensemble is not None

    def test_trainer_default_ticker(self):
        from backend.models.train import ModelTrainer
        trainer = ModelTrainer()
        assert trainer.ticker == 'TSLA'


class TestBacktesterWalkForward:
    def test_walk_forward_returns_list(self):
        from backend.utils.backtesting import Backtester
        bt = Backtester(threshold=0.5)

        class DummyModel:
            def fit(self, X, y): pass
            def predict(self, X): return np.random.rand(len(X))

        X = np.random.randn(250, 10)
        y = (np.random.randn(250) > 0).astype(float)
        prices = 100 + np.cumsum(np.random.randn(250) * 0.5)
        results = bt.walk_forward(DummyModel(), X, y, prices, train_size=200, test_size=20)
        assert isinstance(results, list)
        assert len(results) >= 1
        for r in results:
            assert hasattr(r, 'accuracy')
            assert hasattr(r, 'sharpe_ratio')

    def test_walk_forward_with_predict_proba(self):
        from backend.utils.backtesting import Backtester
        bt = Backtester(threshold=0.5)

        class ProbaModel:
            def fit(self, X, y): pass
            def predict_proba(self, X):
                return np.column_stack([np.random.rand(len(X)), np.random.rand(len(X))])

        X = np.random.randn(250, 10)
        y = (np.random.randn(250) > 0).astype(float)
        prices = 100 + np.cumsum(np.random.randn(250) * 0.5)
        results = bt.walk_forward(ProbaModel(), X, y, prices, train_size=200, test_size=20)
        assert len(results) >= 1

    def test_walk_forward_insufficient_data(self):
        from backend.utils.backtesting import Backtester
        bt = Backtester()
        class DummyModel:
            def fit(self, X, y): pass
            def predict(self, X): return np.random.rand(len(X))
        X = np.random.randn(10, 5)
        y = np.random.randn(10)
        prices = np.random.randn(10)
        results = bt.walk_forward(DummyModel(), X, y, prices, train_size=200, test_size=20)
        assert results == []

    def test_backtest_empty_predictions(self):
        from backend.utils.backtesting import Backtester
        bt = Backtester()
        result = bt.run(np.array([]), np.array([]), np.array([]))
        assert result.accuracy == 0
        assert result.total_trades == 0

    def test_backtest_result_to_dict(self):
        from backend.utils.backtesting import BacktestResult
        r = BacktestResult(0.75, 0.8, 0.7, 1.5, 0.1, 50, 0.6, 2.0)
        d = r.to_dict()
        assert d['accuracy'] == 0.75
        assert d['total_trades'] == 50
        assert isinstance(d, dict)

    def test_backtest_all_bullish(self):
        from backend.utils.backtesting import Backtester
        bt = Backtester(threshold=0.5)
        preds = np.ones(20) * 0.8
        actuals = np.random.randn(20)
        prices = 100 + np.cumsum(np.random.randn(20) * 0.5)
        result = bt.run(preds, actuals, prices)
        assert result.total_trades == 20

    def test_backtest_all_bearish(self):
        from backend.utils.backtesting import Backtester
        bt = Backtester(threshold=0.5)
        preds = np.ones(20) * 0.2
        actuals = np.random.randn(20)
        prices = 100 + np.cumsum(np.random.randn(20) * 0.5)
        result = bt.run(preds, actuals, prices)
        assert result.total_trades == 0
        assert result.win_rate == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
