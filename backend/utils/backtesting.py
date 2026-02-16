"""Backtesting module for model validation."""
import numpy as np
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class BacktestResult:
    """Results from backtesting."""
    accuracy: float
    precision: float
    recall: float
    sharpe_ratio: float
    max_drawdown: float
    total_trades: int
    win_rate: float
    profit_factor: float
    
    def to_dict(self) -> Dict:
        return {
            'accuracy': round(self.accuracy, 4),
            'precision': round(self.precision, 4),
            'recall': round(self.recall, 4),
            'sharpe_ratio': round(self.sharpe_ratio, 4),
            'max_drawdown': round(self.max_drawdown, 4),
            'total_trades': self.total_trades,
            'win_rate': round(self.win_rate, 4),
            'profit_factor': round(self.profit_factor, 4)
        }

class Backtester:
    """Backtest predictions against historical data."""
    
    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold
    
    def run(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        prices: np.ndarray
    ) -> BacktestResult:
        """Run backtest on predictions vs actuals."""
        n = len(predictions)
        if n == 0:
            return self._empty_result()
        
        # Convert probabilities to signals
        signals = (predictions > self.threshold).astype(int)
        actual_direction = (actuals > 0).astype(int)
        
        # Accuracy metrics
        correct = (signals == actual_direction).sum()
        accuracy = correct / n
        
        # Precision/Recall for bullish predictions
        true_pos = ((signals == 1) & (actual_direction == 1)).sum()
        false_pos = ((signals == 1) & (actual_direction == 0)).sum()
        false_neg = ((signals == 0) & (actual_direction == 1)).sum()
        
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        
        # Trading metrics
        returns = self._calculate_returns(signals, actuals, prices)
        sharpe = self._sharpe_ratio(returns)
        max_dd = self._max_drawdown(returns)
        
        # Win rate and profit factor
        trades = signals.sum()
        wins = ((signals == 1) & (actuals > 0)).sum()
        win_rate = wins / trades if trades > 0 else 0
        
        gains = actuals[signals == 1]
        profit_factor = gains[gains > 0].sum() / abs(gains[gains < 0].sum()) if gains[gains < 0].sum() != 0 else 0
        
        return BacktestResult(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            total_trades=int(trades),
            win_rate=win_rate,
            profit_factor=profit_factor
        )
    
    def _calculate_returns(self, signals: np.ndarray, actuals: np.ndarray, prices: np.ndarray) -> np.ndarray:
        """Calculate strategy returns."""
        # Simple: return actual return when signal is 1, 0 otherwise
        return signals * actuals
    
    def _sharpe_ratio(self, returns: np.ndarray, risk_free: float = 0.0) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        excess = returns - risk_free / 252
        return np.sqrt(252) * excess.mean() / returns.std()
    
    def _max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        if len(returns) == 0:
            return 0.0
        cumulative = (1 + returns).cumprod()
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak) / peak
        return abs(drawdown.min())
    
    def _empty_result(self) -> BacktestResult:
        return BacktestResult(0, 0, 0, 0, 0, 0, 0, 0)
    
    def walk_forward(
        self,
        model,
        features: np.ndarray,
        targets: np.ndarray,
        prices: np.ndarray,
        train_size: int = 200,
        test_size: int = 20
    ) -> List[BacktestResult]:
        """Walk-forward validation."""
        results = []
        n = len(features)
        
        for start in range(0, n - train_size - test_size, test_size):
            train_end = start + train_size
            test_end = train_end + test_size
            
            # Train on window
            X_train = features[start:train_end]
            y_train = targets[start:train_end]
            model.fit(X_train, y_train)
            
            # Test on next window
            X_test = features[train_end:test_end]
            y_test = targets[train_end:test_end]
            p_test = prices[train_end:test_end]
            
            preds = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_test)
            results.append(self.run(preds, y_test, p_test))
        
        return results
