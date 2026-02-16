"""Training pipeline for all models using real data."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from typing import Dict, Tuple
from backend.data.real_providers import generate_ohlcv, get_options_chain, get_fundamentals, get_sentiment, get_macro_data
from backend.features.feature_store import FeatureStore
from backend.models.ensemble import EnsembleModel


class ModelTrainer:
    """Training pipeline for stock prediction models."""

    def __init__(self, ticker: str = 'TSLA'):
        self.ticker = ticker
        self.feature_store = FeatureStore()
        self.ensemble = EnsembleModel()

    def generate_training_data(self, days: int = 500) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate features and targets from real data."""
        price_df = generate_ohlcv(self.ticker, "daily", days)
        spot = float(price_df['close'].iloc[-1])
        options = get_options_chain(self.ticker, spot)
        fundamentals = get_fundamentals(self.ticker)
        sentiment = get_sentiment(self.ticker)
        macro = get_macro_data()

        features = self.feature_store.compute_all_features(
            price_df, options, fundamentals, sentiment, macro
        )

        close = price_df['close'].values
        n = len(close)

        # Short: next-day return
        y_short = np.zeros(n)
        y_short[:-1] = (close[1:] - close[:-1]) / close[:-1]

        # Medium: 5-day return
        y_medium = np.zeros(n)
        for i in range(n - 5):
            y_medium[i] = (close[i + 5] - close[i]) / close[i]

        # Long: 20-day return
        y_long = np.zeros(n)
        for i in range(n - 20):
            y_long[i] = (close[i + 20] - close[i]) / close[i]

        return features.values, y_short, y_medium, y_long

    def train(self, days: int = 500) -> Dict[str, float]:
        """Train all models and return scores."""
        X, y_short, y_medium, y_long = self.generate_training_data(days)

        split = int(len(X) * 0.8)
        X_train = X[:split]

        scores = self.ensemble.train_all(
            X_train, y_short[:split], y_medium[:split], y_long[:split]
        )

        # Validate on remaining 20%
        X_val = X[split:]
        for horizon, y in [('short', y_short), ('medium', y_medium), ('long', y_long)]:
            pred, _, _ = self.ensemble.predict(X_val, horizon)
            actual = y[split:].mean()
            scores[f'{horizon}_val'] = 1 - abs(pred - actual) / (abs(actual) + 1e-6)

        return scores

    def save_models(self):
        self.ensemble.save()

    def load_models(self) -> bool:
        return self.ensemble.load()


if __name__ == '__main__':
    trainer = ModelTrainer('TSLA')
    print("Training models on real data...")
    scores = trainer.train(days=500)
    print("Training scores:", scores)
    trainer.save_models()
    print("Models saved.")
