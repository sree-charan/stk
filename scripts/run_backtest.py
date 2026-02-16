#!/usr/bin/env python3
"""Full 2-year backtest with detailed metrics."""
import sys
sys.path.insert(0, '.')

import numpy as np
from backend.data.real_providers.price_data import get_ohlcv
from backend.data.real_providers.options_data import get_options_chain
from backend.data.real_providers.fundamentals import get_fundamentals
from backend.data.real_providers.sentiment_data import get_sentiment
from backend.data.real_providers.macro_data import get_macro_data
from backend.features.feature_store import FeatureStore
from backend.models.ensemble import EnsembleModel

def run_backtest():
    print("=" * 70)
    print("STOCK PREDICTION BACKTEST - 2 YEAR ANALYSIS")
    print("=" * 70)
    
    # Generate 2 years of data
    
    print("\nFetching real data...")
    price_df = get_ohlcv("TSLA", "daily", 730)
    print(f"Fetched {len(price_df)} trading days")
    
    # Generate supporting data
    options = get_options_chain("TSLA", price_df['close'].iloc[-1])
    fundamentals = get_fundamentals("TSLA")
    sentiment = get_sentiment("TSLA")
    macro = get_macro_data()
    
    # Compute features
    print("\nComputing features...")
    store = FeatureStore()
    features_df = store.compute_all_features(price_df, options, fundamentals, sentiment, macro)
    print(f"Feature matrix: {features_df.shape}")
    
    # Create targets (future returns)
    prices = price_df['close'].values
    y_short = np.zeros(len(prices))  # 1-day return
    y_medium = np.zeros(len(prices))  # 5-day return
    y_long = np.zeros(len(prices))  # 20-day return
    
    for i in range(len(prices) - 20):
        y_short[i] = (prices[i+1] / prices[i]) - 1 if i+1 < len(prices) else 0
        y_medium[i] = (prices[min(i+5, len(prices)-1)] / prices[i]) - 1
        y_long[i] = (prices[min(i+20, len(prices)-1)] / prices[i]) - 1
    
    X = features_df.values
    
    # Train/test split (80/20 time-based)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_short_train, y_short_test = y_short[:split_idx], y_short[split_idx:]
    y_medium_train, y_medium_test = y_medium[:split_idx], y_medium[split_idx:]
    y_long_train, y_long_test = y_long[:split_idx], y_long[split_idx:]
    prices_test = prices[split_idx:]  # noqa: F841
    
    print(f"\nTrain: {len(X_train)} days | Test: {len(X_test)} days")
    
    # Train models
    print("\n" + "-" * 70)
    print("TRAINING MODELS")
    print("-" * 70)
    
    ensemble = EnsembleModel()
    scores = ensemble.train_all(X_train, y_short_train, y_medium_train, y_long_train)
    
    for name, score in scores.items():
        print(f"  {name}: R² = {score:.4f}")
    
    # Run predictions on test set
    print("\n" + "-" * 70)
    print("BACKTEST RESULTS")
    print("-" * 70)
    
    results = {'short': [], 'medium': [], 'long': []}
    actuals = {'short': y_short_test, 'medium': y_medium_test, 'long': y_long_test}
    
    for i in range(len(X_test)):
        preds = ensemble.predict_all_horizons(X_test[i])
        for h in ['short', 'medium', 'long']:
            results[h].append(preds[h]['prediction'])
    
    # Calculate metrics for each horizon
    for horizon, horizon_name in [('short', '1-day'), ('medium', '5-day'), ('long', '20-day')]:
        preds = np.array(results[horizon])
        actual = actuals[horizon]
        
        # Directional accuracy
        pred_dir = (preds > 0).astype(int)
        actual_dir = (actual > 0).astype(int)
        accuracy = (pred_dir == actual_dir).mean()
        
        # Precision/Recall for bullish
        tp = ((pred_dir == 1) & (actual_dir == 1)).sum()
        fp = ((pred_dir == 1) & (actual_dir == 0)).sum()
        fn = ((pred_dir == 0) & (actual_dir == 1)).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Trading metrics
        returns = pred_dir * actual
        sharpe = np.sqrt(252) * returns.mean() / (returns.std() + 1e-10)
        
        cumulative = (1 + returns).cumprod()
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak) / (peak + 1e-10)
        max_dd = abs(drawdown.min())
        
        # Win rate
        trades = pred_dir.sum()
        wins = ((pred_dir == 1) & (actual > 0)).sum()
        win_rate = wins / trades if trades > 0 else 0
        
        # Profit factor
        gains = actual[pred_dir == 1]
        profit_factor = gains[gains > 0].sum() / (abs(gains[gains < 0].sum()) + 1e-10)
        
        print(f"\n{horizon_name.upper()} HORIZON:")
        print(f"  Directional Accuracy: {accuracy*100:.1f}% ({int(accuracy*len(actual))}/{len(actual)})")
        print(f"  Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}")
        print(f"  Sharpe Ratio: {sharpe:.2f}")
        print(f"  Max Drawdown: {max_dd*100:.1f}%")
        print(f"  Win Rate: {win_rate*100:.1f}%")
        print(f"  Profit Factor: {profit_factor:.2f}")
    
    # Overall summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    short_acc = ((np.array(results['short']) > 0) == (actuals['short'] > 0)).mean()
    medium_acc = ((np.array(results['medium']) > 0) == (actuals['medium'] > 0)).mean()
    long_acc = ((np.array(results['long']) > 0) == (actuals['long'] > 0)).mean()
    
    print(f"\nTest Period: {len(X_test)} trading days")
    print(f"Short-term accuracy:  {short_acc*100:.1f}%")
    print(f"Medium-term accuracy: {medium_acc*100:.1f}%")
    print(f"Long-term accuracy:   {long_acc*100:.1f}%")
    
    all_pass = short_acc >= 0.55 and medium_acc >= 0.55 and long_acc >= 0.55
    print(f"\n{'✓ ALL HORIZONS >= 55%' if all_pass else '✗ SOME HORIZONS < 55% - NEEDS IMPROVEMENT'}")
    
    return all_pass

if __name__ == "__main__":
    success = run_backtest()
    sys.exit(0 if success else 1)
