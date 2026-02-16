#!/usr/bin/env python3
"""Validate all features with statistics."""
import sys
sys.path.insert(0, '.')

import numpy as np
from backend.data.mock_generators.price_data import generate_ohlcv
from backend.data.mock_generators.options_data import get_options_chain
from backend.data.mock_generators.fundamentals import get_fundamentals
from backend.data.mock_generators.sentiment_data import get_sentiment
from backend.data.mock_generators.macro_data import get_macro_data
from backend.features.feature_store import FeatureStore

def validate_features():
    print("=" * 70)
    print("FEATURE VALIDATION - ALL FEATURES FOR TSLA")
    print("=" * 70)
    
    # Generate data
    print("\nGenerating TSLA data...")
    price_df = generate_ohlcv("TSLA", base_price=250, seed=42)
    spot = price_df['close'].iloc[-1]
    
    options = get_options_chain("TSLA", spot)
    fundamentals = get_fundamentals("TSLA")
    sentiment = get_sentiment("TSLA")
    macro = get_macro_data()
    
    # Compute features
    store = FeatureStore()
    features_df = store.compute_all_features(price_df, options, fundamentals, sentiment, macro)
    
    print(f"\nTotal features: {len(features_df.columns)}")
    print(f"Data points: {len(features_df)}")
    
    # Statistics for each feature
    print("\n" + "-" * 70)
    print("FEATURE STATISTICS")
    print("-" * 70)
    print(f"{'Feature':<35} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10} {'Issues':<10}")
    print("-" * 70)
    
    issues = []
    for col in features_df.columns:
        vals = features_df[col]
        nan_count = vals.isna().sum()
        inf_count = np.isinf(vals).sum()
        
        clean = vals.replace([np.inf, -np.inf], np.nan).dropna()
        if len(clean) > 0:
            min_v, max_v = clean.min(), clean.max()
            mean_v, std_v = clean.mean(), clean.std()
        else:
            min_v = max_v = mean_v = std_v = 0
        
        issue_str = ""
        if nan_count > 0:
            issue_str += f"NaN:{nan_count} "
            issues.append((col, f"{nan_count} NaN values"))
        if inf_count > 0:
            issue_str += f"Inf:{inf_count}"
            issues.append((col, f"{inf_count} infinite values"))
        
        print(f"{col:<35} {min_v:>10.3f} {max_v:>10.3f} {mean_v:>10.3f} {std_v:>10.3f} {issue_str:<10}")
    
    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    counts = store.feature_count()
    print("\nFeature counts by tier:")
    for tier, count in counts.items():
        if tier != 'total':
            print(f"  {tier}: {count}")
    print(f"  TOTAL: {counts['total']}")
    
    print(f"\nActual features computed: {len(features_df.columns)}")
    
    if issues:
        print(f"\n⚠ ISSUES FOUND ({len(issues)}):")
        for feat, issue in issues[:10]:
            print(f"  - {feat}: {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more")
    else:
        print("\n✓ NO ISSUES - All features valid")
    
    # Specific validations
    print("\n" + "-" * 70)
    print("SPECIFIC VALIDATIONS")
    print("-" * 70)
    
    checks = []
    
    # RSI should be 0-100
    rsi_cols = [c for c in features_df.columns if 'rsi' in c.lower()]
    for col in rsi_cols:
        vals = features_df[col].dropna()
        if len(vals) > 0:
            in_range = (vals >= 0) & (vals <= 100)
            pct = in_range.mean() * 100
            checks.append((col, "0-100 range", pct >= 99, f"{pct:.1f}% in range"))
    
    # Volume should be positive (not volume_change which can be negative)
    vol_cols = [c for c in features_df.columns if c in ['volume_sma20', 'volume_ratio', 'dollar_volume']]
    for col in vol_cols[:3]:
        vals = features_df[col].dropna()
        if len(vals) > 0:
            positive = (vals >= 0).mean() * 100
            checks.append((col, "non-negative", positive >= 99, f"{positive:.1f}% positive"))
    
    for name, check, passed, detail in checks:
        status = "✓" if passed else "✗"
        print(f"  {status} {name}: {check} - {detail}")
    
    all_valid = len(issues) == 0 and all(c[2] for c in checks)
    print(f"\n{'✓ ALL FEATURES VALID' if all_valid else '✗ SOME FEATURES HAVE ISSUES'}")
    
    return all_valid

if __name__ == "__main__":
    success = validate_features()
    sys.exit(0 if success else 1)
