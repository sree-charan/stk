"""Test all feature engineering modules."""
import sys
sys.path.insert(0, '/home/charsree/.workspace/stock-chat-assistant')

from backend.data.mock_generators import get_ohlcv, get_options_chain, get_fundamentals, get_sentiment, get_macro_data
from backend.features import FeatureStore

def test_features():
    print("Generating mock data...")
    price_df = get_ohlcv('TSLA', days=100)
    options = get_options_chain('TSLA', price_df['close'].iloc[-1])
    fundamentals = get_fundamentals('TSLA', quarters=8)
    sentiment = get_sentiment('TSLA')
    news_df = sentiment.get('news')
    news = news_df.to_dict('records') if news_df is not None and not news_df.empty else []
    macro = get_macro_data()
    vix_data = macro['vix'].iloc[-1]
    rates = macro['interest_rates'].iloc[-1]
    macro_dict = {
        'vix': vix_data['vix'],
        'vix_3m': vix_data['vix_3m'],
        'fed_funds': rates['fed_funds_rate'],
        'treasury_10y': rates['treasury_10y'],
        'treasury_2y': rates['treasury_2y']
    }
    
    print("Computing features...")
    store = FeatureStore()
    features = store.compute_all_features(
        price_df=price_df,
        options_data=options,
        fundamentals=fundamentals,
        news=news,
        macro_data=macro_dict
    )
    
    print(f"\nFeature matrix shape: {features.shape}")
    print(f"Total features: {len(features.columns)}")
    print("\nFeature counts by tier:")
    for tier, count in FeatureStore.feature_count().items():
        print(f"  {tier}: {count}")
    
    print("\nSample features (last row):")
    sample = features.iloc[-1]
    for col in ['return_1d', 'rsi_14', 'put_call_ratio', 'eps', 'news_sentiment_avg', 'vix']:
        if col in sample:
            print(f"  {col}: {sample[col]:.4f}")
    
    # Verify no NaN or inf
    nan_count = features.isna().sum().sum()
    inf_count = ((features == float('inf')) | (features == float('-inf'))).sum().sum()
    print(f"\nData quality: NaN={nan_count}, Inf={inf_count}")
    
    print("\n✓ All features computed successfully!")
    return features

if __name__ == '__main__':
    test_features()
