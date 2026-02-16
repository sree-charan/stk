#!/usr/bin/env python3
"""Demo prediction with full breakdown."""
import sys
sys.path.insert(0, '.')

import numpy as np
from backend.data.real_providers.price_data import get_ohlcv as generate_ohlcv
from backend.data.real_providers.options_data import get_options_chain
from backend.data.real_providers.fundamentals import get_fundamentals
from backend.data.real_providers.sentiment_data import get_sentiment
from backend.data.real_providers.macro_data import get_macro_data
from backend.features.feature_store import FeatureStore
from backend.models.ensemble import EnsembleModel
from backend.utils.invalidation import calculate_invalidation

def demo_prediction():
    print("=" * 70)
    print("TSLA PREDICTION - FULL BREAKDOWN")
    print("=" * 70)
    
    # Generate data
    price_df = generate_ohlcv("TSLA", "daily", 730)
    spot = price_df['close'].iloc[-1]
    
    options = get_options_chain("TSLA", spot)
    fundamentals = get_fundamentals("TSLA")
    sentiment = get_sentiment("TSLA")
    macro = get_macro_data()
    
    print(f"\nCurrent TSLA Price: ${spot:.2f}")
    print(f"Today's Range: ${price_df['low'].iloc[-1]:.2f} - ${price_df['high'].iloc[-1]:.2f}")
    print(f"Volume: {price_df['volume'].iloc[-1]:,}")
    
    # Compute features
    store = FeatureStore()
    features_df = store.compute_all_features(price_df, options, fundamentals, sentiment, macro)
    feature_names = features_df.columns.tolist()
    X = features_df.iloc[-1].values
    
    print("\n" + "-" * 70)
    print("FEATURE VALUES (Latest)")
    print("-" * 70)
    
    # Group features by tier
    tiers = {
        'Price/Volume': [n for n in feature_names if any(x in n.lower() for x in ['price', 'volume', 'vwap', 'gap', 'range'])],
        'Technical': [n for n in feature_names if any(x in n.lower() for x in ['rsi', 'macd', 'bb_', 'sma', 'ema', 'atr', 'adx', 'stoch', 'obv', 'mfi'])],
        'Options': [n for n in feature_names if any(x in n.lower() for x in ['iv', 'put', 'call', 'pcr', 'oi', 'gamma', 'delta', 'skew'])],
        'Fundamentals': [n for n in feature_names if any(x in n.lower() for x in ['pe', 'pb', 'ps', 'eps', 'revenue', 'margin', 'debt', 'roe', 'fcf'])],
        'Sentiment': [n for n in feature_names if any(x in n.lower() for x in ['sentiment', 'news', 'social', 'analyst', 'insider'])],
        'Macro': [n for n in feature_names if any(x in n.lower() for x in ['vix', 'yield', 'dxy', 'sector', 'market', 'fed', 'gdp', 'cpi'])]
    }
    
    for tier_name, tier_features in tiers.items():
        if tier_features:
            print(f"\n{tier_name}:")
            for feat in tier_features[:8]:
                idx = feature_names.index(feat)
                print(f"  {feat}: {X[idx]:.4f}")
            if len(tier_features) > 8:
                print(f"  ... and {len(tier_features) - 8} more")
    
    # Train models on available data
    print("\n" + "-" * 70)
    print("MODEL OUTPUTS")
    print("-" * 70)
    
    ensemble = EnsembleModel()
    
    # Train on available data
    prices = price_df['close'].values
    y = np.diff(prices) / prices[:-1]
    y = np.append(y, 0)
    y_medium = np.zeros(len(prices))
    y_long = np.zeros(len(prices))
    for i in range(len(prices) - 20):
        y_medium[i] = (prices[min(i+5, len(prices)-1)] / prices[i]) - 1
        y_long[i] = (prices[min(i+20, len(prices)-1)] / prices[i]) - 1
    
    ensemble.train_all(features_df.values, y, y_medium, y_long)
    
    # Get predictions
    results = ensemble.predict_all_horizons(X)
    
    for horizon, data in results.items():
        horizon_name = {'short': '1-HOUR', 'medium': '5-DAY', 'long': '60-DAY'}[horizon]
        pred = data['prediction']
        conf = data['confidence']
        direction = data['direction'].upper()
        breakdown = data['breakdown']
        
        print(f"\n{horizon_name} PREDICTION:")
        print(f"  Direction: {direction}")
        print(f"  Expected Return: {pred*100:+.2f}%")
        print(f"  Confidence: {conf*100:.0f}%")
        print("  Model Breakdown:")
        print(f"    XGBoost: {breakdown['xgb']['prediction']*100:+.3f}% (weight: {breakdown['xgb']['weight']:.1f})")
        print(f"    LSTM:    {breakdown['lstm']['prediction']*100:+.3f}% (weight: {breakdown['lstm']['weight']:.1f})")
    
    # Invalidation levels
    print("\n" + "-" * 70)
    print("INVALIDATION LEVELS")
    print("-" * 70)
    
    atr = features_df['atr_14'].iloc[-1] if 'atr_14' in features_df.columns else spot * 0.02
    
    for horizon in ['short', 'medium', 'long']:
        direction = results[horizon]['direction']
        inv = calculate_invalidation(spot, direction, horizon, atr)
        horizon_name = {'short': '1-Hour', 'medium': '5-Day', 'long': '60-Day'}[horizon]
        
        print(f"\n{horizon_name} ({direction.upper()}):")
        print(f"  Stop Loss: ${inv['stop_loss']:.2f}")
        print(f"  Take Profit: ${inv['take_profit']:.2f}")
        print(f"  Invalidation: {inv['condition']}")
    
    # Key signals
    print("\n" + "-" * 70)
    print("KEY SIGNALS")
    print("-" * 70)
    
    def get_feat(name):
        return X[feature_names.index(name)] if name in feature_names else None
    
    bullish = []
    bearish = []
    
    # RSI signals
    rsi = get_feat('rsi_14')
    if rsi is not None:
        if rsi < 30:
            bullish.append(f"RSI oversold ({rsi:.1f})")
        elif rsi < 40:
            bullish.append(f"RSI approaching oversold ({rsi:.1f})")
        elif rsi > 70:
            bearish.append(f"RSI overbought ({rsi:.1f})")
        elif rsi > 60:
            bearish.append(f"RSI approaching overbought ({rsi:.1f})")
    
    # MACD signals (correct feature name: macd_hist)
    macd_hist = get_feat('macd_hist')
    if macd_hist is not None:
        if macd_hist > 0.5:
            bullish.append(f"MACD histogram positive ({macd_hist:.2f})")
        elif macd_hist > 0:
            bullish.append(f"MACD histogram slightly positive ({macd_hist:.2f})")
        elif macd_hist < -0.5:
            bearish.append(f"MACD histogram negative ({macd_hist:.2f})")
        else:
            bearish.append(f"MACD histogram slightly negative ({macd_hist:.2f})")
    
    # Bollinger Band position
    bb = get_feat('bb_position')
    if bb is not None:
        if bb < 0.2:
            bullish.append(f"Near lower Bollinger Band ({bb:.2f})")
        elif bb > 0.8:
            bearish.append(f"Near upper Bollinger Band ({bb:.2f})")
        elif bb < 0.35:
            bullish.append(f"Below BB midline ({bb:.2f})")
        elif bb > 0.55:
            bearish.append(f"Above BB midline ({bb:.2f})")
    
    # Volume signals
    vol_ratio = get_feat('volume_ratio')
    if vol_ratio is not None:
        if vol_ratio > 1.5:
            bullish.append(f"Volume surge ({vol_ratio:.1f}x average)")
        elif vol_ratio > 1.1:
            bullish.append(f"Above average volume ({vol_ratio:.1f}x)")
        elif vol_ratio < 0.7:
            bearish.append(f"Low volume ({vol_ratio:.1f}x average)")
    
    # VWAP signals
    vwap_dist = get_feat('vwap_distance')
    vwap = get_feat('vwap')
    if vwap_dist is not None and vwap is not None:
        if vwap_dist > 0.005:
            bullish.append(f"Trading above VWAP (${vwap:.2f})")
        elif vwap_dist < -0.005:
            bearish.append(f"Trading below VWAP (${vwap:.2f})")
    
    # Sentiment signals (correct feature name: news_sentiment_avg)
    sent = get_feat('news_sentiment_avg')
    if sent is not None:
        if sent > 0.2:
            bullish.append(f"Positive news sentiment ({sent:.2f})")
        elif sent > 0.1:
            bullish.append(f"Slightly positive sentiment ({sent:.2f})")
        elif sent < -0.2:
            bearish.append(f"Negative news sentiment ({sent:.2f})")
        elif sent < -0.1:
            bearish.append(f"Slightly negative sentiment ({sent:.2f})")
    
    # ADX trend strength
    adx = get_feat('adx_14')
    ret_5d = get_feat('return_5d')
    if adx is not None and adx > 20:
        if ret_5d is not None and ret_5d > 0:
            bullish.append(f"Strong uptrend (ADX {adx:.1f})")
        elif ret_5d is not None and ret_5d < 0:
            bearish.append(f"Strong downtrend (ADX {adx:.1f})")
    
    # Put/Call ratio
    pcr = get_feat('put_call_ratio')
    if pcr is not None:
        if pcr > 1.1:
            bullish.append(f"High put/call ratio ({pcr:.2f}) - contrarian bullish")
        elif pcr < 0.8:
            bearish.append(f"Low put/call ratio ({pcr:.2f}) - contrarian bearish")
    
    # IV percentile
    iv_pct = get_feat('iv_percentile')
    if iv_pct is not None:
        if iv_pct < 0.3:
            bullish.append(f"Low IV percentile ({iv_pct:.0%}) - cheap options")
        elif iv_pct > 0.7:
            bearish.append(f"High IV percentile ({iv_pct:.0%}) - expensive options")
    
    # Price momentum
    ret_1d = get_feat('return_1d')
    if ret_1d is not None:
        if ret_1d > 0.02:
            bullish.append(f"Strong daily gain ({ret_1d*100:.1f}%)")
        elif ret_1d > 0.005:
            bullish.append(f"Positive daily return ({ret_1d*100:.2f}%)")
        elif ret_1d < -0.02:
            bearish.append(f"Strong daily loss ({ret_1d*100:.1f}%)")
        elif ret_1d < -0.005:
            bearish.append(f"Negative daily return ({ret_1d*100:.2f}%)")
    
    # MFI (Money Flow Index)
    mfi = get_feat('mfi_14')
    if mfi is not None:
        if mfi < 30:
            bullish.append(f"MFI oversold ({mfi:.1f})")
        elif mfi > 70:
            bearish.append(f"MFI overbought ({mfi:.1f})")
        elif mfi < 40:
            bullish.append(f"MFI approaching oversold ({mfi:.1f})")
        elif mfi > 55:
            bearish.append(f"MFI elevated ({mfi:.1f})")
    
    # Williams %R
    williams = get_feat('williams_r')
    if williams is not None:
        if williams < -80:
            bullish.append(f"Williams %R oversold ({williams:.1f})")
        elif williams > -20:
            bearish.append(f"Williams %R overbought ({williams:.1f})")
        elif williams < -60:
            bullish.append(f"Williams %R approaching oversold ({williams:.1f})")
        elif williams > -40:
            bearish.append(f"Williams %R approaching overbought ({williams:.1f})")
    
    # Price vs moving averages
    sma_50 = get_feat('sma_50')
    if sma_50 is not None and spot > 0:
        pct_above = (spot - sma_50) / sma_50 * 100
        if pct_above > 5:
            bullish.append(f"Price {pct_above:.1f}% above 50-SMA (${sma_50:.2f})")
        elif pct_above < -5:
            bearish.append(f"Price {abs(pct_above):.1f}% below 50-SMA (${sma_50:.2f})")
    
    # VIX level
    vix = get_feat('vix')
    if vix is not None:
        if vix < 15:
            bullish.append(f"Low VIX ({vix:.1f}) - market calm")
        elif vix > 25:
            bearish.append(f"High VIX ({vix:.1f}) - market fear")
        elif vix > 18:
            bearish.append(f"Elevated VIX ({vix:.1f}) - caution")
    
    # Stochastic oscillator
    stoch_k = get_feat('stoch_k')
    stoch_d = get_feat('stoch_d')  # noqa: F841
    if stoch_k is not None:
        if stoch_k < 20:
            bullish.append(f"Stochastic oversold ({stoch_k:.1f})")
        elif stoch_k > 80:
            bearish.append(f"Stochastic overbought ({stoch_k:.1f})")
    
    # OBV trend
    obv_slope = get_feat('obv_slope')
    if obv_slope is not None:
        if obv_slope > 0.01:
            bullish.append("OBV rising (accumulation)")
        elif obv_slope < -0.01:
            bearish.append("OBV falling (distribution)")
    
    # ATR volatility
    atr_pct = get_feat('atr_percent')
    if atr_pct is not None:
        if atr_pct > 0.03:
            bearish.append(f"High volatility (ATR {atr_pct*100:.1f}%)")
        elif atr_pct < 0.015:
            bullish.append(f"Low volatility (ATR {atr_pct*100:.1f}%)")
    
    # Yield curve
    yield_curve = get_feat('yield_curve')
    if yield_curve is not None:
        if yield_curve < -0.2:
            bearish.append(f"Inverted yield curve ({yield_curve:.2f})")
        elif yield_curve > 0.5:
            bullish.append(f"Steep yield curve ({yield_curve:.2f})")
    
    # Sector rotation
    sector_rot = get_feat('sector_rotation')
    if sector_rot is not None:
        if sector_rot < -0.1:
            bearish.append(f"Negative sector rotation ({sector_rot:.2f})")
        elif sector_rot > 0.1:
            bullish.append(f"Positive sector rotation ({sector_rot:.2f})")
    
    print("\nBULLISH SIGNALS:")
    for s in bullish[:5] or ["None significant"]:
        print(f"  + {s}")
    
    print("\nBEARISH SIGNALS:")
    for s in bearish[:5] or ["None significant"]:
        print(f"  - {s}")
    
    print("\n" + "=" * 70)
    print("PREDICTION COMPLETE")
    print("=" * 70)
    
    return True

if __name__ == "__main__":
    demo_prediction()
