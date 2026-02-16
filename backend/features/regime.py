"""Market regime detection features."""
import numpy as np
import pandas as pd


def compute_regime_features(price_df: pd.DataFrame, macro_data: dict | None = None) -> dict:
    """Compute market regime features from price data and macro data.

    Returns dict of feature_name -> scalar value (for the latest bar).
    """
    close = price_df['close'].values.astype(float)
    high = price_df['high'].values.astype(float)
    low = price_df['low'].values.astype(float)
    n = len(close)
    features = {}

    # 1. Regime label: bull/bear/sideways based on SMA50 vs SMA200
    sma50 = np.mean(close[-50:]) if n >= 50 else np.mean(close)
    sma200 = np.mean(close[-200:]) if n >= 200 else np.mean(close)
    if sma50 > sma200 * 1.01:
        regime = 'bull'
    elif sma50 < sma200 * 0.99:
        regime = 'bear'
    else:
        regime = 'sideways'
    features['regime_bull'] = 1.0 if regime == 'bull' else 0.0
    features['regime_bear'] = 1.0 if regime == 'bear' else 0.0
    features['regime_sideways'] = 1.0 if regime == 'sideways' else 0.0
    features['regime_label'] = regime  # for display, not used as numeric feature

    # 2. Volatility regime based on VIX
    vix = None
    if macro_data and 'vix' in macro_data:
        vix_df = macro_data['vix']
        if hasattr(vix_df, 'iloc') and len(vix_df) > 0:
            vix = float(vix_df['vix'].iloc[-1]) if 'vix' in vix_df.columns else None
    if vix is not None:
        if vix < 15:
            vol_regime = 'low'
        elif vix <= 25:
            vol_regime = 'normal'
        else:
            vol_regime = 'high'
    else:
        # Estimate from price data using realized volatility
        if n >= 20:
            returns = np.diff(close[-21:]) / close[-21:-1]
            rv = np.std(returns) * np.sqrt(252) * 100
            vol_regime = 'low' if rv < 15 else ('high' if rv > 30 else 'normal')
        else:
            vol_regime = 'normal'
    features['vol_regime_low'] = 1.0 if vol_regime == 'low' else 0.0
    features['vol_regime_normal'] = 1.0 if vol_regime == 'normal' else 0.0
    features['vol_regime_high'] = 1.0 if vol_regime == 'high' else 0.0
    features['vol_regime_label'] = vol_regime

    # 3. Trend strength: ADX-based
    adx = _compute_adx(high, low, close, period=14)
    if adx < 20:
        trend = 'no_trend'
    elif adx <= 40:
        trend = 'trending'
    else:
        trend = 'strong_trend'
    features['trend_no_trend'] = 1.0 if trend == 'no_trend' else 0.0
    features['trend_trending'] = 1.0 if trend == 'trending' else 0.0
    features['trend_strong_trend'] = 1.0 if trend == 'strong_trend' else 0.0
    features['adx_value'] = adx

    # 4. Momentum regime: 20-day rate of change
    if n >= 21:
        roc20 = (close[-1] - close[-21]) / close[-21] * 100
    else:
        roc20 = 0.0
    if roc20 > 5:
        mom = 'strong_up'
    elif roc20 > 0:
        mom = 'up'
    elif roc20 > -5:
        mom = 'down'
    else:
        mom = 'strong_down'
    features['mom_strong_up'] = 1.0 if mom == 'strong_up' else 0.0
    features['mom_up'] = 1.0 if mom == 'up' else 0.0
    features['mom_down'] = 1.0 if mom == 'down' else 0.0
    features['mom_strong_down'] = 1.0 if mom == 'strong_down' else 0.0
    features['momentum_roc20'] = roc20

    return features


def format_regime_display(features: dict) -> str:
    """Format regime for CLI display."""
    regime = features.get('regime_label', 'sideways')
    vol = features.get('vol_regime_label', 'normal')
    icons = {'bull': '🐂 Bull', 'bear': '🐻 Bear', 'sideways': '↔ Sideways'}
    return f"Market Regime: {icons.get(regime, regime)} ({vol} volatility)"


def _compute_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
    """Compute ADX value for the latest bar (vectorized)."""
    n = len(close)
    if n < period + 1:
        return 20.0  # default neutral

    # True Range (vectorized)
    tr = np.zeros(n)
    hl = high[1:] - low[1:]
    hc = np.abs(high[1:] - close[:-1])
    lc = np.abs(low[1:] - close[:-1])
    tr[1:] = np.maximum(hl, np.maximum(hc, lc))

    # +DM / -DM (vectorized)
    up = np.diff(high)
    down = -np.diff(low)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    plus_mask = (up > down) & (up > 0)
    minus_mask = (down > up) & (down > 0)
    plus_dm[1:] = np.where(plus_mask, up, 0)
    minus_dm[1:] = np.where(minus_mask, down, 0)

    # Wilder's smoothing (sequential by nature)
    atr = np.mean(tr[1:period + 1])
    pdi_sum = np.mean(plus_dm[1:period + 1])
    mdi_sum = np.mean(minus_dm[1:period + 1])

    dx_values = np.empty(max(0, n - period - 1))
    for i in range(period + 1, n):
        atr = atr - atr / period + tr[i]
        pdi_sum = pdi_sum - pdi_sum / period + plus_dm[i]
        mdi_sum = mdi_sum - mdi_sum / period + minus_dm[i]
        pdi = 100 * pdi_sum / atr if atr > 0 else 0
        mdi = 100 * mdi_sum / atr if atr > 0 else 0
        di_sum = pdi + mdi
        dx_values[i - period - 1] = 100 * abs(pdi - mdi) / di_sum if di_sum > 0 else 0

    if len(dx_values) == 0:
        return 20.0
    adx = float(np.mean(dx_values[-period:])) if len(dx_values) >= period else float(np.mean(dx_values))
    return adx
