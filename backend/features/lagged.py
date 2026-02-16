"""Lagged features, sequence features, and price-derived memory features."""
import numpy as np
import pandas as pd
from typing import List

# Top 30 features to lag (most informative for stock prediction)
TOP_FEATURES = [
    'return_1d', 'return_5d', 'return_20d', 'volume_ratio', 'volume_change',
    'rsi_14', 'macd', 'macd_hist', 'bb_position', 'bb_width',
    'stoch_k', 'stoch_d', 'atr_14', 'adx_14', 'cci',
    'obv_slope', 'mfi_14', 'williams_r', 'roc_10', 'trix',
    'vwap_distance', 'high_low_range', 'close_position', 'gap',
    'up_volume_ratio', 'dollar_volume', 'volume_sma20', 'volume_std',
    'price_to_vwap', 'vwap_slope',
]

LAG_PERIODS = [1, 2, 5, 10, 20]


def compute_lagged_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """Add lagged versions of top features to give the model memory."""
    result = {}
    available = [f for f in TOP_FEATURES if f in features_df.columns]

    for feat in available:
        col = features_df[feat]
        for lag in LAG_PERIODS:
            result[f'{feat}_lag{lag}'] = col.shift(lag).fillna(0)
        # Rate of change: today - 5d ago
        result[f'{feat}_roc5'] = (col - col.shift(5)).fillna(0)
        # Rolling stats
        result[f'{feat}_mean5'] = col.rolling(5, min_periods=1).mean().fillna(0)
        result[f'{feat}_std5'] = col.rolling(5, min_periods=1).std().fillna(0)

    return pd.DataFrame(result, index=features_df.index)


def compute_price_derived_features(price_df: pd.DataFrame) -> pd.DataFrame:
    """Compute price-derived sequence features."""
    close = price_df['close'].values.astype(float)
    volume = price_df['volume'].values.astype(float)
    high = price_df['high'].values.astype(float)
    low = price_df['low'].values.astype(float)
    n = len(close)
    result = {}

    # Consecutive up/down days
    daily_ret = np.zeros(n)
    daily_ret[1:] = (close[1:] - close[:-1]) / np.maximum(close[:-1], 1e-8)
    up_days = np.zeros(n)
    down_days = np.zeros(n)
    for i in range(1, n):
        if daily_ret[i] > 0:
            up_days[i] = up_days[i - 1] + 1
        else:
            down_days[i] = down_days[i - 1] + 1 if daily_ret[i] < 0 else 0
    result['consecutive_up_days'] = up_days
    result['consecutive_down_days'] = down_days

    # Drawdown from 20d high / rally from 20d low
    high_20d = pd.Series(high).rolling(20, min_periods=1).max().values
    low_20d = pd.Series(low).rolling(20, min_periods=1).min().values
    result['drawdown_from_20d_high'] = (close - high_20d) / np.maximum(high_20d, 1e-8)
    result['rally_from_20d_low'] = (close - low_20d) / np.maximum(low_20d, 1e-8)

    # Days since last 5% drop
    days_since_drop = np.full(n, 100.0)
    for i in range(1, n):
        if daily_ret[i] < -0.05:
            days_since_drop[i] = 0
        else:
            days_since_drop[i] = days_since_drop[i - 1] + 1
    result['days_since_last_5pct_drop'] = days_since_drop

    # Volatility expansion: current ATR / 20d avg ATR
    tr = np.maximum(high - low, np.maximum(
        np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr_14 = pd.Series(tr).rolling(14, min_periods=1).mean().values
    atr_20d_avg = pd.Series(atr_14).rolling(20, min_periods=1).mean().values
    result['volatility_expansion'] = atr_14 / np.maximum(atr_20d_avg, 1e-8)

    # Price acceleration: 5d return - previous 5d return
    ret_5d = np.zeros(n)
    ret_5d[5:] = (close[5:] - close[:-5]) / np.maximum(close[:-5], 1e-8)
    prev_ret_5d = np.zeros(n)
    prev_ret_5d[10:] = ret_5d[5:-5]
    result['price_acceleration'] = ret_5d - prev_ret_5d

    # Volume trend 5d (slope of volume over last 5 days)
    vol_series = pd.Series(volume)
    vol_mean5 = vol_series.rolling(5, min_periods=1).mean()
    vol_mean5_prev = vol_mean5.shift(5).fillna(vol_mean5)
    result['volume_trend_5d'] = ((vol_mean5 - vol_mean5_prev) / np.maximum(vol_mean5_prev, 1e-8)).values

    # Additional price-derived features
    # Intraday range relative to price
    result['intraday_range_pct'] = (high - low) / np.maximum(close, 1e-8)
    # Close relative to day's range
    day_range = high - low
    result['close_in_range'] = np.where(day_range > 0, (close - low) / np.maximum(day_range, 1e-8), 0.5)
    # 10d return
    ret_10d = np.zeros(n)
    ret_10d[10:] = (close[10:] - close[:-10]) / np.maximum(close[:-10], 1e-8)
    result['return_10d'] = ret_10d
    # Drawdown from 50d high
    high_50d = pd.Series(high).rolling(50, min_periods=1).max().values
    result['drawdown_from_50d_high'] = (close - high_50d) / np.maximum(high_50d, 1e-8)
    # Rally from 50d low
    low_50d = pd.Series(low).rolling(50, min_periods=1).min().values
    result['rally_from_50d_low'] = (close - low_50d) / np.maximum(low_50d, 1e-8)
    # Volume spike (today vs 20d avg)
    vol_sma20 = vol_series.rolling(20, min_periods=1).mean()
    result['volume_spike'] = (volume / np.maximum(vol_sma20.values, 1e-8))
    # Price range expansion (today's range vs 10d avg range)
    range_10d = pd.Series(day_range).rolling(10, min_periods=1).mean().values
    result['range_expansion'] = day_range / np.maximum(range_10d, 1e-8)
    # Gap from previous close
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    open_prices = price_df['open'].values.astype(float) if 'open' in price_df.columns else close
    result['gap_pct'] = (open_prices - prev_close) / np.maximum(prev_close, 1e-8)
    # Consecutive higher highs / lower lows
    higher_highs = np.zeros(n)
    lower_lows = np.zeros(n)
    for i in range(1, n):
        if high[i] > high[i-1]:
            higher_highs[i] = higher_highs[i-1] + 1
        if low[i] < low[i-1]:
            lower_lows[i] = lower_lows[i-1] + 1
    result['consecutive_higher_highs'] = higher_highs
    result['consecutive_lower_lows'] = lower_lows
    # 3d return momentum
    ret_3d = np.zeros(n)
    ret_3d[3:] = (close[3:] - close[:-3]) / np.maximum(close[:-3], 1e-8)
    result['return_3d'] = ret_3d
    # Volatility ratio (5d vs 20d)
    std_5d = pd.Series(daily_ret).rolling(5, min_periods=1).std().fillna(0).values
    std_20d = pd.Series(daily_ret).rolling(20, min_periods=1).std().fillna(1e-8).values
    result['volatility_ratio_5_20'] = std_5d / np.maximum(std_20d, 1e-8)
    # Mean reversion signal (distance from 10d SMA)
    sma_10 = pd.Series(close).rolling(10, min_periods=1).mean().values
    result['distance_from_sma10'] = (close - sma_10) / np.maximum(sma_10, 1e-8)
    # Overnight return proxy (open vs prev close)
    result['overnight_return'] = (open_prices - prev_close) / np.maximum(prev_close, 1e-8)
    # Intraday return (close vs open)
    result['intraday_return'] = (close - open_prices) / np.maximum(open_prices, 1e-8)
    # Upper/lower shadow ratios
    body = np.abs(close - open_prices)
    upper_shadow = high - np.maximum(close, open_prices)
    lower_shadow = np.minimum(close, open_prices) - low
    candle_range = np.maximum(high - low, 1e-8)
    result['upper_shadow_ratio'] = upper_shadow / candle_range
    result['lower_shadow_ratio'] = lower_shadow / candle_range
    result['body_ratio'] = body / candle_range
    # 20d return
    ret_20d = np.zeros(n)
    ret_20d[20:] = (close[20:] - close[:-20]) / np.maximum(close[:-20], 1e-8)
    result['return_20d_derived'] = ret_20d
    # Distance from 20d SMA
    sma_20 = pd.Series(close).rolling(20, min_periods=1).mean().values
    result['distance_from_sma20'] = (close - sma_20) / np.maximum(sma_20, 1e-8)
    # Volume acceleration
    vol_5d = vol_series.rolling(5, min_periods=1).mean()
    vol_10d = vol_series.rolling(10, min_periods=1).mean()
    result['volume_acceleration'] = ((vol_5d - vol_10d) / np.maximum(vol_10d, 1e-8)).values
    # High-low range 5d avg
    result['avg_range_5d'] = pd.Series(day_range / np.maximum(close, 1e-8)).rolling(5, min_periods=1).mean().values
    # Price momentum score (weighted returns)
    result['momentum_score'] = (0.5 * ret_3d + 0.3 * ret_5d + 0.2 * ret_10d)

    # --- Feature interaction terms ---
    # RSI × volume ratio: oversold + high volume = stronger reversal signal
    # (computed from price-derived proxies since base features aren't available here)
    # Momentum divergence: price going up but volume going down
    result['price_vol_divergence'] = ret_5d * np.where(
        result['volume_trend_5d'] != 0, -np.sign(result['volume_trend_5d']), 0)
    # Volatility-adjusted momentum: momentum relative to recent volatility
    result['vol_adj_momentum'] = result['momentum_score'] / np.maximum(std_5d, 1e-8)
    # Mean reversion strength: distance from SMA × volatility expansion
    result['mean_reversion_strength'] = result['distance_from_sma10'] * result['volatility_expansion']
    # Trend consistency: up days ratio over last 10 days
    up_mask = (daily_ret > 0).astype(float)
    result['up_ratio_10d'] = pd.Series(up_mask).rolling(10, min_periods=1).mean().values
    # Breakout signal: close near 20d high + expanding volume
    result['breakout_signal'] = (1 - np.abs(result['drawdown_from_20d_high'])) * result['volume_spike']
    # Exhaustion signal: many consecutive up days + declining volume
    result['exhaustion_signal'] = up_days * np.maximum(-result['volume_trend_5d'], 0)

    # --- Additional medium/long-term predictive features ---
    # 50d return (long-term momentum)
    ret_50d = np.zeros(n)
    ret_50d[50:] = (close[50:] - close[:-50]) / np.maximum(close[:-50], 1e-8)
    result['return_50d'] = ret_50d
    # Distance from 50d SMA (trend following)
    sma_50 = pd.Series(close).rolling(50, min_periods=1).mean().values
    result['distance_from_sma50'] = (close - sma_50) / np.maximum(sma_50, 1e-8)
    # SMA crossover: 10d SMA vs 50d SMA (golden/death cross proxy)
    result['sma_10_50_ratio'] = sma_10 / np.maximum(sma_50, 1e-8) - 1.0
    # Momentum reversal: sign change in 5d return
    prev_ret_5d_arr = np.zeros(n)
    prev_ret_5d_arr[5:] = ret_5d[:-5]
    result['momentum_reversal'] = np.sign(ret_5d) * np.sign(prev_ret_5d_arr) * -1  # -1 when same sign, +1 when reversed
    # Volume-price confirmation: price up + volume up = confirmed
    result['vol_price_confirm'] = np.sign(daily_ret) * np.sign(np.diff(np.concatenate([[0], volume])))
    # Relative volume (today vs 5d avg)
    vol_sma5 = vol_series.rolling(5, min_periods=1).mean().values
    result['relative_volume_5d'] = volume / np.maximum(vol_sma5, 1e-8)

    # Momentum persistence: autocorrelation of 5d returns (trending vs mean-reverting)
    ret_5d_series = pd.Series(ret_5d)
    result['return_autocorr_5d'] = ret_5d_series.rolling(20, min_periods=5).apply(
        lambda x: x.autocorr(lag=1) if len(x) > 5 else 0, raw=False).fillna(0).values
    # Volatility trend: is volatility increasing or decreasing
    std_20d_series = pd.Series(daily_ret).rolling(20, min_periods=1).std().fillna(0)
    std_20d_prev = std_20d_series.shift(10).fillna(std_20d_series)
    result['volatility_trend'] = ((std_20d_series - std_20d_prev) / np.maximum(std_20d_prev, 1e-8)).values
    # RSI-like momentum from returns (avoids needing base features)
    gains = pd.Series(np.maximum(daily_ret, 0)).rolling(14, min_periods=1).mean()
    losses = pd.Series(np.maximum(-daily_ret, 0)).rolling(14, min_periods=1).mean()
    rs = gains / np.maximum(losses, 1e-8)
    result['price_rsi_14'] = (100 - 100 / (1 + rs)).values

    # Historical UP base rate (rolling) — captures positive drift
    up_mask_series = pd.Series((daily_ret > 0).astype(float))
    result['up_rate_20d'] = up_mask_series.rolling(20, min_periods=1).mean().values
    result['up_rate_60d'] = up_mask_series.rolling(60, min_periods=1).mean().values
    # Trend regime: ratio of 20d mean return to 20d std (Sharpe-like)
    ret_mean_20d = pd.Series(daily_ret).rolling(20, min_periods=1).mean().values
    ret_std_20d = pd.Series(daily_ret).rolling(20, min_periods=1).std().fillna(1e-8).values
    result['trend_sharpe_20d'] = ret_mean_20d / np.maximum(ret_std_20d, 1e-8)
    # Cumulative return position: where are we in the recent range
    ret_cum_20d = pd.Series(daily_ret).rolling(20, min_periods=1).sum().values
    ret_cum_60d = pd.Series(daily_ret).rolling(60, min_periods=1).sum().values
    result['cum_return_20d'] = ret_cum_20d
    result['cum_return_60d'] = ret_cum_60d

    return pd.DataFrame(result, index=price_df.index)


def compute_sequence_features(features_df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Flatten last `window` days of top features into a single row per day."""
    available = [f for f in TOP_FEATURES if f in features_df.columns]
    result = {}
    for feat in available:
        col = features_df[feat].values
        for t in range(window):
            shifted = np.zeros(len(col))
            if t > 0:
                shifted[t:] = col[:-t]
            else:
                shifted = col.copy()
            result[f'{feat}_t{t}'] = shifted
    return pd.DataFrame(result, index=features_df.index)


def lagged_feature_names(base_features: List[str]) -> List[str]:
    """Return all lagged feature names for given base features."""
    available = [f for f in TOP_FEATURES if f in base_features]
    names = []
    for feat in available:
        for lag in LAG_PERIODS:
            names.append(f'{feat}_lag{lag}')
        names.append(f'{feat}_roc5')
        names.append(f'{feat}_mean5')
        names.append(f'{feat}_std5')
    return names


PRICE_DERIVED_NAMES = [
    'consecutive_up_days', 'consecutive_down_days',
    'drawdown_from_20d_high', 'rally_from_20d_low',
    'days_since_last_5pct_drop', 'volatility_expansion',
    'price_acceleration', 'volume_trend_5d',
    'intraday_range_pct', 'close_in_range', 'return_10d',
    'drawdown_from_50d_high', 'rally_from_50d_low',
    'volume_spike', 'range_expansion', 'gap_pct',
    'consecutive_higher_highs', 'consecutive_lower_lows',
    'return_3d', 'volatility_ratio_5_20', 'distance_from_sma10',
    'overnight_return', 'intraday_return',
    'upper_shadow_ratio', 'lower_shadow_ratio', 'body_ratio',
    'return_20d_derived', 'distance_from_sma20', 'volume_acceleration',
    'avg_range_5d', 'momentum_score',
    'price_vol_divergence', 'vol_adj_momentum', 'mean_reversion_strength',
    'up_ratio_10d', 'breakout_signal', 'exhaustion_signal',
    'return_50d', 'distance_from_sma50', 'sma_10_50_ratio',
    'momentum_reversal', 'vol_price_confirm', 'relative_volume_5d',
    'return_autocorr_5d', 'volatility_trend', 'price_rsi_14',
    'up_rate_20d', 'up_rate_60d', 'trend_sharpe_20d',
    'cum_return_20d', 'cum_return_60d',
]


def sequence_feature_names(base_features: List[str], window: int = 20) -> List[str]:
    """Return all sequence feature names."""
    available = [f for f in TOP_FEATURES if f in base_features]
    return [f'{feat}_t{t}' for feat in available for t in range(window)]
