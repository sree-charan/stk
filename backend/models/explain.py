"""SHAP explanations and conviction tiers for ML predictions."""
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Human-readable feature name mapping
_READABLE_NAMES = {
    'rsi_14': 'RSI(14)', 'rsi_14_lag1': 'RSI yesterday', 'rsi_14_lag5': 'RSI 5d ago',
    'macd': 'MACD', 'macd_hist': 'MACD histogram', 'macd_lag1': 'MACD yesterday',
    'bb_position': 'Bollinger Band position', 'bb_width': 'Bollinger Band width',
    'stoch_k': 'Stochastic %K', 'stoch_d': 'Stochastic %D',
    'atr_14': 'ATR(14)', 'adx_14': 'ADX(14)', 'cci': 'CCI',
    'obv_slope': 'OBV slope', 'mfi_14': 'Money Flow Index',
    'williams_r': 'Williams %R', 'roc_10': 'Rate of Change(10)',
    'return_1d': '1-day return', 'return_5d': '5-day return', 'return_20d': '20-day return',
    'volume_ratio': 'Volume ratio', 'volume_change': 'Volume change',
    'vwap_distance': 'Distance from VWAP', 'price_to_vwap': 'Price/VWAP',
    'consecutive_up_days': 'Consecutive up days', 'consecutive_down_days': 'Consecutive down days',
    'drawdown_from_20d_high': 'Drawdown from 20d high', 'rally_from_20d_low': 'Rally from 20d low',
    'days_since_last_5pct_drop': 'Days since 5% drop', 'volatility_expansion': 'Volatility expansion',
    'price_acceleration': 'Price acceleration', 'volume_trend_5d': 'Volume trend (5d)',
    'return_vs_spy': 'Return vs SPY', 'return_vs_sector': 'Return vs sector',
    'spy_rsi': 'SPY RSI', 'spy_macd': 'SPY MACD',
    'correlation_with_spy_20d': 'Correlation with SPY', 'sector_momentum': 'Sector momentum',
    'distance_from_sma10': 'Distance from SMA(10)', 'distance_from_sma20': 'Distance from SMA(20)',
    'momentum_score': 'Momentum score', 'volatility_ratio_5_20': 'Volatility ratio (5d/20d)',
    'volume_spike': 'Volume spike', 'range_expansion': 'Range expansion',
    'relative_strength_20d': 'Relative strength vs SPY', 'beta_spy_20d': 'Beta to SPY',
    'price_vol_divergence': 'Price-volume divergence', 'vol_adj_momentum': 'Vol-adjusted momentum',
    'mean_reversion_strength': 'Mean reversion strength', 'up_ratio_10d': 'Up days ratio (10d)',
    'breakout_signal': 'Breakout signal', 'exhaustion_signal': 'Exhaustion signal',
    'return_50d': '50-day return', 'distance_from_sma50': 'Distance from SMA(50)',
    'sma_10_50_ratio': 'SMA(10)/SMA(50) ratio', 'momentum_reversal': 'Momentum reversal',
    'vol_price_confirm': 'Volume-price confirmation', 'relative_volume_5d': 'Relative volume (5d)',
    'trix': 'TRIX', 'high_low_range': 'High-Low range', 'close_position': 'Close position in range',
    'gap': 'Gap', 'up_volume_ratio': 'Up volume ratio', 'dollar_volume': 'Dollar volume',
    'volume_sma20': 'Avg volume (20d)', 'volume_std': 'Volume volatility',
    'vwap_slope': 'VWAP slope',
    # Technical indicators (tier2)
    'sma_5': 'SMA(5)', 'sma_10': 'SMA(10)', 'sma_20': 'SMA(20)', 'sma_50': 'SMA(50)',
    'ema_5': 'EMA(5)', 'ema_10': 'EMA(10)', 'ema_20': 'EMA(20)', 'ema_50': 'EMA(50)',
    'bb_lower': 'Bollinger Lower', 'bb_upper': 'Bollinger Upper',
    'vwap': 'VWAP', 'vwap_upper': 'VWAP Upper', 'vwap_lower': 'VWAP Lower',
    'vwap_band_width': 'VWAP Band Width',
    'obv': 'OBV', 'obv_sma': 'OBV SMA', 'adl_slope': 'ADL Slope',
    'vpt': 'Volume Price Trend', 'emv': 'Ease of Movement',
    'roc_5': 'ROC(5)', 'roc_20': 'ROC(20)', 'roc_60': 'ROC(60)',
    'rsi_7': 'RSI(7)', 'momentum_10': 'Momentum(10)',
    'volatility_20': 'Volatility(20d)', 'trend_strength': 'Trend Strength',
    'supertrend': 'SuperTrend', 'psar_dist': 'Parabolic SAR Distance',
    'ichimoku_tenkan': 'Ichimoku Tenkan', 'ichimoku_kijun': 'Ichimoku Kijun',
    'ichimoku_tk_cross': 'Ichimoku TK Cross', 'ichimoku_cloud_pos': 'Ichimoku Cloud Position',
    'ichimoku_cloud_width': 'Ichimoku Cloud Width',
    'keltner_lower': 'Keltner Lower', 'keltner_upper': 'Keltner Upper',
    'keltner_position': 'Keltner Position',
    'atr_band_lower': 'ATR Band Lower', 'atr_band_upper': 'ATR Band Upper', 'atrp': 'ATR%',
    'donchian_position': 'Donchian Position', 'donchian_width': 'Donchian Width',
    'donchian_breakout': 'Donchian Breakout',
    'pivot_distance': 'Pivot Distance', 'pivot_r1_dist': 'Pivot R1 Distance',
    'pivot_s1_dist': 'Pivot S1 Distance', 'price_channel_pos': 'Price Channel Position',
    'fib_236': 'Fib 23.6%', 'fib_382': 'Fib 38.2%', 'fib_500': 'Fib 50%', 'fib_618': 'Fib 61.8%',
    'cmf': 'Chaikin Money Flow', 'kvo': 'Klinger Volume Osc',
    'force_index': 'Force Index', 'mass_index': 'Mass Index',
    'connors_rsi': 'Connors RSI', 'cmo': 'Chande Momentum',
    'coppock_curve': 'Coppock Curve', 'dpo': 'Detrended Price Osc',
    'kst': 'KST', 'stc': 'Schaff Trend Cycle', 'rvi': 'Relative Vigor Index',
    'ultimate_osc': 'Ultimate Oscillator', 'vortex': 'Vortex',
    'squeeze_mom': 'Squeeze Momentum', 'choppiness': 'Choppiness Index',
    'hma_dist': 'HMA Distance', 'linear_reg_slope': 'Linear Regression Slope',
    'adxr': 'ADXR', 'aroon_up': 'Aroon Up', 'aroon_down': 'Aroon Down',
    'elder_bull': 'Elder Bull Power', 'elder_bear': 'Elder Bear Power',
    'williams_r_7': 'Williams %R(7)', 'macd_signal': 'MACD Signal',
    # Tier1 extras
    'higher_high': 'Higher High', 'lower_low': 'Lower Low',
    'inside_bar': 'Inside Bar', 'outside_bar': 'Outside Bar',
    # Microstructure
    'volume_clock': 'Volume Clock', 'trade_intensity': 'Trade Intensity',
    'kyle_lambda': 'Kyle Lambda', 'amihud_illiquidity': 'Amihud Illiquidity',
    'roll_spread': 'Roll Spread', 'hasbrouck_info': 'Hasbrouck Info',
    'spy_volatility_20d': 'SPY volatility (20d)', 'correlation_with_sector_20d': 'Correlation with sector',
    'intraday_range_pct': 'Intraday range %', 'close_in_range': 'Close position in range',
    'return_10d': '10-day return', 'drawdown_from_50d_high': 'Drawdown from 50d high',
    'rally_from_50d_low': 'Rally from 50d low', 'gap_pct': 'Gap %',
    'consecutive_higher_highs': 'Consecutive higher highs', 'consecutive_lower_lows': 'Consecutive lower lows',
    'return_3d': '3-day return', 'overnight_return': 'Overnight return', 'intraday_return': 'Intraday return',
    'upper_shadow_ratio': 'Upper shadow ratio', 'lower_shadow_ratio': 'Lower shadow ratio',
    'body_ratio': 'Candle body ratio', 'return_20d_derived': '20-day return',
    'volume_acceleration': 'Volume acceleration', 'avg_range_5d': 'Avg range (5d)',
    'return_autocorr_5d': 'Return autocorrelation (5d)', 'volatility_trend': 'Volatility trend',
    'price_rsi_14': 'Price RSI(14)',
    'regime_bull': 'Bull market regime', 'regime_bear': 'Bear market regime',
    'regime_sideways': 'Sideways regime', 'vol_regime_high': 'High volatility regime',
    'vol_regime_low': 'Low volatility regime', 'vol_regime_normal': 'Normal volatility regime',
    'trend_trending': 'Trending', 'trend_strong_trend': 'Strong trend',
    'trend_no_trend': 'No trend', 'adx_value': 'ADX value',
    'mom_strong_up': 'Strong upward momentum', 'mom_up': 'Upward momentum',
    'mom_down': 'Downward momentum', 'mom_strong_down': 'Strong downward momentum',
    'momentum_roc20': 'Momentum (20d ROC)',
    # Options
    'put_call_ratio': 'Put/Call ratio', 'put_call_volume': 'Put/Call volume',
    'put_call_oi': 'Put/Call open interest', 'atm_iv': 'ATM implied volatility',
    'avg_iv': 'Avg implied volatility', 'iv_skew': 'IV skew',
    'iv_term_structure': 'IV term structure', 'max_pain': 'Max pain',
    'total_volume': 'Options total volume', 'total_oi': 'Options total OI',
    # Fundamentals
    'eps': 'EPS', 'eps_surprise': 'EPS surprise', 'eps_growth_qoq': 'EPS growth (QoQ)',
    'eps_growth_yoy': 'EPS growth (YoY)', 'revenue_growth_qoq': 'Revenue growth (QoQ)',
    'revenue_growth_yoy': 'Revenue growth (YoY)', 'gross_margin': 'Gross margin',
    'operating_margin': 'Operating margin', 'pe_ratio': 'P/E ratio',
    'price_to_book': 'Price/Book', 'debt_to_equity': 'Debt/Equity',
    # Sentiment
    'call_sentiment': 'Earnings call sentiment', 'filing_sentiment': 'SEC filing sentiment',
    'news_sentiment': 'News sentiment', 'news_volume': 'News volume',
    'social_sentiment': 'Social media sentiment',
    # Institutional
    'institutional_ownership': 'Institutional ownership', 'institutional_change': 'Institutional change',
    'insider_buying': 'Insider buying', 'short_interest': 'Short interest',
    'days_to_cover': 'Days to cover',
    # Macro
    'fed_funds_rate': 'Fed funds rate', 'treasury_10y': '10Y Treasury yield',
    'treasury_2y': '2Y Treasury yield', 'yield_curve': 'Yield curve',
    'vix': 'VIX', 'vix_term_structure': 'VIX term structure',
    'sp500_return_20d': 'S&P 500 20d return', 'market_breadth': 'Market breadth',
}


def _readable_name(feat) -> str:
    """Convert feature name to human-readable form."""
    feat = str(feat)
    if feat in _READABLE_NAMES:
        return _READABLE_NAMES[feat]
    # Handle lagged/derived patterns
    for base, readable in _READABLE_NAMES.items():
        if feat.startswith(base + '_lag'):
            lag = feat.split('_lag')[-1]
            return f"{readable} {lag}d ago"
        if feat.startswith(base + '_roc5'):
            return f"{readable} 5d change"
        if feat.startswith(base + '_mean5'):
            return f"{readable} 5d avg"
        if feat.startswith(base + '_std5'):
            return f"{readable} 5d volatility"
    # Sequence features
    if '_t' in feat and feat.split('_t')[-1].isdigit():
        base = '_'.join(feat.split('_')[:-1])
        t = feat.split('_t')[-1]
        base_name = _READABLE_NAMES.get(base, base)
        return f"{base_name} {t}d ago" if t != '0' else base_name
    return feat.replace('_', ' ').title()


def _base_feature(name) -> str:
    """Extract base feature name, stripping lag/roc/mean/std/sequence suffixes."""
    name = str(name)
    import re
    # Sequence features: rsi_14_t5 -> rsi_14
    if re.search(r'_t\d+$', name):
        return re.sub(r'_t\d+$', '', name)
    # Lagged: rsi_14_lag5 -> rsi_14 (any lag number)
    m = re.search(r'_lag\d+$', name)
    if m:
        return name[:m.start()]
    # Rolling stats
    for suffix in ('_roc5', '_mean5', '_std5'):
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return name


def _deduplicate_factors(factors: List[Tuple[str, float, float]], top_n: int) -> List[Tuple[str, float, float]]:
    """Keep only the most influential entry per base feature."""
    seen_bases = set()
    result = []
    for name, shap_val, feat_val in factors:
        base = _base_feature(name)
        if base in seen_bases:
            continue
        seen_bases.add(base)
        result.append((name, shap_val, feat_val))
        if len(result) >= top_n:
            break
    return result


def compute_shap_explanation(model, X_row: np.ndarray, feature_names: List[str],
                             top_n: int = 5) -> Dict:
    """Compute SHAP values for a single prediction and return human-readable explanation."""
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_row.reshape(1, -1))
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        sv = shap_values.flatten()
    except Exception:
        return {'bullish_factors': [], 'bearish_factors': [], 'raw_shap': None}

    pairs = list(zip(feature_names[:len(sv)], sv, X_row[:len(sv)]))
    # Sort by SHAP value descending for bullish, ascending for bearish
    pairs_bull = sorted(pairs, key=lambda x: x[1], reverse=True)
    pairs_bear = sorted(pairs, key=lambda x: x[1])

    bullish = _deduplicate_factors(
        [(n, s, v) for n, s, v in pairs_bull if s > 0], top_n)
    bearish = _deduplicate_factors(
        [(n, s, v) for n, s, v in pairs_bear if s < 0], top_n)

    total_abs = sum(abs(s) for _, s, _ in pairs) or 1.0
    return {
        'bullish_factors': [
            {'feature': _readable_name(n), 'influence': round(abs(s) / total_abs * 100, 1),
             'value': round(v, 4), 'raw_name': n}
            for n, s, v in bullish
        ],
        'bearish_factors': [
            {'feature': _readable_name(n), 'influence': round(abs(s) / total_abs * 100, 1),
             'value': round(v, 4), 'raw_name': n}
            for n, s, v in bearish
        ],
        'raw_shap': sv.tolist() if sv is not None else None,
    }


def _fmt_val(v: float) -> str:
    """Format feature value for display — compact large numbers."""
    av = abs(v)
    if av >= 1e9:
        return f"{v/1e9:.1f}B"
    if av >= 1e6:
        return f"{v/1e6:.1f}M"
    if av >= 1e3:
        return f"{v/1e3:.1f}K"
    if av >= 10:
        return f"{v:.1f}"
    return f"{v:.2f}"


def format_shap_explanation(explanation: Dict, direction: str = 'bullish') -> str:
    """Format SHAP explanation as human-readable text, direction-aware."""
    lines = []
    if direction == 'bearish':
        if explanation.get('bearish_factors'):
            lines.append("Why SELL:")
            for f in explanation['bearish_factors']:
                lines.append(f"  📉 {f['feature']} ({_fmt_val(f['value'])}) → {f['influence']:.0f}% influence")
        if explanation.get('bullish_factors'):
            lines.append("Why not:")
            for f in explanation['bullish_factors']:
                lines.append(f"  📈 {f['feature']} ({_fmt_val(f['value'])}) → {f['influence']:.0f}% influence")
    else:
        if explanation.get('bullish_factors'):
            lines.append("Why BUY:")
            for f in explanation['bullish_factors']:
                lines.append(f"  📈 {f['feature']} ({_fmt_val(f['value'])}) → +{f['influence']:.0f}% influence")
        if explanation.get('bearish_factors'):
            lines.append("Why not:")
            for f in explanation['bearish_factors']:
                lines.append(f"  📉 {f['feature']} ({_fmt_val(f['value'])}) → -{f['influence']:.0f}% influence")
    return '\n'.join(lines) if lines else "No explanation available"


# --- Conviction Tiers ---

def get_conviction_tier(prob: float, conf_interval_width: Optional[float] = None) -> Tuple[str, str, str]:
    """Return (tier, label, emoji) based on classifier probability.
    
    prob: probability of the predicted direction (0.5-1.0)
    conf_interval_width: optional conformal interval width — narrow intervals
        boost conviction, wide intervals reduce it.
    """
    # Adjust probability based on conformal interval width if available
    adjusted = prob
    if conf_interval_width is not None:
        # Narrow interval (<5%) → boost up to +0.03, wide (>15%) → penalize up to -0.03
        width_pct = conf_interval_width * 100
        if width_pct < 5:
            adjusted = min(1.0, prob + 0.03)
        elif width_pct > 15:
            adjusted = max(0.5, prob - 0.03)

    if adjusted > 0.65:
        return 'HIGH', 'high conviction', '🟢'
    elif adjusted > 0.55:
        return 'MODERATE', 'moderate conviction', '🟡'
    else:
        return 'LOW', 'low conviction', '⚪'


def format_conviction_verdict(direction: str, prob: float, wf_accuracy: Optional[float] = None) -> str:
    """Format prediction with conviction tier."""
    tier, label, emoji = get_conviction_tier(prob)
    pct = int(prob * 100)

    if direction == 'bullish':
        if tier == 'HIGH':
            action = 'BUY'
        else:
            action = 'LEAN BUY'
        color_emoji = '🟢' if tier == 'HIGH' else emoji
    elif direction == 'bearish':
        if tier == 'HIGH':
            action = 'SELL'
        else:
            action = 'LEAN SELL'
        color_emoji = '🔴' if tier == 'HIGH' else emoji
    else:
        action = 'HOLD'
        color_emoji = '🟡'

    verdict = f"{color_emoji} {action} ({pct}% — {label})"
    if wf_accuracy is not None:
        verdict += f"  Model accuracy: {int(wf_accuracy * 100)}%"
    return verdict


def compute_volatility_zscore(predicted_return: float, historical_std: float) -> Tuple[float, str]:
    """Compute z-score of predicted return relative to stock's volatility."""
    if historical_std < 1e-8:
        return 0.0, "normal noise"
    z = predicted_return / historical_std
    abs_z = abs(z)
    if abs_z > 2.0:
        desc = "very unusual"
    elif abs_z > 1.0:
        desc = "unusual"
    else:
        desc = "normal noise"
    return round(z, 2), desc


# --- Conformal Prediction Intervals ---

def compute_conformal_interval(predicted_return: float, residuals: List[float],
                               alpha: float = 0.1) -> Tuple[float, float]:
    """Compute conformal prediction interval from WF residuals.

    Returns (lower, upper) bounds at (1-alpha) confidence level.
    Uses split conformal method: quantile of |residuals| as half-width.
    """
    if not residuals or len(residuals) < 10:
        return predicted_return - 0.05, predicted_return + 0.05
    abs_res = sorted(abs(r) for r in residuals)
    q_idx = int(np.ceil((1 - alpha) * (len(abs_res) + 1))) - 1
    q_idx = min(q_idx, len(abs_res) - 1)
    half_width = abs_res[q_idx]
    return round(predicted_return - half_width, 6), round(predicted_return + half_width, 6)


def format_conformal_interval(lower: float, upper: float) -> str:
    """Format prediction interval for display."""
    return f"[{lower*100:+.2f}%, {upper*100:+.2f}%]"


def uncertainty_label(lower: float, upper: float) -> str:
    """Return uncertainty label based on conformal interval width."""
    width = (upper - lower) * 100
    if width < 5:
        return '🎯 low uncertainty'
    elif width < 15:
        return '📊 moderate uncertainty'
    else:
        return '🌫️ high uncertainty'


# --- Feature Drift Detection ---

def detect_feature_drift(current_values: np.ndarray, train_means: np.ndarray,
                         train_stds: np.ndarray, feature_names: List[str],
                         threshold: float = 3.0) -> List[Dict]:
    """Detect features that are far outside training distribution.

    Returns list of drifted features with z-scores.
    """
    if len(current_values) != len(train_means):
        return []
    drifted = []
    safe_stds = np.where(train_stds > 1e-8, train_stds, 1.0)
    z_scores = (current_values - train_means) / safe_stds
    for i in range(len(z_scores)):
        if abs(z_scores[i]) > threshold:
            name = feature_names[i] if i < len(feature_names) else f'feature_{i}'
            drifted.append({
                'feature': _readable_name(name),
                'raw_name': name,
                'z_score': round(float(z_scores[i]), 1),
                'current': round(float(current_values[i]), 4),
                'train_mean': round(float(train_means[i]), 4),
            })
    drifted.sort(key=lambda x: abs(x['z_score']), reverse=True)
    return drifted[:10]


# --- Model Health Score ---

def compute_model_health(wf_accuracy: Dict[str, float], brier_scores: Optional[Dict] = None,
                         has_calibration: bool = False, model_age_days: Optional[int] = None) -> Dict:
    """Compute overall model health from multiple quality signals.

    Returns dict with score (0-100), grade (A-F), and component breakdown.
    """
    components = {}

    # WF accuracy component (0-50 points): 50% → 0, 55% → 17, 60% → 33, 65%+ → 50
    if wf_accuracy:
        avg_wf = float(np.mean([v for v in wf_accuracy.values() if isinstance(v, (int, float))]))
        wf_score = max(0, min(50, (avg_wf - 0.50) * 333))
        components['wf_accuracy'] = {'value': round(avg_wf, 4), 'score': round(wf_score, 1)}
    else:
        wf_score = 0
        components['wf_accuracy'] = {'value': None, 'score': 0}

    # Brier score component (0-25 points): 0.25 → 0, 0.22 → 7.5, 0.15 → 25
    brier_score = 0
    if brier_scores:
        brier_vals = []
        for h in ('short', 'medium', 'long'):
            b = brier_scores.get(h, {})
            val = b.get('calibrated', b.get('raw'))
            if val is not None:
                brier_vals.append(val)
        if brier_vals:
            avg_brier = float(np.mean(brier_vals))
            # More generous: 0.25 → 5, 0.20 → 15, 0.15 → 25
            brier_score = max(0, min(25, (0.27 - avg_brier) * 208))
            components['brier'] = {'value': round(avg_brier, 4), 'score': round(brier_score, 1)}

    # Calibration bonus (0-10 points)
    cal_score = 10 if has_calibration else 0
    components['calibration'] = {'value': has_calibration, 'score': cal_score}

    # Freshness penalty (0-15 points): 0 days → 15, 30+ days → 0
    fresh_score = 15
    if model_age_days is not None:
        fresh_score = max(0, 15 - model_age_days * 0.5)
    components['freshness'] = {'value': model_age_days, 'score': round(fresh_score, 1)}

    total = wf_score + brier_score + cal_score + fresh_score
    total = min(100, max(0, total))

    if total >= 80:
        grade = 'A'
    elif total >= 60:
        grade = 'B'
    elif total >= 40:
        grade = 'C'
    elif total >= 20:
        grade = 'D'
    else:
        grade = 'F'

    return {'score': round(total, 1), 'grade': grade, 'components': components}


def format_model_health(health: Dict) -> str:
    """Format model health for display."""
    grade = health['grade']
    score = health['score']
    icons = {'A': '🟢', 'B': '🟡', 'C': '🟠', 'D': '🔴', 'F': '⛔'}
    return f"{icons.get(grade, '⚪')} Model Health: {grade} ({score:.0f}/100)"


def check_health_degradation(model_dir) -> Optional[str]:
    """Check if model health is degrading across retrains. Returns warning or None."""
    trend_path = Path(model_dir) / 'health_trend.json'
    if not trend_path.exists():
        return None
    import json
    trend = json.loads(trend_path.read_text())
    if len(trend) < 2:
        return None
    latest = trend[-1].get('score', 0)
    prev = trend[-2].get('score', 0)
    if latest < 40:
        return f"⚠ Model health critically low ({latest:.0f}/100) — retrain recommended"
    if len(trend) >= 3:
        scores = [t.get('score', 0) for t in trend[-3:]]
        if all(scores[i] > scores[i + 1] for i in range(len(scores) - 1)):
            return f"⚠ Health declining: {' → '.join(f'{s:.0f}' for s in scores)} — consider retraining with more data"
    if prev - latest > 15:
        return f"⚠ Health dropped {prev - latest:.0f} points since last retrain"
    return None


def save_health_trend(model_dir, health: Dict) -> None:
    """Append current health score to trend file for tracking across retrains."""
    import json
    from datetime import datetime
    trend_path = model_dir / 'health_trend.json'
    history = []
    if trend_path.exists():
        try:
            history = json.loads(trend_path.read_text())
        except Exception:
            history = []
    history.append({
        'timestamp': datetime.now().isoformat(),
        'score': health['score'],
        'grade': health['grade'],
    })
    # Keep last 20 entries
    history = history[-20:]
    trend_path.write_text(json.dumps(history))


def format_health_trend(model_dir) -> Optional[str]:
    """Format health trend for display. Returns None if no trend data."""
    import json
    trend_path = model_dir / 'health_trend.json'
    if not trend_path.exists():
        return None
    try:
        history = json.loads(trend_path.read_text())
    except Exception:
        return None
    if len(history) < 2:
        return None
    prev = history[-2]['score']
    curr = history[-1]['score']
    delta = curr - prev
    if delta > 2:
        arrow = '📈'
        desc = 'improving'
    elif delta < -2:
        arrow = '📉'
        desc = 'declining'
    else:
        arrow = '➡️'
        desc = 'stable'
    return f"{arrow} Health trend: {desc} ({prev:.0f} → {curr:.0f})"


def save_feature_changelog(model_dir, top_features: List[str]) -> None:
    """Save top features for changelog tracking across retrains."""
    import json
    from datetime import datetime
    path = model_dir / 'feature_changelog.json'
    history = []
    if path.exists():
        try:
            history = json.loads(path.read_text())
        except Exception:
            history = []
    history.append({
        'timestamp': datetime.now().isoformat(),
        'top_features': top_features[:20],
    })
    history = history[-10:]
    path.write_text(json.dumps(history))


def format_feature_changelog(model_dir) -> Optional[str]:
    """Format feature importance changes between last two retrains."""
    import json
    path = model_dir / 'feature_changelog.json'
    if not path.exists():
        return None
    try:
        history = json.loads(path.read_text())
    except Exception:
        return None
    if len(history) < 2:
        return None
    prev_set = set(history[-2]['top_features'])
    curr_set = set(history[-1]['top_features'])
    new_feats = curr_set - prev_set
    dropped = prev_set - curr_set
    if not new_feats and not dropped:
        return None
    parts = []
    if new_feats:
        names = [_readable_name(f) for f in list(new_feats)[:3]]
        parts.append(f"new: {', '.join(names)}")
    if dropped:
        names = [_readable_name(f) for f in list(dropped)[:3]]
        parts.append(f"dropped: {', '.join(names)}")
    return f"Feature changes: {'; '.join(parts)}"


def calibrate_platt(probs: np.ndarray, labels: np.ndarray) -> Optional[object]:
    """Fit Platt scaling (logistic regression) calibrator on OOS probabilities.

    Returns a fitted LogisticRegression or None on failure.
    """
    if len(probs) < 20:
        return None
    try:
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
        lr.fit(probs.reshape(-1, 1), labels)
        return lr
    except Exception:
        return None


def apply_platt(calibrator, prob: float) -> float:
    """Apply Platt scaling calibrator to a single probability."""
    if calibrator is None:
        return prob
    try:
        return float(calibrator.predict_proba(np.array([[prob]]))[0, 1])
    except Exception:
        return prob


def compute_calibration_curve(probs: np.ndarray, actuals: np.ndarray,
                              n_bins: int = 5) -> List[Dict]:
    """Compute calibration curve from OOS probabilities and actual outcomes.

    Returns list of {predicted, actual, count} dicts for each bin.
    """
    if len(probs) < 10:
        return []
    bins = np.linspace(0, 1, n_bins + 1)
    curve = []
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if i == n_bins - 1:
            mask = (probs >= bins[i]) & (probs <= bins[i + 1])
        count = int(mask.sum())
        if count > 0:
            curve.append({
                'predicted': round(float(probs[mask].mean()), 4),
                'actual': round(float(actuals[mask].mean()), 4),
                'count': count,
            })
    return curve


def format_ascii_calibration_curve(curve: List[Dict], width: int = 40) -> str:
    """Render calibration curve as ASCII bar chart.

    Each bin shows predicted vs actual probability with a visual bar.
    """
    if not curve:
        return "No calibration data"
    lines = ["  Predicted  Actual   Gap    Chart"]
    lines.append("  " + "-" * (width + 25))
    for b in curve:
        pred = b['predicted']
        act = b['actual']
        n = b.get('count', 0)
        gap = abs(pred - act)
        # Bar: filled to 'actual', marker at 'predicted'
        bar_len = width
        act_pos = int(act * bar_len)
        pred_pos = int(pred * bar_len)
        bar = list('·' * bar_len)
        for i in range(min(act_pos, bar_len)):
            bar[i] = '█'
        if 0 <= pred_pos < bar_len:
            bar[pred_pos] = '|'
        quality = '✓' if gap < 0.05 else ('~' if gap < 0.10 else '✗')
        lines.append(f"  {pred:5.2f}    {act:5.2f}   {gap:4.2f} {quality} {''.join(bar)}  (n={n})")
    lines.append(f"  {'':5s}    {'':5s}         {'0':>{1}}{'':·<{width-2}}{'1'}")
    return '\n'.join(lines)


def compute_ensemble_diversity(bag_probs: List[np.ndarray]) -> Dict:
    """Measure diversity among bagged model predictions.

    Higher diversity = models disagree more = better ensemble.
    Returns dict with mean_disagreement (0-1) and pairwise correlations.
    """
    if len(bag_probs) < 2:
        return {'diversity': 0.0, 'avg_correlation': 1.0, 'description': 'single model'}
    n_pairs = 0
    total_disagree = 0.0
    correlations = []
    for i in range(len(bag_probs)):
        for j in range(i + 1, len(bag_probs)):
            dirs_i = (bag_probs[i] > 0.5).astype(int)
            dirs_j = (bag_probs[j] > 0.5).astype(int)
            total_disagree += float(np.mean(dirs_i != dirs_j))
            n_pairs += 1
            if len(bag_probs[i]) > 2:
                with np.errstate(invalid='ignore'):
                    c = np.corrcoef(bag_probs[i], bag_probs[j])[0, 1]
                if not np.isnan(c):
                    correlations.append(float(c))
    mean_disagree = total_disagree / max(n_pairs, 1)
    avg_corr = float(np.mean(correlations)) if correlations else 1.0
    if mean_disagree > 0.15:
        desc = 'high diversity'
    elif mean_disagree > 0.05:
        desc = 'moderate diversity'
    else:
        desc = 'low diversity'
    return {
        'diversity': round(mean_disagree, 4),
        'avg_correlation': round(avg_corr, 4),
        'description': desc,
    }


def best_horizon_recommendation(wf_accuracy: Dict[str, float]) -> Optional[Dict]:
    """Recommend the best horizon for a ticker based on walk-forward accuracy.
    
    Returns dict with 'horizon', 'accuracy', 'label', 'reason' or None if no data.
    """
    if not wf_accuracy:
        return None
    valid = {h: v for h, v in wf_accuracy.items() if isinstance(v, (int, float)) and h in ('short', 'medium', 'long')}
    if not valid:
        return None
    best_h = max(valid, key=valid.get)
    acc = valid[best_h]
    labels = {'short': '1-day', 'medium': '5-day', 'long': '20-day'}
    if acc < 0.53:
        return {'horizon': best_h, 'accuracy': acc, 'label': labels.get(best_h, best_h),
                'reason': 'No horizon shows reliable accuracy — treat all predictions with caution'}
    reasons = {
        'short': 'Short-term momentum patterns are strongest for this ticker',
        'medium': 'Medium-term trend patterns are most reliable for this ticker',
        'long': 'Longer-term cycles are most predictable for this ticker',
    }
    return {'horizon': best_h, 'accuracy': acc, 'label': labels.get(best_h, best_h),
            'reason': reasons.get(best_h, '')}


def signal_quality_assessment(prob: float, wf_accuracy: Optional[float],
                               conviction_tier: str, drift_count: int = 0) -> Dict:
    """Assess overall signal quality combining multiple factors.
    
    Returns dict with 'quality' (HIGH/MODERATE/LOW/UNRELIABLE), 'icon', 'warnings'.
    """
    warnings = []
    score = 0  # 0-10

    # Conviction contributes 0-4 points
    if conviction_tier == 'HIGH':
        score += 4
    elif conviction_tier == 'MODERATE':
        score += 2
    else:
        score += 0
        warnings.append('Low model conviction')

    # WF accuracy contributes 0-4 points
    if wf_accuracy is not None:
        if wf_accuracy >= 0.60:
            score += 4
        elif wf_accuracy >= 0.55:
            score += 2
        elif wf_accuracy >= 0.52:
            score += 1
        else:
            warnings.append(f'Walk-forward accuracy only {wf_accuracy*100:.0f}%')
    else:
        warnings.append('No walk-forward validation data')

    # Drift penalty
    if drift_count >= 3:
        score -= 2
        warnings.append(f'{drift_count} features show distribution drift')
    elif drift_count >= 1:
        score -= 1

    # Probability extremity contributes 0-2 points
    extremity = abs(prob - 0.5)
    if extremity > 0.15:
        score += 2
    elif extremity > 0.05:
        score += 1

    score = max(0, min(10, score))
    if score >= 7:
        return {'quality': 'HIGH', 'icon': '🟢', 'score': score, 'warnings': warnings}
    elif score >= 4:
        return {'quality': 'MODERATE', 'icon': '🟡', 'score': score, 'warnings': warnings}
    elif score >= 2:
        return {'quality': 'LOW', 'icon': '⚪', 'score': score, 'warnings': warnings}
    else:
        return {'quality': 'UNRELIABLE', 'icon': '🔴', 'score': score, 'warnings': warnings}


def format_signal_quality(sq: Dict) -> str:
    """Format signal quality for display."""
    text = f"Signal quality: {sq['icon']} {sq['quality']}"
    if sq['warnings']:
        text += f" — {'; '.join(sq['warnings'])}"
    return text
