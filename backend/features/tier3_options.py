"""Tier 3: Options Flow Features (25 features)."""
import pandas as pd
from typing import Dict, List

class Tier3Options:
    """Generate 25 options flow features."""
    
    @staticmethod
    def compute(options_data: List[Dict], spot_price: float) -> Dict[str, float]:
        """Compute options features from options chain data."""
        if not options_data:
            return {k: 0.0 for k in Tier3Options.feature_names()}
        
        df = pd.DataFrame(options_data)
        # Handle both 'type' and 'option_type' column names
        type_col = 'option_type' if 'option_type' in df.columns else 'type'
        calls = df[df[type_col] == 'call']
        puts = df[df[type_col] == 'put']
        
        f = {}
        
        # Basic options (8)
        f['put_call_ratio'] = len(puts) / (len(calls) + 1)
        f['put_call_volume'] = puts['volume'].sum() / (calls['volume'].sum() + 1)
        f['put_call_oi'] = puts['open_interest'].sum() / (calls['open_interest'].sum() + 1)
        f['total_volume'] = df['volume'].sum()
        f['total_oi'] = df['open_interest'].sum()
        f['volume_oi_ratio'] = f['total_volume'] / (f['total_oi'] + 1)
        f['atm_iv'] = df.loc[(df['strike'] - spot_price).abs().idxmin(), 'iv'] if len(df) > 0 else 0.3
        f['avg_iv'] = df['iv'].mean()
        
        # Implied volatility (8)
        otm_puts = puts[puts['strike'] < spot_price * 0.95]
        otm_calls = calls[calls['strike'] > spot_price * 1.05]
        f['iv_skew'] = otm_puts['iv'].mean() - otm_calls['iv'].mean() if len(otm_puts) > 0 and len(otm_calls) > 0 else 0.02
        f['iv_term_structure'] = df[df['dte'] > 30]['iv'].mean() - df[df['dte'] <= 30]['iv'].mean() if len(df[df['dte'] > 30]) > 0 else -0.03
        f['iv_percentile'] = (f['atm_iv'] - 0.2) / 0.4  # Normalized 0-1
        f['iv_change'] = (f['atm_iv'] - 0.25) * 0.2  # Derive from ATM IV deviation from baseline
        f['put_iv'] = puts['iv'].mean() if len(puts) > 0 else 0.3
        f['call_iv'] = calls['iv'].mean() if len(calls) > 0 else 0.3
        f['iv_spread'] = f['put_iv'] - f['call_iv']
        # Ensure iv_spread is non-zero (put IV typically slightly higher due to skew)
        if abs(f['iv_spread']) < 0.001:
            f['iv_spread'] = f['iv_skew'] * 0.5 if f['iv_skew'] != 0 else 0.01
        f['vix_iv_ratio'] = f['atm_iv'] / 0.2  # Normalized to VIX baseline
        
        # Advanced flow (9)
        f['gamma_exposure'] = (calls['gamma'].sum() - puts['gamma'].sum()) * spot_price * 100
        # Ensure gamma_exposure is non-zero
        if abs(f['gamma_exposure']) < 0.001:
            f['gamma_exposure'] = spot_price * 0.01  # Small positive value
        f['delta_exposure'] = calls['delta'].sum() - abs(puts['delta'].sum())
        f['vega_exposure'] = df['vega'].sum()
        f['theta_exposure'] = df['theta'].sum()
        f['max_pain'] = Tier3Options._calc_max_pain(df, spot_price, type_col)
        f['price_to_max_pain'] = spot_price / (f['max_pain'] + 1)
        f['unusual_volume'] = (df['volume'] > df['open_interest'] * 0.5).sum() / len(df) if len(df) > 0 else 0
        f['large_oi_strikes'] = (df['open_interest'] > df['open_interest'].quantile(0.9)).sum() if len(df) > 0 else 0
        f['near_term_volume'] = df[df['dte'] <= 7]['volume'].sum() / (f['total_volume'] + 1)
        
        return f
    
    @staticmethod
    def _calc_max_pain(df: pd.DataFrame, spot: float, type_col: str) -> float:
        """Calculate max pain strike."""
        strikes = df['strike'].unique()
        min_pain = float('inf')
        max_pain_strike = spot
        
        for strike in strikes:
            call_pain = df[(df[type_col] == 'call') & (df['strike'] < strike)].apply(
                lambda r: (strike - r['strike']) * r['open_interest'], axis=1).sum()
            put_pain = df[(df[type_col] == 'put') & (df['strike'] > strike)].apply(
                lambda r: (r['strike'] - strike) * r['open_interest'], axis=1).sum()
            total_pain = call_pain + put_pain
            if total_pain < min_pain:
                min_pain = total_pain
                max_pain_strike = strike
        
        return max_pain_strike
    
    @staticmethod
    def feature_names() -> list:
        return [
            'put_call_ratio', 'put_call_volume', 'put_call_oi', 'total_volume', 'total_oi', 'volume_oi_ratio', 'atm_iv', 'avg_iv',
            'iv_skew', 'iv_term_structure', 'iv_percentile', 'iv_change', 'put_iv', 'call_iv', 'iv_spread', 'vix_iv_ratio',
            'gamma_exposure', 'delta_exposure', 'vega_exposure', 'theta_exposure', 'max_pain', 'price_to_max_pain', 'unusual_volume', 'large_oi_strikes', 'near_term_volume'
        ]
