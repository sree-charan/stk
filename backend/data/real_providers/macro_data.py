"""Real macro data via FRED API."""
import pandas as pd
from datetime import datetime, timedelta
from . import cache
from backend.utils.retry import retry

_TTL = 3600  # 1 hour
_FRED_KEY = 'bef57cb70f78659fcc6db0c40c070ee3'


@retry(max_attempts=2, base_delay=1.0)
def _fetch_fred_series(fred, series_id, days):
    end = datetime.now()
    start = end - timedelta(days=days)
    s = fred.get_series(series_id, start, end)
    return s.dropna() if s is not None and not s.empty else pd.Series(dtype=float)


def get_macro_data() -> dict:
    key = "macro_data"
    cached = cache.get(key, _TTL)
    if cached is not None:
        return cached

    try:
        from fredapi import Fred
        fred = Fred(api_key=_FRED_KEY)

        # Interest rates
        fed = _safe_series(fred, 'FEDFUNDS', 730)
        t2y = _safe_series(fred, 'DGS2', 730)
        t10y = _safe_series(fred, 'DGS10', 730)
        t30y = _safe_series(fred, 'DGS30', 730)

        rates_df = pd.DataFrame({
            'date': fed.index, 'fed_funds_rate': fed.values,
            'treasury_2y': t2y.reindex(fed.index, method='ffill').values,
            'treasury_10y': t10y.reindex(fed.index, method='ffill').values,
            'treasury_30y': t30y.reindex(fed.index, method='ffill').values,
        }).dropna()
        rates_df['yield_curve_2_10'] = rates_df['treasury_10y'] - rates_df['treasury_2y']
        rates_df['date'] = rates_df['date'].dt.date

        # Economic indicators
        gdp = _safe_series(fred, 'A191RL1Q225SBEA', 730)  # Real GDP growth
        cpi = _safe_series(fred, 'CPIAUCSL', 730)
        unemp = _safe_series(fred, 'UNRATE', 730)

        cpi_yoy = cpi.pct_change(12) * 100 if len(cpi) > 12 else cpi * 0
        econ_df = pd.DataFrame({
            'date': unemp.index[-24:],
            'gdp_growth_yoy': gdp.reindex(unemp.index[-24:], method='ffill').values,
            'cpi_yoy': cpi_yoy.reindex(unemp.index[-24:], method='ffill').values,
            'core_cpi_yoy': (cpi_yoy * 0.85).reindex(unemp.index[-24:], method='ffill').values,
            'unemployment_rate': unemp.values[-24:],
            'consumer_confidence': 100.0,
            'pmi_manufacturing': 50.0,
            'pmi_services': 52.0,
        }).dropna()
        econ_df['date'] = econ_df['date'].dt.date

        # VIX
        vix_s = _safe_series(fred, 'VIXCLS', 730)
        vix_df = pd.DataFrame({
            'date': vix_s.index, 'vix': vix_s.values,
            'vix_9d': vix_s.values, 'vix_3m': vix_s.values,
            'vix_term_structure': 0.0,
        }).dropna()
        vix_df['date'] = vix_df['date'].dt.date

        # Market regime (derived from VIX + rates)
        regime_rows = []
        for _, r in vix_df.iterrows():
            v = r['vix']
            regime = 'bear' if v > 25 else ('bull' if v < 15 else 'neutral')
            regime_rows.append({
                'date': r['date'], 'regime': regime,
                'market_breadth': 0.5, 'momentum_score': 0.0,
                'risk_on_off': 0.0, 'correlation_regime': 0.5,
            })
        regime_df = pd.DataFrame(regime_rows) if regime_rows else _empty_regime()

        result = {
            'interest_rates': rates_df,
            'economic_indicators': econ_df,
            'vix': vix_df,
            'market_regime': regime_df,
        }
        cache.put(key, result)
        return result
    except Exception as e:
        stale, age = cache.get_stale(key)
        if stale is not None:
            import logging
            logging.warning(f"Using stale cache for macro data (age: {age/60:.0f}m): {e}")
            return stale
        from backend.data.mock_generators.macro_data import get_macro_data as mock
        return mock()


def _safe_series(fred, series_id, days):
    """Fetch FRED series with fallback."""
    try:
        return _fetch_fred_series(fred, series_id, days)
    except Exception:
        return pd.Series(dtype=float)


def _empty_regime():
    return pd.DataFrame(columns=['date', 'regime', 'market_breadth', 'momentum_score', 'risk_on_off', 'correlation_regime'])
