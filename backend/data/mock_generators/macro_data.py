"""Mock macro data generator - GDP, CPI, rates, VIX."""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

def generate_interest_rates(days: int = 730, seed: Optional[int] = None) -> pd.DataFrame:
    """Generate interest rate data."""
    if seed:
        np.random.seed(seed)
    
    data = []
    end_date = datetime.now()
    
    # Start values
    fed_funds = 5.0
    t2y = 4.5
    t10y = 4.2
    t30y = 4.5
    
    for d in range(days):
        date = end_date - timedelta(days=d)
        
        # Random walk with mean reversion
        fed_funds += np.random.normal(0, 0.01) - 0.001 * (fed_funds - 4.0)
        t2y += np.random.normal(0, 0.02) - 0.001 * (t2y - fed_funds)
        t10y += np.random.normal(0, 0.02) - 0.001 * (t10y - 4.0)
        t30y += np.random.normal(0, 0.02) - 0.001 * (t30y - 4.2)
        
        fed_funds = np.clip(fed_funds, 0, 8)
        t2y = np.clip(t2y, 0, 8)
        t10y = np.clip(t10y, 0, 8)
        t30y = np.clip(t30y, 0, 8)
        
        data.append({
            'date': date.date(),
            'fed_funds_rate': round(fed_funds, 2),
            'treasury_2y': round(t2y, 2),
            'treasury_10y': round(t10y, 2),
            'treasury_30y': round(t30y, 2),
            'yield_curve_2_10': round(t10y - t2y, 2)
        })
    
    return pd.DataFrame(data).sort_values('date')


def generate_economic_indicators(months: int = 24, seed: Optional[int] = None) -> pd.DataFrame:
    """Generate monthly economic indicators."""
    if seed:
        np.random.seed(seed)
    
    data = []
    end_date = datetime.now()
    
    gdp_growth = 2.5
    cpi = 3.5
    unemployment = 4.0
    
    for m in range(months):
        date = end_date - timedelta(days=30 * m)
        
        # Random walk
        gdp_growth += np.random.normal(0, 0.3)
        cpi += np.random.normal(0, 0.2)
        unemployment += np.random.normal(0, 0.1)
        
        gdp_growth = np.clip(gdp_growth, -3, 6)
        cpi = np.clip(cpi, 0, 10)
        unemployment = np.clip(unemployment, 3, 10)
        
        data.append({
            'date': date.date(),
            'gdp_growth_yoy': round(gdp_growth, 2),
            'cpi_yoy': round(cpi, 2),
            'core_cpi_yoy': round(cpi * 0.85, 2),
            'unemployment_rate': round(unemployment, 1),
            'consumer_confidence': round(100 + np.random.normal(0, 10), 1),
            'pmi_manufacturing': round(50 + np.random.normal(0, 5), 1),
            'pmi_services': round(52 + np.random.normal(0, 5), 1)
        })
    
    return pd.DataFrame(data).sort_values('date')


def generate_vix(days: int = 730, seed: Optional[int] = None) -> pd.DataFrame:
    """Generate VIX data."""
    if seed:
        np.random.seed(seed)
    
    data = []
    end_date = datetime.now()
    vix = 18.0
    
    for d in range(days):
        date = end_date - timedelta(days=d)
        
        # VIX mean reverts to ~18
        vix += np.random.normal(0, 1.5) - 0.05 * (vix - 18)
        
        # Occasional spikes
        if np.random.random() < 0.02:
            vix += np.random.uniform(5, 20)
        
        vix = np.clip(vix, 10, 80)
        
        data.append({
            'date': date.date(),
            'vix': round(vix, 2),
            'vix_9d': round(vix * np.random.uniform(0.9, 1.1), 2),
            'vix_3m': round(vix * np.random.uniform(0.95, 1.05), 2),
            'vix_term_structure': round(np.random.uniform(-0.1, 0.1), 3)
        })
    
    return pd.DataFrame(data).sort_values('date')


def generate_market_regime(days: int = 730, seed: Optional[int] = None) -> pd.DataFrame:
    """Generate market regime indicators."""
    if seed:
        np.random.seed(seed)
    
    data = []
    end_date = datetime.now()
    
    regime = 'neutral'
    regime_duration = 0
    
    for d in range(days):
        date = end_date - timedelta(days=d)
        
        regime_duration += 1
        if regime_duration > np.random.randint(30, 90):
            regime = np.random.choice(['bull', 'bear', 'neutral'])
            regime_duration = 0
        
        # Regime-dependent metrics
        if regime == 'bull':
            breadth = np.random.uniform(0.5, 0.8)
            momentum = np.random.uniform(0.3, 0.7)
        elif regime == 'bear':
            breadth = np.random.uniform(0.2, 0.5)
            momentum = np.random.uniform(-0.7, -0.3)
        else:
            breadth = np.random.uniform(0.4, 0.6)
            momentum = np.random.uniform(-0.2, 0.2)
        
        data.append({
            'date': date.date(),
            'regime': regime,
            'market_breadth': round(breadth, 3),
            'momentum_score': round(momentum, 3),
            'risk_on_off': round(np.random.uniform(-1, 1), 3),
            'correlation_regime': round(np.random.uniform(0.3, 0.8), 3)
        })
    
    return pd.DataFrame(data).sort_values('date')


def get_macro_data() -> dict:
    """Main interface - get all macro data."""
    seed = 12345
    
    return {
        'interest_rates': generate_interest_rates(seed=seed),
        'economic_indicators': generate_economic_indicators(seed=seed),
        'vix': generate_vix(seed=seed),
        'market_regime': generate_market_regime(seed=seed)
    }


if __name__ == "__main__":
    data = get_macro_data()
    print("Interest rates:")
    print(data['interest_rates'].tail())
    print("\nVIX:")
    print(data['vix'].tail())
    print("\nMarket regime:")
    print(data['market_regime'].tail())
