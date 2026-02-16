"""Tier 7: Macro & Cross-Asset Features (15 features)."""
import pandas as pd
from typing import Dict, Union

class Tier7Macro:
    """Generate 15 macro and cross-asset features."""
    
    @staticmethod
    def compute(macro_data: Union[Dict, None]) -> Dict[str, float]:
        """Compute macro features from economic data."""
        f = {}
        
        if not macro_data:
            macro_data = {}
        
        # Extract latest values from DataFrames if present
        rates = macro_data.get('interest_rates')
        if isinstance(rates, pd.DataFrame) and len(rates) > 0:
            latest = rates.iloc[-1]
            fed_funds = float(latest.get('fed_funds_rate', 5.0))
            t10y = float(latest.get('treasury_10y', 4.0))
            t2y = float(latest.get('treasury_2y', 4.5))
        else:
            fed_funds = macro_data.get('fed_funds', 5.0)
            t10y = macro_data.get('treasury_10y', 4.0)
            t2y = macro_data.get('treasury_2y', 4.5)
        
        # Interest rates (5)
        f['fed_funds_rate'] = fed_funds
        f['treasury_10y'] = t10y
        f['treasury_2y'] = t2y
        f['yield_curve'] = t10y - t2y
        f['rate_change_3m'] = macro_data.get('rate_change_3m', 0.05)  # Small positive change
        
        # VIX data
        vix_df = macro_data.get('vix')
        if isinstance(vix_df, pd.DataFrame) and len(vix_df) > 0:
            latest = vix_df.iloc[-1]
            vix = float(latest.get('vix', 20))
            vix_3m = float(latest.get('vix_3m', 22))
        else:
            vix = macro_data.get('vix', 20) if not isinstance(macro_data.get('vix'), pd.DataFrame) else 20
            vix_3m = macro_data.get('vix_3m', 22)
        
        # Market regime (5)
        f['vix'] = vix
        f['vix_term_structure'] = vix_3m - vix
        f['sp500_return_20d'] = macro_data.get('sp500_return_20d', 0.02)  # Small positive return
        
        regime_df = macro_data.get('market_regime')
        if isinstance(regime_df, pd.DataFrame) and len(regime_df) > 0:
            latest = regime_df.iloc[-1]
            f['market_breadth'] = float(latest.get('market_breadth', 0.5))
            f['sector_rotation'] = float(latest.get('momentum_score', 0))
        else:
            f['market_breadth'] = macro_data.get('advance_decline', 0.5)
            f['sector_rotation'] = macro_data.get('sector_momentum', 0)
        
        # Economic indicators (5)
        econ_df = macro_data.get('economic_indicators')
        if isinstance(econ_df, pd.DataFrame) and len(econ_df) > 0:
            latest = econ_df.iloc[-1]
            f['gdp_growth'] = float(latest.get('gdp_growth_yoy', 2.0))
            f['cpi_yoy'] = float(latest.get('cpi_yoy', 3.0))
            f['unemployment'] = float(latest.get('unemployment_rate', 4.0))
            f['pmi'] = float(latest.get('pmi_manufacturing', 52))
            f['consumer_confidence'] = float(latest.get('consumer_confidence', 100))
        else:
            f['gdp_growth'] = macro_data.get('gdp_growth', 2.0)
            f['cpi_yoy'] = macro_data.get('cpi', 3.0)
            f['unemployment'] = macro_data.get('unemployment', 4.0)
            f['pmi'] = macro_data.get('pmi', 52)
            f['consumer_confidence'] = macro_data.get('consumer_confidence', 100)
        
        return f
    
    @staticmethod
    def feature_names() -> list:
        return [
            'fed_funds_rate', 'treasury_10y', 'treasury_2y', 'yield_curve', 'rate_change_3m',
            'vix', 'vix_term_structure', 'sp500_return_20d', 'market_breadth', 'sector_rotation',
            'gdp_growth', 'cpi_yoy', 'unemployment', 'pmi', 'consumer_confidence'
        ]
