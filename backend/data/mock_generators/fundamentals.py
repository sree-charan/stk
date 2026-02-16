"""Mock fundamentals data generator - earnings, margins, valuation."""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

def generate_quarterly_fundamentals(
    symbol: str,
    quarters: int = 8,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """Generate quarterly fundamental data."""
    if seed:
        np.random.seed(seed)
    
    # Base metrics vary by symbol
    base_revenue = 10e9 + np.random.uniform(0, 20e9)
    base_eps = 1.0 + np.random.uniform(0, 2)
    growth_rate = np.random.uniform(-0.05, 0.15)
    
    data = []
    end_date = datetime.now()
    
    for q in range(quarters):
        quarter_date = end_date - timedelta(days=90 * q)
        quarter_num = ((quarter_date.month - 1) // 3) + 1
        
        # Growth with some noise
        growth_mult = (1 + growth_rate) ** (quarters - q - 1)
        noise = np.random.normal(1, 0.05)
        
        revenue = base_revenue * growth_mult * noise
        eps = base_eps * growth_mult * noise
        
        # Margins with slight variation
        gross_margin = np.random.uniform(0.20, 0.45)
        operating_margin = gross_margin * np.random.uniform(0.3, 0.6)
        net_margin = operating_margin * np.random.uniform(0.6, 0.9)
        
        # Estimates and surprises
        eps_estimate = eps * np.random.uniform(0.9, 1.1)
        eps_surprise = (eps - eps_estimate) / abs(eps_estimate) * 100
        
        revenue_estimate = revenue * np.random.uniform(0.95, 1.05)
        revenue_surprise = (revenue - revenue_estimate) / revenue_estimate * 100
        
        data.append({
            'symbol': symbol,
            'quarter': f"Q{quarter_num} {quarter_date.year}",
            'date': quarter_date.date(),
            'revenue': round(revenue, 0),
            'revenue_estimate': round(revenue_estimate, 0),
            'revenue_surprise_pct': round(revenue_surprise, 2),
            'eps': round(eps, 2),
            'eps_estimate': round(eps_estimate, 2),
            'eps_surprise_pct': round(eps_surprise, 2),
            'gross_margin': round(gross_margin, 4),
            'operating_margin': round(operating_margin, 4),
            'net_margin': round(net_margin, 4),
            'revenue_growth_yoy': round(growth_rate + np.random.normal(0, 0.03), 4),
            'eps_growth_yoy': round(growth_rate + np.random.normal(0, 0.05), 4)
        })
    
    return pd.DataFrame(data)


def generate_valuation_metrics(
    symbol: str,
    current_price: float,
    fundamentals: pd.DataFrame
) -> dict:
    """Generate valuation metrics."""
    latest = fundamentals.iloc[0]
    
    # Annualized metrics
    annual_eps = latest['eps'] * 4
    annual_revenue = latest['revenue'] * 4
    
    # Shares outstanding (estimated from market cap range)
    shares = np.random.uniform(500e6, 3e9)
    market_cap = current_price * shares
    
    pe_ratio = current_price / annual_eps if annual_eps > 0 else 0
    ps_ratio = market_cap / annual_revenue if annual_revenue > 0 else 0
    
    # Book value and other metrics
    book_value_per_share = current_price / np.random.uniform(3, 15)
    pb_ratio = current_price / book_value_per_share
    
    return {
        'symbol': symbol,
        'market_cap': round(market_cap, 0),
        'pe_ratio': round(pe_ratio, 2),
        'forward_pe': round(pe_ratio * np.random.uniform(0.8, 1.1), 2),
        'ps_ratio': round(ps_ratio, 2),
        'pb_ratio': round(pb_ratio, 2),
        'ev_ebitda': round(np.random.uniform(8, 25), 2),
        'peg_ratio': round(pe_ratio / max(1, latest['eps_growth_yoy'] * 100), 2),
        'dividend_yield': round(np.random.uniform(0, 0.03), 4),
        'shares_outstanding': int(shares)
    }


def get_fundamentals(symbol: str, quarters: int = 8) -> dict:
    """Main interface - get fundamental data for a symbol."""
    seed = sum(ord(c) for c in symbol) * 42
    
    base_prices = {"TSLA": 250, "AAPL": 180, "NVDA": 450, "MSFT": 380, "GOOGL": 140}
    price = base_prices.get(symbol, 100 + (seed % 400))
    
    quarterly = generate_quarterly_fundamentals(symbol, quarters, seed)
    valuation = generate_valuation_metrics(symbol, price, quarterly)
    
    return {
        'quarterly': quarterly,
        'valuation': valuation,
        'latest_quarter': quarterly.iloc[0].to_dict()
    }


if __name__ == "__main__":
    data = get_fundamentals("TSLA")
    print("Quarterly data:")
    print(data['quarterly'].head())
    print("\nValuation:", data['valuation'])
