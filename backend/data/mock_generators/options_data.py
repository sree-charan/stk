"""Mock options data generator - full options chains."""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
from scipy.stats import norm

def black_scholes_iv_to_price(S: float, K: float, T: float, r: float, iv: float, option_type: str) -> float:
    """Calculate option price from IV using Black-Scholes."""
    if T <= 0:
        return max(0, S - K) if option_type == 'call' else max(0, K - S)
    
    d1 = (np.log(S/K) + (r + iv**2/2)*T) / (iv*np.sqrt(T))
    d2 = d1 - iv*np.sqrt(T)
    
    if option_type == 'call':
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)


def generate_iv_surface(spot: float, strikes: np.ndarray, dte: int, base_iv: float = 0.35) -> np.ndarray:
    """Generate IV with smile/skew."""
    moneyness = strikes / spot
    
    # IV skew - higher IV for OTM puts
    skew = 0.1 * (1 - moneyness)
    
    # IV smile - higher IV for far OTM options
    smile = 0.05 * (moneyness - 1)**2
    
    # Term structure - lower IV for longer dated
    term_adj = 1 - 0.002 * min(dte, 90)
    
    ivs = (base_iv + skew + smile) * term_adj
    return np.clip(ivs, 0.1, 1.5)


def generate_options_chain(
    symbol: str,
    spot_price: float,
    as_of_date: Optional[datetime] = None,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """Generate full options chain for a symbol."""
    if seed:
        np.random.seed(seed)
    if not as_of_date:
        as_of_date = datetime.now()
    
    # Strike range: ±30% from spot
    strike_min = spot_price * 0.7
    strike_max = spot_price * 1.3
    strike_step = max(1, round(spot_price * 0.025))  # ~2.5% steps
    strikes = np.arange(strike_min, strike_max, strike_step)
    
    # Expirations: weekly for 4 weeks, monthly for 6 months
    expirations = []
    for i in range(1, 5):  # Weekly
        exp = as_of_date + timedelta(days=i*7)
        expirations.append(exp)
    for i in range(1, 7):  # Monthly
        exp = as_of_date + timedelta(days=30*i)
        expirations.append(exp)
    
    base_iv = 0.25 + np.random.uniform(-0.1, 0.2)  # Symbol-specific base IV
    r = 0.05  # Risk-free rate
    
    options = []
    for exp in expirations:
        dte = (exp - as_of_date).days
        if dte <= 0:
            continue
        T = dte / 365
        
        ivs = generate_iv_surface(spot_price, strikes, dte, base_iv)
        
        for strike, iv in zip(strikes, ivs):
            for opt_type in ['call', 'put']:
                price = black_scholes_iv_to_price(spot_price, strike, T, r, iv, opt_type)
                
                # Volume and OI - higher near ATM
                atm_factor = np.exp(-((strike/spot_price - 1)**2) * 20)
                volume = int(np.random.exponential(5000) * atm_factor)
                oi = int(np.random.exponential(20000) * atm_factor)
                
                # Greeks (simplified)
                d1 = (np.log(spot_price/strike) + (r + iv**2/2)*T) / (iv*np.sqrt(T)) if T > 0 else 0
                delta = norm.cdf(d1) if opt_type == 'call' else norm.cdf(d1) - 1
                gamma = norm.pdf(d1) / (spot_price * iv * np.sqrt(T)) if T > 0 else 0
                theta = -spot_price * norm.pdf(d1) * iv / (2*np.sqrt(T)) / 365 if T > 0 else 0
                vega = spot_price * norm.pdf(d1) * np.sqrt(T) / 100 if T > 0 else 0
                
                options.append({
                    'symbol': symbol,
                    'expiration': exp.date(),
                    'strike': round(strike, 2),
                    'option_type': opt_type,
                    'bid': round(max(0.01, price * 0.98), 2),
                    'ask': round(price * 1.02, 2),
                    'last': round(price, 2),
                    'volume': volume,
                    'open_interest': oi,
                    'iv': round(iv, 4),
                    'delta': round(delta, 4),
                    'gamma': round(gamma, 6),
                    'theta': round(theta, 4),
                    'vega': round(vega, 4),
                    'dte': dte
                })
    
    return pd.DataFrame(options)


def get_options_chain(symbol: str, spot_price: Optional[float] = None) -> pd.DataFrame:
    """Main interface - get options chain for a symbol."""
    seed = sum(ord(c) for c in symbol) * 42
    
    if not spot_price:
        base_prices = {"TSLA": 250, "AAPL": 180, "NVDA": 450, "MSFT": 380, "GOOGL": 140}
        spot_price = base_prices.get(symbol, 100 + (seed % 400))
    
    return generate_options_chain(symbol, spot_price, seed=seed)


def get_put_call_ratio(chain: pd.DataFrame) -> dict:
    """Calculate put/call ratios from chain."""
    calls = chain[chain['option_type'] == 'call']
    puts = chain[chain['option_type'] == 'put']
    
    return {
        'volume_ratio': puts['volume'].sum() / max(1, calls['volume'].sum()),
        'oi_ratio': puts['open_interest'].sum() / max(1, calls['open_interest'].sum()),
        'total_call_volume': calls['volume'].sum(),
        'total_put_volume': puts['volume'].sum()
    }


if __name__ == "__main__":
    chain = get_options_chain("TSLA", 250)
    print(f"Generated {len(chain)} options contracts")
    print(chain.head(10))
    print("\nPut/Call ratios:", get_put_call_ratio(chain))
