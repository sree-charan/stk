"""Mock price data generator - 2 years of OHLCV data."""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

def generate_ohlcv(
    symbol: str = "TSLA",
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    base_price: float = 250.0,
    daily_volatility: float = 0.02,
    avg_volume: int = 50_000_000,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """Generate realistic OHLCV data with trends, gaps, and reversals."""
    if seed:
        np.random.seed(seed)
    
    if not end_date:
        end_date = datetime.now()
    if not start_date:
        start_date = end_date - timedelta(days=730)  # 2 years
    
    # Generate daily data first
    trading_days = pd.bdate_range(start_date, end_date)
    n_days = len(trading_days)
    
    # Price simulation with regime changes
    prices = [base_price]
    regime = 0  # 0=neutral, 1=bull, -1=bear
    regime_duration = 0
    
    for i in range(1, n_days):
        # Regime changes every 20-60 days
        regime_duration += 1
        if regime_duration > np.random.randint(20, 60):
            regime = np.random.choice([-1, 0, 1])
            regime_duration = 0
        
        drift = regime * 0.001  # Trend component
        shock = np.random.normal(drift, daily_volatility)
        
        # Occasional gaps (earnings, news)
        if np.random.random() < 0.02:
            shock += np.random.choice([-1, 1]) * np.random.uniform(0.03, 0.08)
        
        new_price = prices[-1] * (1 + shock)
        prices.append(max(new_price, 1.0))  # Floor at $1
    
    # Generate OHLCV for each day
    data = []
    for i, (date, close) in enumerate(zip(trading_days, prices)):
        intraday_vol = daily_volatility * 0.6
        high = close * (1 + abs(np.random.normal(0, intraday_vol)))
        low = close * (1 - abs(np.random.normal(0, intraday_vol)))
        
        # Open near previous close with possible gap
        if i > 0:
            gap = np.random.normal(0, 0.005)
            open_price = prices[i-1] * (1 + gap)
        else:
            open_price = close * (1 + np.random.normal(0, intraday_vol))
        
        # Ensure OHLC consistency
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        # Volume with some randomness
        vol_mult = 1 + np.random.normal(0, 0.3)
        volume = int(avg_volume * max(0.3, vol_mult))
        
        # Higher volume on big moves
        if abs(close/prices[max(0,i-1)] - 1) > 0.03:
            volume = int(volume * 1.5)
        
        data.append({
            'date': date,
            'symbol': symbol,
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close, 2),
            'volume': volume
        })
    
    return pd.DataFrame(data)


def generate_minute_data(daily_df: pd.DataFrame, minutes_per_day: int = 390) -> pd.DataFrame:
    """Expand daily data to minute-level (6.5 hours trading)."""
    minute_data = []
    
    for _, row in daily_df.iterrows():
        date = row['date']
        daily_open = row['open']
        daily_high = row['high']
        daily_low = row['low']
        daily_close = row['close']
        daily_volume = row['volume']
        
        # Generate minute prices using random walk
        prices = [daily_open]
        for _ in range(minutes_per_day - 1):
            change = np.random.normal(0, 0.0005)
            prices.append(prices[-1] * (1 + change))
        
        # Scale to match daily OHLC
        prices = np.array(prices)
        prices = (prices - prices.min()) / (prices.max() - prices.min() + 1e-10)
        prices = prices * (daily_high - daily_low) + daily_low
        prices[-1] = daily_close  # Ensure close matches
        
        # Volume distribution (U-shaped)
        vol_weights = np.concatenate([
            np.linspace(1.5, 0.5, minutes_per_day // 3),
            np.ones(minutes_per_day // 3) * 0.5,
            np.linspace(0.5, 1.5, minutes_per_day - 2 * (minutes_per_day // 3))
        ])
        vol_weights = vol_weights / vol_weights.sum()
        volumes = (vol_weights * daily_volume).astype(int)
        
        market_open = datetime.combine(date.date(), datetime.strptime("09:30", "%H:%M").time())
        
        for i in range(minutes_per_day):
            timestamp = market_open + timedelta(minutes=i)
            minute_data.append({
                'timestamp': timestamp,
                'symbol': row['symbol'],
                'open': round(prices[max(0, i-1)] if i > 0 else prices[i], 2),
                'high': round(max(prices[max(0,i-1):i+1]) if i > 0 else prices[i], 2),
                'low': round(min(prices[max(0,i-1):i+1]) if i > 0 else prices[i], 2),
                'close': round(prices[i], 2),
                'volume': max(100, volumes[i])
            })
    
    return pd.DataFrame(minute_data)


def get_ohlcv(symbol: str, timeframe: str = "daily", days: int = 730) -> pd.DataFrame:
    """Main interface - get OHLCV data for a symbol."""
    # Use symbol hash for consistent random seed
    seed = sum(ord(c) for c in symbol) * 42
    
    # Different base prices per symbol
    base_prices = {"TSLA": 250, "AAPL": 180, "NVDA": 450, "MSFT": 380, "GOOGL": 140}
    base_price = base_prices.get(symbol, 100 + (seed % 400))
    
    daily_df = generate_ohlcv(symbol, base_price=base_price, seed=seed)
    
    if timeframe == "minute":
        return generate_minute_data(daily_df.tail(days))
    return daily_df.tail(days)


if __name__ == "__main__":
    # Test generation
    df = get_ohlcv("TSLA", "daily", 30)
    print(f"Generated {len(df)} daily bars")
    print(df.head())
