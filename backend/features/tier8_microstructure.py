"""Tier 8: Basic Microstructure Features (6 features)."""
import numpy as np
import pandas as pd

class Tier8Microstructure:
    """Generate 6 basic microstructure features."""
    
    @staticmethod
    def compute(df: pd.DataFrame) -> pd.DataFrame:
        """Compute microstructure features from OHLCV data."""
        f = pd.DataFrame(index=df.index)
        
        # Basic microstructure (6)
        f['bid_ask_spread'] = (df['high'] - df['low']) / df['close'] * 0.1  # Proxy
        f['trade_imbalance'] = np.sign(df['close'] - df['open']) * df['volume'] / df['volume'].rolling(20).mean()
        f['price_impact'] = abs(df['close'].pct_change()) / (df['volume'] / df['volume'].mean() + 0.01)
        f['order_flow_toxicity'] = abs(df['close'] - (df['high'] + df['low']) / 2) / (df['high'] - df['low'] + 0.01)
        f['volume_clock'] = df['volume'].cumsum() / df['volume'].sum()
        f['realized_spread'] = (df['close'] - df['close'].shift(1)).abs() / df['close'].shift(1)
        
        return f.fillna(0)
    
    @staticmethod
    def feature_names() -> list:
        return ['bid_ask_spread', 'trade_imbalance', 'price_impact', 'order_flow_toxicity', 'volume_clock', 'realized_spread']
