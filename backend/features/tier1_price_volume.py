"""Tier 1: Price & Volume Features (23 features)."""
import numpy as np
import pandas as pd

class Tier1PriceVolume:
    """Generate 23 price and volume features."""
    
    @staticmethod
    def compute(df: pd.DataFrame) -> pd.DataFrame:
        """Compute all Tier 1 features from OHLCV data."""
        f = pd.DataFrame(index=df.index)
        
        # OHLCV features (6)
        f['return_1d'] = df['close'].pct_change()
        f['return_5d'] = df['close'].pct_change(5)
        f['return_20d'] = df['close'].pct_change(20)
        f['high_low_range'] = (df['high'] - df['low']) / df['close']
        f['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
        f['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        
        # VWAP features (7)
        typical = (df['high'] + df['low'] + df['close']) / 3
        f['vwap'] = (typical * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
        f['price_to_vwap'] = df['close'] / f['vwap']
        f['vwap_slope'] = f['vwap'].pct_change(5)
        f['vwap_distance'] = (df['close'] - f['vwap']) / f['vwap']
        vwap_std = ((typical - f['vwap']).pow(2) * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
        vwap_band = np.sqrt(vwap_std.clip(lower=0))
        f['vwap_upper'] = (f['vwap'] + 2 * vwap_band - df['close']) / (df['close'] + 1e-8)
        f['vwap_lower'] = (df['close'] - (f['vwap'] - 2 * vwap_band)) / (df['close'] + 1e-8)
        f['vwap_band_width'] = (4 * vwap_band) / (f['vwap'] + 1e-8)
        
        # Volume features (6)
        f['volume_sma20'] = df['volume'].rolling(20).mean()
        f['volume_ratio'] = df['volume'] / f['volume_sma20']
        f['volume_change'] = df['volume'].pct_change()
        f['volume_std'] = df['volume'].rolling(20).std() / f['volume_sma20']
        f['up_volume_ratio'] = df['volume'].where(df['close'] > df['open'], 0).rolling(20).sum() / df['volume'].rolling(20).sum()
        f['dollar_volume'] = df['close'] * df['volume']
        
        # Market structure (4)
        f['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
        f['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
        f['inside_bar'] = ((df['high'] < df['high'].shift(1)) & (df['low'] > df['low'].shift(1))).astype(int)
        f['outside_bar'] = ((df['high'] > df['high'].shift(1)) & (df['low'] < df['low'].shift(1))).astype(int)
        
        return f.fillna(0)
    
    @staticmethod
    def feature_names() -> list:
        return [
            'return_1d', 'return_5d', 'return_20d', 'high_low_range', 'close_position', 'gap',
            'vwap', 'price_to_vwap', 'vwap_slope', 'vwap_distance', 'vwap_upper', 'vwap_lower', 'vwap_band_width',
            'volume_sma20', 'volume_ratio', 'volume_change', 'volume_std', 'up_volume_ratio', 'dollar_volume',
            'higher_high', 'lower_low', 'inside_bar', 'outside_bar'
        ]
