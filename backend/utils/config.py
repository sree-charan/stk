"""Configuration settings."""
from dataclasses import dataclass
from typing import List

@dataclass
class Config:
    """System configuration."""
    # Model settings
    SHORT_HORIZON_HOURS: int = 1
    MEDIUM_HORIZON_DAYS: int = 5
    LONG_HORIZON_DAYS: int = 60
    
    # Prediction thresholds
    BULLISH_THRESHOLD: float = 0.6
    BEARISH_THRESHOLD: float = 0.4
    CONFIDENCE_MIN: float = 0.55
    
    # Invalidation rules
    MAX_PREDICTION_AGE_HOURS: int = 4
    VOLATILITY_SPIKE_THRESHOLD: float = 2.0
    NEWS_INVALIDATION_KEYWORDS: List[str] = None
    
    # Feature settings
    FEATURE_LOOKBACK_DAYS: int = 100
    MIN_DATA_POINTS: int = 50
    
    # Supported tickers
    TICKERS: List[str] = None
    
    def __post_init__(self):
        if self.NEWS_INVALIDATION_KEYWORDS is None:
            self.NEWS_INVALIDATION_KEYWORDS = ['earnings', 'merger', 'acquisition', 'fda', 'lawsuit', 'bankruptcy']
        if self.TICKERS is None:
            self.TICKERS = ['TSLA', 'AAPL', 'NVDA', 'MSFT', 'GOOGL', 'AMZN', 'META', 'AMD', 'NFLX', 'SPY']

config = Config()
