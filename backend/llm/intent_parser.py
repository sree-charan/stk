"""Intent parser for extracting user intent from messages."""
import re
from typing import Optional
from dataclasses import dataclass

@dataclass
class Intent:
    action: str  # 'predict', 'analyze', 'compare', 'help', 'unknown'
    ticker: Optional[str] = None
    horizon: str = 'short'  # 'short', 'medium', 'long', 'all'
    raw_query: str = ''

class IntentParser:
    """Parse user messages to extract intent and parameters."""
    
    TICKER_PATTERN = r'\b([A-Z]{1,5})\b'
    
    HORIZON_KEYWORDS = {
        'short': ['1h', '1 hour', 'hour', 'short', 'intraday', 'today', 'now'],
        'medium': ['5d', '5 day', 'week', 'medium', 'weekly'],
        'long': ['60d', '60 day', 'month', 'long', 'monthly', 'quarter']
    }
    
    ACTION_KEYWORDS = {
        'predict': ['predict', 'forecast', 'will', 'going', 'price', 'target'],
        'analyze': ['analyze', 'analysis', 'look', 'check', 'what about', 'how is'],
        'compare': ['compare', 'vs', 'versus', 'better', 'which'],
        'help': ['help', 'how', 'what can', 'commands']
    }
    
    COMMON_TICKERS = {'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'AMD', 'NFLX', 'SPY', 'QQQ'}
    
    def parse(self, message: str) -> Intent:
        """Parse user message and return Intent."""
        message_lower = message.lower()
        
        # Extract action
        action = 'analyze'  # default
        for act, keywords in self.ACTION_KEYWORDS.items():
            if any(kw in message_lower for kw in keywords):
                action = act
                break
        
        # Extract ticker
        ticker = self._extract_ticker(message)
        
        # Extract horizon
        horizon = 'all'  # default to all horizons
        for h, keywords in self.HORIZON_KEYWORDS.items():
            if any(kw in message_lower for kw in keywords):
                horizon = h
                break
        
        return Intent(action=action, ticker=ticker, horizon=horizon, raw_query=message)
    
    def _extract_ticker(self, message: str) -> Optional[str]:
        """Extract stock ticker from message."""
        # Find all uppercase words that could be tickers
        matches = re.findall(self.TICKER_PATTERN, message)
        
        # Filter to known tickers or valid-looking ones
        for match in matches:
            if match in self.COMMON_TICKERS:
                return match
            # Accept 2-5 char uppercase as potential ticker
            if 2 <= len(match) <= 5 and match.isalpha():
                return match
        
        return None
    
    def get_help_text(self) -> str:
        """Return help text for users."""
        return """
**Stock Chat Help**

I can analyze stocks and provide predictions. Try:
- "Analyze TSLA" - Full analysis with all timeframes
- "What's the 1-hour outlook for AAPL?" - Short-term prediction
- "NVDA 5-day forecast" - Medium-term prediction
- "Long-term view on MSFT" - 60-day prediction

**Supported tickers**: Any US stock symbol (e.g., AAPL, TSLA, GOOGL)
**Timeframes**: 1-hour, 5-day, 60-day
"""
