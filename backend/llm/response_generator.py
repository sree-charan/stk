"""Response generator combining predictions with natural language."""
from typing import Dict, Optional
from .mock_llm import MockLLM
from .intent_parser import Intent

class ResponseGenerator:
    """Generate formatted responses for the chat interface."""
    
    def __init__(self):
        self.llm = MockLLM()
    
    def generate(self, intent: Intent, predictions: Optional[Dict] = None, error: Optional[str] = None) -> str:
        """Generate response based on intent and predictions."""
        if error:
            return f"❌ {error}"
        
        if intent.action == 'help':
            return self._help_response()
        
        if not intent.ticker:
            return "Please specify a stock ticker (e.g., 'Analyze TSLA' or 'AAPL forecast')"
        
        if not predictions:
            return f"Unable to generate predictions for {intent.ticker}"
        
        if intent.horizon == 'all':
            return self.llm.generate_summary(intent.ticker, predictions)
        else:
            pred = predictions.get(intent.horizon, predictions.get('short', {}))
            return self.llm.generate(intent.ticker, pred, intent.horizon)
    
    def _help_response(self) -> str:
        return """
🤖 **Stock Chat**

I analyze stocks using ML models and provide predictions.

**Commands:**
• `Analyze [TICKER]` - Full analysis (all timeframes)
• `[TICKER] 1h` - 1-hour prediction
• `[TICKER] 5d` - 5-day prediction  
• `[TICKER] 60d` - 60-day prediction
• `help` - Show this message

**Examples:**
• "Analyze TSLA"
• "AAPL short term outlook"
• "What's NVDA doing?"

**Note:** Predictions use real market data from Yahoo Finance, FRED, and NewsAPI.
"""
    
    def format_prediction_card(self, ticker: str, horizon: str, prediction: Dict) -> Dict:
        """Format prediction as structured card data."""
        return {
            'ticker': ticker,
            'horizon': horizon,
            'direction': prediction.get('direction', 'neutral'),
            'prediction': prediction.get('prediction', 0),
            'confidence': prediction.get('confidence', 0.5),
            'breakdown': prediction.get('breakdown', {}),
            'timestamp': None  # Will be set by API
        }
