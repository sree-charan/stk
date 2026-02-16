"""Test LLM components."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from backend.llm import IntentParser, ResponseGenerator, MockLLM


class TestIntentParser:
    def setup_method(self):
        self.parser = IntentParser()
    
    def test_parse_analyze(self):
        intent = self.parser.parse("Analyze TSLA")
        assert intent.action == "analyze"
        assert intent.ticker == "TSLA"
    
    def test_parse_predict(self):
        intent = self.parser.parse("Predict AAPL price")
        assert intent.action == "predict"
        assert intent.ticker == "AAPL"
    
    def test_parse_horizon_short(self):
        intent = self.parser.parse("TSLA 1h outlook")
        assert intent.horizon == "short"
    
    def test_parse_horizon_medium(self):
        intent = self.parser.parse("MSFT 5 day forecast")
        assert intent.horizon == "medium"
    
    def test_parse_horizon_long(self):
        intent = self.parser.parse("NVDA monthly view")
        assert intent.horizon == "long"
    
    def test_parse_help(self):
        intent = self.parser.parse("help me")
        assert intent.action == "help"
    
    def test_no_ticker(self):
        intent = self.parser.parse("what's happening")
        assert intent.ticker is None


class TestMockLLM:
    def setup_method(self):
        self.llm = MockLLM()
    
    def test_generate_bullish(self):
        pred = {"direction": "bullish", "confidence": 0.8, "prediction": 0.05}
        response = self.llm.generate("TSLA", pred, "short")
        assert "TSLA" in response
    
    def test_generate_bearish(self):
        pred = {"direction": "bearish", "confidence": 0.7, "prediction": -0.03}
        response = self.llm.generate("AAPL", pred, "medium")
        assert "AAPL" in response
    
    def test_generate_summary(self):
        preds = {
            "short": {"direction": "bullish", "confidence": 0.8, "prediction": 0.02},
            "medium": {"direction": "neutral", "confidence": 0.5, "prediction": 0.001},
            "long": {"direction": "bearish", "confidence": 0.6, "prediction": -0.05}
        }
        summary = self.llm.generate_summary("NVDA", preds)
        assert "NVDA" in summary
        assert "1-Hour" in summary


class TestResponseGenerator:
    def setup_method(self):
        self.gen = ResponseGenerator()
    
    def test_help_response(self):
        from backend.llm.intent_parser import Intent
        intent = Intent(action="help")
        response = self.gen.generate(intent)
        assert "Stock Chat" in response
    
    def test_no_ticker_response(self):
        from backend.llm.intent_parser import Intent
        intent = Intent(action="analyze", ticker=None)
        response = self.gen.generate(intent)
        assert "ticker" in response.lower()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
