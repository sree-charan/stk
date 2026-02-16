"""Test API endpoints."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from fastapi.testclient import TestClient
import warnings
warnings.filterwarnings('ignore')

from backend.api.server import app

client = TestClient(app)


class TestHealthEndpoint:
    def test_health(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert "status" in response.json()


class TestChatEndpoint:
    def test_chat_with_ticker(self):
        response = client.post("/chat", json={"message": "Analyze TSLA"})
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert data.get("ticker") == "TSLA"
    
    def test_chat_help(self):
        response = client.post("/chat", json={"message": "help"})
        assert response.status_code == 200
        assert "Stock Chat" in response.json()["response"]
    
    def test_chat_no_ticker(self):
        response = client.post("/chat", json={"message": "what's happening"})
        assert response.status_code == 200
        assert "ticker" in response.json()["response"].lower()


class TestPredictEndpoint:
    def test_predict_all(self):
        response = client.post("/predict", json={"ticker": "AAPL"})
        assert response.status_code == 200
        data = response.json()
        assert data["ticker"] == "AAPL"
        assert "predictions" in data
    
    def test_predict_single_horizon(self):
        response = client.post("/predict", json={"ticker": "MSFT", "horizon": "short"})
        assert response.status_code == 200
        data = response.json()
        assert data["horizon"] == "short"


class TestBacktestEndpoint:
    def test_backtest_returns_results(self):
        response = client.get("/backtest/AAPL")
        assert response.status_code == 200
        data = response.json()
        assert "ticker" in data
        assert "backtest" in data
        assert "accuracy" in data["backtest"]
        assert "sharpe_ratio" in data["backtest"]

class TestTickersEndpoint:
    def test_get_tickers(self):
        response = client.get("/tickers")
        assert response.status_code == 200
        assert "tickers" in response.json()
        assert len(response.json()["tickers"]) > 0


class TestAPIErrorClassification:
    """Test that API returns proper HTTP status codes for different error types."""

    def test_classify_not_found(self):
        from backend.api.server import _classify_http_error
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            _classify_http_error(ValueError("No price data found for 'ZZZZ'"), "ZZZZ")
        assert exc_info.value.status_code == 404

    def test_classify_rate_limit(self):
        from backend.api.server import _classify_http_error
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            _classify_http_error(Exception("429 Too Many Requests"), "TSLA")
        assert exc_info.value.status_code == 429

    def test_classify_network_error(self):
        from backend.api.server import _classify_http_error
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            _classify_http_error(ConnectionError("Connection timeout"), "TSLA")
        assert exc_info.value.status_code == 503

    def test_classify_generic_error(self):
        from backend.api.server import _classify_http_error
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            _classify_http_error(RuntimeError("something unexpected"), "TSLA")
        assert exc_info.value.status_code == 500

    def test_predict_invalid_ticker_returns_404(self):
        from unittest.mock import patch
        with patch('backend.api.server.get_predictions', side_effect=ValueError("No price data found for 'ZZZZ'")):
            response = client.get("/predict/ZZZZ")
            assert response.status_code == 404

    def test_predict_rate_limit_returns_429(self):
        from unittest.mock import patch
        with patch('backend.api.server.get_predictions', side_effect=Exception("429 rate limit exceeded")):
            response = client.get("/predict/TSLA")
            assert response.status_code == 429

    def test_predict_network_error_returns_503(self):
        from unittest.mock import patch
        with patch('backend.api.server.get_predictions', side_effect=ConnectionError("Connection timeout")):
            response = client.get("/predict/TSLA")
            assert response.status_code == 503


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
