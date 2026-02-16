"""WebSocket endpoint tests."""
from fastapi.testclient import TestClient
from backend.api.server import app

client = TestClient(app)

class TestWebSocket:
    def test_websocket_connect(self):
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"message": "help"})
            data = ws.receive_json()
            assert data["type"] == "response"
            assert "help" in data["content"].lower() or "analyze" in data["content"].lower()
    
    def test_websocket_analyze(self):
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"message": "analyze AAPL"})
            # First message is status
            data = ws.receive_json()
            assert data["type"] == "status"
            # Second is prediction
            data = ws.receive_json()
            assert data["type"] == "prediction"
            assert data["ticker"] == "AAPL"
            assert "predictions" in data
    
    def test_websocket_no_ticker(self):
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"message": "analyze"})
            data = ws.receive_json()
            assert data["type"] == "response"
            assert "ticker" in data["content"].lower()
