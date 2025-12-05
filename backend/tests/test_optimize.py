from fastapi.testclient import TestClient
from app.main import app
import pytest

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    # It now serves index.html, so we just check status


def test_optimize_endpoint():
    payload = {
        "tickers": ["META", "AVGO", "GLD"],
        "from": "2021-01-01",
        "to": "2023-01-01",
        "risk_free_rate": 0.02,
        "allow_short": False
    }
    response = client.post("/api/optimize", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    # Check structure
    assert "weights" in data
    assert "expected_return" in data
    assert "portfolio_std" in data
    assert "sharpe" in data
    assert "frontier" in data
    
    # Check weights sum to approx 1
    weights = data["weights"]
    total_weight = sum(weights.values())
    assert abs(total_weight - 1.0) < 0.01
    
    # Check frontier has points
    assert len(data["frontier"]) > 0
