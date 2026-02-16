"""Shared test fixtures for stock-chat-assistant tests."""
import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def price_df():
    """Generate a default price DataFrame with 500 business days."""
    np.random.seed(42)
    days = 500
    dates = pd.date_range("2023-01-01", periods=days, freq="B")
    close = 100 + np.cumsum(np.random.randn(days) * 0.5)
    close = np.maximum(close, 10)
    return pd.DataFrame({
        "date": dates, "symbol": "TEST",
        "open": close - 0.5, "high": close + 1, "low": close - 1,
        "close": close, "volume": np.random.randint(1_000_000, 50_000_000, days),
    })


@pytest.fixture
def macro_data():
    """Generate standard macro data dict with VIX, rates, and indicators."""
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    return {
        "interest_rates": pd.DataFrame({
            "date": dates.date, "fed_funds_rate": 5.25, "treasury_2y": 4.5,
            "treasury_10y": 4.2, "treasury_30y": 4.4, "yield_curve_2_10": -0.3,
        }),
        "economic_indicators": pd.DataFrame({
            "date": dates.date[:24], "gdp_growth_yoy": 2.5, "cpi_yoy": 3.2,
            "core_cpi_yoy": 2.8, "unemployment_rate": 3.7,
            "consumer_confidence": 100, "pmi_manufacturing": 50, "pmi_services": 52,
        }),
        "vix": pd.DataFrame({
            "date": dates.date, "vix": 18.0, "vix_9d": 17.0,
            "vix_3m": 19.0, "vix_term_structure": 0.0,
        }),
    }


@pytest.fixture
def hourly_df():
    """Generate hourly price DataFrame with 200 bars."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    close = 100 + np.cumsum(np.random.randn(n) * 0.1)
    return pd.DataFrame({
        "date": dates, "symbol": "TEST",
        "open": close - 0.1, "high": close + 0.5, "low": close - 0.5,
        "close": close, "volume": np.random.randint(10_000, 1_000_000, n),
    })
