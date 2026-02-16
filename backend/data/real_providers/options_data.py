"""Real options data via yfinance."""
import pandas as pd
import yfinance as yf
from typing import Optional
from . import cache
from backend.utils.retry import retry

_TTL = 600  # 10 min


@retry(max_attempts=2, base_delay=1.0)
def _fetch_options(symbol: str):
    ticker = yf.Ticker(symbol)
    exps = ticker.options
    if not exps:
        raise ValueError("No options data")
    rows = []
    for exp in exps[:6]:
        chain = ticker.option_chain(exp)
        for opt_type, df in [("call", chain.calls), ("put", chain.puts)]:
            for _, r in df.iterrows():
                rows.append({
                    "symbol": symbol, "expiration": exp,
                    "strike": r.get("strike", 0), "option_type": opt_type,
                    "bid": r.get("bid", 0), "ask": r.get("ask", 0),
                    "last": r.get("lastPrice", 0), "volume": int(r.get("volume", 0) or 0),
                    "open_interest": int(r.get("openInterest", 0) or 0),
                    "iv": r.get("impliedVolatility", 0.3),
                    "delta": 0, "gamma": 0, "theta": 0, "vega": 0,
                    "dte": max(1, (pd.Timestamp(exp) - pd.Timestamp.now()).days),
                })
    if not rows:
        raise ValueError("No options data")
    return pd.DataFrame(rows)


def get_options_chain(symbol: str, spot_price: Optional[float] = None) -> pd.DataFrame:
    key = f"options_{symbol}"
    cached = cache.get(key, _TTL)
    if cached is not None:
        return cached

    try:
        result = _fetch_options(symbol)
        cache.put(key, result)
        return result
    except Exception as e:
        stale, age = cache.get_stale(key)
        if stale is not None:
            import logging
            logging.warning(f"Using stale cache for {symbol} options (age: {age/60:.0f}m): {e}")
            return stale
        return _mock_fallback(symbol, spot_price)


def _mock_fallback(symbol, spot_price):
    from backend.data.mock_generators.options_data import get_options_chain as mock
    return mock(symbol, spot_price)


def get_put_call_ratio(chain: pd.DataFrame) -> dict:
    calls = chain[chain["option_type"] == "call"]
    puts = chain[chain["option_type"] == "put"]
    return {
        "volume_ratio": puts["volume"].sum() / max(1, calls["volume"].sum()),
        "oi_ratio": puts["open_interest"].sum() / max(1, calls["open_interest"].sum()),
        "total_call_volume": int(calls["volume"].sum()),
        "total_put_volume": int(puts["volume"].sum()),
    }
