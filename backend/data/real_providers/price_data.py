"""Real price data via yfinance."""
import logging
import pandas as pd
import yfinance as yf
from . import cache
from backend.utils.retry import retry

_DAILY_TTL = 300  # 5 min
logger = logging.getLogger(__name__)


@retry(max_attempts=3, base_delay=1.0)
def _fetch_history(symbol: str, period: str, interval: str) -> pd.DataFrame:
    import io
    import contextlib
    ticker = yf.Ticker(symbol)
    # Suppress yfinance's noisy error output
    f = io.StringIO()
    with contextlib.redirect_stderr(f), contextlib.redirect_stdout(f):
        hist = ticker.history(period=period, interval=interval, prepost=True)
    if hist.empty:
        raise ValueError(f"No price data found for '{symbol}'. Check the ticker symbol.")
    return hist


def get_ohlcv(symbol: str, timeframe: str = "daily", days: int = 730) -> pd.DataFrame:
    """Fetch real OHLCV data from Yahoo Finance."""
    key = f"price_{symbol}_{timeframe}_{days}"
    cached = cache.get(key, _DAILY_TTL)
    if cached is not None:
        return cached

    try:
        period = "2y" if days >= 365 else f"{days}d"
        interval = "1d" if timeframe == "daily" else "1m"
        hist = _fetch_history(symbol, period, interval)

        df = hist.reset_index()
        col_map = {"Date": "date", "Datetime": "date", "Open": "open", "High": "high",
                    "Low": "low", "Close": "close", "Volume": "volume"}
        df = df.rename(columns={c: col_map.get(c, c) for c in df.columns})
        if "date" not in df.columns:
            df = df.rename(columns={df.columns[0]: "date"})
        df["symbol"] = symbol
        df = df[["date", "symbol", "open", "high", "low", "close", "volume"]].tail(days)
        cache.put(key, df)
        return df
    except Exception as e:
        msg = str(e).lower()
        # Don't fallback for clearly invalid tickers
        if "no price data" in msg or "no data" in msg or "not found" in msg or "delisted" in msg:
            raise ValueError(f"Ticker '{symbol}' not found. Check the symbol and try again.") from e
        stale, age = cache.get_stale(key)
        if stale is not None:
            logger.warning(f"Using stale cache for {symbol} price (age: {age/60:.0f}m): {e}")
            return stale
        from backend.data.mock_generators.price_data import get_ohlcv as mock_ohlcv
        logger.warning(f"Using mock data for {symbol}: {e}")
        return mock_ohlcv(symbol, timeframe, days)


# Alias for backward compat
generate_ohlcv = get_ohlcv
