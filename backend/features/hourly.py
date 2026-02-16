"""Hourly feature computation for short-term (1h) predictions."""
import numpy as np
import pandas as pd


def fetch_hourly_data(ticker: str) -> pd.DataFrame | None:
    """Fetch 60 days of 1-hour interval data from yfinance."""
    import yfinance as yf
    import io
    import contextlib
    t = yf.Ticker(ticker)
    f = io.StringIO()
    with contextlib.redirect_stderr(f), contextlib.redirect_stdout(f):
        hist = t.history(period="60d", interval="1h")
    if hist is None or hist.empty or len(hist) < 20:
        return None
    df = hist.reset_index()
    col_map = {"Datetime": "date", "Date": "date", "Open": "open", "High": "high",
               "Low": "low", "Close": "close", "Volume": "volume"}
    df = df.rename(columns={c: col_map.get(c, c) for c in df.columns})
    if "date" not in df.columns:
        df = df.rename(columns={df.columns[0]: "date"})
    df["symbol"] = ticker
    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=["close"])
    return df[["date", "symbol", "open", "high", "low", "close", "volume"]]


def compute_hourly_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute a subset of features suitable for hourly data.

    Features: RSI, MACD, Bollinger, VWAP, volume ratio, returns, ATR.
    """
    close = df['close'].values.astype(float)
    high = df['high'].values.astype(float)
    low = df['low'].values.astype(float)
    volume = df['volume'].values.astype(float)
    n = len(close)
    feats = {}

    # Returns
    ret = np.zeros(n)
    ret[1:] = (close[1:] - close[:-1]) / np.maximum(close[:-1], 1e-8)
    feats['return_1h'] = ret

    # RSI 14
    feats['rsi_14'] = _rsi(close, 14)

    # MACD
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    macd_line = ema12 - ema26
    signal_line = _ema(macd_line, 9)
    feats['macd'] = macd_line
    feats['macd_signal'] = signal_line
    feats['macd_hist'] = macd_line - signal_line

    # Bollinger Bands
    sma20 = _sma(close, 20)
    std20 = _rolling_std(close, 20)
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    bb_range = np.maximum(bb_upper - bb_lower, 1e-8)
    feats['bb_position'] = (close - bb_lower) / bb_range

    # VWAP
    cum_vol = np.cumsum(volume)
    cum_vp = np.cumsum(close * volume)
    vwap = np.where(cum_vol > 0, cum_vp / cum_vol, close)
    feats['vwap_distance'] = (close - vwap) / np.maximum(close, 1e-8)

    # Volume ratio (vs 20-bar average)
    avg_vol = _sma(volume, 20)
    feats['volume_ratio'] = np.where(avg_vol > 0, volume / np.maximum(avg_vol, 1e-8), 1.0)

    # ATR (vectorized)
    tr = np.zeros(n)
    if n > 1:
        hl = high[1:] - low[1:]
        hc = np.abs(high[1:] - close[:-1])
        lc = np.abs(low[1:] - close[:-1])
        tr[1:] = np.maximum(hl, np.maximum(hc, lc))
    feats['atr_14'] = _sma(tr, 14)

    # Stochastic (vectorized via sliding window)
    period = 14
    k = np.zeros(n)
    if n >= period:
        from numpy.lib.stride_tricks import sliding_window_view
        h_win = sliding_window_view(high, period)
        l_win = sliding_window_view(low, period)
        h_max = h_win.max(axis=1)
        l_min = l_win.min(axis=1)
        rng = np.maximum(h_max - l_min, 1e-8)
        k[period - 1:] = (close[period - 1:] - l_min) / rng * 100
    feats[f'stoch_k_{period}'] = k

    # SMA distances
    for p in (5, 10, 20):
        sma = _sma(close, p)
        feats[f'sma_{p}_dist'] = (close - sma) / np.maximum(close, 1e-8)

    # EMA distances
    for p in (5, 10):
        ema = _ema(close, p)
        feats[f'ema_{p}_dist'] = (close - ema) / np.maximum(close, 1e-8)

    result = pd.DataFrame(feats, index=df.index)
    return result.fillna(0).replace([np.inf, -np.inf], 0)


HOURLY_FEATURE_COUNT = 16


def _sma(arr, period):
    result = np.zeros(len(arr))
    if len(arr) >= period:
        cumsum = np.cumsum(arr)
        result[period - 1:] = (cumsum[period - 1:] - np.concatenate(([0], cumsum[:-period]))) / period
    result[:period] = arr[:period].mean() if period <= len(arr) else 0
    return result


def _rolling_std(arr, period):
    result = np.zeros(len(arr))
    if len(arr) >= period:
        from numpy.lib.stride_tricks import sliding_window_view
        windows = sliding_window_view(arr, period)
        result[period - 1:] = np.std(windows, axis=1)
    return result


def _ema(arr, period):
    s = pd.Series(arr)
    return s.ewm(span=period, adjust=False).mean().values


def _rsi(close, period=14):
    n = len(close)
    rsi = np.full(n, 50.0)
    if n < period + 1:
        return rsi
    deltas = np.diff(close)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        rs = avg_gain / max(avg_loss, 1e-8)
        rsi[i + 1] = 100 - 100 / (1 + rs)
    return rsi
