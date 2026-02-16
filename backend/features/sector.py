"""Sector-relative features: compare stock to SPY and sector ETF."""
import numpy as np
import pandas as pd

SECTOR_ETF_MAP = {
    'XLK': ['AAPL', 'MSFT', 'NVDA', 'GOOG', 'GOOGL', 'META', 'AVGO', 'ADBE', 'CRM', 'AMD', 'INTC', 'ORCL', 'CSCO', 'QCOM', 'TXN', 'NOW', 'AMAT', 'MU', 'LRCX', 'KLAC', 'SNPS', 'CDNS', 'MRVL', 'PANW', 'CRWD', 'FTNT'],
    'XLF': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'SCHW', 'AXP', 'USB', 'PNC', 'TFC', 'COF', 'BK', 'STT'],
    'XLE': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY', 'HAL', 'DVN', 'HES', 'FANG', 'BKR'],
    'XLV': ['UNH', 'JNJ', 'LLY', 'PFE', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'BMY', 'AMGN', 'MDT', 'ISRG', 'GILD', 'CVS', 'ELV', 'CI', 'VRTX', 'REGN', 'ZTS'],
    'XLY': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'LOW', 'SBUX', 'TJX', 'BKNG', 'CMG', 'ORLY', 'AZO', 'ROST', 'DHI', 'LEN', 'GM', 'F'],
    'XLI': ['CAT', 'HON', 'UNP', 'UPS', 'RTX', 'BA', 'DE', 'LMT', 'GE', 'MMM', 'FDX', 'WM', 'EMR', 'ITW', 'ETN', 'NSC', 'CSX'],
    'XLU': ['NEE', 'DUK', 'SO', 'D', 'AEP', 'SRE', 'EXC', 'XEL', 'ED', 'WEC', 'ES', 'AWK', 'AEE', 'CMS', 'DTE'],
    'XLP': ['PG', 'KO', 'PEP', 'COST', 'WMT', 'PM', 'MO', 'CL', 'MDLZ', 'GIS', 'KMB', 'SYY', 'HSY', 'K', 'STZ'],
    'XLRE': ['PLD', 'AMT', 'CCI', 'EQIX', 'PSA', 'SPG', 'O', 'DLR', 'WELL', 'AVB', 'EQR', 'VTR', 'ARE', 'MAA', 'UDR'],
    'XLB': ['LIN', 'APD', 'SHW', 'ECL', 'FCX', 'NEM', 'NUE', 'DOW', 'DD', 'VMC', 'MLM', 'PPG', 'ALB', 'CF', 'MOS'],
}


def get_sector_etf(ticker: str) -> str:
    """Map ticker to its sector ETF. Default to SPY."""
    ticker = ticker.upper()
    for etf, tickers in SECTOR_ETF_MAP.items():
        if ticker in tickers:
            return etf
    return 'SPY'


def compute_sector_features(price_df: pd.DataFrame, spy_df: pd.DataFrame = None,
                            sector_df: pd.DataFrame = None) -> pd.DataFrame:
    """Compute sector-relative features. If SPY/sector data unavailable, returns zeros."""
    n = len(price_df)
    close = price_df['close'].values.astype(float)
    result = {}

    stock_ret = np.zeros(n)
    stock_ret[1:] = (close[1:] - close[:-1]) / np.maximum(close[:-1], 1e-8)

    if spy_df is not None and len(spy_df) >= n:
        spy_close = spy_df['close'].values[-n:].astype(float)
        spy_ret = np.zeros(n)
        spy_ret[1:] = (spy_close[1:] - spy_close[:-1]) / np.maximum(spy_close[:-1], 1e-8)
        result['return_vs_spy'] = stock_ret - spy_ret
        # SPY RSI (simplified 14-period)
        delta = pd.Series(spy_close).diff().fillna(0)
        gain = delta.clip(lower=0).rolling(14, min_periods=1).mean()
        loss = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
        rs = gain / np.maximum(loss, 1e-8)
        result['spy_rsi'] = (100 - 100 / (1 + rs)).values
        # Correlation with SPY over 20 days
        corr = pd.Series(stock_ret).rolling(20, min_periods=5).corr(pd.Series(spy_ret)).fillna(0).values
        result['correlation_with_spy_20d'] = corr
        # SPY MACD (12/26 EMA diff)
        spy_s = pd.Series(spy_close)
        ema12 = spy_s.ewm(span=12, min_periods=1).mean()
        ema26 = spy_s.ewm(span=26, min_periods=1).mean()
        result['spy_macd'] = (ema12 - ema26).values
        # SPY volatility (20d rolling std of returns)
        result['spy_volatility_20d'] = pd.Series(spy_ret).rolling(20, min_periods=1).std().fillna(0).values
        # Relative strength vs SPY (20d cumulative)
        cum_stock = pd.Series(stock_ret).rolling(20, min_periods=1).sum().fillna(0).values
        cum_spy = pd.Series(spy_ret).rolling(20, min_periods=1).sum().fillna(0).values
        result['relative_strength_20d'] = cum_stock - cum_spy
        # Beta to SPY (20d rolling)
        cov = pd.Series(stock_ret).rolling(20, min_periods=5).cov(pd.Series(spy_ret)).fillna(0).values
        spy_var = pd.Series(spy_ret).rolling(20, min_periods=5).var().fillna(1e-8).values
        result['beta_spy_20d'] = cov / np.maximum(spy_var, 1e-8)
    else:
        result['return_vs_spy'] = np.zeros(n)
        result['spy_rsi'] = np.full(n, 50.0)
        result['correlation_with_spy_20d'] = np.zeros(n)
        result['spy_macd'] = np.zeros(n)
        result['spy_volatility_20d'] = np.zeros(n)
        result['relative_strength_20d'] = np.zeros(n)
        result['beta_spy_20d'] = np.ones(n)

    if sector_df is not None and len(sector_df) >= n:
        sec_close = sector_df['close'].values[-n:].astype(float)
        sec_ret = np.zeros(n)
        sec_ret[1:] = (sec_close[1:] - sec_close[:-1]) / np.maximum(sec_close[:-1], 1e-8)
        result['return_vs_sector'] = stock_ret - sec_ret
        # Sector momentum (20d return)
        sec_mom = np.zeros(n)
        sec_mom[20:] = (sec_close[20:] - sec_close[:-20]) / np.maximum(sec_close[:-20], 1e-8)
        result['sector_momentum'] = sec_mom
        # Correlation with sector
        result['correlation_with_sector_20d'] = pd.Series(stock_ret).rolling(20, min_periods=5).corr(pd.Series(sec_ret)).fillna(0).values
    else:
        result['return_vs_sector'] = np.zeros(n)
        result['sector_momentum'] = np.zeros(n)
        result['correlation_with_sector_20d'] = np.zeros(n)

    return pd.DataFrame(result, index=price_df.index)


SECTOR_FEATURE_NAMES = [
    'return_vs_spy', 'spy_rsi', 'correlation_with_spy_20d',
    'spy_macd', 'spy_volatility_20d', 'relative_strength_20d', 'beta_spy_20d',
    'return_vs_sector', 'sector_momentum', 'correlation_with_sector_20d',
]
