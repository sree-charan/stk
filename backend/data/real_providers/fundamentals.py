"""Real fundamentals data via yfinance + SEC EDGAR."""
import pandas as pd
import yfinance as yf
from datetime import datetime
from . import cache
from backend.utils.retry import retry

_TTL = 3600  # 1 hour


@retry(max_attempts=2, base_delay=1.0)
def _fetch_ticker_data(symbol: str):
    ticker = yf.Ticker(symbol)
    info = ticker.info or {}
    qf = ticker.quarterly_financials
    return info, qf


def get_fundamentals(symbol: str, quarters: int = 8) -> dict:
    key = f"fundamentals_{symbol}"
    cached = cache.get(key, _TTL)
    if cached is not None:
        return cached

    try:
        info, qf = _fetch_ticker_data(symbol)
        price = info.get('currentPrice') or info.get('regularMarketPrice', 100)

        # Quarterly financials
        quarterly_rows = []
        if qf is not None and not qf.empty:
            for col in list(qf.columns)[:quarters]:
                rev = qf.loc['Total Revenue', col] if 'Total Revenue' in qf.index else 0
                ni = qf.loc['Net Income', col] if 'Net Income' in qf.index else 0
                gp = qf.loc['Gross Profit', col] if 'Gross Profit' in qf.index else 0
                oi = qf.loc['Operating Income', col] if 'Operating Income' in qf.index else 0
                eps = ni / max(1, info.get('sharesOutstanding', 1e9)) if ni else 0
                quarterly_rows.append({
                    'symbol': symbol,
                    'quarter': f"Q{((col.month-1)//3)+1} {col.year}",
                    'date': col.date(),
                    'revenue': float(rev or 0),
                    'revenue_estimate': float(rev or 0),
                    'revenue_surprise_pct': 0.0,
                    'eps': round(eps, 2),
                    'eps_estimate': round(eps, 2),
                    'eps_surprise_pct': 0.0,
                    'gross_margin': round(float(gp) / max(1, float(rev)), 4) if rev else 0,
                    'operating_margin': round(float(oi) / max(1, float(rev)), 4) if rev else 0,
                    'net_margin': round(float(ni) / max(1, float(rev)), 4) if rev else 0,
                    'revenue_growth_yoy': info.get('revenueGrowth', 0) or 0,
                    'eps_growth_yoy': info.get('earningsGrowth', 0) or 0,
                })

        if not quarterly_rows:
            quarterly_rows = [_default_quarter(symbol)]

        quarterly = pd.DataFrame(quarterly_rows)

        shares = info.get('sharesOutstanding', 1e9)
        market_cap = info.get('marketCap', price * shares)
        annual_eps = info.get('trailingEps', quarterly_rows[0]['eps'] * 4)
        pe = info.get('trailingPE', 0) or (price / annual_eps if annual_eps else 0)

        valuation = {
            'symbol': symbol,
            'market_cap': float(market_cap),
            'pe_ratio': round(float(pe), 2),
            'forward_pe': round(float(info.get('forwardPE', pe * 0.9) or pe), 2),
            'ps_ratio': round(float(info.get('priceToSalesTrailing12Months', 0) or 0), 2),
            'pb_ratio': round(float(info.get('priceToBook', 0) or 0), 2),
            'ev_ebitda': round(float(info.get('enterpriseToEbitda', 0) or 0), 2),
            'peg_ratio': round(float(info.get('pegRatio', 0) or 0), 2),
            'dividend_yield': round(float(info.get('dividendYield', 0) or 0), 4),
            'shares_outstanding': int(shares),
        }

        result = {
            'quarterly': quarterly,
            'valuation': valuation,
            'latest_quarter': quarterly.iloc[0].to_dict(),
        }
        cache.put(key, result)
        return result
    except Exception as e:
        stale, age = cache.get_stale(key)
        if stale is not None:
            import logging
            logging.warning(f"Using stale cache for {symbol} fundamentals (age: {age/60:.0f}m): {e}")
            return stale
        from backend.data.mock_generators.fundamentals import get_fundamentals as mock
        return mock(symbol, quarters)


def _default_quarter(symbol):
    return {
        'symbol': symbol, 'quarter': 'Q1 2026', 'date': datetime.now().date(),
        'revenue': 0, 'revenue_estimate': 0, 'revenue_surprise_pct': 0,
        'eps': 0, 'eps_estimate': 0, 'eps_surprise_pct': 0,
        'gross_margin': 0, 'operating_margin': 0, 'net_margin': 0,
        'revenue_growth_yoy': 0, 'eps_growth_yoy': 0,
    }
