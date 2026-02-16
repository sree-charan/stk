"""Real sentiment data via NewsAPI + VADER."""
import pandas as pd
from datetime import datetime
from . import cache
from backend.utils.retry import retry

_TTL = 900  # 15 min


@retry(max_attempts=2, base_delay=1.0)
def _fetch_articles(symbol: str):
    from newsapi import NewsApiClient
    newsapi = NewsApiClient(api_key='4adf7244903142a4986d99294d6ea3ee')
    return newsapi.get_everything(q=symbol, sort_by='publishedAt', language='en', page_size=50)


def get_sentiment(symbol: str) -> dict:
    key = f"sentiment_{symbol}"
    cached = cache.get(key, _TTL)
    if cached is not None:
        return cached

    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()

        articles = _fetch_articles(symbol)

        rows = []
        for a in articles.get('articles', []):
            text = f"{a.get('title', '')} {a.get('description', '')}"
            score = analyzer.polarity_scores(text)['compound']
            rows.append({
                'symbol': symbol,
                'date': pd.Timestamp(a.get('publishedAt', datetime.now().isoformat())).date(),
                'timestamp': pd.Timestamp(a.get('publishedAt', datetime.now().isoformat())),
                'headline': a.get('title', ''),
                'sentiment': round(score, 3),
                'source': a.get('source', {}).get('name', 'Unknown'),
                'relevance': 0.8,
            })

        news_df = pd.DataFrame(rows) if rows else _empty_news(symbol)

        # Stub earnings/filings/social with minimal data matching mock interface
        result = {
            'news': news_df.sort_values('timestamp', ascending=False) if not news_df.empty else news_df,
            'earnings_calls': _stub_earnings(symbol),
            'sec_filings': _stub_filings(symbol),
            'social': _stub_social(symbol),
        }
        cache.put(key, result)
        return result
    except Exception as e:
        stale, age = cache.get_stale(key)
        if stale is not None:
            import logging
            logging.warning(f"Using stale cache for {symbol} sentiment (age: {age/60:.0f}m): {e}")
            return stale
        from backend.data.mock_generators.sentiment_data import get_sentiment as mock
        return mock(symbol)


def _empty_news(symbol):
    return pd.DataFrame(columns=['symbol', 'date', 'timestamp', 'headline', 'sentiment', 'source', 'relevance'])


def _stub_earnings(symbol):
    now = datetime.now()
    rows = [{
        'symbol': symbol, 'quarter': f"Q{((now.month-1)//3)+1} {now.year}",
        'date': now.date(), 'management_tone': 0.0, 'qa_sentiment': 0.0,
        'guidance_sentiment': 0.0, 'overall_sentiment': 0.0,
        'word_count': 0, 'uncertainty_words': 0, 'positive_words': 0, 'negative_words': 0,
    }]
    return pd.DataFrame(rows)


def _stub_filings(symbol):
    rows = [{
        'symbol': symbol, 'filing_type': '10-Q', 'date': datetime.now().date(),
        'sentiment_score': 0.0, 'risk_factor_change': 0.0, 'readability_score': 45.0,
        'new_risk_factors': 0, 'removed_risk_factors': 0,
    }]
    return pd.DataFrame(rows)


def _stub_social(symbol):
    rows = [{
        'symbol': symbol, 'date': datetime.now().date(),
        'twitter_mentions': 0, 'reddit_mentions': 0, 'sentiment_score': 0.0,
        'bullish_pct': 0.5, 'bearish_pct': 0.5, 'volume_change': 0.0,
    }]
    return pd.DataFrame(rows)
