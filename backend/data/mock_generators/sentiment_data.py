"""Mock sentiment data generator - news, filings, sentiment scores."""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

HEADLINES = [
    "{symbol} beats earnings expectations",
    "{symbol} misses revenue targets",
    "Analysts upgrade {symbol} to buy",
    "{symbol} announces new product launch",
    "{symbol} CEO discusses growth strategy",
    "Institutional investors increase {symbol} holdings",
    "{symbol} faces regulatory scrutiny",
    "{symbol} expands into new markets",
    "Short interest in {symbol} rises",
    "{symbol} announces stock buyback program"
]

def generate_news_sentiment(
    symbol: str,
    days: int = 30,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """Generate news articles with sentiment scores."""
    if seed:
        np.random.seed(seed)
    
    data = []
    end_date = datetime.now()
    
    for d in range(days):
        date = end_date - timedelta(days=d)
        n_articles = np.random.poisson(8)  # ~8 articles per day
        
        for _ in range(n_articles):
            headline = np.random.choice(HEADLINES).format(symbol=symbol)
            
            # Sentiment based on headline keywords
            if any(w in headline.lower() for w in ['beats', 'upgrade', 'growth', 'buyback']):
                sentiment = np.random.uniform(0.3, 0.9)
            elif any(w in headline.lower() for w in ['misses', 'scrutiny', 'short']):
                sentiment = np.random.uniform(-0.9, -0.3)
            else:
                sentiment = np.random.uniform(-0.3, 0.3)
            
            data.append({
                'symbol': symbol,
                'date': date.date(),
                'timestamp': date - timedelta(hours=np.random.randint(0, 24)),
                'headline': headline,
                'sentiment': round(sentiment, 3),
                'source': np.random.choice(['Reuters', 'Bloomberg', 'CNBC', 'WSJ', 'MarketWatch']),
                'relevance': round(np.random.uniform(0.5, 1.0), 2)
            })
    
    return pd.DataFrame(data).sort_values('timestamp', ascending=False)


def generate_earnings_call_sentiment(symbol: str, quarters: int = 4, seed: Optional[int] = None) -> pd.DataFrame:
    """Generate earnings call transcript sentiment."""
    if seed:
        np.random.seed(seed)
    
    data = []
    end_date = datetime.now()
    
    for q in range(quarters):
        date = end_date - timedelta(days=90 * q)
        
        # Sentiment components
        mgmt_tone = np.random.uniform(-0.5, 0.8)
        qa_sentiment = np.random.uniform(-0.5, 0.8)
        guidance_sentiment = np.random.uniform(-0.5, 0.8)
        
        data.append({
            'symbol': symbol,
            'quarter': f"Q{((date.month-1)//3)+1} {date.year}",
            'date': date.date(),
            'management_tone': round(mgmt_tone, 3),
            'qa_sentiment': round(qa_sentiment, 3),
            'guidance_sentiment': round(guidance_sentiment, 3),
            'overall_sentiment': round((mgmt_tone + qa_sentiment + guidance_sentiment) / 3, 3),
            'word_count': np.random.randint(5000, 15000),
            'uncertainty_words': np.random.randint(10, 50),
            'positive_words': np.random.randint(50, 200),
            'negative_words': np.random.randint(20, 100)
        })
    
    return pd.DataFrame(data)


def generate_sec_filings(symbol: str, filings: int = 10, seed: Optional[int] = None) -> pd.DataFrame:
    """Generate SEC filing sentiment analysis."""
    if seed:
        np.random.seed(seed)
    
    data = []
    end_date = datetime.now()
    
    filing_types = ['10-K', '10-Q', '8-K', '10-Q', '8-K', '8-K']
    
    for i in range(filings):
        date = end_date - timedelta(days=30 * i)
        filing_type = filing_types[i % len(filing_types)]
        
        # Risk factor changes
        risk_change = np.random.uniform(-0.2, 0.2)
        
        # Ensure new != removed for non-zero filing_length_change
        new_rf = np.random.randint(1, 5)
        rem_rf = np.random.randint(0, new_rf)  # Always less than new
        
        data.append({
            'symbol': symbol,
            'filing_type': filing_type,
            'date': date.date(),
            'sentiment_score': round(np.random.uniform(-0.3, 0.5), 3),
            'risk_factor_change': round(risk_change, 3),
            'readability_score': round(np.random.uniform(30, 60), 1),
            'new_risk_factors': new_rf,
            'removed_risk_factors': rem_rf
        })
    
    return pd.DataFrame(data)


def generate_social_sentiment(symbol: str, days: int = 30, seed: Optional[int] = None) -> pd.DataFrame:
    """Generate social media sentiment metrics."""
    if seed:
        np.random.seed(seed)
    
    data = []
    end_date = datetime.now()
    
    for d in range(days):
        date = end_date - timedelta(days=d)
        
        # Social metrics
        mentions = int(np.random.exponential(5000))
        sentiment = np.random.uniform(-0.5, 0.5)
        
        data.append({
            'symbol': symbol,
            'date': date.date(),
            'twitter_mentions': mentions,
            'reddit_mentions': int(mentions * np.random.uniform(0.1, 0.3)),
            'sentiment_score': round(sentiment, 3),
            'bullish_pct': round(0.5 + sentiment * 0.3, 3),
            'bearish_pct': round(0.5 - sentiment * 0.3, 3),
            'volume_change': round(np.random.uniform(-0.5, 2.0), 2)
        })
    
    return pd.DataFrame(data)


def get_sentiment(symbol: str) -> dict:
    """Main interface - get all sentiment data for a symbol."""
    seed = sum(ord(c) for c in symbol) * 42
    
    return {
        'news': generate_news_sentiment(symbol, seed=seed),
        'earnings_calls': generate_earnings_call_sentiment(symbol, seed=seed),
        'sec_filings': generate_sec_filings(symbol, seed=seed),
        'social': generate_social_sentiment(symbol, seed=seed)
    }


if __name__ == "__main__":
    data = get_sentiment("TSLA")
    print("News sentiment:")
    print(data['news'].head())
    print("\nEarnings call sentiment:")
    print(data['earnings_calls'])
