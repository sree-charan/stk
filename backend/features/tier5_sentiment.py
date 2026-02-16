"""Tier 5: Sentiment & NLP Features (25 features)."""
import numpy as np
from typing import Dict, List

class Tier5Sentiment:
    """Generate 25 sentiment and NLP features."""
    
    @staticmethod
    def compute(news: List[Dict], filings: List[Dict] = None, earnings_calls: List[Dict] = None) -> Dict[str, float]:
        """Compute sentiment features from text data."""
        f = {}
        
        # Earnings call features (8) - use actual field names from generator
        if earnings_calls and len(earnings_calls) > 0:
            latest_call = earnings_calls[0]  # Most recent first
            f['call_sentiment'] = latest_call.get('overall_sentiment', latest_call.get('management_tone', 0))
            f['call_uncertainty'] = latest_call.get('uncertainty_words', 29) / 100
            total_words = latest_call.get('word_count', 7000)
            f['call_positive_ratio'] = latest_call.get('positive_words', 64) / (total_words + 1)
            f['call_negative_ratio'] = latest_call.get('negative_words', 97) / (total_words + 1)
            f['call_forward_looking'] = latest_call.get('guidance_sentiment', 0.3) + 0.5  # Normalize to 0-1
            f['call_risk_mentions'] = latest_call.get('uncertainty_words', 29) / 200
            f['call_guidance_change'] = latest_call.get('guidance_sentiment', 0)
            f['call_qa_sentiment'] = latest_call.get('qa_sentiment', 0)
        else:
            f.update({k: 0.0 for k in ['call_sentiment', 'call_uncertainty', 'call_positive_ratio', 
                                        'call_negative_ratio', 'call_forward_looking', 'call_risk_mentions',
                                        'call_guidance_change', 'call_qa_sentiment']})
        
        # SEC filings features (6) - use actual field names from generator
        if filings and len(filings) > 0:
            latest_filing = filings[0]  # Most recent first
            prev_filing = filings[1] if len(filings) > 1 else latest_filing  # noqa: F841
            f['filing_sentiment'] = latest_filing.get('sentiment_score', 0)
            f['filing_risk_change'] = latest_filing.get('risk_factor_change', 0)
            f['filing_complexity'] = latest_filing.get('readability_score', 45) / 60  # Normalize
            f['filing_length_change'] = (latest_filing.get('new_risk_factors', 1) - latest_filing.get('removed_risk_factors', 0)) / 10
            f['filing_litigation_risk'] = latest_filing.get('new_risk_factors', 1) / 20
            f['filing_md_a_sentiment'] = latest_filing.get('sentiment_score', 0) * 0.8
        else:
            f.update({k: 0.0 for k in ['filing_sentiment', 'filing_risk_change', 'filing_complexity',
                                        'filing_length_change', 'filing_litigation_risk', 'filing_md_a_sentiment']})
        
        # News sentiment (6)
        if news and len(news) > 0:
            recent_news = news[:50] if len(news) > 50 else news
            sentiments = [n.get('sentiment', 0) for n in recent_news]
            f['news_sentiment_avg'] = np.mean(sentiments)
            f['news_sentiment_std'] = np.std(sentiments) if len(sentiments) > 1 else 0.5
            f['news_volume'] = len(recent_news)
            f['news_sentiment_trend'] = np.mean(sentiments[:10]) - np.mean(sentiments[-10:]) if len(sentiments) > 10 else 0
            f['news_positive_ratio'] = sum(1 for s in sentiments if s > 0.2) / (len(sentiments) + 1)
            f['news_negative_ratio'] = sum(1 for s in sentiments if s < -0.2) / (len(sentiments) + 1)
        else:
            f.update({k: 0.0 for k in ['news_sentiment_avg', 'news_sentiment_std', 'news_volume',
                                        'news_sentiment_trend', 'news_positive_ratio', 'news_negative_ratio']})
        
        # Alternative sentiment (5) - derive from available data
        if news and len(news) > 0:
            sentiments = [n.get('sentiment', 0) for n in news[:20]]
            avg_sent = np.mean(sentiments)
            f['social_sentiment'] = avg_sent * 0.8  # Proxy
            f['analyst_sentiment'] = avg_sent * 1.1  # Proxy
            f['insider_sentiment'] = avg_sent * 0.5  # Derived
            f['short_interest_sentiment'] = -abs(avg_sent) * 0.3  # Typically negative
            f['options_sentiment'] = avg_sent * 0.7  # Derived
        else:
            f.update({k: 0.0 for k in ['social_sentiment', 'analyst_sentiment', 'insider_sentiment',
                                        'short_interest_sentiment', 'options_sentiment']})
        
        return f
    
    @staticmethod
    def feature_names() -> list:
        return [
            'call_sentiment', 'call_uncertainty', 'call_positive_ratio', 'call_negative_ratio', 
            'call_forward_looking', 'call_risk_mentions', 'call_guidance_change', 'call_qa_sentiment',
            'filing_sentiment', 'filing_risk_change', 'filing_complexity', 'filing_length_change', 
            'filing_litigation_risk', 'filing_md_a_sentiment',
            'news_sentiment_avg', 'news_sentiment_std', 'news_volume', 'news_sentiment_trend', 
            'news_positive_ratio', 'news_negative_ratio',
            'social_sentiment', 'analyst_sentiment', 'insider_sentiment', 'short_interest_sentiment', 'options_sentiment'
        ]
