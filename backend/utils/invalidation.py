"""Invalidation rules engine for predictions."""
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional


def calculate_invalidation(price: float, direction: str, horizon: str, atr: float) -> Dict:
    """Calculate invalidation levels for a prediction."""
    # Use percentage-based stops capped by ATR
    # Max stop distance as % of price per horizon
    max_pct = {'short': 0.03, 'medium': 0.08, 'long': 0.15}  # 3%, 8%, 15%
    pct = max_pct.get(horizon, 0.05)

    # ATR-based distance, but capped at max percentage
    atr_pct = atr / price if price > 0 else 0.02
    stop_dist = min(atr_pct, pct) * price
    target_dist = stop_dist * 1.5

    if direction == 'bullish':
        stop = price - stop_dist
        target = price + target_dist
        condition = f"Close below ${stop:.2f}"
    elif direction == 'bearish':
        stop = price + stop_dist
        target = price - target_dist
        condition = f"Close above ${stop:.2f}"
    else:
        stop = price - stop_dist * 0.5
        target = price + stop_dist * 0.5
        condition = f"Move beyond ${price - stop_dist:.2f} - ${price + stop_dist:.2f}"

    return {'stop_loss': round(stop, 2), 'take_profit': round(target, 2), 'condition': condition}

@dataclass
class InvalidationResult:
    """Result of invalidation check."""
    is_valid: bool
    reasons: List[str]
    severity: str  # 'none', 'warning', 'critical'

class InvalidationEngine:
    """Check if predictions should be invalidated."""
    
    def __init__(self, volatility_threshold: float = 2.0, max_age_hours: int = 4):
        self.volatility_threshold = volatility_threshold
        self.max_age_hours = max_age_hours
        self.invalidation_keywords = ['earnings', 'merger', 'acquisition', 'fda', 'lawsuit', 'bankruptcy', 'recall']
    
    def check(
        self,
        prediction_time: datetime,
        current_price: float,
        prediction_price: float,
        recent_volatility: float,
        historical_volatility: float,
        news: Optional[List[Dict]] = None
    ) -> InvalidationResult:
        """Check if a prediction should be invalidated."""
        reasons = []
        severity = 'none'
        
        # Rule 1: Age check
        age = datetime.now() - prediction_time
        if age > timedelta(hours=self.max_age_hours):
            reasons.append(f"Prediction is {age.total_seconds()/3600:.1f}h old (max {self.max_age_hours}h)")
            severity = 'warning'
        
        # Rule 2: Volatility spike
        if historical_volatility > 0:
            vol_ratio = recent_volatility / historical_volatility
            if vol_ratio > self.volatility_threshold:
                reasons.append(f"Volatility spike: {vol_ratio:.1f}x normal")
                severity = 'critical'
        
        # Rule 3: Price moved significantly
        if prediction_price > 0:
            price_move = abs(current_price - prediction_price) / prediction_price
            if price_move > 0.05:  # 5% move
                reasons.append(f"Price moved {price_move*100:.1f}% since prediction")
                severity = 'critical' if price_move > 0.1 else 'warning'
        
        # Rule 4: Material news
        if news:
            for article in news:
                headline = article.get('headline', '').lower()
                if any(kw in headline for kw in self.invalidation_keywords):
                    reasons.append(f"Material news detected: {article.get('headline', '')[:50]}")
                    severity = 'critical'
                    break
        
        return InvalidationResult(
            is_valid=len(reasons) == 0,
            reasons=reasons,
            severity=severity
        )
    
    def should_refresh(self, result: InvalidationResult) -> bool:
        """Determine if prediction should be refreshed."""
        return result.severity in ('warning', 'critical')
