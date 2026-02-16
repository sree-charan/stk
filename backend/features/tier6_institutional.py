"""Tier 6: Institutional Data Features (5 features)."""
from typing import Dict

class Tier6Institutional:
    """Generate 5 institutional data features."""
    
    @staticmethod
    def compute(institutional_data: Dict = None) -> Dict[str, float]:
        """Compute institutional features."""
        if not institutional_data:
            institutional_data = {}
        
        return {
            'institutional_ownership': institutional_data.get('institutional_pct', 0.7),
            'institutional_change': institutional_data.get('ownership_change', 0.01),  # Small positive change
            'insider_buying': institutional_data.get('insider_buys', 0.15),  # Net insider buying
            'short_interest': institutional_data.get('short_interest', 0.05),
            'days_to_cover': institutional_data.get('days_to_cover', 2.0),
        }
    
    @staticmethod
    def feature_names() -> list:
        return ['institutional_ownership', 'institutional_change', 'insider_buying', 'short_interest', 'days_to_cover']
