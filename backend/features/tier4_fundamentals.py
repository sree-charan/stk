"""Tier 4: Fundamental Data Features (20 features)."""
from typing import Dict, List

class Tier4Fundamentals:
    """Generate 20 fundamental features."""
    
    @staticmethod
    def compute(fundamentals: List[Dict], price: float) -> Dict[str, float]:
        """Compute fundamental features from earnings/financial data."""
        if not fundamentals:
            return {k: 0.0 for k in Tier4Fundamentals.feature_names()}
        
        latest = fundamentals[0]  # Most recent is first
        prev = fundamentals[1] if len(fundamentals) > 1 else latest
        year_ago = fundamentals[3] if len(fundamentals) > 3 else latest  # noqa: F841
        
        f = {}
        
        # Earnings metrics (8) - use actual field names from generator
        f['eps'] = latest.get('eps', 1.0)
        f['eps_surprise'] = latest.get('eps_surprise_pct', 0) / 100  # Convert pct to decimal
        f['eps_growth_qoq'] = (latest.get('eps', 1) - prev.get('eps', 1)) / (abs(prev.get('eps', 1)) + 0.01)
        f['eps_growth_yoy'] = latest.get('eps_growth_yoy', 0)
        f['revenue'] = latest.get('revenue', 10e9) / 1e9  # Normalize to billions
        f['revenue_surprise'] = latest.get('revenue_surprise_pct', 0) / 100
        f['revenue_growth_qoq'] = (latest.get('revenue', 10e9) - prev.get('revenue', 10e9)) / (prev.get('revenue', 10e9) + 1)
        f['revenue_growth_yoy'] = latest.get('revenue_growth_yoy', 0)
        
        # Margins (6)
        f['gross_margin'] = latest.get('gross_margin', 0.3)
        f['operating_margin'] = latest.get('operating_margin', 0.15)
        f['net_margin'] = latest.get('net_margin', 0.1)
        f['margin_expansion'] = latest.get('gross_margin', 0.3) - prev.get('gross_margin', 0.3)
        f['fcf_margin'] = f['net_margin'] * 0.8  # Approximate FCF margin
        f['roa'] = f['net_margin'] * 0.5  # Approximate ROA
        
        # Valuation (6)
        eps_ttm = sum(q.get('eps', 0.5) for q in fundamentals[:4]) if len(fundamentals) >= 4 else latest.get('eps', 1) * 4
        f['pe_ratio'] = price / (eps_ttm + 0.01)
        f['forward_pe'] = price / (latest.get('eps_estimate', eps_ttm / 4) * 4 + 0.01)
        f['peg_ratio'] = f['pe_ratio'] / (max(0.01, f['eps_growth_yoy']) * 100 + 1)
        f['ps_ratio'] = price / (f['revenue'] * 4 + 0.01)  # Price to sales (revenue in billions)
        f['pb_ratio'] = price / (price / 5 + 0.01)  # Approximate P/B
        f['ev_ebitda'] = f['pe_ratio'] * 0.8  # Approximate EV/EBITDA
        
        return f
    
    @staticmethod
    def feature_names() -> list:
        return [
            'eps', 'eps_surprise', 'eps_growth_qoq', 'eps_growth_yoy', 'revenue', 'revenue_surprise', 'revenue_growth_qoq', 'revenue_growth_yoy',
            'gross_margin', 'operating_margin', 'net_margin', 'margin_expansion', 'fcf_margin', 'roa',
            'pe_ratio', 'forward_pe', 'peg_ratio', 'ps_ratio', 'pb_ratio', 'ev_ebitda'
        ]
