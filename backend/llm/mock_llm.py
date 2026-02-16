"""Mock LLM for offline operation."""
from typing import Dict

class MockLLM:
    """Simulates LLM responses for stock analysis."""
    
    def __init__(self):
        self.templates = {
            'bullish': [
                "Based on my analysis, {ticker} shows bullish signals with {confidence:.0%} confidence.",
                "The technical and fundamental indicators suggest {ticker} may rise {return_pct:.1%} over the {horizon}.",
                "I'm seeing positive momentum in {ticker}. Key factors: {factors}."
            ],
            'bearish': [
                "My analysis indicates bearish pressure on {ticker} with {confidence:.0%} confidence.",
                "Technical indicators suggest {ticker} may decline {return_pct:.1%} over the {horizon}.",
                "Caution advised for {ticker}. Negative signals: {factors}."
            ],
            'neutral': [
                "{ticker} appears range-bound with mixed signals. Confidence: {confidence:.0%}.",
                "No clear directional bias for {ticker} over the {horizon}.",
                "Waiting for clearer signals on {ticker}. Current outlook: neutral."
            ]
        }
        
        self.factor_templates = {
            'bullish': ['strong momentum', 'positive sentiment', 'institutional buying', 'favorable macro'],
            'bearish': ['weak momentum', 'negative sentiment', 'institutional selling', 'macro headwinds'],
            'neutral': ['mixed signals', 'consolidation pattern', 'awaiting catalyst']
        }
    
    def generate(self, ticker: str, prediction: Dict, horizon: str = 'short') -> str:
        """Generate natural language response from prediction."""
        direction = prediction.get('direction', 'neutral')
        confidence = prediction.get('confidence', 0.5)
        if confidence > 1:
            confidence = confidence / 100
        return_pct = abs(prediction.get('prediction', 0))
        
        horizon_map = {'short': '1 hour', 'medium': '5 days', 'long': '60 days'}
        horizon_text = horizon_map.get(horizon, horizon)
        
        # Deterministic template selection based on confidence level
        idx = 0 if confidence > 0.6 else (1 if confidence > 0.5 else 2)
        template = self.templates[direction][idx % len(self.templates[direction])]
        factors = ', '.join(self.factor_templates[direction][:2])
        
        return template.format(
            ticker=ticker,
            confidence=confidence,
            return_pct=return_pct,
            horizon=horizon_text,
            factors=factors
        )
    
    def generate_summary(self, ticker: str, all_predictions: Dict) -> str:
        """Generate comprehensive summary for all horizons."""
        lines = [f"📊 **{ticker} Analysis Summary**\n"]
        
        # Handle new format with 'horizons' list
        if 'horizons' in all_predictions:
            price = all_predictions.get('current_price', 0)
            lines.append(f"Current Price: ${price:.2f}")
            if all_predictions.get('predictability'):
                level, icon, desc = all_predictions['predictability']
                lines.append(f"Predictability: {icon} {level} — {desc}")
            lines.append("")
            directions = []
            for h in all_predictions['horizons']:
                d = h['direction'].lower()
                directions.append(d)
                emoji = '🟢' if d == 'bullish' else '🔴' if d == 'bearish' else '🟡'
                conf = h['confidence'] / 100 if h['confidence'] > 1 else h['confidence']
                verdict = h.get('conviction_verdict', '')
                if verdict:
                    lines.append(f"{emoji} **{h['name']}**: {verdict}")
                else:
                    lines.append(f"{emoji} **{h['name']}**: {h['direction']} ({conf:.0%} confidence)")
                lines.append(f"   Expected: {h.get('expected_return', 0):+.2f}%")
                if h.get('vol_zscore') is not None:
                    lines.append(f"   ({h['vol_zscore']:+.1f}σ — {h.get('vol_zscore_desc', '')})")
                lines.append(f"   Invalidation: {h.get('invalidation', 'N/A')}")
                if h.get('explanation'):
                    lines.append(f"   {h['explanation']}")
                if h.get('mtf_verdict'):
                    lines.append(f"   Timeframe vote: {h['mtf_verdict']}")
                if h.get('predictability'):
                    pred_icons = {'HIGH': '🟢', 'MODERATE': '🟡', 'LOW': '⚪'}
                    lines.append(f"   Predictability: {pred_icons.get(h['predictability'], '⚪')} {h['predictability']}")
                if h.get('conf_interval'):
                    lines.append(f"   90% range: {h['conf_interval']}")
                if h.get('uncertainty'):
                    lines.append(f"   {h['uncertainty']}")
                if h.get('signal_quality'):
                    lines.append(f"   Signal quality: {h['signal_quality']}")
                lines.append("")
        else:
            # Old format with short/medium/long keys
            directions = []
            for horizon, pred in all_predictions.items():
                if horizon in ('short', 'medium', 'long'):
                    d = pred['direction']
                    directions.append(d)
                    emoji = '🟢' if d == 'bullish' else '🔴' if d == 'bearish' else '🟡'
                    horizon_name = {'short': '1-Hour', 'medium': '5-Day', 'long': '60-Day'}[horizon]
                    conf = pred['confidence'] if pred['confidence'] <= 1 else pred['confidence'] / 100
                    lines.append(f"{emoji} **{horizon_name}**: {d.title()} ({conf:.0%} confidence)")
                    lines.append(f"   Expected return: {pred['prediction']*100:+.2f}%\n")
        
        if directions.count('bullish') >= 2:
            lines.append("💡 **Overall**: Bullish bias across timeframes")
        elif directions.count('bearish') >= 2:
            lines.append("💡 **Overall**: Bearish bias across timeframes")
        else:
            lines.append("💡 **Overall**: Mixed signals - consider waiting for clarity")

        # Model health and drift warnings
        if all_predictions.get('model_health'):
            grade = all_predictions['model_health']
            icons = {'A': '🟢', 'B': '🟡', 'C': '🟠', 'D': '🔴', 'F': '⛔'}
            lines.append(f"\n{icons.get(grade, '⚪')} Model Health: {grade}")
        if all_predictions.get('drift_warnings'):
            lines.append(f"⚠ {all_predictions['drift_warnings']} features outside training range")
        if all_predictions.get('health_trend'):
            lines.append(all_predictions['health_trend'])
        if all_predictions.get('health_degradation'):
            lines.append(all_predictions['health_degradation'])
        if all_predictions.get('past_accuracy'):
            lines.append(f"\n📋 Past prediction accuracy: {all_predictions['past_accuracy']}")
        if all_predictions.get('best_horizon'):
            lines.append(f"\n🎯 Best horizon: {all_predictions['best_horizon']}")

        return '\n'.join(lines)
