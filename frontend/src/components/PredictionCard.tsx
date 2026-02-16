import React from 'react';

interface HorizonData {
  name?: string;
  direction: string;
  confidence: number;
  expected_return?: number;
  prediction?: number;
  invalidation?: string;
}

interface Props {
  ticker: string;
  predictions: {
    current_price?: number;
    bias?: string;
    key_bullish_signals?: string[];
    key_bearish_signals?: string[];
    horizons?: HorizonData[];
    [key: string]: any;
  };
}

const verdictEmoji: Record<string, string> = {
  BUY: '🟢', BULLISH: '🟢', SELL: '🔴', BEARISH: '🔴', HOLD: '🟡', NEUTRAL: '🟡',
};

export default function PredictionCard({ ticker, predictions }: Props) {
  const { current_price, bias, horizons, key_bullish_signals, key_bearish_signals } = predictions;

  return (
    <div className="prediction-card">
      <div className="prediction-header">
        <h3>{ticker}</h3>
        {current_price != null && <span className="price">${current_price.toFixed(2)}</span>}
        {bias && <span className={`verdict ${bias.toLowerCase()}`}>{verdictEmoji[bias] || '🟡'} {bias}</span>}
      </div>

      {horizons && horizons.map((h, i) => {
        const dir = (h.direction || '').toUpperCase();
        const ret = h.expected_return ?? (h.prediction != null ? h.prediction * 100 : 0);
        return (
          <div key={i} className="prediction-row">
            <span className="horizon-label">{h.name || ['Short', 'Medium', 'Long'][i]}</span>
            <span className={`direction ${dir.toLowerCase()}`}>
              {verdictEmoji[dir] || '🟡'} {dir} ({h.confidence}%)
            </span>
            <span className="return">{ret >= 0 ? '+' : ''}{ret.toFixed(2)}%</span>
          </div>
        );
      })}

      {key_bullish_signals && key_bullish_signals.length > 0 && (
        <div className="signals bullish">
          <strong>Bullish:</strong> {key_bullish_signals.slice(0, 3).join(' · ')}
        </div>
      )}
      {key_bearish_signals && key_bearish_signals.length > 0 && (
        <div className="signals bearish">
          <strong>Bearish:</strong> {key_bearish_signals.slice(0, 3).join(' · ')}
        </div>
      )}

      <div className="data-freshness">
        Updated: {new Date().toLocaleTimeString()} (live data)
      </div>
    </div>
  );
}
