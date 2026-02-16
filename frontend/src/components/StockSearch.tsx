import React, { useState } from 'react';

interface Props {
  onSearch: (ticker: string) => void;
  disabled?: boolean;
}

const POPULAR_TICKERS = ['TSLA', 'AAPL', 'NVDA', 'MSFT', 'GOOGL', 'AMZN', 'META'];

export default function StockSearch({ onSearch, disabled }: Props) {
  const [ticker, setTicker] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (ticker.trim()) {
      onSearch(ticker.toUpperCase());
      setTicker('');
    }
  };

  return (
    <div style={{ marginBottom: '1rem' }}>
      <form onSubmit={handleSubmit} style={{ display: 'flex', gap: '0.5rem', marginBottom: '0.5rem' }}>
        <input
          value={ticker}
          onChange={e => setTicker(e.target.value.toUpperCase())}
          placeholder="Enter ticker..."
          disabled={disabled}
          style={{ flex: 1, padding: '0.5rem', borderRadius: '0.25rem', border: '1px solid #475569', background: '#1e293b', color: '#e2e8f0' }}
        />
        <button type="submit" disabled={disabled || !ticker.trim()} style={{ padding: '0.5rem 1rem', background: '#2563eb', color: 'white', border: 'none', borderRadius: '0.25rem', cursor: 'pointer' }}>
          Analyze
        </button>
      </form>
      <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
        {POPULAR_TICKERS.map(t => (
          <button
            key={t}
            onClick={() => onSearch(t)}
            disabled={disabled}
            style={{ padding: '0.25rem 0.5rem', background: '#334155', color: '#e2e8f0', border: 'none', borderRadius: '0.25rem', cursor: 'pointer', fontSize: '0.75rem' }}
          >
            {t}
          </button>
        ))}
      </div>
    </div>
  );
}
