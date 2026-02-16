import React from 'react';

export default function LoadingIndicator() {
  return (
    <div className="message assistant" style={{ display: 'flex', gap: '4px' }}>
      <span className="dot" style={{ animation: 'pulse 1s infinite', animationDelay: '0s' }}>●</span>
      <span className="dot" style={{ animation: 'pulse 1s infinite', animationDelay: '0.2s' }}>●</span>
      <span className="dot" style={{ animation: 'pulse 1s infinite', animationDelay: '0.4s' }}>●</span>
      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 0.3; }
          50% { opacity: 1; }
        }
      `}</style>
    </div>
  );
}
