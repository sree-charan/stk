import React from 'react';
import PredictionCard from './PredictionCard';
import LoadingIndicator from './LoadingIndicator';

interface Message {
  id: number;
  type: 'user' | 'assistant' | 'error';
  content: string;
  predictions?: Record<string, any>;
  ticker?: string;
}

interface Props {
  messages: Message[];
  chatEndRef: React.RefObject<HTMLDivElement>;
  loading?: boolean;
}

export default function ChatInterface({ messages, chatEndRef, loading }: Props) {
  return (
    <div className="chat-container">
      {messages.map(msg => (
        <div key={msg.id} className={`message ${msg.type}`}>
          <div style={{ whiteSpace: 'pre-wrap' }}>{msg.content}</div>
          {msg.predictions && msg.ticker && (
            <PredictionCard ticker={msg.ticker} predictions={msg.predictions} />
          )}
        </div>
      ))}
      {loading && <LoadingIndicator />}
      <div ref={chatEndRef} />
    </div>
  );
}
