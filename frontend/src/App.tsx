import React, { useState, useRef, useEffect } from 'react';
import ChatInterface from './components/ChatInterface';
import PredictionCard from './components/PredictionCard';

interface Message {
  id: number;
  type: 'user' | 'assistant' | 'error';
  content: string;
  predictions?: Record<string, any>;
  ticker?: string;
}

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

export default function App() {
  const [messages, setMessages] = useState<Message[]>([
    { id: 0, type: 'assistant', content: '👋 Welcome! I can analyze stocks and provide predictions. Try "Analyze TSLA" or type "help" for more options.' }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    const ws = new WebSocket(`${API_URL.replace('http', 'ws')}/ws`);
    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'status') return;
      setLoading(false);
      setMessages(prev => [...prev, {
        id: Date.now(),
        type: data.type === 'error' ? 'error' : 'assistant',
        content: data.content,
        predictions: data.predictions,
        ticker: data.ticker
      }]);
    };
    wsRef.current = ws;
    return () => ws.close();
  }, []);

  const sendMessage = async () => {
    if (!input.trim() || loading) return;
    
    const userMsg: Message = { id: Date.now(), type: 'user', content: input };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setLoading(true);

    if (connected && wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ message: input }));
    } else {
      try {
        const res = await fetch(`${API_URL}/chat`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: input })
        });
        const data = await res.json();
        setMessages(prev => [...prev, {
          id: Date.now(),
          type: 'assistant',
          content: data.response,
          predictions: data.predictions,
          ticker: data.ticker
        }]);
      } catch (e) {
        setMessages(prev => [...prev, { id: Date.now(), type: 'error', content: 'Failed to connect to server' }]);
      }
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="header">
        <h1>📈 Stock Chat Assistant</h1>
        <div className="status">{connected ? '🟢 Connected' : '🔴 Disconnected (using REST)'}</div>
      </header>
      <ChatInterface messages={messages} chatEndRef={chatEndRef} loading={loading} />
      <div className="input-container">
        <input
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && sendMessage()}
          placeholder="Ask about a stock (e.g., Analyze TSLA)"
          aria-label="Chat message input"
          disabled={loading}
        />
        <button onClick={sendMessage} disabled={loading || !input.trim()}>
          {loading ? '...' : 'Send'}
        </button>
      </div>
    </div>
  );
}
