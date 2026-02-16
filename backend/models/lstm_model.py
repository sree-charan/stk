"""LSTM model for sequence-based prediction."""
import numpy as np
import pickle
from pathlib import Path
from typing import Tuple, Optional


def _get_torch():
    import torch
    return torch


def _get_nn():
    import torch.nn as nn
    return nn


class LSTMNet:
    """LSTM network for time series prediction (lazy torch import)."""

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2):
        _get_torch()  # ensure torch is available
        nn = _get_nn()

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
                self.fc = nn.Linear(hidden_size, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :])

        self._net = _Net()
        self._net_class = _Net

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        return getattr(self._net, name)

    def __call__(self, *args, **kwargs):
        return self._net(*args, **kwargs)

    def state_dict(self):
        return self._net.state_dict()

    def load_state_dict(self, sd):
        return self._net.load_state_dict(sd)

    def train(self, mode=True):
        return self._net.train(mode)

    def eval(self):
        return self._net.eval()

    def parameters(self):
        return self._net.parameters()

    def to(self, device):
        self._net = self._net.to(device)
        return self


class LSTMModel:
    """LSTM wrapper for stock prediction."""
    
    def __init__(self, seq_length: int = 20, hidden_size: int = 64):
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.model: Optional[LSTMNet] = None
        self.input_size: Optional[int] = None
        self._trained = False
        self._device = None

    @property
    def device(self):
        if self._device is None:
            torch = _get_torch()
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return self._device
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training."""
        X_seq, y_seq = [], []
        for i in range(len(X) - self.seq_length):
            X_seq.append(X[i:i + self.seq_length])
            y_seq.append(y[i + self.seq_length])
        return np.array(X_seq), np.array(y_seq)
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 10) -> float:
        """Train LSTM model."""
        torch = _get_torch()
        nn = _get_nn()
        self.input_size = X.shape[1]
        self.model = LSTMNet(self.input_size, self.hidden_size).to(self.device)
        
        X_seq, y_seq = self._create_sequences(X, y)
        if len(X_seq) == 0:
            return 0.0
        
        X_t = torch.FloatTensor(X_seq).to(self.device)
        y_t = torch.FloatTensor(y_seq).unsqueeze(1).to(self.device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        self.model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            pred = self.model(X_t)
            loss = criterion(pred, y_t)
            loss.backward()
            optimizer.step()
        
        self._trained = True
        self.model.eval()
        with torch.no_grad():
            pred = self.model(X_t).cpu().numpy().flatten()
        
        # Return R² score
        ss_res = np.sum((y_seq - pred) ** 2)
        ss_tot = np.sum((y_seq - np.mean(y_seq)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using last seq_length samples."""
        if not self._trained or self.model is None:
            return np.array([0.0])
        
        torch = _get_torch()
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Use last seq_length rows or pad
        if len(X) < self.seq_length:
            pad = np.zeros((self.seq_length - len(X), X.shape[1]))
            X = np.vstack([pad, X])
        
        X_seq = X[-self.seq_length:].reshape(1, self.seq_length, -1)
        X_t = torch.FloatTensor(X_seq).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            return self.model(X_t).cpu().numpy().flatten()
    
    def predict_proba(self, X: np.ndarray) -> Tuple[float, float]:
        """Return prediction and confidence."""
        pred = self.predict(X)
        val = pred[0] if len(pred) > 0 else 0.0
        confidence = min(0.9, 0.4 + abs(val) * 1.5)
        return float(val), confidence
    
    def save(self, path: Path):
        """Save model."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            'state_dict': self.model.state_dict() if self.model else None,
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'seq_length': self.seq_length,
            'trained': self._trained
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, path: Path) -> bool:
        """Load model."""
        if not path.exists():
            return False
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.input_size = data['input_size']
        self.hidden_size = data['hidden_size']
        self.seq_length = data['seq_length']
        self._trained = data['trained']
        if data['state_dict'] and self.input_size:
            self.model = LSTMNet(self.input_size, self.hidden_size).to(self.device)
            self.model.load_state_dict(data['state_dict'])
            self.model.eval()
        return True
