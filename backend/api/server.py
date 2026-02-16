"""FastAPI server with REST and WebSocket endpoints."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from datetime import datetime
from typing import Dict, Optional
import json
import numpy as np

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.llm import IntentParser, ResponseGenerator
from backend.models.ensemble import EnsembleModel
from backend.features.feature_store import FeatureStore
from backend.data.real_providers import generate_ohlcv, get_options_chain, get_fundamentals, get_sentiment, get_macro_data
from backend.utils.invalidation import InvalidationEngine
from backend.utils.backtesting import Backtester


def _classify_http_error(e: Exception, ticker: str = ""):
    """Convert exceptions to appropriate HTTPException with proper status codes."""
    msg = str(e).lower()
    if "no price data" in msg or "no data" in msg or "not found" in msg:
        raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' not found") from e
    if "rate limit" in msg or "429" in msg or "too many" in msg:
        raise HTTPException(status_code=429, detail="Rate limited — try again shortly") from e
    if "connection" in msg or "timeout" in msg or "network" in msg:
        raise HTTPException(status_code=503, detail=f"Data source unavailable: {e}") from e
    raise HTTPException(status_code=500, detail=str(e)) from e

ROOT_DIR = Path(__file__).parent.parent.parent
FRONTEND_BUILD = ROOT_DIR / 'frontend' / 'build'

def convert_numpy(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def extract_signals(features, current_price: float):
    """Extract top 5 bullish and bearish signals from features."""
    bullish, bearish = [], []
    
    def get_feat(name):
        if hasattr(features, 'iloc'):
            last = features.iloc[-1]
            return last[name] if name in features.columns else None
        return None
    
    # RSI
    rsi = get_feat('rsi_14')
    if rsi is not None:
        if rsi < 30: bullish.append(f"RSI oversold ({rsi:.1f})")
        elif rsi < 40: bullish.append(f"RSI approaching oversold ({rsi:.1f})")
        elif rsi > 70: bearish.append(f"RSI overbought ({rsi:.1f})")
        elif rsi > 60: bearish.append(f"RSI approaching overbought ({rsi:.1f})")
    
    # MACD
    macd_hist = get_feat('macd_hist')
    if macd_hist is not None:
        if macd_hist > 0.5: bullish.append(f"MACD histogram positive ({macd_hist:.2f})")
        elif macd_hist > 0: bullish.append(f"MACD slightly positive ({macd_hist:.2f})")
        elif macd_hist < -0.5: bearish.append(f"MACD histogram negative ({macd_hist:.2f})")
        else: bearish.append(f"MACD slightly negative ({macd_hist:.2f})")
    
    # Bollinger Bands
    bb = get_feat('bb_position')
    if bb is not None:
        if bb < 0.2: bullish.append(f"Near lower Bollinger ({bb:.2f})")
        elif bb > 0.8: bearish.append(f"Near upper Bollinger ({bb:.2f})")
        elif bb < 0.35: bullish.append(f"Below BB midline ({bb:.2f})")
        elif bb > 0.55: bearish.append(f"Above BB midline ({bb:.2f})")
    
    # Volume
    vol_ratio = get_feat('volume_ratio')
    if vol_ratio is not None:
        if vol_ratio > 1.5: bullish.append(f"Volume surge ({vol_ratio:.1f}x avg)")
        elif vol_ratio > 1.1: bullish.append(f"Above avg volume ({vol_ratio:.1f}x)")
        elif vol_ratio < 0.7: bearish.append(f"Low volume ({vol_ratio:.1f}x)")
    
    # VWAP
    vwap = get_feat('vwap')
    vwap_dist = get_feat('vwap_distance')
    if vwap is not None and vwap_dist is not None:
        if vwap_dist > 0.005: bullish.append(f"Above VWAP (${vwap:.2f})")
        elif vwap_dist < -0.005: bearish.append(f"Below VWAP (${vwap:.2f})")
    
    # Sentiment
    sent = get_feat('news_sentiment_avg')
    if sent is not None:
        if sent > 0.1: bullish.append(f"Positive sentiment ({sent:.2f})")
        elif sent < -0.1: bearish.append(f"Negative sentiment ({sent:.2f})")
    
    # ADX trend
    adx = get_feat('adx_14')
    ret_5d = get_feat('return_5d')
    if adx is not None and adx > 20 and ret_5d is not None:
        if ret_5d > 0: bullish.append(f"Strong uptrend (ADX {adx:.1f})")
        else: bearish.append(f"Strong downtrend (ADX {adx:.1f})")
    
    # IV percentile
    iv_pct = get_feat('iv_percentile')
    if iv_pct is not None:
        if iv_pct < 0.3: bullish.append(f"Low IV ({iv_pct:.0%})")
        elif iv_pct > 0.7: bearish.append(f"High IV ({iv_pct:.0%})")
    
    # 50-SMA
    sma_50 = get_feat('sma_50')
    if sma_50 is not None and current_price > 0:
        pct = (current_price - sma_50) / sma_50 * 100
        if pct > 5: bullish.append(f"Price {pct:.1f}% above 50-SMA")
        elif pct < -5: bearish.append(f"Price {abs(pct):.1f}% below 50-SMA")
    
    # Williams %R
    williams = get_feat('williams_r')
    if williams is not None:
        if williams < -80: bullish.append(f"Williams %R oversold ({williams:.1f})")
        elif williams > -20: bearish.append(f"Williams %R overbought ({williams:.1f})")
        elif williams < -60: bullish.append(f"Williams %R approaching oversold ({williams:.1f})")
        elif williams > -40: bearish.append(f"Williams %R approaching overbought ({williams:.1f})")
    
    # MFI
    mfi = get_feat('mfi_14')
    if mfi is not None:
        if mfi < 30: bullish.append(f"MFI oversold ({mfi:.1f})")
        elif mfi > 70: bearish.append(f"MFI overbought ({mfi:.1f})")
        elif mfi < 40: bullish.append(f"MFI approaching oversold ({mfi:.1f})")
        elif mfi > 55: bearish.append(f"MFI elevated ({mfi:.1f})")
    
    # VIX
    vix = get_feat('vix')
    if vix is not None:
        if vix < 15: bullish.append(f"Low VIX ({vix:.1f})")
        elif vix > 25: bearish.append(f"High VIX ({vix:.1f})")
        elif vix > 18: bearish.append(f"Elevated VIX ({vix:.1f})")
    
    # Yield curve
    yield_curve = get_feat('yield_curve')
    if yield_curve is not None:
        if yield_curve < -0.2: bearish.append(f"Inverted yield curve ({yield_curve:.2f})")
        elif yield_curve > 0.5: bullish.append(f"Steep yield curve ({yield_curve:.2f})")
    
    # Sector rotation
    sector_rot = get_feat('sector_rotation')
    if sector_rot is not None:
        if sector_rot < -0.1: bearish.append(f"Negative sector rotation ({sector_rot:.2f})")
        elif sector_rot > 0.1: bullish.append(f"Positive sector rotation ({sector_rot:.2f})")
    
    # Put/Call ratio
    pcr = get_feat('put_call_ratio')
    if pcr is not None:
        if pcr < 0.7: bullish.append(f"Low put/call ratio ({pcr:.2f})")
        elif pcr > 1.2: bearish.append(f"High put/call ratio ({pcr:.2f})")
    
    # Daily return momentum
    ret_1d = get_feat('return_1d')
    if ret_1d is not None:
        if ret_1d > 0.02: bullish.append(f"Strong daily gain ({ret_1d*100:.1f}%)")
        elif ret_1d < -0.02: bearish.append(f"Strong daily loss ({abs(ret_1d)*100:.1f}%)")
    
    # EMA crossover
    ema_5 = get_feat('ema_5')
    ema_20 = get_feat('ema_20')
    if ema_5 is not None and ema_20 is not None:
        if ema_5 > ema_20 * 1.01: bullish.append(f"EMA5 above EMA20 (${ema_5:.2f})")
        elif ema_5 < ema_20 * 0.99: bearish.append(f"EMA5 below EMA20 (${ema_5:.2f})")
    
    # ATR volatility
    atr = get_feat('atr_14')
    if atr is not None and current_price > 0:
        atr_pct = atr / current_price * 100
        if atr_pct < 1.5: bullish.append(f"Low volatility (ATR {atr_pct:.1f}%)")
        elif atr_pct > 3: bearish.append(f"High volatility (ATR {atr_pct:.1f}%)")
    
    # OBV trend
    obv_slope = get_feat('obv_slope')
    if obv_slope is not None:
        if obv_slope > 0: bullish.append("Positive OBV trend")
        elif obv_slope < 0: bearish.append("Negative OBV trend")
    
    return bullish[:5] or ["Momentum positive"], bearish[:5] or ["Caution advised"]

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app):
    load_models()
    yield

app = FastAPI(title="Stock Chat", version="1.2.1", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
intent_parser = IntentParser()
response_generator = ResponseGenerator()
feature_store = FeatureStore()
ensemble = EnsembleModel()
invalidation_engine = InvalidationEngine()
backtester = Backtester()
_models_loaded = False
_prediction_cache: Dict[str, Dict] = {}  # ticker -> {time, price, predictions}

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    predictions: Optional[Dict] = None
    ticker: Optional[str] = None

class PredictionRequest(BaseModel):
    ticker: str
    horizon: str = 'all'

def load_models():
    global _models_loaded
    if not _models_loaded:
        _models_loaded = ensemble.load()
    return _models_loaded

def get_predictions(ticker: str) -> Dict:
    """Get predictions for a ticker with caching and invalidation."""
    global _prediction_cache
    load_models()
    
    price_df = generate_ohlcv(ticker, "daily", 100)
    current_price = float(price_df['close'].iloc[-1])
    
    # Check cache
    if ticker in _prediction_cache:
        cached = _prediction_cache[ticker]
        vol_recent = price_df['close'].pct_change().tail(5).std()
        vol_hist = price_df['close'].pct_change().std()
        result = invalidation_engine.check(
            prediction_time=cached['time'], current_price=current_price,
            prediction_price=cached['price'], recent_volatility=vol_recent,
            historical_volatility=vol_hist
        )
        if result.is_valid:
            return cached['predictions']
    
    # Generate predictions
    options = get_options_chain(ticker, current_price)
    fundamentals = get_fundamentals(ticker)
    sentiment = get_sentiment(ticker)
    news = sentiment['news'].to_dict('records')
    macro = get_macro_data()
    
    features = feature_store.compute_all_features(price_df, options, fundamentals, news, macro)
    raw_pred = ensemble.predict_all_horizons(features.values)
    raw_pred = convert_numpy(raw_pred)
    
    # Build response with all required fields
    horizons = []
    for name, label in [('short', '1-Hour'), ('medium', '5-Day'), ('long', '60-Day')]:
        p = raw_pred[name]
        stop = current_price * (0.97 if p['direction'] == 'bullish' else 1.03)
        target = current_price * (1.05 if p['direction'] == 'bullish' else 0.95)  # noqa: F841
        horizons.append({
            'name': label, 'direction': p['direction'].upper(),
            'confidence': int(p['confidence'] * 100),
            'expected_return': round(p['prediction'] * 100, 2),
            'invalidation': f"Close {'below' if p['direction']=='bullish' else 'above'} ${stop:.2f}"
        })
    
    # Extract comprehensive signals from features
    bullish, bearish = extract_signals(features, current_price)
    
    # Get primary horizon (short-term) for top-level fields
    primary = horizons[0] if horizons else {}
    stop = current_price * (0.97 if primary.get('direction') == 'BULLISH' else 1.03)
    
    predictions = {
        'symbol': ticker, 'current_price': current_price,
        'bias': 'BUY' if primary.get('direction') == 'BULLISH' else ('SELL' if primary.get('direction') == 'BEARISH' else 'HOLD'),
        'confidence': primary.get('confidence'),
        'invalidation_price': round(stop, 2),
        'key_bullish_signals': bullish or ['Momentum positive'],
        'key_bearish_signals': bearish or ['Caution advised'],
        'horizons': horizons, 'bullish_signals': bullish or ['Momentum positive'],
        'bearish_signals': bearish or ['Caution advised'], 'raw': raw_pred
    }
    
    _prediction_cache[ticker] = {'time': datetime.now(), 'price': current_price, 'predictions': predictions}
    return predictions

@app.get("/health")
async def health():
    return {"status": "healthy", "models_loaded": _models_loaded}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process chat message and return response."""
    intent = intent_parser.parse(request.message)
    
    if intent.action == 'help':
        return ChatResponse(response=response_generator.generate(intent))
    
    if not intent.ticker:
        return ChatResponse(response="Please specify a stock ticker (e.g., 'Analyze TSLA')")
    
    try:
        predictions = get_predictions(intent.ticker)
        response = response_generator.generate(intent, predictions)
        return ChatResponse(response=response, predictions=predictions, ticker=intent.ticker)
    except Exception as e:
        return ChatResponse(response=f"Error analyzing {intent.ticker}: {str(e)}")

@app.post("/predict")
async def predict_post(request: PredictionRequest):
    """Get predictions for a ticker (POST)."""
    try:
        predictions = get_predictions(request.ticker)
        if request.horizon != 'all':
            return {"ticker": request.ticker, "horizon": request.horizon, "prediction": predictions.get(request.horizon)}
        return {"ticker": request.ticker, "predictions": predictions}
    except HTTPException:
        raise
    except Exception as e:
        _classify_http_error(e, request.ticker)

@app.get("/predict/{ticker}")
async def predict_get(ticker: str, horizon: str = 'all'):
    """Get predictions for a ticker (GET)."""
    try:
        predictions = get_predictions(ticker)
        cached = _prediction_cache.get(ticker, {})
        if horizon != 'all':
            return {"symbol": ticker, "ticker": ticker, "horizon": horizon, 
                    "current_price": cached.get('price', 0), "prediction": predictions.get(horizon)}
        return {"symbol": ticker, "ticker": ticker, "current_price": cached.get('price', 0), "predictions": predictions}
    except HTTPException:
        raise
    except Exception as e:
        _classify_http_error(e, ticker)

@app.get("/tickers")
async def get_tickers():
    """Get list of supported tickers."""
    return {"tickers": list(intent_parser.COMMON_TICKERS)}

@app.get("/backtest/{ticker}")
async def backtest(ticker: str):
    """Run backtest for a ticker."""
    import numpy as np
    try:
        price_df = generate_ohlcv(ticker, "daily", 365)
        options = get_options_chain(ticker, price_df['close'].iloc[-1])
        fundamentals = get_fundamentals(ticker)
        sentiment = get_sentiment(ticker)
        macro = get_macro_data()
        
        features = feature_store.compute_all_features(
            price_df, options, fundamentals, 
            sentiment['news'].to_dict('records'), macro
        )
        
        # Generate targets (next day returns)
        returns = price_df['close'].pct_change().shift(-1).fillna(0).values
        prices = price_df['close'].values
        
        # Get predictions for all rows (use confidence instead of probability)
        preds = np.array([ensemble.predict_all_horizons(features.iloc[i:i+1].values)['short']['confidence'] 
                         for i in range(len(features))])
        
        result = backtester.run(preds, returns, prices)
        return {"ticker": ticker, "backtest": result.to_dict()}
    except HTTPException:
        raise
    except Exception as e:
        _classify_http_error(e, ticker)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []
    
    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)
    
    def disconnect(self, ws: WebSocket):
        self.active.remove(ws)
    
    async def broadcast(self, message: str):
        for ws in self.active:
            await ws.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time chat."""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            
            intent = intent_parser.parse(msg.get('message', ''))
            
            if intent.action == 'help':
                response = response_generator.generate(intent)
                await websocket.send_json({"type": "response", "content": response})
                continue
            
            if not intent.ticker:
                await websocket.send_json({"type": "response", "content": "Please specify a ticker"})
                continue
            
            # Send processing status
            await websocket.send_json({"type": "status", "content": f"Analyzing {intent.ticker}..."})
            
            try:
                predictions = get_predictions(intent.ticker)
                response = response_generator.generate(intent, predictions)
                await websocket.send_json({
                    "type": "prediction",
                    "ticker": intent.ticker,
                    "content": response,
                    "predictions": predictions
                })
            except Exception as e:
                await websocket.send_json({"type": "error", "content": str(e)})
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Serve frontend static files
if FRONTEND_BUILD.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_BUILD / "static")), name="static")
    
    @app.get("/")
    async def serve_frontend():
        return FileResponse(str(FRONTEND_BUILD / "index.html"))
    
    @app.get("/{path:path}")
    async def serve_frontend_path(path: str):
        file_path = FRONTEND_BUILD / path
        if file_path.exists() and file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(FRONTEND_BUILD / "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
