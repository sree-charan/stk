# Stock Chat

ML-powered stock analysis with real market data, FastAPI backend, React frontend, and CLI.

## Features

- 204 features across 8 tiers (price/volume, technical, options, fundamentals, sentiment, institutional, macro, microstructure)
- ML models: XGBoost (short/medium/long) + LSTM + Ensemble
- Real data from Yahoo Finance, FRED, NewsAPI, SEC EDGAR
- FastAPI backend with REST + WebSocket
- React frontend with real-time chat
- CLI (`stk`) with 50+ commands for analysis, portfolio, watchlist, alerts, and more
- Backtesting on real historical data
- 705 automated tests

## Quick Start

```bash
# Install
pip install -e .

# Start API server
cd backend && python -m uvicorn api.server:app --reload --port 8000

# Use CLI
stk analyze TSLA
stk price AAPL
stk chat "should I buy GOOGL?"
```

## Setup

### Requirements
- Python 3.10+
- pip

### API Keys

Yahoo Finance and SEC EDGAR require no keys. For full functionality, configure:

```bash
stk config set fred-key YOUR_FRED_API_KEY    # FRED macro data
stk config set news-key YOUR_NEWSAPI_KEY     # NewsAPI sentiment
```

Get keys from:
- FRED: https://fred.stlouisfed.org/docs/api/api_key.html
- NewsAPI: https://newsapi.org/register

### Installation

```bash
git clone https://github.com/sree-charan/stk.git
cd stock-chat
pip install -e .

# Verify installation
stk --version
stk doctor          # check system health
stk doctor --fix    # auto-fix common issues
```

## CLI Usage

### Analysis

```bash
stk analyze TSLA              # full analysis (all timeframes)
stk analyze TSLA --short      # short-term only (1 hour)
stk analyze TSLA --medium     # medium-term only (5 days)
stk analyze TSLA --long       # long-term only (60 days)
stk analyze TSLA --verbose    # include raw feature values
stk analyze TSLA --json       # JSON output
stk analyze TSLA --mock       # offline mode with mock data
```

### Quick Actions

```bash
stk price AAPL                # current price + change
stk news TSLA                 # recent news with sentiment scores
stk earnings MSFT             # valuation + recent quarters
```

### Portfolio Management

```bash
stk hold TSLA --entry 250 --qty 10   # add position
stk sell TSLA                         # remove position
stk positions                         # show all with live P&L
stk positions --sort pnl              # sort by P&L
stk positions --group-by sector       # group by sector
stk positions-export --format csv     # export to CSV/JSON
stk positions-import positions.csv    # import from file
stk position-history TSLA             # price history for position
```

### Portfolio Analytics

```bash
stk portfolio-risk                 # VaR, beta, volatility, Sharpe
stk portfolio-optimize             # min-variance, max-Sharpe weights
stk portfolio-rebalance            # trades needed to rebalance
stk portfolio-rebalance --strategy equal_weight
stk portfolio-correlation          # position correlation matrix
stk portfolio-diversification      # sector breakdown + grade
stk portfolio-stress               # stress test all scenarios
stk portfolio-stress -s market_crash  # specific scenario
stk portfolio-performance          # performance metrics
stk portfolio-summary              # combined overview
stk portfolio-history              # value over time
stk portfolio-history --save       # save snapshot
stk tax-report                     # capital gains report
stk what-if NVDA --qty 50          # simulate adding position
stk what-if TSLA --qty 5 --remove  # simulate removing
```

### Watchlist

```bash
stk watch NVDA                # add to watchlist
stk unwatch NVDA              # remove
stk watchlist                 # show with current prices
stk watchlist --signals       # include buy/sell/hold signals
stk watchlist --sort change   # sort by price change
stk watchlist-export          # export watchlist
stk watchlist-import list.txt # import watchlist
```

### Alerts

```bash
stk alerts add TSLA --above 300   # alert when price goes above
stk alerts add AAPL --below 170   # alert when price drops below
stk alerts list                    # show active alerts
stk alerts list --all              # include paused alerts
stk alerts check                   # check alerts against live prices
stk alerts remove 1               # remove alert by ID
stk alerts clear                   # remove all alerts
stk alerts auto TSLA              # auto-set stop-loss + target alerts
stk alerts smart TSLA             # AI-suggested alert levels
stk alerts pause 1                # pause alert
stk alerts resume 1               # resume alert
stk alerts history                # alert trigger history
stk alerts stats                  # alert statistics
stk alerts schedule               # scheduled alert checks
stk alerts export                 # export alerts
stk alerts import alerts.json     # import alerts
```

### Screening & Scanning

```bash
stk screen                         # oversold stocks in top 10
stk screen --criteria volume       # volume spikes
stk scan 'rsi<30 AND volume>2'     # flexible filter expressions
stk scan --preset oversold         # use saved preset
stk scan --watch                   # scan watchlist only
stk scan-save myfilter 'rsi<30'    # save preset
stk scan-delete myfilter           # delete preset
stk scan-presets                   # list saved presets
```

### Compare & Correlate

```bash
stk compare TSLA AAPL GOOGL        # side-by-side comparison
stk compare TSLA AAPL --chart      # with sparkline charts
stk compare TSLA AAPL --correlation # with correlation matrix
stk correlate TSLA AAPL            # correlation analysis
stk correlate TSLA AAPL --days 90  # custom period
```

### Replay & Backtest

```bash
stk replay TSLA --date 2025-06-15  # historical prediction replay
stk replay-range TSLA --start 2025-01-01 --end 2025-06-01
stk backtest TSLA --days 180       # backtest with metrics
```

### Chat

```bash
stk chat "is AAPL a buy?"         # natural language query
stk chat "should I sell TSLA?"    # sell analysis
stk chat "compare AAPL and MSFT"  # comparison query
```

### Dashboard & Overview

```bash
stk dashboard                 # live TUI with alerts (Ctrl+C to exit)
stk summary                   # combined portfolio + watchlist
stk heatmap                   # sector performance heatmap
stk momentum                  # momentum ranking
stk top-movers                # biggest movers from watchlist
stk sector Technology         # sector-level screening
stk history TSLA              # prediction history with accuracy
```

### Export & Utilities

```bash
stk export TSLA --format html -o tsla.html  # export analysis
stk cache-clean                    # remove stale cache entries
stk doctor                         # system health check
stk doctor --fix                   # auto-fix issues
```

### Config

```bash
stk config set fred-key <key>      # set FRED API key
stk config set news-key <key>      # set NewsAPI key
stk config get fred-key            # get config value
stk config show                    # show all config (keys masked)
stk config list                    # alias for show
stk config reset                   # reset to defaults
```

## Data Sources

| Source | Data | Key Required |
|--------|------|-------------|
| Yahoo Finance (yfinance) | Price, options, fundamentals, analyst ratings | No |
| FRED API | GDP, CPI, rates, VIX, unemployment | Yes |
| NewsAPI | News headlines for sentiment analysis | Yes |
| SEC EDGAR | Company filings | No |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/chat` | POST | Chat with assistant |
| `/predict/{ticker}` | GET | Get predictions |
| `/predict` | POST | Get predictions (POST) |
| `/backtest/{ticker}` | GET | Run backtest |
| `/tickers` | GET | List supported tickers |
| `/ws` | WebSocket | Real-time chat |

## Architecture

```
stock-chat/
├── backend/
│   ├── api/              # FastAPI server
│   ├── data/
│   │   ├── real_providers/  # Yahoo Finance, FRED, NewsAPI, SEC EDGAR
│   │   └── mock_generators/ # Fallback mock data (offline mode)
│   ├── features/         # 8 feature tiers (204 total)
│   ├── llm/              # Intent parsing + response generation
│   ├── models/           # XGBoost, LSTM, Ensemble
│   └── utils/            # Backtesting, invalidation, config
├── cli/                  # Click CLI with rich formatting
│   ├── main.py           # All CLI commands
│   ├── engine.py         # Analysis engine (signal extraction, horizons, portfolio)
│   ├── db.py             # SQLite for positions/watchlist/alerts
│   ├── config.py         # Config management (~/.stk/config.json)
│   └── errors.py         # Custom error types (InvalidTickerError, NetworkError, etc.)
├── frontend/             # React chat UI
├── tests/                # 705 test cases
└── scripts/              # Demo prediction, backtest
```

## Testing

```bash
python -m pytest tests/ -v         # 705 tests
python -m pytest tests/ -q         # quick summary
python scripts/demo_prediction.py  # real TSLA prediction
python scripts/run_backtest.py     # 2-year backtest
```

## Caching

API responses are cached in `~/.stk/cache/` with TTL:
- Price data: 5 minutes
- Options: 10 minutes
- Sentiment: 15 minutes
- Fundamentals: 1 hour
- Macro data: 1 hour

If an API call fails, the system falls back to mock data with a staleness warning.

## Error Handling

The system handles failures gracefully:
- **Invalid tickers**: Clear error message, no traceback
- **Network failures**: Falls back to cached/mock data with warning
- **Rate limits**: Automatic retry with backoff, cached data as fallback
- **Missing data**: Features compute with available data, NaN-safe calculations

## Shell Completion

Enable tab completion for all `stk` commands:

```bash
# Bash
stk completion bash --install   # adds to ~/.bashrc
# or manually:
eval "$(_STK_COMPLETE=bash_source stk)"

# Zsh
stk completion zsh --install    # adds to ~/.zshrc

# Fish
stk completion fish --install   # adds to ~/.config/fish/completions/
```

## Troubleshooting

### Common Issues

**`stk` command not found**
```bash
pip install -e .   # reinstall in editable mode
# or run directly:
python -m cli.main --help
```

**API key errors (FRED/NewsAPI)**
```bash
stk config set fred-key YOUR_KEY
stk config set news-key YOUR_KEY
stk doctor   # verify connectivity
```

**Stale or missing data**
```bash
stk cache-clean          # clear expired cache
stk analyze TSLA --mock  # use mock data as fallback
```

**Yahoo Finance rate limiting**
- The system caches price data for 5 minutes. If you see errors, wait and retry.
- Use `stk doctor` to check API connectivity.

**Database issues**
```bash
stk doctor --fix   # recreates database and directories if needed
```

**Import errors after update**
```bash
pip install -e .   # reinstall to pick up new dependencies
```

## License

MIT
