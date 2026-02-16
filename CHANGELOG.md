# Changelog

## 1.2.3 — Type Annotations & Refactoring

- Full parameter type annotations on all engine.py helper functions (Callable, pd.DataFrame, np.ndarray)
- Extracted `_eval_op` helper from `scan_tickers` to reduce complexity
- 6 new tests for `_eval_op` operator evaluation
- 705 automated tests

## 1.2.2 — Test Coverage & Developer Docs

- Added CONTRIBUTING.md with development workflow, code style, and project structure
- 15 new tests: LSTM save/load, edge cases (untrained predict, short input, 1D input, empty sequences), ModelTrainer init, walk-forward backtesting (proba model, insufficient data), BacktestResult serialization, all-bullish/all-bearish scenarios
- 699 automated tests

## 1.2.1 — API Error Handling & Quality

- API endpoints now return proper HTTP status codes:
  - 404 for invalid/unknown tickers
  - 429 for rate-limited requests
  - 503 for network/data source failures
  - 500 for unexpected errors
- Migrated from deprecated `on_event("startup")` to FastAPI lifespan
- Fixed version consistency across setup.py, CLI, and CHANGELOG
- Frontend: replaced deprecated `onKeyPress` with `onKeyDown`, added aria-label
- Added `py.typed` marker for PEP 561 type checking support
- Added `Makefile` for common dev tasks (test, lint, install, clean, dev)
- 6 new integration tests: retry backoff timing, multi-ticker analysis, provider fallback
- 684 automated tests

## 1.2.0 — Engine Refactoring & Quality

- Refactored `get_analysis` into smaller helper functions:
  - `_extract_signals`: signal extraction from 40+ technical indicators
  - `_build_horizons`: prediction horizon construction with entry/exit levels
  - `_get_ticker_info`: company info lookup with graceful fallback
  - `_check_threshold`: reusable threshold comparison for signal generation
- 14 new unit tests for refactored helpers
- 10 advanced signal extraction tests
- 655 automated tests

## 1.1.0 — Invalid Ticker Detection & Error Polish

- Invalid tickers now raise clear errors instead of silently falling back to mock data
- Suppressed noisy yfinance HTTP error output for cleaner UX
- Added 5 new integration tests for invalid ticker handling across commands
- 631 automated tests

## 1.0.1 — Polish & Shell Completion

- Added `stk completion` command for bash/zsh/fish shell completion
- Added `--install` flag to auto-install completion to shell config
- Added `NoDataError` and `ConfigError` for better error classification
- Updated README with shell completion documentation
- 626 automated tests

## 1.0.0 — Real Data + CLI Release

### Real Data Integration
- Replaced all 5 mock data generators with real API providers
  - Yahoo Finance: price, options, fundamentals, analyst ratings
  - FRED API: GDP, CPI, fed funds rate, VIX, unemployment
  - NewsAPI + VADER: news sentiment analysis
  - SEC EDGAR: company filings
- File-based cache with configurable TTL per data type
- Graceful fallback to mock data on API failures
- All 204 features compute on real data

### ML Models
- Retrained XGBoost (short/medium/long) + LSTM on real historical data
- Ensemble model combines all predictions
- Backtesting on real 2-year data

### CLI (`stk`)
- 50+ commands for analysis, portfolio, watchlist, alerts, and more
- SQLite storage for positions, watchlist, alerts, predictions
- Rich terminal output with tables, panels, colors
- Key commands: analyze, price, news, earnings, chat, dashboard
- Portfolio: hold, sell, positions, risk, optimize, rebalance, stress, diversification
- Watchlist: watch, unwatch, watchlist, top-movers, momentum
- Alerts: add, remove, check, auto, smart, schedule, pause/resume
- Screening: scan, screen, sector, heatmap, compare, correlate
- Utilities: export, replay, backtest, history, doctor, config

### API Server
- All endpoints return real data
- WebSocket support for real-time chat
- Error handling for API failures and rate limits

### Frontend
- Loading indicators for real data fetches
- Data freshness timestamps
- Prediction cards with buy/sell/hold verdicts

### Testing
- 618 automated tests across 16 test files
- Integration tests for CLI → engine → backend flow
- Edge case and error handling coverage
