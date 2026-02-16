# Contributing

## Setup

```bash
git clone https://github.com/sree-charan/stk.git
cd stock-chat-assistant
pip install -e .
stk doctor          # verify everything works
```

## Development Workflow

1. Create a branch: `git checkout -b feature/my-change`
2. Make changes
3. Run tests: `make test`
4. Commit and push

## Project Structure

- `backend/data/real_providers/` — API integrations (Yahoo Finance, FRED, NewsAPI, SEC EDGAR)
- `backend/data/mock_generators/` — Fallback mock data for offline/testing
- `backend/features/` — 8 feature tiers (204 features total)
- `backend/models/` — XGBoost, LSTM, Ensemble
- `backend/api/` — FastAPI server
- `cli/` — Click CLI (`stk` command)
- `tests/` — All test files

## Running Tests

```bash
make test           # quick run
make test-v         # verbose
make test-cov       # with coverage report
```

## Adding a CLI Command

1. Add the command function in `cli/main.py`
2. Use `@cli.command()` decorator
3. Use `rich` for output (Console, Table, Panel)
4. Add tests in `tests/test_cli.py`

## Adding a Data Provider

1. Create provider in `backend/data/real_providers/`
2. Add cache TTL in the provider
3. Add fallback to mock data on failure
4. Add tests in `tests/test_providers.py`

## Code Style

- Use type hints for function signatures
- Handle errors gracefully — no raw tracebacks in CLI
- Cache API responses to avoid rate limits
- Follow existing naming conventions (snake_case for functions/variables)

## Make Targets

```bash
make help       # list all targets
make install    # pip install -e .
make test       # run tests
make lint       # type checking
make clean      # remove caches
make dev        # start API server
make demo       # run demo prediction
make backtest   # run backtest
make doctor     # system health check
```
