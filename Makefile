.PHONY: test lint install dev clean help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install package in editable mode
	pip install -e .

test: ## Run all tests
	python -m pytest tests/ -q

test-v: ## Run tests with verbose output
	python -m pytest tests/ -v

test-cov: ## Run tests with coverage
	python -m pytest tests/ --cov=cli --cov=backend --cov-report=term-missing -q

lint: ## Run type checking
	python -m mypy cli/ backend/ --ignore-missing-imports --no-error-summary 2>/dev/null || true

clean: ## Remove caches and build artifacts
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf .pytest_cache *.egg-info build dist
	rm -rf .mypy_cache

dev: ## Start API server in dev mode
	cd backend && python -m uvicorn api.server:app --reload --port 8000

demo: ## Run demo prediction
	python scripts/demo_prediction.py

backtest: ## Run backtest
	python scripts/run_backtest.py

doctor: ## Run system health check
	stk doctor
