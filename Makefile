.DEFAULT_GOAL := help

.PHONY: help install lint typecheck test cost-anomaly

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install all deps via uv
	uv sync --dev

lint: ## Lint + format check
	uv run ruff check src/ tests/ examples/ && uv run ruff format --check src/ tests/ examples/

typecheck: ## Run pyright
	uv run pyright src/

test: ## Run test suite
	uv run pytest -v

cost-anomaly: ## Run the cost anomaly example end-to-end
	uv run python examples/cost_anomaly_agent/run_evals.py
