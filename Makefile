.DEFAULT_GOAL := help

.PHONY: help install lint typecheck test demo figures cost-anomaly

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

demo: ## Run the offline support-agent demo: run, compare v2, calibrate
	uv run aek run examples/support_agent/evals.py --repeats 5
	uv run aek run examples/support_agent/evals.py:suite_v2 --repeats 5 --out /tmp/aek-v2.json -q
	-uv run aek compare examples/support_agent/baseline.json /tmp/aek-v2.json
	uv run aek calibrate examples/support_agent/evals.py:reply_suite examples/support_agent/labels.jsonl

figures: ## Regenerate the README chart from the demo baseline
	uv run python scripts/make_figures.py

cost-anomaly: ## Run the cost anomaly example end-to-end (needs ANTHROPIC_API_KEY)
	uv run python examples/cost_anomaly_agent/run_evals.py
