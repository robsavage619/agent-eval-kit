# agent-eval-kit

Rob's global conventions (`~/.claude/CLAUDE.md`) apply; this file is the project overlay.

A small Python package (`agent_eval_kit`, 0.1.0) for running golden-set evals against LangGraph/Claude agents: load cases from JSONL, run the agent concurrently, score with a judge, report, and diff two runs for regressions. Public GitHub repo (`robsavage619/agent-eval-kit`); users are developers wiring it into their own CI.

## Commands

Baseline 2026-11-15, run from the repo root with `.venv/bin/<tool>` (the Makefile wraps the same commands in `uv run`; `make` itself was not run). The venv is Python 3.13.13 although the project targets 3.12.

- Tests: `.venv/bin/pytest -q` -> 9 passed, 0.24s, no API key needed.
- Lint: `.venv/bin/ruff check --no-cache src/ tests/ examples/` -> 1 error today (I001 import order in `examples/cost_anomaly_agent/run_evals.py`, auto-fixable).
- Format: `.venv/bin/ruff format --no-cache --check src/ tests/ examples/` -> 4 files in `examples/cost_anomaly_agent/` would be reformatted.
- Types: `.venv/bin/pyright src/` -> 11 errors today, all `.text` access on non-text content blocks at `src/agent_eval_kit/judges/llm_as_judge.py:58`.
- So CI's lint and typecheck steps fail on the current tree; fixing is not part of unrelated work.
- Not run: `make cost-anomaly` and the NL-to-SQL example (need `ANTHROPIC_API_KEY`, paid calls; the latter also needs a downloaded Chinook DB).

## Layout

- `src/agent_eval_kit/harness.py`: `run_evals()` and `EvalRun`; `golden_set.py`: JSONL cases; `regression.py`: `diff_runs()`; `cost_tracker.py`: per-case cost and the model pricing table.
- `src/agent_eval_kit/judges/`: `ExactMatchJudge`, `NumericToleranceJudge`, `LLMJudge` (Claude Haiku), shared `base.py`.
- `src/agent_eval_kit/reporters/`: `markdown.py`, `junit.py`.
- `tests/test_harness.py`: the only test file.
- `examples/`: `cost_anomaly_agent/` (synthetic data, 20 cases) and `nl_to_sql_agent/` (10 cases); not part of the wheel.
- `docs/why-eval.md`: rationale essay. `README.md` is the user-facing reference.

## Load-bearing rules

- Config lives in `pyproject.toml`: ruff line length 100, rules E/F/I/UP/B/SIM with E501 ignored, pyright basic, pytest `asyncio_mode = "auto"`.
- CI (`.github/workflows/ci.yml`) runs lint, `pyright src/`, and `pytest` on pushes and PRs to `main`; keep local checks identical to those commands.
- Judge cost ordering (README, "Judge selection guide"): prefer exact-match or numeric-tolerance judges, use `LLMJudge` only for free-text output.
- README says to update the pricing table in `cost_tracker.py` as models change; it is hand-maintained.
- Do not call the Anthropic API from tests; the current tests are offline.

## Git

Conventional commits (`feat:`, `docs:` in the log). Commit to the current default branch (`main`). Ask before pushing.
