# agent-eval-kit

Rob's global conventions (`~/.claude/CLAUDE.md`) apply; this file is the project overlay.

A Python package (`agent_eval_kit`, 0.2.0) and CLI (`aek`) for golden-set evals of LLM agents that report how far to trust their own numbers: confidence intervals, repeated trials with pass^k / pass@k, noise-aware run comparison (exact McNemar, critical-case gating, minimum detectable effect), judge calibration against human labels, and trajectory (tool-call) evals. Public GitHub repo (`robsavage619/agent-eval-kit`); users are developers wiring it into their own CI. The repo root is also a composite GitHub Action (`action.yml`).

## Commands

- Tests: `uv run pytest -q` (offline, no API key).
- Lint: `uv run ruff check src/ tests/ examples/` and `uv run ruff format --check src/ tests/ examples/`.
- Types: `uv run pyright src/`.
- Demo (no API key): `uv run aek run examples/support_agent/evals.py --repeats 5`, `...evals.py:suite_v2`, `aek compare examples/support_agent/baseline.json <run.json>`, `aek calibrate examples/support_agent/evals.py:reply_suite examples/support_agent/labels.jsonl`.
- README chart: `python scripts/make_figures.py` regenerates `docs/assets/reliability.svg` from `examples/support_agent/baseline.json`.
- Not run in CI: `make cost-anomaly` and the NL-to-SQL example (need `ANTHROPIC_API_KEY`, paid calls; NL-to-SQL also needs the Chinook DB).

## Layout

- `src/agent_eval_kit/stats.py`: Wilson, exact McNemar, bootstrap, kappa, pass^k / pass@k, clustered SE, power. Every function has a reference-value test in `tests/test_stats.py`.
- `harness.py`: `run_evals()` (repeats, per-case cost scope, agent/judge errors captured), `EvalRun`, `CaseSummary`, `RunSpec`.
- `compare.py`: `compare_runs()` and `min_flips()`; `regression.py`: the 0.1 `diff_runs()`, kept for compatibility.
- `calibration.py`: `calibrate_judge()`, `load_labels()`.
- `trajectory.py`: `TrajectoryJudge`, `extract_tool_calls()`, `Step`.
- `judges/`: `ExactMatchJudge`, `NumericToleranceJudge`, `LLMJudge`, `AllOf`/`AnyOf`/`Not`/`Weighted`; `base.py` holds `describe()`/`fingerprint()`.
- `reporters/`: markdown (run, comparison, calibration) and JUnit.
- `suite.py`: `Suite` and `load_suite("path.py:var")`; `cli.py`: `aek run|compare|calibrate|power|report`.
- `examples/support_agent/`: offline demo (simulated agent, v1 and v2, 25 cases, 40 labels, committed `baseline.json`). `examples/cost_anomaly_agent/` and `nl_to_sql_agent/` call real models.
- `docs/methods.md`: every formula and threshold, with references. Keep it in sync with `stats.py`, `compare.py`, `calibration.py`.

## Load-bearing rules

- Statistics are computed on per-case summaries, never pooled trials. Do not change that without updating `docs/methods.md`.
- `pydantic` is the only runtime dependency; `anthropic` is the `llm` extra and is imported lazily in `LLMJudge`.
- `LLMJudge` computes pass/fail from the score in code and raises `JudgeError` on unparseable replies; never turn that back into a silent 0.
- Regenerate `examples/support_agent/baseline.json` and the README chart whenever demo behaviour or report format changes, and update the README numbers to match real output.
- Do not call the Anthropic API from tests.
- Prose (README, docs) follows Rob's human-voice rules: no em dashes, no hype words.

## Git

Conventional commits. Work on a branch and ask before pushing.
