# agent-eval-kit

> Run evals against your agents. Catch regressions before they ship.

[![CI](https://github.com/robsavage619/agent-eval-kit/actions/workflows/ci.yml/badge.svg)](https://github.com/robsavage619/agent-eval-kit/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Why this exists

Agent prompts drift silently. A well-intentioned change to a system prompt or tool schema can
break 20% of your golden cases without a single error in your logs. This kit makes that failure
visible before it ships — a lightweight eval harness you run in CI, not a monitoring platform you
pay for.

<br />

## Quickstart

```bash
uv add agent-eval-kit        # or: pip install agent-eval-kit
```

```python
import asyncio
from agent_eval_kit import GoldenSet, run_evals
from agent_eval_kit.judges import ExactMatchJudge
from agent_eval_kit.reporters import write_markdown

golden = GoldenSet.from_jsonl("evals/golden_set.jsonl")

async def my_agent(input: dict) -> dict:
    ...  # call your LangGraph graph or Claude agent here

async def main():
    run = await run_evals(golden, my_agent, ExactMatchJudge())
    print(write_markdown(run))
    assert run.pass_rate >= 0.9, f"Pass rate {run.pass_rate:.1%} below threshold"

asyncio.run(main())
```

<br />

## Engineering highlights

- **Eval harness with golden sets** — cases stored as JSONL, load and filter by tag, run concurrently with configurable semaphore
- **Three judge types** — `ExactMatchJudge` for deterministic outputs, `NumericToleranceJudge` for numeric fields (relative + absolute tolerance), `LLMJudge` (Claude Haiku) for semantic evaluation at low cost
- **Regression detection** — `diff_runs()` diffs two `EvalRun`s and surfaces cases that newly regressed or improved
- **Cost tracking** — per-case and total USD tracked via `CostTracker`; models tiered (Sonnet for agent, Haiku for judge) to minimize spend
- **CI-ready reporters** — Markdown for humans, JUnit XML for GitHub Actions / Jenkins test result integration

<br />

## What it does

```
GoldenSet  ──► run_evals() ──► EvalRun
                    │               │
               Judge (3 types)    diff_runs() ──► RegressionReport
                    │
              CostTracker ──► reporters (Markdown, JUnit)
```

**`harness.py`** — `run_evals(golden, agent, judge)` drives concurrent evaluation with a semaphore,
returns an `EvalRun` with per-case results, pass rate, mean score, and total cost.

**`regression.py`** — `diff_runs(baseline, current)` compares two runs and returns a
`RegressionReport` with explicit lists of regressions and improvements. Exit non-zero in CI if
`report.has_regressions`.

**`cost_tracker.py`** — thread-safe accumulator; call `tracker.record(model, in_tok, out_tok)`
after each LLM response, then `tracker.pop_cost()` once per eval case.

**`reporters/`** — `write_markdown()` returns a table for PR comments; `write_junit()` writes
XML that GitHub Actions can parse as test results.

<br />

## Examples

### [Cost Anomaly Agent](examples/cost_anomaly_agent/) — FinOps-flavored

A LangGraph agent that detects spend anomalies in AWS Cost & Usage Report data. Demonstrates:
- Synthetic CUR-style data generation (no real AWS account needed)
- 20-case golden set with seeded anomalies
- Custom judge that validates anomaly detection range
- Cost tracking across a multi-node graph

```bash
# Generate golden set (deterministic, no API key)
python examples/cost_anomaly_agent/generate_golden_set.py

# Run evals (requires ANTHROPIC_API_KEY for the recommendation node)
make cost-anomaly
```

### [NL-to-SQL Agent](examples/nl_to_sql_agent/) — generic data

A Claude-powered natural language to SQL agent over the public Chinook music database.
Demonstrates LLM-as-judge for semantic SQL correctness.

<br />

## Methodology

See [docs/why-eval.md](docs/why-eval.md) — a practical guide to building eval harnesses that
catch the failure modes that matter, without turning into a full-time monitoring project.

Key principles:
1. **Golden sets over spot checks** — 20 well-chosen cases beat ad-hoc testing
2. **Regression budget** — track pass rate over time; a 5% drop is a merge blocker
3. **Cheap judges first** — exact match → numeric tolerance → LLM judge, in that order
4. **Human in the loop at the boundary** — evals gate merges, humans gate deploys

<br />

## Configuration

No config file needed. Pass a `CostTracker` to cap spend:

```python
from agent_eval_kit.cost_tracker import CostTracker

tracker = CostTracker()
run = await run_evals(golden, agent, judge, tracker=tracker)
print(f"Total cost: ${tracker.total_usd:.4f}")
```

Set `concurrency` in `run_evals()` to control parallelism (default: 4).

<br />

## Stack

```
Python 3.12 · LangGraph · Anthropic Claude (Sonnet for agents, Haiku for judges)
uv · ruff · pyright · pytest · pydantic v2
```

<br />

---

<div align="center">

Built by [Rob Savage](https://github.com/robsavage619) · MIT License

</div>
