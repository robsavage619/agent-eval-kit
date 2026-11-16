# agent-eval-kit

[![CI](https://github.com/robsavage619/agent-eval-kit/actions/workflows/ci.yml/badge.svg)](https://github.com/robsavage619/agent-eval-kit/actions/workflows/ci.yml)
![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A small golden-set eval harness for LangGraph and Claude agents. You write test cases as JSONL, point the harness at an async agent function, pick a judge, and get a pass rate, per-case scores, latency, and token cost. Run it again after a prompt or model change and `diff_runs()` tells you exactly which cases went from passing to failing.

It is about 450 lines of library code. There is no server, dashboard, or hosted backend: it is meant to run in CI next to your tests.

## Why

Prompt edits, model upgrades, and tool-schema changes break agents quietly. Nothing throws; a slice of inputs just starts getting worse answers. A pass rate on its own hides that, because one fix can mask two new failures. What you need is the list of specific cases that regressed, on every commit. The longer argument is in [docs/why-eval.md](docs/why-eval.md).

## Install

Not on PyPI yet. Install from GitHub:

```bash
uv add git+https://github.com/robsavage619/agent-eval-kit
```

Requires Python 3.12+. `LLMJudge` and the examples need `ANTHROPIC_API_KEY`; the deterministic judges and the test suite don't.

## Quickstart

A golden set is one JSON object per line. `input` is passed to your agent, `expected` is passed to the judge, and `tags` let you run a subset. Blank lines and lines starting with `#` are skipped.

```jsonl
{"id": "refund_request", "input": {"message": "I want my money back"}, "expected": {"intent": "refund"}, "tags": ["critical"]}
{"id": "shipping_question", "input": {"message": "Where is my order?"}, "expected": {"intent": "order_status"}}
```

Your agent is any `async (dict) -> dict` function. Wrap a compiled LangGraph graph with `graph.ainvoke`, or call the Anthropic SDK directly.

```python
import asyncio

from agent_eval_kit import GoldenSet, run_evals
from agent_eval_kit.judges import ExactMatchJudge
from agent_eval_kit.reporters import write_markdown


async def my_agent(inp: dict) -> dict:
    ...  # return {"intent": ...}


async def main() -> None:
    golden = GoldenSet.from_jsonl("evals/golden_set.jsonl")
    run = await run_evals(golden, my_agent, ExactMatchJudge(), concurrency=4)
    print(write_markdown(run))
    assert run.pass_rate >= 0.9, f"pass rate {run.pass_rate:.1%} below 90%"


asyncio.run(main())
```

`run_evals` returns an `EvalRun` (a pydantic model) with `pass_rate`, `mean_score`, `mean_latency_ms`, `total_cost_usd`, and a `results` list holding each case's score, pass/fail, actual output, and the judge's reason.

## Judges

Every judge returns `(score 0-1, passed, reason)`. Use the cheapest one that can actually tell right from wrong.

| Judge | Passes when | Cost | Good for |
|:---|:---|:---|:---|
| `ExactMatchJudge(keys=None)` | every key in `expected` (or only `keys`) equals the same key in the output | free | labels, classifications, structured fields |
| `NumericToleranceJudge(keys, rel_tol=0.05, abs_tol=0.0)` | each listed field is within `max(abs_tol, rel_tol * abs(expected))` | free | costs, metrics, forecasts |
| `LLMJudge(criteria, threshold=0.7)` | Claude Haiku scores the output against your `criteria` at or above `threshold` | one Haiku call per case | free text, SQL, recommendations |

If the model's reply can't be parsed as JSON, `LLMJudge` scores the case 0 and puts the raw reply in the reason, so a broken judge shows up as failures instead of passing silently.

For anything else, subclass `Judge`:

```python
from typing import Any

from agent_eval_kit.judges import Judge


class CountInRange(Judge):
    async def evaluate(self, expected: dict[str, Any], actual: dict[str, Any]) -> tuple[float, bool, str]:
        n = actual.get("count", 0)
        ok = expected["min"] <= n <= expected["max"]
        return (1.0 if ok else 0.0), ok, f"count={n}"
```

## Catching regressions

Save a run as JSON on `main`, then diff later runs against it.

```python
import sys
from pathlib import Path

from agent_eval_kit import EvalRun, diff_runs

# on main
Path("baseline.json").write_text(run.model_dump_json())

# on your branch
baseline = EvalRun.model_validate_json(Path("baseline.json").read_text())
report = diff_runs(baseline, run)

for r in report.regressions:
    print(f"REGRESSED {r.case_id}: {r.baseline_score:.2f} -> {r.current_score:.2f}")
for r in report.improvements:
    print(f"fixed     {r.case_id}")

sys.exit(1 if report.has_regressions else 0)
```

A regression is a case that passed in the baseline and fails now. An improvement is the reverse. Everything else counts as unchanged, including score drops that don't cross the pass line.

## Cost tracking

`CostTracker` adds up tokens and dollars. Your agent calls `record()` after each model response; the harness calls `pop_cost()` after each case to attribute spend to it. Pass the same tracker to `LLMJudge` so judge calls are counted too.

```python
from agent_eval_kit.cost_tracker import CostTracker

tracker = CostTracker()
judge = LLMJudge(criteria="The answer is factually correct and cites a source.", tracker=tracker)


async def my_agent(inp: dict) -> dict:
    response = client.messages.create(model="claude-sonnet-4-6", ...)
    tracker.record("claude-sonnet-4-6", response.usage.input_tokens, response.usage.output_tokens)
    return {...}


run = await run_evals(golden, my_agent, judge, tracker=tracker)
print(f"${tracker.total_usd:.4f}, tokens in/out: {tracker.total_tokens}")
```

Prices live in a hand-maintained table at the top of [`cost_tracker.py`](src/agent_eval_kit/cost_tracker.py). Models missing from it are priced at $3 / $15 per million input/output tokens, so check the table when you change models.

## Reports and CI

`write_markdown(run, path=None)` returns a summary table plus one row per case, ready for a PR comment or a terminal. `write_junit(run, path)` writes JUnit XML, so each case shows up as a test in GitHub Actions, Jenkins, or GitLab.

```yaml
- name: Run evals
  run: uv run python evals/run.py   # your script: run_evals + write_junit("eval-results.xml") + diff_runs
  env:
    ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}

- name: Publish eval results
  if: always()
  uses: mikepenz/action-junit-report@v4
  with:
    report_paths: eval-results.xml
```

## Examples

**[Cost anomaly agent](examples/cost_anomaly_agent/)** is a three-node LangGraph graph over synthetic AWS cost data: compute mean and standard deviation, flag days above 2.5 sigma, then ask Claude Sonnet for a recommendation. Its 20 cases are generated from seeded data, so no AWS account is needed. It uses a custom judge that checks the anomaly count falls in an expected range.

```bash
uv run python examples/cost_anomaly_agent/generate_golden_set.py   # regenerate cases, no API key
make cost-anomaly                                                   # run evals, needs ANTHROPIC_API_KEY
```

**[NL-to-SQL agent](examples/nl_to_sql_agent/)** turns questions into SQLite queries against the public [Chinook](https://github.com/lerocha/chinook-database) music-store database and grades them with `LLMJudge`. See its README to download the database.

## Limitations

These come straight from the current code; worth knowing before you rely on it.

- If the agent or judge raises on any case, the whole run raises. Catch errors inside your agent function and return them as output if you want them scored as failures.
- `LLMJudge` uses the synchronous Anthropic client, so judge calls block the event loop and effectively run one at a time whatever `concurrency` is set to.
- Per-case cost comes from one shared counter. With `concurrency > 1`, a case can pick up tokens recorded by another case still in flight. Totals are exact; set `concurrency=1` if per-case cost matters.
- `diff_runs` only compares cases present in both runs. Added or removed cases are not reported.

## Development

```bash
make install     # uv sync --dev
make lint        # ruff check + ruff format --check
make typecheck   # pyright src/
make test        # pytest, offline, no API key
```

```
src/agent_eval_kit/
├── golden_set.py       Case, GoldenSet (JSONL load/save, tag filter)
├── harness.py          run_evals, EvalRun, EvalResult
├── regression.py       diff_runs, RegressionReport
├── cost_tracker.py     CostTracker and the pricing table
├── judges/             ExactMatchJudge, NumericToleranceJudge, LLMJudge, Judge base
└── reporters/          write_markdown, write_junit
```

MIT licensed. Built by [Rob Savage](https://github.com/robsavage619).
