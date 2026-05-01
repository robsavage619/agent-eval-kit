<div align="center">
<br />

```
 ██████╗ ██╗   ██╗ █████╗ ██╗          ███████╗██╗   ██╗ █████╗ ██╗
██╔════╝ ██║   ██║██╔══██╗██║          ██╔════╝██║   ██║██╔══██╗██║
██║  ███╗██║   ██║███████║██║    █████╗█████╗  ██║   ██║███████║██║
██║   ██║╚██╗ ██╔╝██╔══██║██║    ╚════╝██╔══╝  ╚██╗ ██╔╝██╔══██║██║
╚██████╔╝ ╚████╔╝ ██║  ██║███████╗    ███████╗  ╚████╔╝ ██║  ██║███████╗
 ╚═════╝   ╚═══╝  ╚═╝  ╚═╝╚══════╝    ╚══════╝   ╚═══╝  ╚═╝  ╚═╝╚══════╝
                     ██╗  ██╗██╗████████╗
                     ██║ ██╔╝██║╚══██╔══╝
                     █████╔╝ ██║   ██║
                     ██╔═██╗ ██║   ██║
                     ██║  ██╗██║   ██║
                     ╚═╝  ╚═╝╚═╝   ╚═╝
```

*Eval harness for LangGraph + Claude agents.*
*Catch regressions before they ship.*

<br />

[![CI](https://github.com/robsavage619/agent-eval-kit/actions/workflows/ci.yml/badge.svg?style=flat-square)](https://github.com/robsavage619/agent-eval-kit/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.12-1e1e2e?style=flat-square&labelColor=1e1e2e&color=cba6f7)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2-1e1e2e?style=flat-square&labelColor=1e1e2e&color=89b4fa)](https://github.com/langchain-ai/langgraph)
[![Claude](https://img.shields.io/badge/Claude-Haiku_Judge-1e1e2e?style=flat-square&labelColor=1e1e2e&color=f38ba8)](https://anthropic.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-1e1e2e?style=flat-square&labelColor=1e1e2e&color=a6e3a1)](LICENSE)

<br />
</div>

---

## The problem

Agent prompts drift silently. A well-intentioned change to a system prompt, tool schema, or model version can break 20% of your inputs without a single error in your logs. You find out from a user, not from CI.

This kit makes that failure visible before it ships — a lightweight eval harness you run on every commit, not a monitoring platform you pay for.

<br />

---

## Quickstart

```bash
uv add agent-eval-kit   # or: pip install agent-eval-kit
```

```python
import asyncio
from agent_eval_kit import GoldenSet, run_evals
from agent_eval_kit.judges import ExactMatchJudge
from agent_eval_kit.reporters import write_markdown

golden = GoldenSet.from_jsonl("evals/golden_set.jsonl")

async def my_agent(input: dict) -> dict:
    ...  # your LangGraph graph or Claude agent

async def main():
    run = await run_evals(golden, my_agent, ExactMatchJudge())
    print(write_markdown(run))
    assert run.pass_rate >= 0.90, f"pass rate {run.pass_rate:.1%} below threshold"

asyncio.run(main())
```

Done. That's the whole integration.

<br />

---

## Architecture

```
                         ┌─────────────────────────────────────┐
                         │           GoldenSet                  │
                         │   cases loaded from JSONL · tagged   │
                         └──────────────────┬──────────────────┘
                                            │
                                            ▼
┌───────────────┐        ┌─────────────────────────────────────┐
│  Your agent   │◄───────│           run_evals()                │
│  (async fn)   │        │  concurrent · semaphore-controlled   │
└───────┬───────┘        └──────────────────┬──────────────────┘
        │                                   │
        │ actual output                     │ per-case result
        ▼                                   ▼
┌───────────────┐        ┌─────────────────────────────────────┐
│     Judge     │        │            EvalRun                   │
│  (3 types)    │        │  pass_rate · mean_score · cost_usd   │
└───────┬───────┘        └──────────────────┬──────────────────┘
        │                                   │
        │ score · pass · reason             │
        ▼                                   ▼
┌───────────────┐        ┌─────────────────────────────────────┐
│  CostTracker  │        │          diff_runs()                 │
│  per-case USD │        │  regressions · improvements · delta  │
└───────────────┘        └──────────────────┬──────────────────┘
                                            │
                               ┌────────────┴──────────────┐
                               ▼                           ▼
                      Markdown report              JUnit XML
                      (PR comments)           (CI test results)
```

<br />

---

## What's in the box

<table>
<tr>
<td width="50%" valign="top">

### Core

**`harness.py`** — `run_evals(golden, agent, judge, concurrency=4)` drives concurrent evaluation with an asyncio semaphore. Returns an `EvalRun` with per-case results, pass rate, mean score, latency, and total cost.

**`golden_set.py`** — `GoldenSet.from_jsonl(path)` loads eval cases. Filter by tag (`golden.filter("critical")`). Save updated sets. Cases are pydantic-validated.

**`regression.py`** — `diff_runs(baseline, current)` compares two `EvalRun`s and returns a `RegressionReport` with explicit lists of regressions and improvements. Use `report.has_regressions` as a CI gate.

**`cost_tracker.py`** — Thread-safe accumulator. Call `tracker.record(model, in_tok, out_tok)` after each LLM response. `tracker.pop_cost()` returns and resets the per-case total. Model pricing table is in the file — update as needed.

</td>
<td width="50%" valign="top">

### Judges

**`ExactMatchJudge`** — Passes when every key in `expected` matches `actual`. Optionally scope to a subset of keys. Zero latency, zero cost. Use for classification, labels, structured JSON outputs.

**`NumericToleranceJudge`** — Passes when numeric fields are within relative or absolute tolerance. Configurable per-field. Use for cost calculations, metrics, predictions with acceptable error ranges.

**`LLMJudge`** — Claude Haiku-powered semantic judge. Scores 0–1, configurable pass threshold. Use for free-text outputs, SQL correctness, recommendation quality. ~$0.001 per case.

### Reporters

**`write_markdown(run)`** — Returns a formatted table suitable for PR comments or terminal output.

**`write_junit(run, path)`** — Writes JUnit XML that GitHub Actions, Jenkins, and most CI systems can parse as native test results with pass/fail counts.

</td>
</tr>
</table>

<br />

---

## Judge selection guide

| Output type | Judge | Cost | Latency |
|:---|:---|:---|:---|
| Classification label | `ExactMatchJudge` | Free | ~0ms |
| Structured JSON fields | `ExactMatchJudge(keys=[...])` | Free | ~0ms |
| Numeric value (cost, metric) | `NumericToleranceJudge` | Free | ~0ms |
| Free-text recommendation | `LLMJudge` | ~$0.001 | ~300ms |
| SQL query correctness | `LLMJudge` | ~$0.001 | ~300ms |
| Multi-criteria output | Compose judges — cheapest first | Varies | Varies |

The pattern: start with exact match or numeric tolerance, escalate to LLM judge only when the output genuinely requires semantic evaluation.

<br />

---

## Examples

### Cost Anomaly Agent — FinOps-flavored

A LangGraph agent that detects spend anomalies in AWS Cost & Usage Report data. No real AWS account needed — uses deterministic synthetic CUR-style data with seeded anomalies.

```
examples/cost_anomaly_agent/
├── synthetic_data.py       # generate fake CUR rows — seeded, reproducible
├── agent.py                # LangGraph graph: stats → detect → recommend
├── golden_set.jsonl        # 20 eval cases (clean baseline + known spikes)
└── run_evals.py            # end-to-end eval runner with custom judge
```

```bash
# Generate golden set (no API key — deterministic synthetic data)
uv run python examples/cost_anomaly_agent/generate_golden_set.py

# Run evals (ANTHROPIC_API_KEY required for the recommendation node)
make cost-anomaly
```

The agent graph:

```
compute_stats ──► detect_anomalies ──► recommend (Claude Sonnet) ──► END
      │                  │                       │
  mean + σ         z-score > 2.5σ          natural language
  calculation       → anomaly list           recommendation
```

<br />

### NL-to-SQL Agent — public Chinook database

A Claude-powered natural language to SQL agent over the [Chinook music database](https://github.com/lerocha/chinook-database) — a public sample schema with artists, albums, tracks, invoices, and customers.

Uses `LLMJudge` to evaluate semantic SQL correctness rather than string matching.

```bash
# Download Chinook SQLite (one-time)
curl -L https://github.com/lerocha/chinook-database/raw/master/ChinookDatabase/DataSources/Chinook_Sqlite.sqlite \
     -o examples/nl_to_sql_agent/chinook.db
```

```
examples/nl_to_sql_agent/
├── agent.py            # Claude Sonnet → SQL → execute → return rows
├── golden_set.jsonl    # 10 NL questions with structural SQL validators
└── README.md
```

<br />

---

## Regression workflow

The pattern for production use:

```bash
# 1. Run evals on main, save as baseline
uv run python -m agent_eval_kit.cli run --save baseline.json

# 2. Make your change (prompt edit, model version, tool schema)

# 3. Run evals on the change, diff against baseline
uv run python -m agent_eval_kit.cli run --diff baseline.json

# 4. CI exits non-zero if has_regressions == True
```

In GitHub Actions:

```yaml
- name: Run evals
  run: uv run pytest evals/ -v
  env:
    ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}

- name: Publish results
  uses: mikepenz/action-junit-report@v4
  with:
    report_paths: eval-results.xml
```

<br />

---

## Methodology

The longer argument for why you need this and how to do it right: [**docs/why-eval.md**](docs/why-eval.md).

Key principles:

| Principle | Why |
|:---|:---|
| **Golden sets over spot checks** | 20 well-chosen cases beat ad-hoc testing. Spot checks only cover what you thought to check. |
| **Regression budget, not 100% target** | Chasing 100% pass rate burns time on edge cases. Set a budget (≤2 new failures per PR) and enforce it. |
| **Cheapest judge first** | Exact match → tolerance → LLM judge. Most outputs can be validated for free. |
| **Evals gate merges, humans gate deploys** | Automate the obvious failures. Reserve human review for the cases that passed automation. |
| **Write cases when you write the agent** | The exercise forces you to specify what the agent is actually supposed to do — which is the hardest part. |

<br />

---

## Stack

```
Language    Python 3.12
Agent       LangGraph 0.2 · langchain-anthropic
AI          Claude Sonnet 4.6 (agents) · Claude Haiku 4.5 (judge)
Validation  pydantic v2
Tooling     uv · ruff · pyright · pytest · pytest-asyncio
```

<br />

---

<div align="center">

[Why Eval?](docs/why-eval.md) · [Cost Anomaly Example](examples/cost_anomaly_agent/) · [NL-to-SQL Example](examples/nl_to_sql_agent/)

<br />

*Built by [Rob Savage](https://github.com/robsavage619) · MIT License*

</div>
