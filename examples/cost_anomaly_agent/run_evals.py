#!/usr/bin/env python3
"""Run evals for the cost anomaly agent and print a markdown report."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from agent_eval_kit import GoldenSet, run_evals
from agent_eval_kit.cost_tracker import CostTracker
from agent_eval_kit.judges.exact_match import ExactMatchJudge
from agent_eval_kit.reporters.markdown import write_markdown

from agent import run_agent


class AnomalyJudge(ExactMatchJudge):
    """Pass when detected anomaly count is within the expected min/max range."""

    async def evaluate(self, expected: dict[str, Any], actual: dict[str, Any]) -> tuple[float, bool, str]:
        actual_count = actual.get("anomaly_count", 0)
        min_count = expected.get("anomaly_count_min", 0)
        max_count = expected.get("anomaly_count_max", 99)
        has_anomalies_exp = expected.get("has_anomalies", False)
        has_anomalies_act = actual_count > 0

        if has_anomalies_exp != has_anomalies_act:
            return 0.0, False, f"has_anomalies={has_anomalies_act} expected={has_anomalies_exp}"
        if not (min_count <= actual_count <= max_count):
            return 0.5, False, f"anomaly_count={actual_count} not in [{min_count},{max_count}]"
        return 1.0, True, f"anomaly_count={actual_count} in [{min_count},{max_count}]"


async def main() -> None:
    golden = GoldenSet.from_jsonl(Path(__file__).parent / "golden_set.jsonl")
    tracker = CostTracker()

    async def agent_fn(inp: dict[str, Any]) -> dict[str, Any]:
        return await run_agent(inp, tracker)

    run = await run_evals(golden, agent_fn, AnomalyJudge(), tracker=tracker, concurrency=4)

    md = write_markdown(run)
    print(md)
    print(f"\nTotal API cost: ${tracker.total_usd:.4f}")

    if run.pass_rate < 0.8:
        print(f"\n⚠  Pass rate {run.pass_rate:.1%} below 80% threshold", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
