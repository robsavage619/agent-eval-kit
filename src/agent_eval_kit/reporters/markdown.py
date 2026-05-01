from __future__ import annotations

from pathlib import Path

from ..harness import EvalRun


def write_markdown(run: EvalRun, path: str | Path | None = None) -> str:
    lines = [
        "# Eval Run Report\n",
        "| Metric | Value |",
        "|:---|:---|",
        f"| Pass rate | {run.pass_rate:.1%} |",
        f"| Mean score | {run.mean_score:.3f} |",
        f"| Mean latency | {run.mean_latency_ms:.0f} ms |",
        f"| Total cost | ${run.total_cost_usd:.4f} |",
        "",
        "## Results\n",
        "| Case | Pass | Score | Latency | Cost | Reason |",
        "|:---|:---:|---:|---:|---:|:---|",
    ]
    for r in run.results:
        status = "✅" if r.passed else "❌"
        lines.append(
            f"| {r.case_id} | {status} | {r.score:.2f} | {r.latency_ms:.0f}ms "
            f"| ${r.cost_usd:.4f} | {r.judge_reason[:80]} |"
        )

    md = "\n".join(lines) + "\n"
    if path:
        Path(path).write_text(md)
    return md
