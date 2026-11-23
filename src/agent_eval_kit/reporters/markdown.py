from __future__ import annotations

from pathlib import Path

from ..calibration import CalibrationReport
from ..compare import Comparison
from ..harness import EvalRun


def _pct(x: float) -> str:
    return f"{x:.1%}"


def _ci(ci: tuple[float, float]) -> str:
    return f"{ci[0]:.0%} to {ci[1]:.0%}"


def _clip(text: str, n: int = 90) -> str:
    text = " ".join(text.split()).replace("|", "\\|")
    return text if len(text) <= n else text[: n - 1] + "…"


def _write(md: str, path: str | Path | None) -> str:
    if path:
        Path(path).write_text(md)
    return md


def write_markdown(run: EvalRun, path: str | Path | None = None) -> str:
    """Report for a single run: headline with its uncertainty, then every case."""
    title = f"# Eval run: {run.name}" if run.name else "# Eval run"
    lines = [
        title,
        "",
        "| Metric | Value |",
        "|:---|:---|",
        f"| Cases | {run.n_cases} x {run.repeats} trial{'s' if run.repeats != 1 else ''} |",
        f"| Pass rate | **{_pct(run.pass_rate)}** (95% CI {_ci(run.pass_rate_ci)}) |",
    ]
    if run.repeats > 1:
        lines += [
            f"| Passed every trial | {_pct(run.pass_all_trials)} |",
            f"| Flaky cases | {len(run.flaky_cases)} |",
        ]
    lines += [
        f"| Mean score | {run.mean_score:.3f} |",
        f"| Mean latency | {run.mean_latency_ms:.0f} ms |",
        f"| Total cost | ${run.total_cost_usd:.4f} |",
    ]
    if run.n_errors:
        lines.append(f"| Errored trials (excluded) | **{run.n_errors}** |")
    lines.append("")
    if run.ci_method == "clustered" and run.n_clusters is not None:
        lines += [
            f"> The interval accounts for {run.n_clusters} clusters of related cases "
            f"(clustered SE {run.clustered_stderr:.3f} vs naive {run.stderr:.3f}).",
            "",
        ]
    elif run.n_cases and run.n_cases < 30:
        lines += [
            f"> With {run.n_cases} cases the interval is wide: one case is "
            f"{1 / run.n_cases:.0%} of the pass rate.",
            "",
        ]

    curve = run.reliability_curve()
    if len(curve) > 1:
        lines += [
            "## Reliability",
            "",
            "Estimated from every trial (unbiased, as in tau-bench and Chen et al. 2021).",
            "pass^k: a case passes *k times in a row*. pass@k: *at least once* in k tries.",
            "",
            "| k | pass^k | pass@k |",
            "|---:|---:|---:|",
        ]
        lines += [f"| {k} | {_pct(hat)} | {_pct(at)} |" for k, hat, at in curve]
        lines.append("")

    tags = run.by_tag()
    if len(tags) > 1:
        lines += ["## By tag", "", "| Tag | Cases | Pass rate | 95% CI |", "|:---|---:|---:|:---|"]
        for ts in tags:
            lines.append(f"| `{ts.tag}` | {ts.n_cases} | {_pct(ts.pass_rate)} | {_ci(ts.ci)} |")
        lines.append("")

    flaky = {c.case_id for c in run.flaky_cases}
    lines += ["## Cases", "", "| Case | Result | Score | Reason |", "|:---|:---:|---:|:---|"]
    for case in run.cases:
        trials = [r for r in run.results if r.case_id == case.case_id]
        if case.trials == 0:
            result = f"💥 {case.errors} error{'s' if case.errors != 1 else ''}"
        elif case.case_id in flaky:
            result = f"⚠️ {case.passes}/{case.trials}"
        elif case.passes == case.trials:
            result = "✅" if case.trials == 1 else f"✅ {case.passes}/{case.trials}"
        else:
            result = "❌" if case.trials == 1 else f"❌ {case.passes}/{case.trials}"
        failing = next((r for r in trials if not r.passed), trials[0])
        lines.append(
            f"| `{case.case_id}` | {result} | {case.mean_score:.2f} "
            f"| {_clip(failing.judge_reason)} |"
        )
    return _write("\n".join(lines) + "\n", path)


_VERDICT = {
    "regression": "❌ **Regression.** More cases broke than chance explains.",
    "improvement": "✅ **Improvement.** More cases were fixed than chance explains.",
    "no_detectable_change": "➖ **No detectable change.** Any movement is within noise.",
}


def write_comparison_markdown(cmp: Comparison, path: str | Path | None = None) -> str:
    """Report for baseline vs current, written to work as a PR comment."""
    lines = ["# Eval comparison", "", _VERDICT[cmp.verdict], ""]
    for warning in cmp.warnings:
        lines += [f"> ⚠️ {warning}", ""]
    if cmp.critical_regressions:
        names = ", ".join(f"`{c}`" for c in cmp.critical_regressions)
        lines += [f"🚨 **Critical case regressed:** {names}", ""]
    lines += [
        "| | Baseline | Current |",
        "|:---|---:|---:|",
        f"| Pass rate | {_pct(cmp.baseline_pass_rate)} | {_pct(cmp.current_pass_rate)} |",
        f"| 95% CI | {_ci(cmp.baseline_ci)} | {_ci(cmp.current_ci)} |",
        "",
        f"Cases regressed: **{cmp.regressed}**, improved: **{cmp.improved}**, "
        f"degraded: **{cmp.degraded}**, flaky in baseline (set aside): **{cmp.flaky}**. "
        f"Exact McNemar p = {cmp.p_value:.3f} (alpha {cmp.alpha}). Mean score change "
        f"{cmp.score_delta:+.3f} (95% CI {cmp.score_delta_ci[0]:+.3f} to "
        f"{cmp.score_delta_ci[1]:+.3f}).",
        "",
        "**What this comparison can see.** With "
        f"{cmp.n_paired} paired cases, a paired test would catch a true change of about "
        f"**{cmp.min_detectable_effect:.0%}** in mean score 80% of the time. Separately, the "
        f"exact test cannot call anything significant until at least **{cmp.flip_floor}** "
        f"cases flip the same way ({cmp.flip_floor / max(cmp.n_paired, 1):.0%} of these "
        "cases). Smaller changes can hide inside a 'no detectable change'.",
        "",
    ]
    sections = [
        ("regressed", "Regressed"),
        ("degraded", "Degraded (passed every baseline trial, now fails some)"),
        ("improved", "Improved"),
        ("flaky", "Flaky in the baseline (set aside)"),
        ("error", "Errored (not compared)"),
        ("new", "New cases"),
        ("removed", "Removed cases"),
    ]
    for status, heading in sections:
        rows = cmp.cases_with(status)  # type: ignore[arg-type]
        if not rows:
            continue
        lines += [f"### {heading}", "", "| Case | Baseline | Current |", "|:---|---:|---:|"]
        for c in rows:
            b = "" if c.baseline_pass_rate is None else _pct(c.baseline_pass_rate)
            n = "" if c.current_pass_rate is None else _pct(c.current_pass_rate)
            flag = " 🚨" if c.case_id in cmp.critical_regressions else ""
            lines.append(f"| `{c.case_id}`{flag} | {b} | {n} |")
        lines.append("")
    return _write("\n".join(lines), path)


_TRUST = {
    "trust": "✅ **Trust.** The judge agrees with people and rarely passes bad output.",
    "use_with_care": "⚠️ **Use with care.** Agreement is decent; read its failures.",
    "do_not_trust": "❌ **Do not trust.** The judge disagrees with people too often.",
    "insufficient_data": "➖ **Not enough labels** to say. Label more examples.",
}


def write_calibration_markdown(rep: CalibrationReport, path: str | Path | None = None) -> str:
    lines = [
        "# Judge calibration",
        "",
        _TRUST[rep.verdict],
        "",
        "| | Human: pass | Human: fail |",
        "|:---|---:|---:|",
        f"| **Judge: pass** | {rep.true_pass} | {rep.false_pass} |",
        f"| **Judge: fail** | {rep.false_fail} | {rep.true_fail} |",
        "",
        "| Metric | Value |",
        "|:---|:---|",
        f"| Labeled examples | {rep.n} |",
        f"| Agreement | {_pct(rep.agreement)} (95% CI {_ci(rep.agreement_ci)}) |",
        f"| Cohen's kappa | {rep.kappa:.2f} |",
        f"| False-pass rate | **{_pct(rep.false_pass_rate)}** (95% CI {_ci(rep.false_pass_ci)}) |",
        f"| False-fail rate | {_pct(rep.false_fail_rate)} |",
        f"| Judge fingerprint | `{rep.judge}` |",
        "",
    ]
    for warning in rep.warnings:
        lines += [f"> ⚠️ {warning}", ""]
    if rep.disagreements:
        lines += ["## Disagreements", "", "| Example | Kind | Judge said | Notes |"]
        lines += ["|:---|:---|:---|:---|"]
        for d in sorted(rep.disagreements, key=lambda d: d.kind != "false_pass"):
            kind = "false pass" if d.kind == "false_pass" else "false fail"
            lines.append(
                f"| `{d.id}` | {kind} | {_clip(d.judge_reason, 60)} | {_clip(d.notes, 60)} |"
            )
        lines.append("")
    return _write("\n".join(lines), path)
