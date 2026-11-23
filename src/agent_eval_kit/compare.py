"""Compare two runs without mistaking noise for a regression.

A golden set of 20 cases moves by 5 points every time one case flips, and a
model at temperature > 0 flips cases on its own. ``compare_runs`` separates
the things a plain pass-rate diff mixes together:

- cases that were already flaky in the baseline, which say nothing about the
  change and are set aside,
- cases that now fail on a majority of trials (regressed) or now pass
  (improved),
- cases that passed every baseline trial and now fail some but still pass
  most (degraded): reported, and fatal for critical cases, but not counted in
  the significance test,
- cases with errors, which are reported and excluded,
- whether the net movement is larger than chance (exact McNemar test).

Cases tagged ``critical`` are held to a stricter rule: any drop in pass rate
fails the gate.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .harness import CaseSummary, EvalRun
from .stats import bootstrap_mean_ci, mcnemar_exact, min_detectable_effect, standard_error

Status = Literal[
    "regressed", "improved", "degraded", "unchanged", "flaky", "error", "new", "removed"
]
Verdict = Literal["regression", "improvement", "no_detectable_change"]


class CaseChange(BaseModel):
    case_id: str
    status: Status
    baseline_pass_rate: float | None
    current_pass_rate: float | None
    baseline_score: float | None
    current_score: float | None
    critical: bool = False


class Comparison(BaseModel):
    baseline_name: str = ""
    current_name: str = ""
    baseline_pass_rate: float
    current_pass_rate: float
    baseline_ci: tuple[float, float]
    current_ci: tuple[float, float]
    score_delta: float
    score_delta_ci: tuple[float, float]
    score_delta_se: float = 0.0
    min_detectable_effect: float = 0.0
    """Smallest true change in mean score a paired test on these cases would
    catch 80% of the time (Miller 2024). An estimate, not a guarantee."""
    flip_floor: int = 0
    """Fewest cases that must flip the same way before the exact McNemar test
    can call anything significant at ``alpha``."""
    n_paired: int = 0
    regressed: int
    improved: int
    degraded: int = 0
    flaky: int
    errored: int = 0
    p_value: float
    alpha: float
    verdict: Verdict
    critical_regressions: list[str] = Field(default_factory=list)
    changes: list[CaseChange] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def should_fail(self) -> bool:
        """Fail a CI gate on a real regression, or on any critical case getting worse."""
        return self.verdict == "regression" or bool(self.critical_regressions)

    def cases_with(self, status: Status) -> list[CaseChange]:
        return [c for c in self.changes if c.status == status]


def min_flips(alpha: float = 0.05) -> int:
    """Fewest one-directional flips the exact McNemar test can call significant."""
    b = 1
    while mcnemar_exact(b, 0) >= alpha:
        b += 1
    return b


def _status(base: CaseSummary, curr: CaseSummary) -> Status:
    if base.trials == 0 or curr.trials == 0:
        return "error"
    if base.flaky:
        return "flaky"
    if base.passing and not curr.passing:
        return "regressed"
    if not base.passing and curr.passing:
        return "improved"
    if curr.pass_rate < base.pass_rate:
        return "degraded"
    return "unchanged"


def compare_runs(
    baseline: EvalRun,
    current: EvalRun,
    alpha: float = 0.05,
    critical_tag: str = "critical",
    seed: int = 0,
) -> Comparison:
    base_cases = {c.case_id: c for c in baseline.cases}
    curr_cases = {c.case_id: c for c in current.cases}

    changes: list[CaseChange] = []
    paired: list[tuple[CaseSummary, CaseSummary]] = []
    for case_id in list(base_cases) + [k for k in curr_cases if k not in base_cases]:
        base = base_cases.get(case_id)
        curr = curr_cases.get(case_id)
        either = curr or base
        critical = either is not None and critical_tag in either.tags
        if base is None or curr is None:
            status: Status = "new" if base is None else "removed"
        else:
            status = _status(base, curr)
            if status != "error":
                paired.append((base, curr))
        changes.append(
            CaseChange(
                case_id=case_id,
                status=status,
                baseline_pass_rate=base.pass_rate if base and base.trials else None,
                current_pass_rate=curr.pass_rate if curr and curr.trials else None,
                baseline_score=base.mean_score if base and base.trials else None,
                current_score=curr.mean_score if curr and curr.trials else None,
                critical=critical,
            )
        )

    deltas = [c.mean_score - b.mean_score for b, c in paired]
    n = len(paired)

    # Power. The observed per-case differences already contain each run's
    # sampling noise, so their variance is used on its own. When the runs barely
    # differ that variance is near zero, so floor it at the noise you would see
    # between two runs of the *same* agent.
    diff_var = 0.0
    if n > 1:
        mean_d = sum(deltas) / n
        diff_var = sum((d - mean_d) ** 2 for d in deltas) / (n - 1)
    if baseline.repeats > 1 or current.repeats > 1:
        noise = sum(
            b.score_variance / max(b.trials, 1) + c.score_variance / max(c.trials, 1)
            for b, c in paired
        ) / max(n, 1)
    else:
        # One trial per case measures no sampling noise; use the binomial
        # variance at a smoothed baseline pass rate as a stand-in.
        passes = sum(b.passes for b, _ in paired)
        p = (passes + 1) / (n + 2)
        noise = 2 * p * (1 - p)
    mde = min_detectable_effect(n, max(diff_var, noise))

    warnings: list[str] = []
    bs, cs = baseline.spec, current.spec
    if bs.dataset and cs.dataset and bs.dataset != cs.dataset:
        warnings.append(
            f"golden set changed ({bs.dataset} -> {cs.dataset}); only shared cases are compared"
        )
    if bs.judge and cs.judge and bs.judge != cs.judge:
        warnings.append(
            f"judge config changed ({bs.judge} -> {cs.judge}); score changes may come from the judge"
        )
    if baseline.repeats != current.repeats:
        warnings.append(
            f"trials per case differ ({baseline.repeats} vs {current.repeats}); "
            "flakiness is easier to see with more trials"
        )
    errored = sum(c.status == "error" for c in changes)
    if errored or current.n_errors:
        warnings.append(
            f"{current.n_errors} trial(s) errored in the current run; "
            f"{errored} case(s) could not be compared"
        )

    regressed = sum(c.status == "regressed" for c in changes)
    improved = sum(c.status == "improved" for c in changes)
    p_value = mcnemar_exact(regressed, improved)
    if p_value < alpha and regressed > improved:
        verdict: Verdict = "regression"
    elif p_value < alpha and improved > regressed:
        verdict = "improvement"
    else:
        verdict = "no_detectable_change"

    critical_regressions = [
        c.case_id
        for c in changes
        if c.critical
        and c.baseline_pass_rate is not None
        and c.current_pass_rate is not None
        and c.current_pass_rate < c.baseline_pass_rate
    ]

    return Comparison(
        baseline_name=baseline.name,
        current_name=current.name,
        baseline_pass_rate=baseline.pass_rate,
        current_pass_rate=current.pass_rate,
        baseline_ci=baseline.pass_rate_ci,
        current_ci=current.pass_rate_ci,
        score_delta=sum(deltas) / n if n else 0.0,
        score_delta_ci=bootstrap_mean_ci(deltas, seed=seed),
        score_delta_se=standard_error(deltas),
        min_detectable_effect=mde,
        flip_floor=min_flips(alpha),
        n_paired=n,
        regressed=regressed,
        improved=improved,
        degraded=sum(c.status == "degraded" for c in changes),
        flaky=sum(c.status == "flaky" for c in changes),
        errored=errored,
        p_value=p_value,
        alpha=alpha,
        verdict=verdict,
        critical_regressions=critical_regressions,
        changes=changes,
        warnings=warnings,
    )
