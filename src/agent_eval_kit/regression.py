"""Case-level diff between two runs (the 0.1 API).

Every flipped case counts. For a verdict that accounts for flaky cases and
small golden sets, use ``compare_runs`` in ``agent_eval_kit.compare``.
"""

from __future__ import annotations

from pydantic import BaseModel

from .harness import EvalRun


class Regression(BaseModel):
    case_id: str
    baseline_passed: bool
    current_passed: bool
    baseline_score: float
    current_score: float
    delta: float


class RegressionReport(BaseModel):
    regressions: list[Regression]
    improvements: list[Regression]
    unchanged: int
    regression_count: int
    improvement_count: int

    @property
    def has_regressions(self) -> bool:
        return self.regression_count > 0


def diff_runs(baseline: EvalRun, current: EvalRun) -> RegressionReport:
    base_map = {c.case_id: c for c in baseline.cases}
    regressions: list[Regression] = []
    improvements: list[Regression] = []
    unchanged = 0

    for curr in current.cases:
        base = base_map.get(curr.case_id)
        if base is None:
            continue
        rec = Regression(
            case_id=curr.case_id,
            baseline_passed=base.passing,
            current_passed=curr.passing,
            baseline_score=base.mean_score,
            current_score=curr.mean_score,
            delta=curr.mean_score - base.mean_score,
        )
        if base.passing and not curr.passing:
            regressions.append(rec)
        elif not base.passing and curr.passing:
            improvements.append(rec)
        else:
            unchanged += 1

    return RegressionReport(
        regressions=regressions,
        improvements=improvements,
        unchanged=unchanged,
        regression_count=len(regressions),
        improvement_count=len(improvements),
    )
