from __future__ import annotations

from pydantic import BaseModel

from .harness import EvalResult, EvalRun


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
    base_map: dict[str, EvalResult] = {r.case_id: r for r in baseline.results}
    curr_map: dict[str, EvalResult] = {r.case_id: r for r in current.results}

    regressions: list[Regression] = []
    improvements: list[Regression] = []
    unchanged = 0

    for case_id, curr in curr_map.items():
        base = base_map.get(case_id)
        if base is None:
            continue
        delta = curr.score - base.score
        rec = Regression(
            case_id=case_id,
            baseline_passed=base.passed,
            current_passed=curr.passed,
            baseline_score=base.score,
            current_score=curr.score,
            delta=delta,
        )
        if base.passed and not curr.passed:
            regressions.append(rec)
        elif not base.passed and curr.passed:
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
