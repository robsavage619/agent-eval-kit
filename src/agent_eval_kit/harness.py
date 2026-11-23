from __future__ import annotations

import asyncio
import os
import subprocess
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from .cost_tracker import CostTracker
from .golden_set import Case, GoldenSet
from .judges.base import Judge
from .stats import (
    bootstrap_mean_ci,
    clustered_standard_error,
    pass_at_k,
    pass_hat_k,
    standard_error,
    t_critical_95,
    wilson_interval,
)

CIMethod = Literal["wilson", "bootstrap", "clustered", "none"]


class EvalResult(BaseModel):
    """One trial of one case."""

    case_id: str
    trial: int = 0
    passed: bool
    score: float
    actual: dict[str, Any]
    expected: dict[str, Any]
    latency_ms: float
    cost_usd: float
    judge_reason: str = ""
    error: str | None = None
    tags: list[str] = Field(default_factory=list)
    cluster: str | None = None


class CaseSummary(BaseModel):
    """All trials of one case, rolled up. The unit of analysis for every statistic.

    ``trials`` counts scored trials only. Trials where the agent or the judge
    raised are counted in ``errors`` and kept out of every statistic: an API
    outage is not evidence that the agent got worse.
    """

    case_id: str
    trials: int
    passes: int
    mean_score: float
    errors: int = 0
    score_variance: float = 0.0
    tags: list[str] = Field(default_factory=list)
    cluster: str | None = None

    @property
    def pass_rate(self) -> float:
        return self.passes / self.trials if self.trials else 0.0

    @property
    def flaky(self) -> bool:
        """Passed on some trials and failed on others."""
        return 0 < self.passes < self.trials

    @property
    def passing(self) -> bool:
        """Passed on a majority of trials. With one trial, simply passed."""
        return self.pass_rate > 0.5


class RunSpec(BaseModel):
    """What produced this run, so `compare` can refuse to mix apples and oranges."""

    dataset: str = ""
    judge: str = ""
    judge_config: dict[str, Any] = Field(default_factory=dict)
    git_sha: str = ""
    kit_version: str = ""


class TagStats(BaseModel):
    tag: str
    n_cases: int
    pass_rate: float
    ci: tuple[float, float]


class EvalRun(BaseModel):
    results: list[EvalResult]
    total_cost_usd: float
    pass_rate: float
    mean_score: float
    mean_latency_ms: float
    name: str = ""
    created_at: str = ""
    repeats: int = 1
    cases: list[CaseSummary] = Field(default_factory=list)
    pass_rate_ci: tuple[float, float] = (0.0, 1.0)
    ci_method: CIMethod = "none"
    stderr: float = 0.0
    clustered_stderr: float | None = None
    n_clusters: int | None = None
    n_errors: int = 0
    spec: RunSpec = Field(default_factory=RunSpec)

    @classmethod
    def from_results(
        cls,
        results: list[EvalResult],
        repeats: int | None = None,
        name: str = "",
        spec: RunSpec | None = None,
    ) -> EvalRun:
        cases = _summarize(results)
        scored = [c for c in cases if c.trials > 0]
        if repeats is None:
            repeats = max((c.trials + c.errors for c in cases), default=1)
        ok = [r for r in results if r.error is None]
        rates = [c.pass_rate for c in scored]
        labels = [c.cluster or f"_case:{c.case_id}" for c in scored]
        n_clusters = len(set(labels))
        has_clusters = any(c.cluster is not None for c in scored) and n_clusters >= 2

        ci, method = _interval(scored, repeats)
        cse: float | None = None
        if has_clusters:
            cse = clustered_standard_error(rates, labels)
            mean_rate = sum(rates) / len(rates)
            half = t_critical_95(n_clusters - 1) * cse
            ci, method = (max(0.0, mean_rate - half), min(1.0, mean_rate + half)), "clustered"

        return cls(
            results=results,
            total_cost_usd=sum(r.cost_usd for r in results),
            pass_rate=sum(r.passed for r in ok) / len(ok) if ok else 0.0,
            mean_score=sum(r.score for r in ok) / len(ok) if ok else 0.0,
            mean_latency_ms=sum(r.latency_ms for r in results) / len(results) if results else 0.0,
            name=name,
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
            repeats=repeats,
            cases=cases,
            pass_rate_ci=ci,
            ci_method=method,
            stderr=standard_error(rates),
            clustered_stderr=cse,
            n_clusters=n_clusters if has_clusters else None,
            n_errors=len(results) - len(ok),
            spec=spec or RunSpec(),
        )

    @property
    def n_cases(self) -> int:
        return len(self.cases)

    @property
    def scored_cases(self) -> list[CaseSummary]:
        """Cases with at least one trial that did not error."""
        return [c for c in self.cases if c.trials > 0]

    @property
    def flaky_cases(self) -> list[CaseSummary]:
        return [c for c in self.cases if c.flaky]

    @property
    def pass_all_trials(self) -> float:
        """Share of cases that passed on every scored trial."""
        scored = self.scored_cases
        if not scored:
            return 0.0
        return sum(c.passes == c.trials for c in scored) / len(scored)

    def pass_hat_k(self, k: int) -> float:
        """Estimated chance that k fresh trials of a case all pass, averaged over cases.

        The number that matters for anything a user hits more than once: a case
        that passes 2 runs in 3 fails one user in three. Cases with fewer than
        k scored trials are left out.
        """
        eligible = [c for c in self.scored_cases if c.trials >= k]
        if not eligible:
            return 0.0
        return sum(pass_hat_k(c.passes, c.trials, k) for c in eligible) / len(eligible)

    def pass_at_k(self, k: int) -> float:
        """Estimated chance that at least one of k fresh trials passes."""
        eligible = [c for c in self.scored_cases if c.trials >= k]
        if not eligible:
            return 0.0
        return sum(pass_at_k(c.passes, c.trials, k) for c in eligible) / len(eligible)

    def reliability_curve(self) -> list[tuple[int, float, float]]:
        """(k, pass^k, pass@k) for k = 1 up to the smallest trial count."""
        kmax = min((c.trials for c in self.scored_cases), default=0)
        return [(k, self.pass_hat_k(k), self.pass_at_k(k)) for k in range(1, kmax + 1)]

    def by_tag(self) -> list[TagStats]:
        """Pass rate per tag, so a slice that broke is not hidden by the average."""
        tags = sorted({t for c in self.scored_cases for t in c.tags})
        out = []
        for tag in tags:
            group = [c for c in self.scored_cases if tag in c.tags]
            rate = sum(c.pass_rate for c in group) / len(group)
            ci, _ = _interval(group, self.repeats)
            out.append(TagStats(tag=tag, n_cases=len(group), pass_rate=rate, ci=ci))
        return out

    def summary(self, case_id: str) -> CaseSummary | None:
        return next((c for c in self.cases if c.case_id == case_id), None)


_MIN_CASES_FOR_BOOTSTRAP = 10


def _interval(cases: list[CaseSummary], repeats: int) -> tuple[tuple[float, float], CIMethod]:
    """95% interval for the mean per-case pass rate.

    - one trial per case: Wilson over cases
    - several trials, 10+ cases with some spread: bootstrap over whole cases
    - otherwise: Wilson over all trials, which contains the estimate but is
      somewhat narrow because trials of one case are correlated
    """
    if not cases:
        return (0.0, 1.0), "none"
    if repeats <= 1 and all(c.trials == 1 for c in cases):
        return wilson_interval(sum(c.passes for c in cases), len(cases)), "wilson"
    rates = [c.pass_rate for c in cases]
    if len(cases) >= _MIN_CASES_FOR_BOOTSTRAP and len(set(rates)) > 1:
        return bootstrap_mean_ci(rates), "bootstrap"
    return wilson_interval(sum(c.passes for c in cases), sum(c.trials for c in cases)), "wilson"


def _summarize(results: list[EvalResult]) -> list[CaseSummary]:
    grouped: dict[str, list[EvalResult]] = {}
    for r in results:
        grouped.setdefault(r.case_id, []).append(r)
    out = []
    for case_id, rs in grouped.items():
        ok = [r for r in rs if r.error is None]
        scores = [r.score for r in ok]
        mean = sum(scores) / len(scores) if scores else 0.0
        var = sum((s - mean) ** 2 for s in scores) / (len(scores) - 1) if len(scores) > 1 else 0.0
        out.append(
            CaseSummary(
                case_id=case_id,
                trials=len(ok),
                passes=sum(r.passed for r in ok),
                mean_score=mean,
                errors=len(rs) - len(ok),
                score_variance=var,
                tags=rs[0].tags,
                cluster=rs[0].cluster,
            )
        )
    return out


def _git_sha() -> str:
    sha = os.environ.get("GITHUB_SHA", "")
    if sha:
        return sha[:12]
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


AgentFn = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


async def _run_one(
    case: Case,
    trial: int,
    agent: AgentFn,
    judge: Judge,
    tracker: CostTracker,
) -> EvalResult:
    key = f"{case.id}#{trial}"
    error: str | None = None
    actual: dict[str, Any] = {}
    with tracker.case_scope(key):
        t0 = time.perf_counter()
        try:
            actual = await agent(case.input)
        except Exception as exc:  # an agent crash is a failed case, not a crashed run
            error = f"agent raised {type(exc).__name__}: {exc}"
        latency_ms = (time.perf_counter() - t0) * 1000

        if error is None:
            try:
                score, passed, reason = await judge.evaluate(case.expected, actual)
            except Exception as exc:  # a broken judge is visible, never a silent pass
                error = f"judge raised {type(exc).__name__}: {exc}"
                score, passed, reason = 0.0, False, error
        else:
            score, passed, reason = 0.0, False, error
        cost = tracker.pop_cost(key)

    return EvalResult(
        case_id=case.id,
        trial=trial,
        passed=passed,
        score=score,
        actual=actual,
        expected=case.expected,
        latency_ms=latency_ms,
        cost_usd=cost,
        judge_reason=reason,
        error=error,
        tags=case.tags,
        cluster=case.cluster,
    )


async def run_evals(
    golden_set: GoldenSet,
    agent: AgentFn,
    judge: Judge,
    tracker: CostTracker | None = None,
    concurrency: int = 4,
    repeats: int = 1,
    name: str = "",
) -> EvalRun:
    """Run every case ``repeats`` times and score each trial with ``judge``.

    Use ``repeats`` > 1 for anything with a model in the loop. One trial tells
    you whether a case can pass; several tell you whether it reliably does.
    """
    from . import __version__

    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    tracker = tracker or CostTracker()
    sem = asyncio.Semaphore(concurrency)

    async def bounded(case: Case, trial: int) -> EvalResult:
        async with sem:
            return await _run_one(case, trial, agent, judge, tracker)

    jobs = [bounded(c, t) for c in golden_set for t in range(repeats)]
    results = await asyncio.gather(*jobs)
    spec = RunSpec(
        dataset=golden_set.fingerprint(),
        judge=judge.fingerprint(),
        judge_config=judge.describe(),
        git_sha=_git_sha(),
        kit_version=__version__,
    )
    return EvalRun.from_results(list(results), repeats=repeats, name=name, spec=spec)
