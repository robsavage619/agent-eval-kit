from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel

from .cost_tracker import CostTracker
from .golden_set import Case, GoldenSet
from .judges.base import Judge


class EvalResult(BaseModel):
    case_id: str
    passed: bool
    score: float
    actual: dict[str, Any]
    expected: dict[str, Any]
    latency_ms: float
    cost_usd: float
    judge_reason: str = ""


class EvalRun(BaseModel):
    results: list[EvalResult]
    total_cost_usd: float
    pass_rate: float
    mean_score: float
    mean_latency_ms: float

    @classmethod
    def from_results(cls, results: list[EvalResult]) -> EvalRun:
        n = len(results)
        return cls(
            results=results,
            total_cost_usd=sum(r.cost_usd for r in results),
            pass_rate=sum(r.passed for r in results) / n if n else 0.0,
            mean_score=sum(r.score for r in results) / n if n else 0.0,
            mean_latency_ms=sum(r.latency_ms for r in results) / n if n else 0.0,
        )


AgentFn = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


async def _run_one(
    case: Case,
    agent: AgentFn,
    judge: Judge,
    tracker: CostTracker,
) -> EvalResult:
    t0 = time.perf_counter()
    actual = await agent(case.input)
    latency_ms = (time.perf_counter() - t0) * 1000

    score, passed, reason = await judge.evaluate(case.expected, actual)
    cost = tracker.pop_cost()

    return EvalResult(
        case_id=case.id,
        passed=passed,
        score=score,
        actual=actual,
        expected=case.expected,
        latency_ms=latency_ms,
        cost_usd=cost,
        judge_reason=reason,
    )


async def run_evals(
    golden_set: GoldenSet,
    agent: AgentFn,
    judge: Judge,
    tracker: CostTracker | None = None,
    concurrency: int = 4,
) -> EvalRun:
    tracker = tracker or CostTracker()
    sem = asyncio.Semaphore(concurrency)

    async def bounded(case: Case) -> EvalResult:
        async with sem:
            return await _run_one(case, agent, judge, tracker)

    results = await asyncio.gather(*[bounded(c) for c in golden_set])
    return EvalRun.from_results(list(results))
