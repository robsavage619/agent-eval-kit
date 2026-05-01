from __future__ import annotations

import pytest

from agent_eval_kit import Case, EvalRun, GoldenSet, diff_runs, run_evals
from agent_eval_kit.cost_tracker import CostTracker
from agent_eval_kit.judges import ExactMatchJudge, NumericToleranceJudge
from agent_eval_kit.regression import RegressionReport
from agent_eval_kit.reporters.markdown import write_markdown


def _make_golden(cases: list[dict]) -> GoldenSet:
    return GoldenSet([Case(**c) for c in cases])


async def _exact_agent(inp: dict) -> dict:
    return {"label": inp.get("label")}


async def _wrong_agent(inp: dict) -> dict:
    return {"label": "wrong"}


@pytest.mark.asyncio
async def test_exact_match_all_pass():
    golden = _make_golden(
        [
            {"id": "c1", "input": {"label": "A"}, "expected": {"label": "A"}},
            {"id": "c2", "input": {"label": "B"}, "expected": {"label": "B"}},
        ]
    )
    run = await run_evals(golden, _exact_agent, ExactMatchJudge())
    assert run.pass_rate == 1.0
    assert run.mean_score == 1.0


@pytest.mark.asyncio
async def test_exact_match_all_fail():
    golden = _make_golden(
        [
            {"id": "c1", "input": {"label": "A"}, "expected": {"label": "A"}},
        ]
    )
    run = await run_evals(golden, _wrong_agent, ExactMatchJudge())
    assert run.pass_rate == 0.0


@pytest.mark.asyncio
async def test_numeric_tolerance_pass():
    async def agent(inp: dict) -> dict:
        return {"value": inp["value"] * 1.03}  # 3% off — within 5% tolerance

    golden = _make_golden(
        [
            {"id": "n1", "input": {"value": 100.0}, "expected": {"value": 100.0}},
        ]
    )
    judge = NumericToleranceJudge(keys=["value"], rel_tol=0.05)
    run = await run_evals(golden, agent, judge)
    assert run.pass_rate == 1.0


@pytest.mark.asyncio
async def test_numeric_tolerance_fail():
    async def agent(inp: dict) -> dict:
        return {"value": inp["value"] * 1.20}  # 20% off — outside 5% tolerance

    golden = _make_golden(
        [
            {"id": "n1", "input": {"value": 100.0}, "expected": {"value": 100.0}},
        ]
    )
    judge = NumericToleranceJudge(keys=["value"], rel_tol=0.05)
    run = await run_evals(golden, agent, judge)
    assert run.pass_rate == 0.0


@pytest.mark.asyncio
async def test_regression_detection():
    golden = _make_golden(
        [
            {"id": "c1", "input": {"label": "A"}, "expected": {"label": "A"}},
            {"id": "c2", "input": {"label": "B"}, "expected": {"label": "B"}},
        ]
    )
    judge = ExactMatchJudge()
    baseline = await run_evals(golden, _exact_agent, judge)
    current = await run_evals(golden, _wrong_agent, judge)
    report: RegressionReport = diff_runs(baseline, current)
    assert report.has_regressions
    assert report.regression_count == 2
    assert report.improvement_count == 0


@pytest.mark.asyncio
async def test_cost_tracker():
    tracker = CostTracker()
    tracker.record("claude-haiku-4-5", 1000, 200)
    cost = tracker.pop_cost()
    assert cost > 0
    assert tracker.pop_cost() == 0.0  # reset after pop


def test_golden_set_filter():
    golden = _make_golden(
        [
            {"id": "c1", "input": {}, "expected": {}, "tags": ["spike"]},
            {"id": "c2", "input": {}, "expected": {}, "tags": ["clean"]},
        ]
    )
    filtered = golden.filter("spike")
    assert len(filtered) == 1


def test_markdown_reporter():
    golden = _make_golden(
        [
            {"id": "c1", "input": {"label": "A"}, "expected": {"label": "A"}},
        ]
    )

    import asyncio

    run = asyncio.get_event_loop().run_until_complete(
        run_evals(golden, _exact_agent, ExactMatchJudge())
    )
    md = write_markdown(run)
    assert "Pass rate" in md
    assert "c1" in md


@pytest.mark.asyncio
async def test_eval_run_from_results():
    golden = _make_golden(
        [
            {"id": "c1", "input": {"label": "A"}, "expected": {"label": "A"}},
            {"id": "c2", "input": {"label": "B"}, "expected": {"label": "B"}},
        ]
    )
    run = await run_evals(golden, _exact_agent, ExactMatchJudge(), concurrency=1)
    assert isinstance(run, EvalRun)
    assert len(run.results) == 2
    assert run.total_cost_usd >= 0
