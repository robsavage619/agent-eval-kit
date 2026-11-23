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


@pytest.mark.asyncio
async def test_markdown_reporter():
    golden = _make_golden(
        [
            {"id": "c1", "input": {"label": "A"}, "expected": {"label": "A"}},
        ]
    )
    run = await run_evals(golden, _exact_agent, ExactMatchJudge())
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


@pytest.mark.asyncio
async def test_repeats_detect_flaky_case():
    calls: dict[str, int] = {}

    async def flaky_agent(inp: dict) -> dict:
        n = calls.get(inp["label"], 0)
        calls[inp["label"]] = n + 1
        if inp["label"] == "B" and n % 2 == 1:
            return {"label": "wrong"}
        return {"label": inp["label"]}

    golden = _make_golden(
        [
            {"id": "steady", "input": {"label": "A"}, "expected": {"label": "A"}},
            {"id": "wobbly", "input": {"label": "B"}, "expected": {"label": "B"}},
        ]
    )
    run = await run_evals(golden, flaky_agent, ExactMatchJudge(), repeats=4, concurrency=1)
    assert len(run.results) == 8
    assert [c.case_id for c in run.flaky_cases] == ["wobbly"]
    assert run.pass_rate == 6 / 8
    assert run.pass_all_trials == 0.5
    # Two cases is too few to bootstrap; Wilson over trials is used instead.
    assert run.ci_method == "wilson"
    assert run.pass_rate_ci[0] <= run.pass_rate <= run.pass_rate_ci[1]


@pytest.mark.asyncio
async def test_pass_rate_ci_is_wilson_for_single_trial():
    golden = _make_golden(
        [{"id": f"c{i}", "input": {"label": "A"}, "expected": {"label": "A"}} for i in range(10)]
    )
    run = await run_evals(golden, _exact_agent, ExactMatchJudge())
    lo, hi = run.pass_rate_ci
    assert run.ci_method == "wilson"
    assert hi == pytest.approx(1.0)
    assert 0.65 < lo < 0.75  # 10/10 still leaves real doubt


@pytest.mark.asyncio
async def test_agent_exception_becomes_failed_case():
    async def boom(inp: dict) -> dict:
        raise RuntimeError("tool timed out")

    golden = _make_golden([{"id": "c1", "input": {}, "expected": {"label": "A"}}])
    run = await run_evals(golden, boom, ExactMatchJudge())
    result = run.results[0]
    assert not result.passed
    assert result.error is not None and "tool timed out" in result.error


@pytest.mark.asyncio
async def test_cost_is_attributed_to_the_case_that_spent_it():
    import asyncio

    tracker = CostTracker()

    async def agent(inp: dict) -> dict:
        # The expensive case finishes first; under the 0.1 tracker the cheap case
        # could pop the expensive case's spend.
        await asyncio.sleep(0.01 if inp["label"] == "cheap" else 0)
        tokens = 1_000_000 if inp["label"] == "pricey" else 1_000
        tracker.record("claude-sonnet-4-6", tokens, 0)
        await asyncio.sleep(0.02 if inp["label"] == "pricey" else 0)
        return {"label": inp["label"]}

    golden = _make_golden(
        [
            {"id": "pricey", "input": {"label": "pricey"}, "expected": {"label": "pricey"}},
            {"id": "cheap", "input": {"label": "cheap"}, "expected": {"label": "cheap"}},
        ]
    )
    run = await run_evals(golden, agent, ExactMatchJudge(), tracker=tracker, concurrency=2)
    cost = {r.case_id: r.cost_usd for r in run.results}
    assert cost["pricey"] == pytest.approx(3.0)
    assert cost["cheap"] == pytest.approx(0.003)


def test_duplicate_case_ids_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        _make_golden(
            [
                {"id": "c1", "input": {}, "expected": {}},
                {"id": "c1", "input": {}, "expected": {}},
            ]
        )


@pytest.mark.asyncio
async def test_run_round_trips_through_json():
    golden = _make_golden([{"id": "c1", "input": {"label": "A"}, "expected": {"label": "A"}}])
    run = await run_evals(golden, _exact_agent, ExactMatchJudge(), repeats=2, name="demo")
    again = EvalRun.model_validate_json(run.model_dump_json())
    assert again.cases[0].passes == 2
    assert again.pass_rate_ci == run.pass_rate_ci
    assert again.name == "demo"


@pytest.mark.asyncio
async def test_reliability_curve_tags_and_spec():
    calls = {"n": 0}

    async def agent(inp: dict) -> dict:
        calls["n"] += 1
        if inp["label"] == "B" and calls["n"] % 3 == 0:
            return {"label": "wrong"}
        return {"label": inp["label"]}

    golden = _make_golden(
        [
            {"id": "a", "input": {"label": "A"}, "expected": {"label": "A"}, "tags": ["easy"]},
            {"id": "b", "input": {"label": "B"}, "expected": {"label": "B"}, "tags": ["hard"]},
        ]
    )
    run = await run_evals(golden, agent, ExactMatchJudge(), repeats=3, concurrency=1)
    curve = run.reliability_curve()
    assert [k for k, _, _ in curve] == [1, 2, 3]
    assert curve[0][1] == pytest.approx(run.pass_rate)
    assert curve[-1][1] == pytest.approx(run.pass_all_trials)
    assert {t.tag for t in run.by_tag()} == {"easy", "hard"}
    assert run.spec.dataset == golden.fingerprint()
    assert run.spec.judge == ExactMatchJudge().fingerprint()


@pytest.mark.asyncio
async def test_clusters_widen_the_interval():
    cases = [
        {
            "id": f"t{t}_{i}",
            "input": {"label": "A" if t == 0 else "B"},
            "expected": {"label": "A"},
            "cluster": f"t{t}",
        }
        for t in range(2)
        for i in range(10)
    ]
    run = await run_evals(_make_golden(cases), _exact_agent, ExactMatchJudge())
    assert run.ci_method == "clustered"
    assert run.n_clusters == 2
    assert run.clustered_stderr is not None and run.clustered_stderr > run.stderr


def test_judge_fingerprint_tracks_config():
    assert ExactMatchJudge(keys=["a"]).fingerprint() == ExactMatchJudge(keys=["a"]).fingerprint()
    assert ExactMatchJudge(keys=["a"]).fingerprint() != ExactMatchJudge(keys=["b"]).fingerprint()
    assert NumericToleranceJudge(["x"], rel_tol=0.05).fingerprint() != (
        NumericToleranceJudge(["x"], rel_tol=0.1).fingerprint()
    )


def test_interval_always_contains_the_estimate():
    from agent_eval_kit import EvalResult

    # Every case at 2/3: the old fallback produced (0.72, 1.00) around 0.667.
    results = [
        EvalResult(
            case_id=f"c{i}",
            trial=t,
            passed=t < 2,
            score=float(t < 2),
            actual={},
            expected={},
            latency_ms=1.0,
            cost_usd=0.0,
        )
        for i in range(12)
        for t in range(3)
    ]
    run = EvalRun.from_results(results)
    assert run.repeats == 3  # inferred from the data
    lo, hi = run.pass_rate_ci
    assert lo <= run.pass_rate <= hi
    for tag in run.by_tag():
        assert tag.ci[0] <= tag.pass_rate <= tag.ci[1]


@pytest.mark.asyncio
async def test_errors_are_excluded_from_statistics():
    calls = {"n": 0}

    async def sometimes_down(inp: dict) -> dict:
        calls["n"] += 1
        if calls["n"] % 2 == 0:
            raise ConnectionError("upstream 529")
        return {"label": inp["label"]}

    golden = _make_golden(
        [{"id": f"c{i}", "input": {"label": "A"}, "expected": {"label": "A"}} for i in range(4)]
    )
    run = await run_evals(golden, sometimes_down, ExactMatchJudge(), repeats=2, concurrency=1)
    assert run.n_errors == 4
    assert run.pass_rate == 1.0  # the 4 trials that ran all passed


@pytest.mark.asyncio
async def test_few_clusters_use_t_and_one_cluster_is_ignored():
    one = [
        {"id": f"x{i}", "input": {"label": "A"}, "expected": {"label": "A"}, "cluster": "only"}
        for i in range(6)
    ]
    run = await run_evals(_make_golden(one), _exact_agent, ExactMatchJudge())
    assert run.ci_method == "wilson" and run.n_clusters is None
