from __future__ import annotations

from agent_eval_kit import EvalResult, EvalRun, compare_runs
from agent_eval_kit.reporters import write_comparison_markdown


def _run(outcomes: dict[str, list[bool]], tags: dict[str, list[str]] | None = None) -> EvalRun:
    tags = tags or {}
    results = [
        EvalResult(
            case_id=case_id,
            trial=t,
            passed=p,
            score=1.0 if p else 0.0,
            actual={},
            expected={},
            latency_ms=1.0,
            cost_usd=0.0,
            tags=tags.get(case_id, []),
        )
        for case_id, trials in outcomes.items()
        for t, p in enumerate(trials)
    ]
    repeats = max(len(v) for v in outcomes.values())
    return EvalRun.from_results(results, repeats=repeats)


def test_small_drop_is_not_called_a_regression():
    base = _run({f"c{i}": [True] for i in range(20)})
    curr = _run(
        {**{f"c{i}": [True] for i in range(17)}, **{f"c{i}": [False] for i in range(17, 20)}}
    )
    cmp = compare_runs(base, curr)
    assert cmp.regressed == 3
    assert cmp.verdict == "no_detectable_change"
    assert cmp.p_value > 0.05
    assert not cmp.should_fail


def test_broad_drop_is_a_regression():
    base = _run({f"c{i}": [True] for i in range(20)})
    curr = _run(
        {**{f"c{i}": [True] for i in range(12)}, **{f"c{i}": [False] for i in range(12, 20)}}
    )
    cmp = compare_runs(base, curr)
    assert cmp.regressed == 8
    assert cmp.verdict == "regression"
    assert cmp.should_fail


def test_cases_flaky_in_the_baseline_are_set_aside():
    base = _run({"a": [True, True, True], "b": [True, False, True]})
    curr = _run({"a": [True, True, True], "b": [False, False, True]})
    cmp = compare_runs(base, curr)
    assert cmp.flaky == 1
    assert cmp.regressed == 0


def test_newly_flaky_cases_count_as_regressions():
    # Reviewer case: 20 critical cases go from 3/3 to 1/3. That is a regression.
    tags = {f"c{i}": ["critical"] for i in range(20)}
    base = _run({f"c{i}": [True, True, True] for i in range(20)}, tags)
    curr = _run({f"c{i}": [True, False, False] for i in range(20)}, tags)
    cmp = compare_runs(base, curr)
    assert cmp.regressed == 20
    assert cmp.verdict == "regression"
    assert cmp.should_fail


def test_degraded_critical_case_fails_even_while_mostly_passing():
    tags = {"refund": ["critical"]}
    base = _run({"refund": [True, True, True], "other": [True, True, True]}, tags)
    curr = _run({"refund": [True, True, False], "other": [True, True, True]}, tags)
    cmp = compare_runs(base, curr)
    assert cmp.degraded == 1 and cmp.regressed == 0
    assert cmp.verdict == "no_detectable_change"
    assert cmp.critical_regressions == ["refund"]
    assert cmp.should_fail


def test_errored_cases_are_not_regressions():
    base = _run({f"c{i}": [True] for i in range(10)})
    results = [
        EvalResult(
            case_id=f"c{i}",
            passed=False,
            score=0.0,
            actual={},
            expected={},
            latency_ms=1.0,
            cost_usd=0.0,
            error="judge raised APIError",
        )
        for i in range(10)
    ]
    curr = EvalRun.from_results(results)
    cmp = compare_runs(base, curr)
    assert cmp.errored == 10 and cmp.regressed == 0
    assert any("errored" in w for w in cmp.warnings)


def test_critical_case_fails_the_gate_on_its_own():
    tags = {"refund": ["critical"]}
    base = _run({"refund": [True], "other": [True]}, tags)
    curr = _run({"refund": [False], "other": [True]}, tags)
    cmp = compare_runs(base, curr)
    assert cmp.verdict == "no_detectable_change"
    assert cmp.critical_regressions == ["refund"]
    assert cmp.should_fail


def test_new_and_removed_cases_are_reported_not_scored():
    cmp = compare_runs(_run({"a": [True], "old": [True]}), _run({"a": [True], "new": [False]}))
    statuses = {c.case_id: c.status for c in cmp.changes}
    assert statuses == {"a": "unchanged", "old": "removed", "new": "new"}
    assert cmp.regressed == 0


def test_comparison_markdown_mentions_verdict_and_cases():
    base = _run({f"c{i}": [True] for i in range(10)})
    curr = _run({**{f"c{i}": [True] for i in range(9)}, "c9": [False]})
    md = write_comparison_markdown(compare_runs(base, curr))
    assert "No detectable change" in md
    assert "`c9`" in md


def test_comparison_reports_power_and_spec_warnings():
    from agent_eval_kit.harness import RunSpec

    base = _run({f"c{i}": [True] for i in range(20)})
    curr = _run({f"c{i}": [True] for i in range(20)})
    base.spec = RunSpec(dataset="aaa", judge="j1")
    curr.spec = RunSpec(dataset="bbb", judge="j2")
    cmp = compare_runs(base, curr)
    assert cmp.n_paired == 20
    assert 0 < cmp.min_detectable_effect < 1
    assert len(cmp.warnings) == 2
    md = write_comparison_markdown(cmp)
    assert "golden set changed" in md and "What this comparison can see" in md


def test_more_cases_lower_the_detectable_effect():
    small = compare_runs(
        _run({f"c{i}": [True] for i in range(10)}), _run({f"c{i}": [True] for i in range(10)})
    )
    large = compare_runs(
        _run({f"c{i}": [True] for i in range(200)}), _run({f"c{i}": [True] for i in range(200)})
    )
    # All-passing runs carry no binomial variance; use runs with a 50% pass rate instead.
    half = {f"c{i}": [i % 2 == 0] for i in range(10)}
    half_big = {f"c{i}": [i % 2 == 0] for i in range(200)}
    small = compare_runs(_run(half), _run(half))
    large = compare_runs(_run(half_big), _run(half_big))
    assert large.min_detectable_effect < small.min_detectable_effect
