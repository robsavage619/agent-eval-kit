from __future__ import annotations

from typing import Any

import pytest

from agent_eval_kit import LabeledExample, calibrate_judge
from agent_eval_kit.judges import Judge
from agent_eval_kit.reporters import write_calibration_markdown


class PassEverything(Judge):
    async def evaluate(self, expected: dict[str, Any], actual: dict[str, Any]):
        return 1.0, True, "looks fine"


class Oracle(Judge):
    async def evaluate(self, expected: dict[str, Any], actual: dict[str, Any]):
        ok = actual.get("answer") == expected.get("answer")
        return float(ok), ok, "match" if ok else "mismatch"


def _labels(n_good: int, n_bad: int) -> list[LabeledExample]:
    good = [
        LabeledExample(id=f"g{i}", expected={"answer": 1}, actual={"answer": 1}, human_passed=True)
        for i in range(n_good)
    ]
    bad = [
        LabeledExample(id=f"b{i}", expected={"answer": 1}, actual={"answer": 2}, human_passed=False)
        for i in range(n_bad)
    ]
    return good + bad


@pytest.mark.asyncio
async def test_rubber_stamp_judge_is_caught():
    rep = await calibrate_judge(PassEverything(), _labels(30, 10))
    assert rep.agreement == pytest.approx(0.75)
    assert rep.kappa == pytest.approx(0.0)
    assert rep.false_pass_rate == 1.0
    assert rep.verdict == "do_not_trust"
    assert len(rep.disagreements) == 10
    assert all(d.kind == "false_pass" for d in rep.disagreements)


@pytest.mark.asyncio
async def test_accurate_judge_is_trusted():
    rep = await calibrate_judge(Oracle(), _labels(40, 40))
    assert rep.kappa == pytest.approx(1.0)
    assert rep.false_pass_rate == 0.0
    assert rep.false_pass_ci[1] < 0.1
    assert rep.verdict == "trust"


@pytest.mark.asyncio
async def test_perfect_record_on_few_bad_examples_is_not_enough():
    # 0 false passes out of 25 bad outputs still leaves the upper bound near 13%.
    rep = await calibrate_judge(Oracle(), _labels(25, 25))
    assert rep.false_pass_rate == 0.0
    assert rep.false_pass_ci[1] > 0.1
    assert rep.verdict == "use_with_care"


@pytest.mark.asyncio
async def test_too_few_labels():
    rep = await calibrate_judge(Oracle(), _labels(5, 5))
    assert rep.verdict == "insufficient_data"


@pytest.mark.asyncio
async def test_calibration_markdown():
    md = write_calibration_markdown(await calibrate_judge(PassEverything(), _labels(30, 10)))
    assert "Do not trust" in md
    assert "false pass" in md


class Broken(Judge):
    async def evaluate(self, expected: dict[str, Any], actual: dict[str, Any]):
        raise RuntimeError("model returned prose")


@pytest.mark.asyncio
async def test_judge_errors_are_excluded_and_reported():
    rep = await calibrate_judge(Broken(), _labels(5, 5))
    assert rep.n == 0
    assert len(rep.judge_errors) == 10
    assert any("judge error" in w for w in rep.warnings)


@pytest.mark.asyncio
async def test_unbalanced_labels_warn_and_fingerprint_recorded():
    rep = await calibrate_judge(Oracle(), _labels(38, 2))
    assert any("human-failed" in w for w in rep.warnings)
    assert rep.judge == Oracle().fingerprint()
