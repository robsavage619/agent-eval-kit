from __future__ import annotations

from typing import Any

from .base import Judge


class AllOf(Judge):
    """Pass only when every judge passes. Score is the mean of their scores.

    Typical use: ``AllOf([ExactMatchJudge(keys=["intent"]), TrajectoryJudge()])``
    so a case needs the right answer *and* the right path to it.
    """

    def __init__(self, judges: list[Judge]) -> None:
        if not judges:
            raise ValueError("AllOf needs at least one judge")
        self._judges = judges

    async def evaluate(
        self,
        expected: dict[str, Any],
        actual: dict[str, Any],
    ) -> tuple[float, bool, str]:
        outcomes = [await j.evaluate(expected, actual) for j in self._judges]
        score = sum(s for s, _, _ in outcomes) / len(outcomes)
        passed = all(p for _, p, _ in outcomes)
        failed = [r for _, p, r in outcomes if not p]
        return score, passed, "; ".join(failed) if failed else "all judges passed"


class AnyOf(Judge):
    """Pass when at least one judge passes. Score is the best score.

    Useful when several answers are acceptable and each needs its own check.
    """

    def __init__(self, judges: list[Judge]) -> None:
        if not judges:
            raise ValueError("AnyOf needs at least one judge")
        self._judges = judges

    async def evaluate(
        self,
        expected: dict[str, Any],
        actual: dict[str, Any],
    ) -> tuple[float, bool, str]:
        outcomes = [await j.evaluate(expected, actual) for j in self._judges]
        best = max(outcomes, key=lambda o: (o[1], o[0]))
        if best[1]:
            return best[0], True, best[2]
        return best[0], False, "none passed: " + "; ".join(r for _, _, r in outcomes)


class Not(Judge):
    """Invert a judge: pass when it fails. For "must not" checks."""

    def __init__(self, judge: Judge) -> None:
        self._judge = judge

    async def evaluate(
        self,
        expected: dict[str, Any],
        actual: dict[str, Any],
    ) -> tuple[float, bool, str]:
        score, passed, reason = await self._judge.evaluate(expected, actual)
        return 1.0 - score, not passed, f"not({reason})"


class Weighted(Judge):
    """Weighted mean of several judges, in the style of promptfoo assertions.

    ``judges`` is a list of ``(judge, weight)``. A weight of 0 means observe
    only: the judge runs and its reason is reported, but it never affects the
    score or the outcome (good for latency, cost, or a new check you are not
    ready to gate on).

    With ``threshold`` set, the case passes when the weighted score reaches it.
    Without one, every judge with a positive weight has to pass.
    """

    def __init__(self, judges: list[tuple[Judge, float]], threshold: float | None = None) -> None:
        if not any(w > 0 for _, w in judges):
            raise ValueError("Weighted needs at least one judge with a positive weight")
        if any(w < 0 for _, w in judges):
            raise ValueError("weights must be non-negative")
        self._judges = judges
        self._threshold = threshold

    async def evaluate(
        self,
        expected: dict[str, Any],
        actual: dict[str, Any],
    ) -> tuple[float, bool, str]:
        gating: list[tuple[float, float, bool, str]] = []
        observed: list[str] = []
        for judge, weight in self._judges:
            s, p, r = await judge.evaluate(expected, actual)
            if weight > 0:
                gating.append((weight, s, p, r))
            else:
                observed.append(f"[observe] {r}")
        total = sum(w for w, _, _, _ in gating)
        score = sum(w * s for w, s, _, _ in gating) / total
        if self._threshold is not None:
            passed = score >= self._threshold
        else:
            passed = all(p for _, _, p, _ in gating)
        failed = [r for _, _, p, r in gating if not p]
        reason = "; ".join(failed) if failed else f"weighted score {score:.2f}"
        return score, passed, "; ".join([reason, *observed])
