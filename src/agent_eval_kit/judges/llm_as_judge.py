from __future__ import annotations

import json
import os
import re
from typing import Any

from ..cost_tracker import CostTracker
from .base import Judge

_SYSTEM = """\
You are an objective evaluator. Given an expected output and an actual agent output,
score the actual output from 0.0 to 1.0 against the criteria.
Respond ONLY with a JSON object: {"score": <float>, "reason": "<one sentence>"}.
"""

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


class JudgeError(RuntimeError):
    """The judge could not produce a verdict. Recorded as an error, never as a pass."""


def _parse(raw: str) -> tuple[float, str]:
    """Return (score, reason). Raises ValueError on anything unparseable."""
    parsed = json.loads(_FENCE.sub("", raw).strip())
    score = float(parsed["score"])
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"score {score} outside [0, 1]")
    return score, str(parsed.get("reason", ""))


class LLMJudge(Judge):
    """Claude-graded judge for free-text output.

    Pass/fail comes from ``score >= threshold``, computed here, not from a
    boolean the model reports, so the threshold means the same thing on every
    case. A reply that cannot be parsed raises ``JudgeError``; the harness
    records it as an error on that case, so a broken judge is visible instead
    of passing or failing quietly.

    ``steps`` fixes the evaluation procedure (as in G-Eval): the judge works
    through the same numbered steps on every case, which makes its scores more
    repeatable than a bare criterion.

    Calibrate it against human labels before trusting its pass rate: see
    ``agent_eval_kit.calibration``.
    """

    def __init__(
        self,
        criteria: str,
        threshold: float = 0.7,
        model: str = "claude-haiku-4-5-20251001",
        tracker: CostTracker | None = None,
        max_tokens: int = 256,
        steps: list[str] | None = None,
    ) -> None:
        self._criteria = criteria
        self._steps = steps or []
        self._threshold = threshold
        self._model = model
        self._tracker = tracker
        self._max_tokens = max_tokens
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            import anthropic

            self._client = anthropic.AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        return self._client

    async def evaluate(
        self,
        expected: dict[str, Any],
        actual: dict[str, Any],
    ) -> tuple[float, bool, str]:
        procedure = ""
        if self._steps:
            numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(self._steps, start=1))
            procedure = f"Evaluation steps (follow in order):\n{numbered}\n\n"
        prompt = (
            f"Criteria: {self._criteria}\n\n"
            f"{procedure}"
            f"Expected:\n{json.dumps(expected, indent=2, default=str)}\n\n"
            f"Actual:\n{json.dumps(actual, indent=2, default=str)}"
        )
        response = await self._get_client().messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        if self._tracker:
            self._tracker.record(
                self._model,
                response.usage.input_tokens,
                response.usage.output_tokens,
            )
        raw = "".join(
            getattr(b, "text", "") for b in response.content if getattr(b, "type", "") == "text"
        ).strip()
        try:
            score, reason = _parse(raw)
        except (ValueError, KeyError, TypeError) as exc:
            raise JudgeError(f"unparseable judge reply ({exc}): {raw[:200]}") from exc
        return score, score >= self._threshold, reason
