from __future__ import annotations

import json
import os
from typing import Any

import anthropic

from ..cost_tracker import CostTracker
from .base import Judge

_SYSTEM = """\
You are an objective evaluator. Given an expected output and an actual agent output,
score the actual output from 0.0 to 1.0 and decide if it passes (score >= threshold).
Respond ONLY with a JSON object: {"score": <float>, "passed": <bool>, "reason": "<string>"}.
"""


class LLMJudge(Judge):
    """Claude Haiku-based judge — cheaper than Sonnet, good enough for structured evals."""

    def __init__(
        self,
        criteria: str,
        threshold: float = 0.7,
        model: str = "claude-haiku-4-5-20251001",
        tracker: CostTracker | None = None,
    ) -> None:
        self._criteria = criteria
        self._threshold = threshold
        self._model = model
        self._tracker = tracker
        self._client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    async def evaluate(
        self,
        expected: dict[str, Any],
        actual: dict[str, Any],
    ) -> tuple[float, bool, str]:
        prompt = (
            f"Criteria: {self._criteria}\n\n"
            f"Expected:\n{json.dumps(expected, indent=2)}\n\n"
            f"Actual:\n{json.dumps(actual, indent=2)}\n\n"
            f"Pass threshold: {self._threshold}"
        )
        response = self._client.messages.create(
            model=self._model,
            max_tokens=256,
            system=_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        if self._tracker:
            self._tracker.record(
                self._model,
                response.usage.input_tokens,
                response.usage.output_tokens,
            )
        raw = response.content[0].text.strip()
        try:
            parsed = json.loads(raw)
            score = float(parsed["score"])
            passed = bool(parsed["passed"])
            reason = str(parsed.get("reason", ""))
        except (json.JSONDecodeError, KeyError):
            score = 0.0
            passed = False
            reason = f"judge parse error: {raw[:200]}"
        return score, passed, reason
