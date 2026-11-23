"""The kind of judge people write first: check the reply for keywords.

It is cheap and fast, and it has a blind spot the calibration demo exposes:
a reply that *mentions* a refund passes even when it refuses one.
"""

from __future__ import annotations

from typing import Any

from agent_eval_kit.judges import Judge


class KeywordReplyJudge(Judge):
    async def evaluate(
        self,
        expected: dict[str, Any],
        actual: dict[str, Any],
    ) -> tuple[float, bool, str]:
        reply = str(actual.get("reply", "")).lower()
        keywords = [k.lower() for k in expected.get("keywords", [])]
        missing = [k for k in keywords if k not in reply]
        if not keywords:
            return 1.0, True, "no keywords required"
        score = 1 - len(missing) / len(keywords)
        if missing:
            return score, False, f"missing: {', '.join(missing)}"
        return 1.0, True, "all keywords present"
