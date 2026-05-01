from __future__ import annotations

from typing import Any

from .base import Judge


class ExactMatchJudge(Judge):
    """Pass when every key in expected matches the same key in actual."""

    def __init__(self, keys: list[str] | None = None) -> None:
        self._keys = keys

    async def evaluate(
        self,
        expected: dict[str, Any],
        actual: dict[str, Any],
    ) -> tuple[float, bool, str]:
        keys = self._keys or list(expected.keys())
        mismatches = [k for k in keys if expected.get(k) != actual.get(k)]
        if not mismatches:
            return 1.0, True, "all keys match"
        reasons = [
            f"{k}: expected={expected.get(k)!r} actual={actual.get(k)!r}" for k in mismatches
        ]
        return 0.0, False, "; ".join(reasons)
