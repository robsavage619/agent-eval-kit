from __future__ import annotations

from typing import Any

from .base import Judge


class NumericToleranceJudge(Judge):
    """Pass when numeric fields are within relative or absolute tolerance."""

    def __init__(self, keys: list[str], rel_tol: float = 0.05, abs_tol: float = 0.0) -> None:
        self._keys = keys
        self._rel_tol = rel_tol
        self._abs_tol = abs_tol

    async def evaluate(
        self,
        expected: dict[str, Any],
        actual: dict[str, Any],
    ) -> tuple[float, bool, str]:
        mismatches = []
        for k in self._keys:
            exp_val = expected.get(k)
            act_val = actual.get(k)
            if exp_val is None or act_val is None:
                mismatches.append(f"{k}: missing")
                continue
            tol = max(self._abs_tol, self._rel_tol * abs(float(exp_val)))
            if abs(float(act_val) - float(exp_val)) > tol:
                mismatches.append(f"{k}: expected={exp_val} actual={act_val} tol=±{tol:.4g}")

        if not mismatches:
            return 1.0, True, "all numeric fields within tolerance"
        return 0.0, False, "; ".join(mismatches)
