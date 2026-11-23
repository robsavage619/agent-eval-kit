from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from typing import Any

_SIMPLE = (str, int, float, bool, type(None))


def _describe(value: Any) -> Any:
    if isinstance(value, Judge):
        return value.describe()
    if isinstance(value, _SIMPLE):
        return value
    if isinstance(value, (list, tuple)):
        items = [_describe(v) for v in value]
        return None if any(i is _SKIP for i in items) else items
    if isinstance(value, dict):
        items = {str(k): _describe(v) for k, v in value.items()}
        return None if any(v is _SKIP for v in items.values()) else items
    if callable(value):
        return getattr(value, "__qualname__", type(value).__name__)
    return _SKIP


_SKIP = object()


class Judge(ABC):
    @abstractmethod
    async def evaluate(
        self,
        expected: dict[str, Any],
        actual: dict[str, Any],
    ) -> tuple[float, bool, str]:
        """Return (score 0-1, passed, reason)."""

    def describe(self) -> dict[str, Any]:
        """The judge's class and configuration, for run logs and calibration records.

        Clients, trackers and other live objects are left out; prompts,
        thresholds, models and nested judges are kept.
        """
        config = {}
        for key, value in vars(self).items():
            if key in {"_client", "_tracker"}:
                continue
            described = _describe(value)
            if described is not _SKIP:
                config[key.lstrip("_")] = described
        return {"judge": type(self).__name__, "config": config}

    def fingerprint(self) -> str:
        """Short hash of ``describe()``. Changes whenever the judge's config does."""
        blob = json.dumps(self.describe(), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]
