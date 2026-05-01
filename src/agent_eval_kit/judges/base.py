from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Judge(ABC):
    @abstractmethod
    async def evaluate(
        self,
        expected: dict[str, Any],
        actual: dict[str, Any],
    ) -> tuple[float, bool, str]:
        """Return (score 0–1, passed, reason)."""
