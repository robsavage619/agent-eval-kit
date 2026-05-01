from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class Case(BaseModel):
    id: str
    input: dict[str, Any]
    expected: dict[str, Any]
    tags: list[str] = Field(default_factory=list)
    notes: str = ""


class GoldenSet:
    def __init__(self, cases: list[Case]) -> None:
        self._cases = cases

    @classmethod
    def from_jsonl(cls, path: str | Path) -> GoldenSet:
        cases = []
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                cases.append(Case.model_validate_json(line))
        return cls(cases)

    def save(self, path: str | Path) -> None:
        Path(path).write_text("\n".join(c.model_dump_json() for c in self._cases) + "\n")

    def filter(self, tag: str) -> GoldenSet:
        return GoldenSet([c for c in self._cases if tag in c.tags])

    def __iter__(self):  # type: ignore[override]
        return iter(self._cases)

    def __len__(self) -> int:
        return len(self._cases)
