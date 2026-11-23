from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError


class Case(BaseModel):
    id: str
    input: dict[str, Any]
    expected: dict[str, Any]
    tags: list[str] = Field(default_factory=list)
    notes: str = ""
    cluster: str | None = None
    """Cases sharing a cluster (same scenario, template, or source document)
    are not independent. Set this and the run reports a clustered error."""


class GoldenSet:
    def __init__(self, cases: list[Case]) -> None:
        ids = [c.id for c in cases]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(f"duplicate case ids: {', '.join(dupes)}")
        self._cases = cases

    @classmethod
    def from_jsonl(cls, path: str | Path) -> GoldenSet:
        cases = []
        for lineno, line in enumerate(Path(path).read_text().splitlines(), start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                cases.append(Case.model_validate_json(line))
            except ValidationError as exc:
                raise ValueError(f"{path}:{lineno}: invalid case: {exc}") from exc
        return cls(cases)

    def save(self, path: str | Path) -> None:
        Path(path).write_text("\n".join(c.model_dump_json() for c in self._cases) + "\n")

    def fingerprint(self) -> str:
        """Short hash of the cases, so runs on different datasets are never compared silently."""
        blob = "\n".join(c.model_dump_json() for c in self._cases)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def filter(self, tag: str) -> GoldenSet:
        return GoldenSet([c for c in self._cases if tag in c.tags])

    def __iter__(self):  # type: ignore[override]
        return iter(self._cases)

    def __len__(self) -> int:
        return len(self._cases)
