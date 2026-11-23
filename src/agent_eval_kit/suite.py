"""A suite bundles what the CLI needs: cases, the agent, and the judge.

Write an ``evals.py`` that defines ``suite = Suite(...)`` and point the CLI at
it: ``aek run evals.py``. Use ``path.py:name`` for a different variable name.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .cost_tracker import CostTracker
from .golden_set import GoldenSet
from .harness import AgentFn, EvalRun, run_evals
from .judges.base import Judge


@dataclass
class Suite:
    name: str
    golden_set: GoldenSet
    agent: AgentFn
    judge: Judge
    concurrency: int = 4
    repeats: int = 1
    tracker: CostTracker = field(default_factory=CostTracker)

    async def run(self, repeats: int | None = None, tag: str | None = None) -> EvalRun:
        golden = self.golden_set.filter(tag) if tag else self.golden_set
        return await run_evals(
            golden,
            self.agent,
            self.judge,
            tracker=self.tracker,
            concurrency=self.concurrency,
            repeats=repeats or self.repeats,
            name=self.name,
        )


def load_suite(spec: str) -> Suite:
    """Load ``path/to/evals.py`` (variable ``suite``) or ``path/to/evals.py:name``."""
    path_str, _, attr = spec.partition(":")
    attr = attr or "suite"
    path = Path(path_str).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"suite file not found: {path_str}")

    module_name = f"_aek_suite_{abs(hash(path))}"
    module_spec = importlib.util.spec_from_file_location(module_name, path)
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(module_spec)
    # Let the suite import modules that sit next to it (its agent, helpers).
    # If another suite already imported a sibling with the same name (two
    # folders each with an `agent.py`), set the cached one aside so this suite
    # gets its own, then put it back.
    siblings = {p.stem for p in path.parent.glob("*.py")}
    shadowed = {}
    for name in siblings & set(sys.modules):
        origin = getattr(sys.modules[name], "__file__", None)
        if origin is None or Path(origin).resolve().parent != path.parent:
            shadowed[name] = sys.modules.pop(name)
    sys.path.insert(0, str(path.parent))
    try:
        sys.modules[module_name] = module
        module_spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(path.parent))
        for name in siblings:
            mod = sys.modules.get(name)
            origin = getattr(mod, "__file__", None) if mod else None
            if origin and Path(origin).resolve().parent == path.parent:
                del sys.modules[name]
        sys.modules.update(shadowed)

    suite = getattr(module, attr, None)
    if not isinstance(suite, Suite):
        raise TypeError(f"{path_str} has no Suite named {attr!r}")
    return suite
