"""Judge what an agent did, not only what it said.

An agent can return the right answer for the wrong reason: it skipped the
lookup and guessed, called a write tool it should not have, or looped on the
same tool until it hit a limit. ``TrajectoryJudge`` checks the sequence of tool
calls an agent made against the case's expectations.

Your agent returns its tool calls under ``trajectory`` (or another key you
choose). ``extract_tool_calls`` builds that list from LangGraph / LangChain
messages, Anthropic content blocks, or OpenAI-style tool calls.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Iterable
from typing import Any, Literal

from pydantic import BaseModel, Field

from .judges.base import Judge


class Step(BaseModel):
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)


def _get(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _as_args(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {"_raw": raw}
        return parsed if isinstance(parsed, dict) else {"_raw": parsed}
    return {}


def _from_call(call: Any) -> Step | None:
    fn = _get(call, "function")
    if fn is not None:  # OpenAI style: {"function": {"name", "arguments"}}
        name = _get(fn, "name")
        return Step(tool=name, args=_as_args(_get(fn, "arguments"))) if name else None
    name = _get(call, "name")  # LangChain style: {"name", "args"}
    if not name:
        return None
    return Step(tool=name, args=_as_args(_get(call, "args") or _get(call, "input")))


def extract_tool_calls(messages: Iterable[Any]) -> list[Step]:
    """Pull tool calls, in order, out of a message list.

    Handles LangChain ``AIMessage.tool_calls`` (what LangGraph state holds),
    Anthropic ``tool_use`` content blocks, and OpenAI ``tool_calls`` dicts.
    Anything it does not recognise is skipped.
    """
    steps: list[Step] = []
    for msg in messages:
        calls = _get(msg, "tool_calls")
        if calls:
            steps.extend(s for s in (_from_call(c) for c in calls) if s)
            continue
        content = _get(msg, "content")
        if isinstance(content, list):
            for block in content:
                if _get(block, "type") == "tool_use":
                    step = _from_call(block)
                    if step:
                        steps.append(step)
    return steps


def _normalise(raw: Any) -> list[Step]:
    out: list[Step] = []
    for item in raw or []:
        if isinstance(item, str):
            out.append(Step(tool=item))
        elif isinstance(item, Step):
            out.append(item)
        elif isinstance(item, dict):
            out.append(
                Step(tool=item.get("tool") or item.get("name", ""), args=item.get("args", {}))
            )
    return out


type ArgRule = Literal["exact", "subset", "ignore"] | list[str] | Callable[[dict, dict], bool]
type Order = Literal["in_order", "exact", "any_order", "unordered", "subset"]


def _args_match(rule: ArgRule, want: dict[str, Any], got: dict[str, Any]) -> bool:
    if callable(rule):
        return bool(rule(want, got))
    if rule == "ignore":
        return True
    if rule == "exact":
        return want == got
    keys = rule if isinstance(rule, list) else list(want)
    return all(got.get(k) == want.get(k) for k in keys)


class _Matcher:
    def __init__(self, default: ArgRule, overrides: dict[str, ArgRule]) -> None:
        self._default: ArgRule = default
        self._overrides: dict[str, ArgRule] = overrides

    def __call__(self, required: Step, actual: Step) -> bool:
        if required.tool != actual.tool:
            return False
        rule: ArgRule = self._overrides.get(required.tool, self._default)
        return _args_match(rule, required.args, actual.args)


def _in_order(required: list[Step], actual: list[Step], match: _Matcher) -> bool:
    it = iter(actual)
    return all(any(match(r, a) for a in it) for r in required)


def _unordered(required: list[Step], actual: list[Step], match: _Matcher) -> bool:
    if Counter(s.tool for s in required) != Counter(s.tool for s in actual):
        return False
    pool = list(actual)
    for r in required:
        hit = next((a for a in pool if match(r, a)), None)
        if hit is None:
            return False
        pool.remove(hit)
    return True


def _longest_repeat(actual: list[Step]) -> tuple[str, int]:
    best_tool, best, run = "", 0, 0
    for i, step in enumerate(actual):
        run = run + 1 if i and step.tool == actual[i - 1].tool else 1
        if run > best:
            best_tool, best = step.tool, run
    return best_tool, best


class TrajectoryJudge(Judge):
    """Check an agent's tool calls against ``expected[key]``.

    The expectation is either a list of reference tool calls, or a dict with:

    - ``required``: reference calls (tool names, or ``{"tool", "args"}``)
    - ``forbidden``: tools the agent must never call
    - ``max_steps``: an upper bound on total tool calls
    - ``max_repeats``: an upper bound on calling one tool back to back (loops)
    - ``order``: overrides the judge's default for this case

    ``order`` decides how ``required`` is matched. The names follow
    LangChain's agentevals where they overlap:

    - ``in_order``: every reference call, in order, extra calls allowed between
    - ``exact`` (agentevals "strict"): exactly these calls in this order
    - ``any_order`` (agentevals "superset"): every reference call, any order,
      extras allowed
    - ``unordered``: the same calls in any order, nothing extra
    - ``subset``: no tool outside the reference (calls may be skipped)

    Argument checks: ``args`` is the default rule for every tool ("subset":
    the reference args must match, extra args are fine; "exact"; "ignore"),
    and ``arg_overrides`` sets a rule per tool: one of those strings, a list of
    keys that must match, or a callable ``(reference_args, actual_args) -> bool``.

    The score is the share of checks that passed; the case passes only when
    all of them do.
    """

    def __init__(
        self,
        key: str = "trajectory",
        order: Order = "in_order",
        max_steps: int | None = None,
        max_repeats: int | None = None,
        args: ArgRule = "subset",
        arg_overrides: dict[str, ArgRule] | None = None,
    ) -> None:
        self._key = key
        self._order: Order = order
        self._max_steps = max_steps
        self._max_repeats = max_repeats
        self._args = args
        self._arg_overrides = arg_overrides or {}
        self._match = _Matcher(args, self._arg_overrides)

    async def evaluate(
        self,
        expected: dict[str, Any],
        actual: dict[str, Any],
    ) -> tuple[float, bool, str]:
        raw = expected.get(self._key)
        if raw is None:
            return 1.0, True, "no trajectory expectation"
        spec: dict[str, Any] = {"required": raw} if isinstance(raw, list) else dict(raw)

        steps = _normalise(actual.get(self._key))
        required = _normalise(spec.get("required"))
        forbidden = {s.tool for s in _normalise(spec.get("forbidden"))}
        max_steps: int | None = spec.get("max_steps", self._max_steps)
        max_repeats: int | None = spec.get("max_repeats", self._max_repeats)

        order: Order = spec.get("order", self._order)
        match = self._match
        checks: list[tuple[bool, str]] = []
        if required:
            if order == "exact":
                ok = len(required) == len(steps) and all(
                    match(r, a) for r, a in zip(required, steps, strict=True)
                )
            elif order == "any_order":
                ok = all(any(match(r, a) for a in steps) for r in required)
            elif order == "unordered":
                ok = _unordered(required, steps, match)
            elif order == "subset":
                allowed = {r.tool for r in required}
                ok = all(
                    s.tool in allowed and any(match(r, s) for r in required if r.tool == s.tool)
                    for s in steps
                )
            else:
                ok = _in_order(required, steps, match)
            want = " -> ".join(r.tool for r in required)
            got = " -> ".join(s.tool for s in steps) or "(no tool calls)"
            checks.append((ok, f"required {order} [{want}], got [{got}]"))
        if forbidden:
            hit = sorted({s.tool for s in steps} & forbidden)
            checks.append((not hit, f"forbidden tools called: {', '.join(hit)}" if hit else ""))
        if max_steps is not None:
            checks.append((len(steps) <= max_steps, f"{len(steps)} steps > max {max_steps}"))
        if max_repeats is not None:
            tool, n = _longest_repeat(steps)
            checks.append((n <= max_repeats, f"{tool} called {n}x in a row > max {max_repeats}"))

        if not checks:
            return 1.0, True, "no trajectory checks configured"
        failed = [msg for ok, msg in checks if not ok]
        score = sum(ok for ok, _ in checks) / len(checks)
        if failed:
            return score, False, "; ".join(failed)
        return 1.0, True, f"trajectory ok ({len(steps)} steps)"
