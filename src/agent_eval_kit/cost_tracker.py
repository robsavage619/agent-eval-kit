from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

# Approximate pricing (USD per 1M tokens: input, output). Hand-maintained and
# easy to get stale: check your provider's price list, or pass your own table
# with CostTracker(pricing=...).
_PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-4-7": (15.0, 75.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (0.80, 4.0),
    "claude-haiku-4-5-20251001": (0.80, 4.0),
}
_DEFAULT_PRICING = (3.0, 15.0)

_UNSCOPED = "__unscoped__"

# The case currently being evaluated. asyncio copies context into each task,
# so concurrent cases each see their own value and a model call is charged to
# the case that made it, not to whichever case happens to pop the total next.
_current_case: ContextVar[str | None] = ContextVar("agent_eval_kit_case", default=None)


class CostTracker:
    """Thread-safe token and cost accumulator with per-case attribution.

    Your agent (or a judge) calls ``record()`` after each model response. The
    harness wraps every case in ``case_scope()``, so spend lands on the right
    case even when cases run concurrently.

    Attribution follows ``contextvars``. ``asyncio.to_thread`` and LangChain's
    executors copy the context, so calls made there are attributed correctly;
    a bare ``loop.run_in_executor`` does not, and its spend lands in the
    unscoped bucket (still counted in ``total_usd``, but not on any case).
    """

    def __init__(self, pricing: dict[str, tuple[float, float]] | None = None) -> None:
        self._lock = threading.Lock()
        self._pricing = {**_PRICING, **(pricing or {})}
        self._pending: dict[str, float] = {}
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_usd = 0.0

    def record(self, model: str, input_tokens: int, output_tokens: int) -> float:
        in_price, out_price = self._pricing.get(model, _DEFAULT_PRICING)
        cost = (input_tokens * in_price + output_tokens * out_price) / 1_000_000
        key = _current_case.get() or _UNSCOPED
        with self._lock:
            self._pending[key] = self._pending.get(key, 0.0) + cost
            self._total_input_tokens += input_tokens
            self._total_output_tokens += output_tokens
            self._total_usd += cost
        return cost

    @contextmanager
    def case_scope(self, key: str) -> Iterator[None]:
        token = _current_case.set(key)
        try:
            yield
        finally:
            _current_case.reset(token)

    def pop_cost(self, key: str | None = None) -> float:
        """Return and reset the spend recorded for ``key``.

        With no key, uses the current case scope, or the unscoped bucket when
        called outside the harness (the 0.1 behaviour).
        """
        key = key or _current_case.get() or _UNSCOPED
        with self._lock:
            return self._pending.pop(key, 0.0)

    @property
    def total_usd(self) -> float:
        with self._lock:
            return self._total_usd

    @property
    def total_tokens(self) -> tuple[int, int]:
        with self._lock:
            return self._total_input_tokens, self._total_output_tokens
