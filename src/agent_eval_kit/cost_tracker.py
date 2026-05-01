from __future__ import annotations

import threading

# Approximate pricing (USD per 1M tokens) — update as models change
_PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-4-7": (15.0, 75.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (0.80, 4.0),
    "claude-haiku-4-5-20251001": (0.80, 4.0),
}
_DEFAULT_PRICING = (3.0, 15.0)


class CostTracker:
    """Thread-safe token + cost accumulator.

    Instruments are responsible for calling record() after each LLM response.
    pop_cost() returns and resets the running total — call once per eval case.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending_usd = 0.0
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_usd = 0.0

    def record(self, model: str, input_tokens: int, output_tokens: int) -> float:
        in_price, out_price = _PRICING.get(model, _DEFAULT_PRICING)
        cost = (input_tokens * in_price + output_tokens * out_price) / 1_000_000
        with self._lock:
            self._pending_usd += cost
            self._total_input_tokens += input_tokens
            self._total_output_tokens += output_tokens
            self._total_usd += cost
        return cost

    def pop_cost(self) -> float:
        with self._lock:
            cost = self._pending_usd
            self._pending_usd = 0.0
        return cost

    @property
    def total_usd(self) -> float:
        with self._lock:
            return self._total_usd

    @property
    def total_tokens(self) -> tuple[int, int]:
        with self._lock:
            return self._total_input_tokens, self._total_output_tokens
