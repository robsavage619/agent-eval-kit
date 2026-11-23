"""A simulated customer-support agent, so the demo runs with no API key.

It is deliberately small: keyword triage instead of a model, and a seeded
random number generator standing in for sampling noise on ambiguous messages.
That is enough to produce what real agents produce in evals: correct answers,
wrong answers, right answers reached the wrong way, and cases that pass on one
run and fail on the next.

``version="v2"`` simulates a plausible "small prompt change" that ships two
bugs: refunds under $50 skip the order lookup, and any message that mentions
a refund is routed to refunds even when the customer asked to cancel.
"""

from __future__ import annotations

import random
import re
from typing import Any

_AMOUNT = re.compile(r"\$(\d+(?:\.\d+)?)")
_ORDER = re.compile(r"\b([A-Z]\d{3,})\b")
_AMBIGUOUS = ("not sure", "maybe", "or something", "whatever")


def _intent(message: str, version: str) -> str:
    m = message.lower()
    if any(w in m for w in ("lawyer", "complaint", "manager", "unacceptable")):
        return "escalate"
    if version == "v2" and "refund" in m:
        return "refund"
    if "cancel" in m:
        return "cancel"
    if any(w in m for w in ("refund", "money back", "charged twice")):
        return "refund"
    if any(w in m for w in ("where is", "tracking", "hasn't arrived", "status")):
        return "order_status"
    return "escalate"


def make_agent(version: str = "v1", seed: int = 7, noise: float = 0.35):
    rng = random.Random(seed)
    intents = ["refund", "order_status", "cancel", "escalate"]

    async def agent(inp: dict[str, Any]) -> dict[str, Any]:
        message: str = inp["message"]
        intent = _intent(message, version)
        if any(a in message.lower() for a in _AMBIGUOUS) and rng.random() < noise:
            intent = rng.choice([i for i in intents if i != intent])

        order = _ORDER.search(message)
        order_id = order.group(1) if order else None
        amount_match = _AMOUNT.search(message)
        amount = float(amount_match.group(1)) if amount_match else None

        trajectory: list[dict[str, Any]] = []
        if intent == "refund":
            skip_lookup = version == "v2" and amount is not None and amount < 50
            if order_id and not skip_lookup:
                trajectory.append({"tool": "lookup_order", "args": {"order_id": order_id}})
            if amount is not None and amount > 500:
                trajectory.append({"tool": "escalate_to_human", "args": {"reason": "large refund"}})
            else:
                trajectory.append({"tool": "issue_refund", "args": {"order_id": order_id}})
        elif intent == "order_status" and order_id:
            trajectory.append({"tool": "lookup_order", "args": {"order_id": order_id}})
            trajectory.append({"tool": "get_tracking", "args": {"order_id": order_id}})
        elif intent == "cancel" and order_id:
            trajectory.append({"tool": "lookup_order", "args": {"order_id": order_id}})
            trajectory.append({"tool": "cancel_order", "args": {"order_id": order_id}})
        elif intent == "escalate":
            trajectory.append({"tool": "escalate_to_human", "args": {"reason": "triage"}})

        return {"intent": intent, "trajectory": trajectory}

    return agent
