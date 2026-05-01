from __future__ import annotations

import json
import os
import statistics
from typing import Any, TypedDict

import anthropic
from langgraph.graph import END, StateGraph

from agent_eval_kit.cost_tracker import CostTracker

_MODEL = "claude-sonnet-4-6"
_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    return _client


class AnomalyState(TypedDict):
    daily_totals: dict[str, float]
    mean: float
    std_dev: float
    anomalies: list[dict[str, Any]]
    recommendation: str
    raw_response: str


def _compute_stats(state: AnomalyState) -> AnomalyState:
    values = list(state["daily_totals"].values())
    state["mean"] = statistics.mean(values)
    state["std_dev"] = statistics.stdev(values) if len(values) > 1 else 0.0
    return state


def _detect_anomalies(state: AnomalyState) -> AnomalyState:
    threshold = state["mean"] + 2.5 * state["std_dev"]
    anomalies = [
        {"date": d, "cost_usd": v, "z_score": round((v - state["mean"]) / max(state["std_dev"], 0.01), 2)}
        for d, v in state["daily_totals"].items()
        if v > threshold
    ]
    state["anomalies"] = sorted(anomalies, key=lambda x: x["cost_usd"], reverse=True)
    return state


def _recommend(state: AnomalyState, tracker: CostTracker | None = None) -> AnomalyState:
    if not state["anomalies"]:
        state["recommendation"] = "No anomalies detected. Spend is within normal range."
        state["raw_response"] = ""
        return state

    prompt = (
        f"AWS cost anomalies detected (threshold: mean + 2.5σ = ${state['mean'] + 2.5 * state['std_dev']:.0f}/day):\n"
        f"{json.dumps(state['anomalies'], indent=2)}\n\n"
        "Provide a concise (2-3 sentence) recommendation for the FinOps team. "
        "Include: likely cause category (autoscaling runaway, data transfer spike, misconfigured resource), "
        "immediate action, and monitoring suggestion."
    )
    response = _get_client().messages.create(
        model=_MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    if tracker:
        tracker.record(_MODEL, response.usage.input_tokens, response.usage.output_tokens)
    state["raw_response"] = response.content[0].text
    state["recommendation"] = response.content[0].text
    return state


def build_graph(tracker: CostTracker | None = None) -> Any:
    graph = StateGraph(AnomalyState)
    graph.add_node("compute_stats", _compute_stats)
    graph.add_node("detect_anomalies", _detect_anomalies)
    graph.add_node("recommend", lambda s: _recommend(s, tracker))
    graph.set_entry_point("compute_stats")
    graph.add_edge("compute_stats", "detect_anomalies")
    graph.add_edge("detect_anomalies", "recommend")
    graph.add_edge("recommend", END)
    return graph.compile()


async def run_agent(input_data: dict[str, Any], tracker: CostTracker | None = None) -> dict[str, Any]:
    graph = build_graph(tracker)
    result = await graph.ainvoke({
        "daily_totals": input_data["daily_totals"],
        "mean": 0.0,
        "std_dev": 0.0,
        "anomalies": [],
        "recommendation": "",
        "raw_response": "",
    })
    return {
        "anomaly_dates": [a["date"] for a in result["anomalies"]],
        "anomaly_count": len(result["anomalies"]),
        "recommendation": result["recommendation"],
    }
