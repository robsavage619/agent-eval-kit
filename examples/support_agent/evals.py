"""Suites for the support-agent demo. No API key needed.

aek run examples/support_agent/evals.py --repeats 5
aek run examples/support_agent/evals.py:suite_v2 --repeats 5
aek calibrate examples/support_agent/evals.py:reply_suite examples/support_agent/labels.jsonl
"""

from __future__ import annotations

from pathlib import Path

from agent import make_agent
from reply_judge import KeywordReplyJudge

from agent_eval_kit import GoldenSet, Suite
from agent_eval_kit.judges import AllOf, ExactMatchJudge, TrajectoryJudge

HERE = Path(__file__).parent
golden = GoldenSet.from_jsonl(HERE / "golden_set.jsonl")

# A case passes only with the right intent *and* the right tool calls.
judge = AllOf([ExactMatchJudge(keys=["intent"]), TrajectoryJudge(max_repeats=2)])

suite = Suite(
    name="support-agent v1",
    golden_set=golden,
    agent=make_agent("v1"),
    judge=judge,
    concurrency=1,
)

suite_v2 = Suite(
    name="support-agent v2",
    golden_set=golden,
    agent=make_agent("v2"),
    judge=judge,
    concurrency=1,
)

# For `aek calibrate`: only the judge is used, scored against labels.jsonl.
reply_suite = Suite(
    name="reply keyword judge",
    golden_set=GoldenSet([]),
    agent=make_agent("v1"),
    judge=KeywordReplyJudge(),
)
