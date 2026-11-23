from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_eval_kit import Step, TrajectoryJudge, extract_tool_calls
from agent_eval_kit.judges import AllOf, ExactMatchJudge


@pytest.mark.asyncio
async def test_in_order_allows_extra_steps():
    judge = TrajectoryJudge()
    actual = {"trajectory": ["lookup_order", "check_policy", "issue_refund"]}
    score, passed, _ = await judge.evaluate(
        {"trajectory": ["lookup_order", "issue_refund"]}, actual
    )
    assert passed and score == 1.0


@pytest.mark.asyncio
async def test_skipped_step_fails():
    judge = TrajectoryJudge()
    actual = {"trajectory": ["issue_refund"]}
    _, passed, reason = await judge.evaluate(
        {"trajectory": ["lookup_order", "issue_refund"]}, actual
    )
    assert not passed
    assert "lookup_order" in reason


@pytest.mark.asyncio
async def test_exact_order():
    judge = TrajectoryJudge(order="exact")
    spec = {"trajectory": ["a", "b"]}
    assert (await judge.evaluate(spec, {"trajectory": ["a", "b"]}))[1]
    assert not (await judge.evaluate(spec, {"trajectory": ["a", "x", "b"]}))[1]
    assert not (await judge.evaluate(spec, {"trajectory": ["b", "a"]}))[1]


@pytest.mark.asyncio
async def test_forbidden_loops_and_budget():
    judge = TrajectoryJudge()
    spec = {"trajectory": {"forbidden": ["issue_refund"], "max_steps": 3, "max_repeats": 2}}
    actual = {"trajectory": ["search", "search", "search", "issue_refund"]}
    score, passed, reason = await judge.evaluate(spec, actual)
    assert not passed
    assert score == 0.0
    assert "issue_refund" in reason and "4 steps" in reason and "3x in a row" in reason


@pytest.mark.asyncio
async def test_required_args_must_match():
    judge = TrajectoryJudge()
    spec = {"trajectory": [{"tool": "issue_refund", "args": {"order_id": "A1"}}]}
    ok = {"trajectory": [{"tool": "issue_refund", "args": {"order_id": "A1", "amount": 20}}]}
    wrong = {"trajectory": [{"tool": "issue_refund", "args": {"order_id": "B2"}}]}
    assert (await judge.evaluate(spec, ok))[1]
    assert not (await judge.evaluate(spec, wrong))[1]


@pytest.mark.asyncio
async def test_no_expectation_passes():
    assert (await TrajectoryJudge().evaluate({}, {"trajectory": ["x"]}))[1]


def test_extract_from_langchain_anthropic_and_openai_shapes():
    langchain_msg = SimpleNamespace(
        tool_calls=[{"name": "lookup_order", "args": {"order_id": "A1"}, "id": "1"}],
        content="",
    )
    tool_result = SimpleNamespace(tool_calls=None, content="order found")
    anthropic_msg = {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Refunding now."},
            {"type": "tool_use", "name": "issue_refund", "input": {"order_id": "A1"}},
        ],
    }
    openai_msg = {
        "role": "assistant",
        "tool_calls": [{"function": {"name": "send_email", "arguments": '{"to": "a@b.c"}'}}],
    }
    steps = extract_tool_calls([langchain_msg, tool_result, anthropic_msg, openai_msg])
    assert steps == [
        Step(tool="lookup_order", args={"order_id": "A1"}),
        Step(tool="issue_refund", args={"order_id": "A1"}),
        Step(tool="send_email", args={"to": "a@b.c"}),
    ]


@pytest.mark.asyncio
async def test_all_of_needs_right_answer_and_right_path():
    judge = AllOf([ExactMatchJudge(keys=["intent"]), TrajectoryJudge()])
    expected = {"intent": "refund", "trajectory": ["lookup_order", "issue_refund"]}
    right_answer_wrong_path = {"intent": "refund", "trajectory": ["issue_refund"]}
    score, passed, _ = await judge.evaluate(expected, right_answer_wrong_path)
    assert not passed
    assert score == 0.5


@pytest.mark.asyncio
async def test_subset_and_unordered_modes():
    spec = {"trajectory": {"required": ["search", "summarize"], "order": "subset"}}
    judge = TrajectoryJudge()
    assert (await judge.evaluate(spec, {"trajectory": ["search"]}))[1]
    assert not (await judge.evaluate(spec, {"trajectory": ["search", "delete_db"]}))[1]
    spec_u = {"trajectory": {"required": ["a", "b"], "order": "unordered"}}
    assert (await judge.evaluate(spec_u, {"trajectory": ["b", "a"]}))[1]
    assert not (await judge.evaluate(spec_u, {"trajectory": ["b", "a", "c"]}))[1]


@pytest.mark.asyncio
async def test_arg_overrides():
    judge = TrajectoryJudge(
        arg_overrides={
            "search": "ignore",
            "get_weather": lambda want, got: want["city"].lower() == got.get("city", "").lower(),
        }
    )
    spec = {
        "trajectory": [
            {"tool": "search", "args": {"q": "weather"}},
            {"tool": "get_weather", "args": {"city": "Portland"}},
        ]
    }
    actual = {
        "trajectory": [
            {"tool": "search", "args": {"q": "something else"}},
            {"tool": "get_weather", "args": {"city": "PORTLAND"}},
        ]
    }
    assert (await judge.evaluate(spec, actual))[1]
    assert not (await TrajectoryJudge().evaluate(spec, actual))[1]


@pytest.mark.asyncio
async def test_weighted_any_of_and_not():
    from agent_eval_kit.judges import AnyOf, Not, Weighted

    intent = ExactMatchJudge(keys=["intent"])
    tone = ExactMatchJudge(keys=["tone"])
    actual = {"intent": "refund", "tone": "rude"}
    expected = {"intent": "refund", "tone": "polite"}

    score, passed, reason = await Weighted([(intent, 3), (tone, 1)], threshold=0.7).evaluate(
        expected, actual
    )
    assert passed and score == pytest.approx(0.75)

    # weight 0 = observe only: reported, never gating
    score, passed, reason = await Weighted([(intent, 1), (tone, 0)]).evaluate(expected, actual)
    assert passed and score == 1.0 and "[observe]" in reason

    assert (await AnyOf([tone, intent]).evaluate(expected, actual))[1]
    assert not (await Not(intent).evaluate(expected, actual))[1]
