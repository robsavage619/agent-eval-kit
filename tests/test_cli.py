from __future__ import annotations

from pathlib import Path

from agent_eval_kit import EvalRun
from agent_eval_kit.cli import main

DEMO = Path(__file__).parent.parent / "examples" / "support_agent"


def test_run_writes_json_markdown_and_junit(tmp_path: Path):
    out, md, junit = tmp_path / "run.json", tmp_path / "run.md", tmp_path / "junit.xml"
    code = main(
        [
            "run",
            str(DEMO / "evals.py"),
            "--repeats",
            "3",
            "--out",
            str(out),
            "--markdown",
            str(md),
            "--junit",
            str(junit),
            "-q",
        ]
    )
    assert code == 0
    run = EvalRun.model_validate_json(out.read_text())
    assert run.repeats == 3 and run.n_cases == 25
    assert "Pass rate" in md.read_text()
    assert "<testsuite" in junit.read_text()


def test_min_pass_rate_gate(tmp_path: Path):
    suite = str(DEMO / "evals.py:suite_v2")
    assert main(["run", suite, "--min-pass-rate", "0.99", "-q"]) == 1
    assert main(["run", suite, "--min-pass-rate", "0.10", "-q"]) == 0


def test_compare_flags_v2_regression(tmp_path: Path):
    v1, v2 = tmp_path / "v1.json", tmp_path / "v2.json"
    assert main(["run", str(DEMO / "evals.py"), "--repeats", "5", "--out", str(v1), "-q"]) == 0
    assert (
        main(["run", str(DEMO / "evals.py:suite_v2"), "--repeats", "5", "--out", str(v2), "-q"])
        == 0
    )
    assert main(["compare", str(v1), str(v2), "-q"]) == 1
    assert main(["compare", str(v1), str(v1), "-q"]) == 0


def test_calibrate_catches_keyword_judge(tmp_path: Path):
    md = tmp_path / "cal.md"
    suite = str(DEMO / "evals.py:reply_suite")
    labels = str(DEMO / "labels.jsonl")
    assert main(["calibrate", suite, labels, "--markdown", str(md), "-q"]) == 0
    assert main(["calibrate", suite, labels, "--require", "use_with_care", "-q"]) == 1
    assert "false pass" in md.read_text()


def test_power_command(tmp_path: Path, capsys):
    out = tmp_path / "run.json"
    assert main(["run", str(DEMO / "evals.py"), "--repeats", "3", "--out", str(out), "-q"]) == 0
    capsys.readouterr()
    assert main(["power", str(out), "--delta", "0.1"]) == 0
    text = capsys.readouterr().out
    assert "Plan for about" in text and "Exact test floor" in text


def test_suites_in_different_folders_get_their_own_sibling_modules(tmp_path: Path):
    from agent_eval_kit import load_suite

    for name, label in (("one", "first"), ("two", "second")):
        d = tmp_path / name
        d.mkdir()
        (d / "agent.py").write_text(f"async def agent(inp):\n    return {{'label': {label!r}}}\n")
        (d / "evals.py").write_text(
            "from agent import agent\n"
            "from agent_eval_kit import GoldenSet, Suite\n"
            "from agent_eval_kit.judges import ExactMatchJudge\n"
            f"suite = Suite(name={label!r}, golden_set=GoldenSet([]), agent=agent,"
            " judge=ExactMatchJudge())\n"
        )
    import asyncio

    first = load_suite(str(tmp_path / "one" / "evals.py"))
    second = load_suite(str(tmp_path / "two" / "evals.py"))
    assert asyncio.run(first.agent({}))["label"] == "first"
    assert asyncio.run(second.agent({}))["label"] == "second"


def test_misuse_exits_2_not_1(tmp_path: Path):
    assert main(["run", str(tmp_path / "missing.py"), "-q"]) == 2
    assert main(["compare", str(tmp_path / "a.json"), str(tmp_path / "b.json"), "-q"]) == 2
    assert main(["run", str(DEMO / "evals.py"), "--tag", "no-such-tag", "-q"]) == 2
