from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from ..harness import EvalRun


def write_junit(run: EvalRun, path: str | Path, suite_name: str = "agent-evals") -> None:
    suite = ET.Element("testsuite")
    suite.set("name", suite_name)
    suite.set("tests", str(len(run.results)))
    suite.set("failures", str(sum(1 for r in run.results if not r.passed)))
    suite.set("time", f"{sum(r.latency_ms for r in run.results) / 1000:.3f}")

    for r in run.results:
        case = ET.SubElement(suite, "testcase")
        case.set("name", r.case_id)
        case.set("time", f"{r.latency_ms / 1000:.3f}")
        if not r.passed:
            failure = ET.SubElement(case, "failure")
            failure.set("message", r.judge_reason[:200])
            failure.text = f"score={r.score:.3f}\nexpected={r.expected}\nactual={r.actual}"

    tree = ET.ElementTree(suite)
    ET.indent(tree, space="  ")
    tree.write(str(path), encoding="unicode", xml_declaration=True)
