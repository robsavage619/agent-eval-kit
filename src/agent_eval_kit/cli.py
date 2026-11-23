"""``aek``: run, compare, calibrate, and report from the command line.

Exit codes are the contract with CI: 0 means the gate passed, 1 means it
failed, 2 means the command itself was used wrongly.
"""

from __future__ import annotations

import argparse
import asyncio
import math
import sys
from pathlib import Path

from pydantic import ValidationError

from .calibration import calibrate_judge, load_labels
from .compare import compare_runs, min_flips
from .harness import EvalRun
from .reporters import (
    write_calibration_markdown,
    write_comparison_markdown,
    write_junit,
    write_markdown,
)
from .stats import cases_needed, min_detectable_effect
from .suite import load_suite

_TRUST_ORDER = ["insufficient_data", "do_not_trust", "use_with_care", "trust"]


def _load_run(path: str) -> EvalRun:
    return EvalRun.model_validate_json(Path(path).read_text())


def _cmd_run(args: argparse.Namespace) -> int:
    suite = load_suite(args.suite)
    run = asyncio.run(suite.run(repeats=args.repeats, tag=args.tag))
    if run.n_cases == 0:
        what = f"with tag {args.tag!r}" if args.tag else "in the golden set"
        print(f"error: no cases {what}", file=sys.stderr)
        return 2
    if args.out:
        Path(args.out).write_text(run.model_dump_json(indent=2))
    md = write_markdown(run, args.markdown)
    if args.junit:
        write_junit(run, args.junit)
    if not args.quiet:
        print(md)

    if run.n_errors > args.max_errors:
        print(
            f"FAIL: {run.n_errors} trial(s) errored (agent or judge raised); "
            f"allowed {args.max_errors}. Errors are excluded from the pass rate.",
            file=sys.stderr,
        )
        return 1
    if args.min_pass_rate is not None:
        measured = run.pass_rate_ci[0] if args.gate_on == "lower" else run.pass_rate
        if measured < args.min_pass_rate:
            label = "lower CI bound" if args.gate_on == "lower" else "pass rate"
            print(
                f"FAIL: {label} {measured:.1%} is below {args.min_pass_rate:.1%}", file=sys.stderr
            )
            return 1
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    cmp = compare_runs(_load_run(args.baseline), _load_run(args.current), alpha=args.alpha)
    md = write_comparison_markdown(cmp, args.markdown)
    if args.json:
        Path(args.json).write_text(cmp.model_dump_json(indent=2))
    if not args.quiet:
        print(md)
    if args.strict and cmp.regressed:
        print(f"FAIL: {cmp.regressed} case(s) regressed (--strict)", file=sys.stderr)
        return 1
    if cmp.should_fail:
        reason = "critical case regressed" if cmp.critical_regressions else "regression detected"
        print(f"FAIL: {reason}", file=sys.stderr)
        return 1
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    suite = load_suite(args.suite)
    report = asyncio.run(
        calibrate_judge(suite.judge, load_labels(args.labels), min_examples=args.min_examples)
    )
    md = write_calibration_markdown(report, args.markdown)
    if not args.quiet:
        print(md)
    if args.require and _TRUST_ORDER.index(report.verdict) < _TRUST_ORDER.index(args.require):
        print(f"FAIL: judge verdict {report.verdict} is below {args.require}", file=sys.stderr)
        return 1
    return 0


def _cmd_power(args: argparse.Namespace) -> int:
    run = _load_run(args.baseline)
    trials = args.trials or run.repeats
    cases = run.scored_cases
    if run.repeats > 1:
        within = sum(c.score_variance for c in cases) / max(len(cases), 1)
        source = f"measured within-case variance over {run.repeats} trials"
    else:
        within = run.pass_rate * (1 - run.pass_rate)
        source = "binomial variance at the baseline pass rate (one trial per case)"
    noise = 2 * within / trials

    if args.against:
        other = {c.case_id: c for c in _load_run(args.against).scored_cases}
        deltas = [other[c.case_id].mean_score - c.mean_score for c in cases if c.case_id in other]
        mean_d = sum(deltas) / len(deltas) if deltas else 0.0
        diff_var = sum((d - mean_d) ** 2 for d in deltas) / max(len(deltas) - 1, 1)
        diff_source = f"estimated from {args.against} ({len(deltas)} shared cases)"
    else:
        diff_var = args.diff_sd**2
        diff_source = (
            f"--diff-sd {args.diff_sd} (how much the change moves individual cases "
            "differently; often the largest term, so pass --against a candidate run)"
        )
    variance = max(diff_var, noise)
    by_formula = cases_needed(args.delta, variance)
    flips = min_flips()
    by_floor = math.ceil(flips / args.delta)
    mde = min_detectable_effect(len(cases), variance)
    print(
        f"Baseline: {len(cases)} cases, pass rate {run.pass_rate:.1%}\n"
        f"Noise: {source}, {trials} trial(s) per case\n"
        f"Per-case differences: {diff_source}\n\n"
        f"Paired test (Miller 2024, alpha 0.05, 80% power): about {by_formula} cases to detect "
        f"a {args.delta:.0%} change; the current {len(cases)} cases detect about {mde:.0%}.\n"
        f"Exact test floor: at least {flips} cases must flip the same way, so a {args.delta:.0%} "
        f"change in pass rate needs at least {by_floor} cases.\n\n"
        f"Plan for about {max(by_formula, by_floor)} cases."
    )
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    print(write_markdown(_load_run(args.run), args.markdown))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aek", description="Golden-set evals that check their own checks."
    )
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a suite and write a report")
    run.add_argument("suite", help="path/to/evals.py[:variable]")
    run.add_argument("--repeats", type=int, default=None, help="trials per case (default: suite's)")
    run.add_argument("--tag", help="only run cases with this tag")
    run.add_argument("--out", help="write the run as JSON (input to `aek compare`)")
    run.add_argument("--markdown", help="write the markdown report here")
    run.add_argument("--junit", help="write JUnit XML here")
    run.add_argument("--min-pass-rate", type=float, help="fail below this pass rate (0-1)")
    run.add_argument(
        "--gate-on",
        choices=["point", "lower"],
        default="point",
        help="compare --min-pass-rate to the pass rate or to its lower 95%% bound",
    )
    run.add_argument(
        "--max-errors",
        type=int,
        default=0,
        help="fail if more trials than this errored (default 0)",
    )
    run.add_argument("-q", "--quiet", action="store_true")
    run.set_defaults(func=_cmd_run)

    cmp = sub.add_parser("compare", help="compare a run to a baseline, noise-aware")
    cmp.add_argument("baseline")
    cmp.add_argument("current")
    cmp.add_argument("--alpha", type=float, default=0.05)
    cmp.add_argument("--strict", action="store_true", help="fail on any regressed case")
    cmp.add_argument("--markdown", help="write the markdown report here")
    cmp.add_argument("--json", help="write the comparison as JSON here")
    cmp.add_argument("-q", "--quiet", action="store_true")
    cmp.set_defaults(func=_cmd_compare)

    cal = sub.add_parser("calibrate", help="score the suite's judge against human labels")
    cal.add_argument("suite")
    cal.add_argument("labels", help="JSONL of {id, expected, actual, human_passed}")
    cal.add_argument("--min-examples", type=int, default=30)
    cal.add_argument("--require", choices=_TRUST_ORDER, help="fail if the verdict is worse")
    cal.add_argument("--markdown", help="write the markdown report here")
    cal.add_argument("-q", "--quiet", action="store_true")
    cal.set_defaults(func=_cmd_calibrate)

    pw = sub.add_parser("power", help="how many cases you need to detect a given change")
    pw.add_argument("baseline", help="a saved run (JSON) to take variance from")
    pw.add_argument("--delta", type=float, default=0.05, help="change to detect (default 0.05)")
    pw.add_argument("--trials", type=int, help="trials per case you plan to run")
    pw.add_argument("--against", help="a second run (JSON) to estimate per-case differences from")
    pw.add_argument(
        "--diff-sd",
        type=float,
        default=0.0,
        help="SD of per-case score differences between versions, if you know it (default 0)",
    )
    pw.set_defaults(func=_cmd_power)

    rep = sub.add_parser("report", help="render a saved run as markdown")
    rep.add_argument("run")
    rep.add_argument("--markdown", help="write the markdown report here")
    rep.set_defaults(func=_cmd_report)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (FileNotFoundError, ImportError, TypeError, ValueError, ValidationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
