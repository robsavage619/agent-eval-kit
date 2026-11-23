"""Check the checker: measure a judge against human labels.

An LLM judge is a model grading a model. Before its pass rate means anything,
you need to know how often it agrees with a person, and in particular how often
it passes an output a person would fail. That false-pass rate is the number
that lets bad answers through your CI gate.

Label 30 to 100 real outputs by hand (pass or fail), run ``calibrate_judge``,
and keep the labels in git next to the golden set. Re-run it whenever you
change the judge's prompt, model, or threshold: the report records the judge's
fingerprint so a stale verdict is easy to spot.

Two habits from the literature are worth keeping. Include plenty of bad
outputs, not just good ones, or the false-pass rate cannot be measured. And do
not tune the judge's prompt on the same labels you calibrate against; hold some
back, because people's criteria also drift as they grade (Shankar et al., "Who
Validates the Validators?", 2024).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from .judges.base import Judge
from .stats import cohen_kappa, wilson_interval

Verdict = Literal["trust", "use_with_care", "do_not_trust", "insufficient_data"]


class LabeledExample(BaseModel):
    id: str
    expected: dict[str, Any]
    actual: dict[str, Any]
    human_passed: bool
    notes: str = ""


class Disagreement(BaseModel):
    id: str
    human_passed: bool
    judge_passed: bool
    judge_score: float
    judge_reason: str
    notes: str = ""

    @property
    def kind(self) -> Literal["false_pass", "false_fail"]:
        return "false_pass" if self.judge_passed else "false_fail"


class CalibrationReport(BaseModel):
    n: int
    true_pass: int
    false_pass: int
    false_fail: int
    true_fail: int
    agreement: float
    agreement_ci: tuple[float, float]
    kappa: float
    false_pass_rate: float
    false_pass_ci: tuple[float, float]
    false_fail_rate: float
    verdict: Verdict
    disagreements: list[Disagreement] = Field(default_factory=list)
    judge: str = ""
    """Fingerprint of the judge config this verdict applies to. Any change to
    the prompt, model, or threshold makes it stale: re-run calibration."""
    judge_config: dict[str, Any] = Field(default_factory=dict)
    judge_errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def load_labels(path: str | Path) -> list[LabeledExample]:
    examples = []
    for lineno, line in enumerate(Path(path).read_text().splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            examples.append(LabeledExample.model_validate_json(line))
        except ValueError as exc:
            raise ValueError(f"{path}:{lineno}: invalid label: {exc}") from exc
    return examples


def _verdict(
    n: int,
    kappa: float,
    false_pass_upper: float,
    min_examples: int,
    trust_kappa: float,
    care_kappa: float,
    max_false_pass: float,
) -> Verdict:
    if n < min_examples:
        return "insufficient_data"
    if kappa >= trust_kappa and false_pass_upper <= max_false_pass:
        return "trust"
    if kappa >= care_kappa:
        return "use_with_care"
    return "do_not_trust"


async def calibrate_judge(
    judge: Judge,
    examples: list[LabeledExample],
    concurrency: int = 4,
    min_examples: int = 30,
    trust_kappa: float = 0.8,
    care_kappa: float = 0.6,
    max_false_pass: float = 0.1,
    max_error_rate: float = 0.1,
) -> CalibrationReport:
    """Score ``judge`` against human labels.

    Verdict rules (all adjustable):

    - ``insufficient_data``: fewer than ``min_examples`` labels.
    - ``trust``: kappa at least ``trust_kappa`` and the upper end of the 95%
      interval on the false-pass rate at most ``max_false_pass``.
    - ``use_with_care``: kappa at least ``care_kappa``.
    - ``do_not_trust``: anything else.

    If more than ``max_error_rate`` of the examples produce a judge error, the
    verdict is ``insufficient_data`` regardless: a judge that cannot answer is
    not one you have measured.

    The kappa bands follow the common Landis and Koch reading (0.6 substantial,
    0.8 almost perfect). They are conventions, not laws; tighten them for a
    judge that gates production changes.
    """
    sem = asyncio.Semaphore(concurrency)

    async def score(ex: LabeledExample) -> tuple[LabeledExample, float, bool, str] | str:
        async with sem:
            try:
                s, p, reason = await judge.evaluate(ex.expected, ex.actual)
            except Exception as exc:  # no verdict; keep it out of the agreement table
                return f"{ex.id}: {type(exc).__name__}: {exc}"
            return ex, s, p, reason

    outcomes = await asyncio.gather(*[score(ex) for ex in examples])
    errors = [o for o in outcomes if isinstance(o, str)]
    scored = [o for o in outcomes if not isinstance(o, str)]

    tp = fp = fn = tn = 0
    disagreements: list[Disagreement] = []
    for ex, s, judge_passed, reason in scored:
        if judge_passed and ex.human_passed:
            tp += 1
        elif judge_passed and not ex.human_passed:
            fp += 1
        elif not judge_passed and ex.human_passed:
            fn += 1
        else:
            tn += 1
        if judge_passed != ex.human_passed:
            disagreements.append(
                Disagreement(
                    id=ex.id,
                    human_passed=ex.human_passed,
                    judge_passed=judge_passed,
                    judge_score=s,
                    judge_reason=reason,
                    notes=ex.notes,
                )
            )

    n = tp + fp + fn + tn
    human_fails = fp + tn
    human_passes = tp + fn
    warnings: list[str] = []
    min_per_class = max(1, min_examples // 3)
    if human_fails < min_per_class:
        warnings.append(
            f"only {human_fails} human-failed examples; the false-pass rate needs bad outputs "
            f"to measure (aim for {min_per_class}+)"
        )
    if human_passes < min_per_class:
        warnings.append(
            f"only {human_passes} human-passed examples; the false-fail rate is unreliable "
            f"(aim for {min_per_class}+)"
        )
    if errors:
        warnings.append(f"{len(errors)} example(s) produced a judge error and were excluded")
    error_rate = len(errors) / len(examples) if examples else 0.0
    kappa = cohen_kappa(tp, fp, fn, tn)
    false_pass_ci = wilson_interval(fp, human_fails)
    return CalibrationReport(
        n=n,
        true_pass=tp,
        false_pass=fp,
        false_fail=fn,
        true_fail=tn,
        agreement=(tp + tn) / n if n else 0.0,
        agreement_ci=wilson_interval(tp + tn, n),
        kappa=kappa,
        false_pass_rate=fp / human_fails if human_fails else 0.0,
        false_pass_ci=false_pass_ci,
        false_fail_rate=fn / human_passes if human_passes else 0.0,
        verdict=(
            "insufficient_data"
            if error_rate > max_error_rate
            else _verdict(
                n, kappa, false_pass_ci[1], min_examples, trust_kappa, care_kappa, max_false_pass
            )
        ),
        disagreements=disagreements,
        judge=judge.fingerprint(),
        judge_config=judge.describe(),
        judge_errors=errors,
        warnings=warnings,
    )
