#!/usr/bin/env python3
"""Regenerate the README figures from the committed demo baseline.

    python scripts/make_figures.py

Writes docs/assets/reliability.svg from examples/support_agent/baseline.json,
so the chart in the README is real output, not an illustration.
"""

from __future__ import annotations

from pathlib import Path

from agent_eval_kit import EvalRun

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "examples" / "support_agent" / "baseline.json"
OUT = ROOT / "docs" / "assets" / "reliability.svg"

W, H = 760, 340
L, R, T, B = 70, 180, 40, 60  # margins
INK, MUTED, GRID = "#0f172a", "#64748b", "#e2e8f0"
HAT, AT = "#e11d48", "#0891b2"


def _x(k: int, kmax: int) -> float:
    return L + (k - 1) / (kmax - 1) * (W - L - R)


def _y(v: float, lo: float) -> float:
    return T + (1 - (v - lo) / (1 - lo)) * (H - T - B)


def main() -> None:
    run = EvalRun.model_validate_json(BASELINE.read_text())
    curve = run.reliability_curve()
    kmax = len(curve)
    lo = 0.8
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
        f'font-family="ui-sans-serif, -apple-system, Segoe UI, Helvetica, Arial, sans-serif">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        f'<text x="{L}" y="24" font-size="15" font-weight="600" fill="{INK}">'
        f"Same run, two questions: {run.n_cases} cases x {run.repeats} trials</text>",
    ]
    for v in (0.8, 0.85, 0.9, 0.95, 1.0):
        y = _y(v, lo)
        parts.append(f'<line x1="{L}" x2="{W - R}" y1="{y:.1f}" y2="{y:.1f}" stroke="{GRID}"/>')
        parts.append(
            f'<text x="{L - 10}" y="{y + 4:.1f}" font-size="12" fill="{MUTED}" '
            f'text-anchor="end">{v:.0%}</text>'
        )
    for k, _, _ in curve:
        parts.append(
            f'<text x="{_x(k, kmax):.1f}" y="{H - B + 22}" font-size="12" fill="{MUTED}" '
            f'text-anchor="middle">k = {k}</text>'
        )
    for idx, color, label in (
        (1, HAT, "pass^k: passes every one of k tries"),
        (2, AT, "pass@k: passes at least once in k"),
    ):
        pts = " ".join(f"{_x(c[0], kmax):.1f},{_y(c[idx], lo):.1f}" for c in curve)
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="3"/>')
        for c in curve:
            parts.append(
                f'<circle cx="{_x(c[0], kmax):.1f}" cy="{_y(c[idx], lo):.1f}" r="4.5" fill="{color}"/>'
            )
        last = curve[-1]
        parts.append(
            f'<text x="{W - R + 12}" y="{_y(last[idx], lo) + 4:.1f}" font-size="13" '
            f'font-weight="600" fill="{color}">{last[idx]:.0%}</text>'
        )
        ly = H - 14 if idx == 1 else H - 14
        lx = L if idx == 1 else L + 300
        parts.append(f'<rect x="{lx}" y="{ly - 9}" width="12" height="12" rx="2" fill="{color}"/>')
        parts.append(f'<text x="{lx + 18}" y="{ly + 1}" font-size="12" fill="{INK}">{label}</text>')
    parts.append("</svg>")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(parts) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
