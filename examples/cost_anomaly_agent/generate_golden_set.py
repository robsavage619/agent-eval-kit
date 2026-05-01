#!/usr/bin/env python3
"""Generate golden_set.jsonl from synthetic data. Run once; commit the output."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from synthetic_data import daily_totals, generate_cur_rows

CASES = [
    # (id, seed, anomaly_days, expected_anomaly_count_min)
    ("no_anomaly_baseline", 1, [], 0),
    ("single_spike_day5", 2, [5], 1),
    ("single_spike_day15", 3, [15], 1),
    ("single_spike_day28", 4, [28], 1),
    ("two_spikes_sequential", 5, [10, 11], 2),
    ("two_spikes_spread", 6, [3, 22], 2),
    ("three_spikes", 7, [0, 14, 29], 3),
    ("spike_at_start", 8, [0], 1),
    ("spike_at_end", 9, [29], 1),
    ("no_anomaly_high_variance", 10, [], 0),
    ("double_consecutive_mid", 11, [13, 14], 2),
    ("single_spike_day1", 12, [1], 1),
    ("single_spike_day20", 13, [20], 1),
    ("no_anomaly_stable", 14, [], 0),
    ("spike_weekday_mix", 15, [7], 1),
    ("three_spikes_spread", 16, [2, 16, 27], 3),
    ("no_anomaly_growing", 17, [], 0),
    ("single_spike_day10", 18, [10], 1),
    ("two_spikes_early", 19, [1, 4], 2),
    ("single_spike_day25", 20, [25], 1),
]


def main() -> None:
    out = Path(__file__).parent / "golden_set.jsonl"
    lines = []
    for case_id, seed, anomaly_days, min_count in CASES:
        rows = generate_cur_rows(days=30, seed=seed, anomaly_days=anomaly_days)
        totals = daily_totals(rows)
        case = {
            "id": case_id,
            "input": {"daily_totals": totals},
            "expected": {
                "anomaly_count_min": min_count,
                "anomaly_count_max": max(min_count, len(anomaly_days) + 1),
                "has_anomalies": len(anomaly_days) > 0,
            },
            "tags": ["cost_anomaly"] + (["has_spike"] if anomaly_days else ["clean"]),
            "notes": f"seed={seed} anomaly_days={anomaly_days}",
        }
        lines.append(json.dumps(case))
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(lines)} cases to {out}")


if __name__ == "__main__":
    main()
