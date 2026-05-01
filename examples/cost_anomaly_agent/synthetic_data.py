from __future__ import annotations

import random
from datetime import date, timedelta
from typing import Any


def generate_cur_rows(
    days: int = 30,
    seed: int = 42,
    anomaly_days: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Generate synthetic AWS Cost & Usage Report rows.

    anomaly_days: 0-indexed day offsets that should have a 3–5x cost spike.
    """
    rng = random.Random(seed)
    anomaly_set = set(anomaly_days or [])

    services = ["EC2", "S3", "RDS", "Lambda", "CloudFront", "ElastiCache"]
    # Baseline daily spend per service (USD)
    baseline = {"EC2": 420, "S3": 85, "RDS": 210, "Lambda": 40, "CloudFront": 65, "ElastiCache": 130}

    rows = []
    start = date(2026, 1, 1)
    for day_offset in range(days):
        d = start + timedelta(days=day_offset)
        multiplier = rng.uniform(3.2, 5.0) if day_offset in anomaly_set else rng.uniform(0.85, 1.15)
        for svc in services:
            svc_mult = multiplier if day_offset in anomaly_set else rng.uniform(0.9, 1.1)
            cost = round(baseline[svc] * svc_mult + rng.gauss(0, baseline[svc] * 0.02), 2)
            rows.append({
                "date": d.isoformat(),
                "service": svc,
                "cost_usd": max(0.0, cost),
                "usage_quantity": round(cost / rng.uniform(0.01, 0.05), 1),
                "region": rng.choice(["us-east-1", "us-west-2", "eu-west-1"]),
            })
    return rows


def daily_totals(rows: list[dict[str, Any]]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for row in rows:
        totals[row["date"]] = totals.get(row["date"], 0.0) + row["cost_usd"]
    return totals
