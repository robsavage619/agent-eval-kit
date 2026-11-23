from __future__ import annotations

import pytest

from agent_eval_kit.stats import bootstrap_mean_ci, cohen_kappa, mcnemar_exact, wilson_interval


def test_wilson_matches_reference_values():
    # Reference: Newcombe (1998), 81/263 -> 0.2553 to 0.3662
    lo, hi = wilson_interval(81, 263)
    assert lo == pytest.approx(0.2553, abs=5e-4)
    assert hi == pytest.approx(0.3662, abs=5e-4)


def test_wilson_handles_extremes_and_empty():
    lo, hi = wilson_interval(0, 20)
    assert lo == 0.0 and 0.15 < hi < 0.17
    assert wilson_interval(0, 0) == (0.0, 1.0)
    with pytest.raises(ValueError):
        wilson_interval(5, 3)


def test_mcnemar_exact():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(3, 3) == 1.0
    # 6 discordant pairs all one way: 2 * (1/64) = 0.03125
    assert mcnemar_exact(6, 0) == pytest.approx(0.03125)
    # 3 regressions, 0 improvements is not significant
    assert mcnemar_exact(3, 0) == pytest.approx(0.25)
    assert mcnemar_exact(0, 6) == mcnemar_exact(6, 0)


def test_bootstrap_ci_is_reproducible_and_brackets_mean():
    values = [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0]
    a = bootstrap_mean_ci(values, seed=7)
    b = bootstrap_mean_ci(values, seed=7)
    assert a == b
    assert a[0] < sum(values) / len(values) < a[1]
    assert bootstrap_mean_ci([]) == (0.0, 0.0)
    assert bootstrap_mean_ci([0.4]) == (0.4, 0.4)


def test_kappa():
    assert cohen_kappa(10, 0, 0, 10) == pytest.approx(1.0)
    # A judge that passes everything agrees 90% with a 90%-passing set, kappa 0.
    assert cohen_kappa(tp=9, fp=1, fn=0, tn=0) == pytest.approx(0.0)
    # Classic textbook table: 20/5/10/15 -> 0.4
    assert cohen_kappa(20, 5, 10, 15) == pytest.approx(0.4)


def test_pass_hat_k_and_pass_at_k():
    from agent_eval_kit.stats import pass_at_k, pass_hat_k

    # 3 of 5 trials passed
    assert pass_hat_k(3, 5, 1) == pytest.approx(0.6)
    assert pass_hat_k(3, 5, 2) == pytest.approx(3 / 10)  # C(3,2)/C(5,2)
    assert pass_hat_k(3, 5, 4) == 0.0
    assert pass_at_k(3, 5, 1) == pytest.approx(0.6)
    assert pass_at_k(3, 5, 3) == pytest.approx(1.0)  # only 2 failures exist
    with pytest.raises(ValueError):
        pass_hat_k(1, 3, 4)


def test_clustered_se_is_larger_when_clusters_move_together():
    from agent_eval_kit.stats import clustered_standard_error, standard_error

    # Two templates: one always passes, one always fails.
    values = [1.0] * 10 + [0.0] * 10
    clusters = ["a"] * 10 + ["b"] * 10
    naive = standard_error(values)
    clustered = clustered_standard_error(values, clusters)
    assert clustered > 2 * naive
    # Independent cases (one per cluster) give roughly the naive error.
    solo = clustered_standard_error(values, [str(i) for i in range(20)])
    assert solo == pytest.approx(naive, rel=0.05)


def test_power_formulas_are_inverse():
    from agent_eval_kit.stats import cases_needed, min_detectable_effect

    n = cases_needed(0.05, diff_variance=0.0, within_variance_a=0.09, within_variance_b=0.09)
    mde = min_detectable_effect(n, 0.0, 0.09, 0.09)
    assert mde <= 0.05 < min_detectable_effect(n - 1, 0.0, 0.09, 0.09)
    # More trials per case shrink the noise term and the cases needed.
    assert cases_needed(0.05, 0.0, 0.09, 0.09, trials_a=5, trials_b=5) < n
