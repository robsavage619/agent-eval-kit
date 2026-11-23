# Methods

The statistics behind each number in a report, and the choices made along the way. Every function named here lives in `src/agent_eval_kit/stats.py` and has a test with a reference value.

## The unit of analysis is the case

A run of 25 cases with 5 trials each produces 125 results, but it is still 25 pieces of evidence. Trials of the same case share an input and usually an outcome, so treating them as independent makes every interval too narrow. Following Miller (2024), every statistic is computed on per-case summaries: a case's pass rate across its trials, and its mean score.

## Pass-rate interval

| Situation | Interval | Why |
|:---|:---|:---|
| one trial per case | Wilson score interval over cases | well behaved at small n and near 0% or 100%, where golden sets live |
| several trials per case, 10+ cases with some spread | percentile bootstrap over cases (2,000 resamples, seeded) | resamples whole cases, so correlated trials are not counted as extra evidence |
| several trials, fewer than 10 cases or no spread | Wilson over all trials | always contains the estimate; somewhat narrow, because trials of a case are correlated |
| cases share a `cluster` (2+ clusters) | mean +/- t(g - 1) x cluster-robust SE | cases from one template or scenario move together; t with g - 1 degrees of freedom because the number of clusters g is usually small |

The cluster-robust standard error sums residuals within each cluster before squaring, with the usual small-sample factor g / (g - 1). With one case per cluster it reduces to the ordinary standard error. With a single cluster there is nothing to estimate, and the run falls back to the unclustered interval.

## Errors

A trial where the agent or the judge raised an exception is an error, not a failure. Errors are kept out of every statistic (pass rate, intervals, comparison, calibration) and counted separately. `aek run` fails when more than `--max-errors` trials errored, so an outage stops the gate without being mistaken for the agent getting worse.

## Reliability: pass^k and pass@k

For a case with c passes in n trials, the unbiased estimates for k <= n are:

- pass^k = C(c, k) / C(n, k): the chance that k fresh trials all pass (Yao et al., tau-bench, 2024)
- pass@k = 1 - C(n - c, k) / C(n, k): the chance that at least one passes (Chen et al., 2021)

Both are averaged over cases. At k = 1 both equal the pass rate; at k = n, pass^k equals the share of cases that passed every trial. The gap between the two curves is a direct read on flakiness.

## Comparing two runs

Each case present in both runs gets a status:

- **flaky** if it passed some trials and failed others *in the baseline*. Its movement now says nothing about the change, so it is set aside.
- **regressed** if it passed a majority of baseline trials and fails a majority now. This includes a case that was solid and now passes only sometimes.
- **improved** for the reverse.
- **degraded** if it passed every baseline trial and now fails some, while still passing most.
- **error** if either run has no scored trials for it.
- **unchanged** otherwise. **new** and **removed** cases are listed but not scored.

The verdict is an exact two-sided McNemar test on regressed versus improved counts. Only cases that changed carry information about the difference between runs, which is why the test is paired. "Regression" requires p < alpha and more regressions than improvements. Degraded cases are reported but not counted in the test.

Cases tagged `critical` follow a stricter rule: any drop in pass rate fails the gate, whatever the status or p-value. That includes a critical case that was flaky in the baseline and got worse, which is deliberate: a critical case should not be flaky in the first place.

The report also gives the mean change in per-case score with a paired bootstrap interval. Miller notes the analytic paired standard error is usually enough; the bootstrap is kept because it is assumption-light and costs nothing at this size.

## Minimum detectable effect

A "no detectable change" is only as strong as the comparison's power, so the report states two limits.

**The paired test.** From Miller's sample-size formula, solved for the effect, the smallest true change in mean score a paired comparison of n cases catches 80% of the time at alpha 0.05 is

```
MDE = (z_0.975 + z_0.80) * sqrt( V / n )
```

where V is the variance of per-case score differences between the two runs. The observed differences already contain each run's sampling noise, so they are used as they are, not with noise added on top. When the runs barely differ that variance is close to zero, so V is floored at the noise expected between two runs of the same agent: the within-case trial variances divided by the trial counts, or, with one trial per case, 2p(1 - p) at a smoothed baseline pass rate (passes + 1) / (n + 2).

**The exact test.** The verdict comes from an exact McNemar test, which cannot reach p < 0.05 until at least 6 cases flip the same way. That is a floor in cases, not a power figure: a 5-point change in pass rate needs at least 6 / 0.05 = 120 cases before the test can see it at all.

`aek power` reports both and plans for the larger. Without a second run it does not know how differently a change moves individual cases; pass `--against` a candidate run (or `--diff-sd`) to include that term, which is often the largest.

## Judge calibration

Labels are (expected, actual, human_passed) triples. The judge scores each one, and the report builds a 2x2 table with "pass" as the positive class.

- **Agreement** with a Wilson interval.
- **Cohen's kappa**, because raw agreement flatters a judge when most outputs pass. A judge that passes everything agrees 90% of the time with a 90%-passing label set and has kappa 0.
- **False-pass rate**: the share of human-failed outputs the judge passed, with a Wilson interval. This is the error that lets bad output through a gate.
- **False-fail rate**: the share of human-passed outputs the judge failed.

Verdict rules, all adjustable in `calibrate_judge`:

| Verdict | Rule |
|:---|:---|
| insufficient_data | fewer than 30 labels |
| trust | kappa >= 0.8 and the upper 95% bound on the false-pass rate <= 10% |
| use_with_care | kappa >= 0.6 |
| do_not_trust | anything else |

The kappa bands are the common Landis and Koch reading (0.6 substantial, 0.8 almost perfect). They are conventions, not laws. Note the consequence of using the upper bound: a judge with zero false passes on 25 bad outputs still has an upper bound near 13%, so it cannot earn "trust" until it has been tested on about 35 bad outputs. That is deliberate.

Judge errors (an exception or an unparseable reply) are excluded from the table and reported separately. If more than 10% of examples error, the verdict is `insufficient_data` regardless of the rest. The report records the judge's fingerprint, so a verdict is tied to one exact configuration, and it warns when either class has too few labels to estimate its error rate.

## Fingerprints

A judge's fingerprint is a short SHA-256 of its class name and configuration (prompts, thresholds, models, nested judges, argument rules), leaving out live objects like API clients. A golden set's fingerprint hashes its cases. Both are stored in every run, and `compare` warns when they differ, because a score that moved because the dataset or the judge changed is not a regression in the agent.

## References

- Miller, E. (2024). *Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations.* arXiv:2411.00640.
- Yao, S. et al. (2024). *tau-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains.* arXiv:2406.12045.
- Chen, M. et al. (2021). *Evaluating Large Language Models Trained on Code.* arXiv:2107.03374.
- Shankar, S. et al. (2024). *Who Validates the Validators? Aligning LLM-Assisted Evaluation of LLM Outputs with Human Preferences.* arXiv:2404.12272.
- Liu, Y. et al. (2023). *G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment.* arXiv:2303.16634.
- Landis, J. R. and Koch, G. G. (1977). The measurement of observer agreement for categorical data. *Biometrics* 33(1).
- Newcombe, R. G. (1998). Two-sided confidence intervals for the single proportion. *Statistics in Medicine* 17(8).
