# Why your agent needs an eval harness (and why it's not that hard)

The failure mode I see most often in agentic systems isn't a crash. It's a silent regression —
a well-intentioned prompt tweak that drops accuracy on 15% of inputs, ships on Friday, and shows
up in your metrics two weeks later as "the model started doing something weird."

Monitoring catches it after it lands. Evals catch it before.

## The three failure modes evals are actually solving for

**1. Prompt drift.** You updated your system prompt to handle a new edge case. The new case works.
Three existing cases broke. You didn't notice because you only tested the new case.

**2. Model version drift.** Your LLM provider upgraded the model. The new version reasons
differently on ambiguous inputs. Your agent's pass rate dropped from 94% to 79%. You found out
from a user.

**3. Tool schema drift.** You added a field to a tool's output schema. Your agent was downstream
of that tool and depended on a field being absent. Now it gets confused.

All three look the same in production: "the agent is doing something wrong." Evals make them
distinguishable and catchable before they reach users.

## What a minimal harness actually needs

You don't need a platform. You need four things:

1. **A golden set** — 15–25 cases that cover your input distribution. Stored as JSONL so they're
   diffable in git. Tagged so you can filter by failure mode.

2. **A judge** — something that evaluates actual vs expected. Start with exact match or numeric
   tolerance (deterministic, free). Escalate to LLM-as-judge only for outputs that require
   semantic evaluation. I use Claude Haiku as judge — good enough, ~$0.001 per case.

3. **A regression diff** — the ability to compare today's run against yesterday's and surface
   which cases newly failed. A pass rate number alone is useless; you need the specific cases.

4. **A CI gate** — the harness runs on every merge. If pass rate drops below threshold (I use 85%
   for experimental agents, 95% for production paths), the merge is blocked.

That's it. No dashboards. No data warehouse. One JSONL file and a pytest fixture.

## The case for keeping judges cheap

There's a temptation to use your best model as judge. Don't. The judge's job is to evaluate
outputs, not generate them — it's a classification task, not a reasoning task. Haiku is more than
capable. The cost difference matters: at $0.80/M input tokens vs $15/M for Opus, you can afford
to run evals on every commit instead of every release.

The right tiering:
- **Exact match** — deterministic outputs (classification labels, JSON fields, numeric values)
- **Numeric tolerance** — cost calculations, metric computations (allow 5% relative error)
- **LLM judge** — free-text outputs, SQL correctness, recommendation quality

## The regression budget

Don't try to maintain 100% pass rate. You'll spend all your time on edge cases and never ship.

Instead, set a regression budget: the maximum number of cases you're allowed to break in a single
change. I use:
- 0 regressions on P0 cases (tagged `critical`)
- ≤2 regressions on the full golden set for any non-breaking change
- ≤5 regressions allowed when changing the core prompt or model

When you exceed the budget, you either fix the regressions or explicitly add the newly failing
cases to a `known_failures.jsonl` and get a second set of eyes. The key is that the decision is
visible and intentional, not silent.

## Human in the loop at the right layer

Evals are not a replacement for human review. They're a filter that decides what reaches human
review.

The mental model I use:
- **Evals gate merges** — automated, fast, cheap. Catch the obvious regressions.
- **Humans gate deploys** — someone reads the eval report before a new version goes live.
- **Users gate rollouts** — canary 10% before 100%.

Evals at the merge layer mean humans are only reviewing the diffs that passed automated screening.
That's a better use of review time than "look at this PR, I think it's probably fine."

## What the golden set should look like

Cover these categories at minimum:
- **Happy path** — clean inputs, expected outputs, no edge cases
- **Near-miss inputs** — inputs that look like they should trigger behavior X but shouldn't
- **Known failure modes** — cases you've already caught in prod; regression tests for real bugs
- **Distribution edges** — min/max values, empty inputs, unusual but valid inputs

You don't need 100 cases. You need representative cases. 20 good cases beat 200 random ones.

## The thing people get wrong

They write evals after a production incident. The harness exists to prevent the incident, not to
document it after the fact.

Write your golden set when you write the agent. The exercise of writing test cases forces you to
specify what the agent is actually supposed to do — which is the hardest part of building agents
that work reliably in production.

---

*This methodology is implemented in [agent-eval-kit](https://github.com/robsavage619/agent-eval-kit).*
