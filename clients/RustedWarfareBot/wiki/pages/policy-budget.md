---
title: "The Budget — One Authority Over a Tick's Credits"
tags: [policy, economy, architecture]
related:
  - "[[policy-loop]]"
  - "[[policy-economy]]"
  - "[[policy-production]]"
source_paths:
  - "src/rw_bot/policy/budget.py"
  - "src/rw_bot/policy/campaign.py"
source_git_blobs:
  "src/rw_bot/policy/budget.py": "06e3cb9d18cf4b87be4d309da6b5a9b52a0c226f"
  "src/rw_bot/policy/campaign.py": "ae3700f5a5c413b05f2909de398d1154d8262b2f"
game_version: "1.15 (code 176, build #28)"
fact_checked: 2026-08-17
confidence: high
hubs: [bot-architecture]
---

# The Budget — One Authority Over a Tick's Credits

The bot had two spenders and one balance.

Inside a single observation, the production pass budgeted across every idle
producer using `sample["credits"]`, and the expansion pass then asked the same
field whether it could afford an extractor. Neither knew what the other had just
committed. **Both were correct in isolation and the pair was not.**

With one factory the overlap was small enough to hide — production could spend
at most one tank's 350 before expansion looked. The moment a second producer
existed, the same credit was committed twice and the engine silently refused the
second order, which reads in a run log exactly like an order that was never
sent.

## The fix is an order, not a bigger reserve

Spending is a single decision with a priority, and `rw_bot.policy.budget` is
that decision. A tick opens one `Budget`; every spender claims against it in
order; a claim that does not fit is **refused with a reason** rather than issued
and dropped by the engine.

Priority is expressed by call order rather than by weights attached to each
claim, because the order *is* the policy and burying it in numbers would make it
unreadable. What the loop does, in order:

1. **the opening plan**, because nothing else can proceed without its
   prerequisites;
2. **replacing losses**, because an army dying now cannot wait for income;
3. **more throughput**, when every producer is busy and the bank is the surplus;
4. **more income**, which pays back over the rest of the match.

Attacking costs nothing and is not arbitrated at all.

## What the reserve is actually for

The reserve is **not** the ordering mechanism — the call order is. It exists for
a claim that has not been made yet.

Production only spends when a producer is *idle*. On a tick where every factory
is busy, production claims nothing, so the lower-priority claims would take
everything and leave nothing for the replacement queued a moment later. Holding
a floor that only a protected claim may cross is a forward reservation, which is
a different job from ordering the claims already in hand.

So: **protected** claims (the plan, replacing losses) may draw on the reserve;
**unprotected** ones (expansion, in both its forms) may not.

## Refusals are the informative half

Every claim is recorded whether or not it succeeded. "No pool was taken" has
several causes calling for opposite responses, and a bare count of expansions
cannot tell them apart — the same reasoning that made
[[policy-economy]] carry its counts.

A refusal names what was wanted, what was available, and how much this tick had
already committed:

```
expand:extractorT1 wanted 700 of 450 available past a 350 reserve;
350 already committed this tick
```

`refused_claims` rides on the match report. A high count against a healthy
balance means the priority order is starving something, which is a tuning
question the run can now pose.

## The claim that cannot fail

The plan claims first and protected, and
`rw_bot.policy.build_order.decide` has already refused to act on a price the
sample's own balance cannot cover — so the plan's claim can never be refused. It
is still *made*, because committing the credits is what stops the later
claimants spending them. The code does not branch on its result, and says why.

Coverage found that: the branch testing it was unreachable, and unreachable
error handling is worse than none, because it reads as a case somebody has
thought about.
