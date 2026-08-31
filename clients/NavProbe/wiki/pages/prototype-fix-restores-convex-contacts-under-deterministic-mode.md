---
title: A three-part prototype restores convex contacts under deterministic mode, and its cost is measurable
tags: [warp, determinism, contacts, collision, fix, prototype, upstream]
related: ["[[deterministic-mode-drops-contacts-in-convex-narrowphase]]", "[[tactile-alias-patch-clears-warp-deterministic-compile]]", "[[a-determinism-verdict-needs-a-correctness-oracle]]", "[[kernel-split-fix-restores-convex-contacts-at-upstream-head]]"]
provenance:
  - "mujoco-warp 3.11.0"
  - "warp-lang 1.16.0"
  - "NVIDIA Warp 1.16 user guide, Deterministic Execution"
fact_checked: 2026-08-30
confidence: high
measured_with:
  package: mujoco-warp 3.11.0
  warp: 1.16.0
  backend: warp cuda:0
  device: NVIDIA GeForce RTX 3090 Ti (sm_86, 84 SMs, host austinpc)
  modes: [NOT_GUARANTEED, RUN_TO_RUN]
  deterministic_max_records: 4096
  step_count: 40
  repetitions: 3 (single pair), 4 (six-pair sweep)
  patch: the alias patch plus three prototype patches, all reversible
hubs: [determinism-measurement]
---

# A three-part prototype restores convex contacts under deterministic mode, and its cost is measurable

The failure and its mechanism are on [[deterministic-mode-drops-contacts-in-convex-narrowphase]]: three gates make the contact-slot reservation count depend on EPA results the counting pass cannot compute. This page is the remedy, and what it costs.

Neutralising all three gates makes the reservation count depend only on how many threads enter the kernel — which the counting pass can compute — while the writes stay gated on the execution pass's real EPA results. Under `RUN_TO_RUN`, box-on-box then **rests on the box at 0.598057, the default-mode height, with contact on every step and one distinct digest from three repetitions**.[^13] All three convex pairs recover:

| pair | routes | `RUN_TO_RUN` before | after |
|---|---|---:|---:|
| sphere on plane | `PRIMITIVE` | 80 | 80 |
| sphere on box | `PRIMITIVE` | 74 | 74 |
| mesh on plane | `PRIMITIVE` | 116 | **320** |
| mesh on box | `CONVEX` | **0** | 3100 |
| mesh on mesh | `CONVEX` | **0** | 3600 |
| box on box | `CONVEX` | **0** | 4000 |

**This is a validated direction, not a shippable patch**, and three costs are why. The
shippable version now exists: a kernel split built in upstream's own repository removes all
three costs at once — exact counts, untouched primitive path, no stale slots
([[kernel-split-fix-restores-convex-contacts-at-upstream-head]]). This page stays as the
record of how the direction was validated and what the naive remedy costs.

- **It changes primitive pairs too.** `write_contact` is shared by both narrowphases, so making its reservation unconditional inflates counts wherever the `detected` early-return used to fire — mesh-on-plane moves 116 → 320 despite being a primitive pair. Sphere pairs are untouched only because their contacts are always detected. An earlier draft of this page claimed primitive pairs would be unaffected; that was wrong.
- **Over-reserved slots are counted but never written**, so they hold whatever the previous step left there. Physics came out correct on the one pair where it was checked end to end, which is not the same as safe.
- **It is slow.** Convex-pair wall times move from roughly 13-28 s to 197-292 s, from removing an early-out and running `MJ_MAXCONPAIR` = 50 loop iterations per candidate where 1-4 were needed.

A production version would gate all three changes behind `wp.static(deterministic)` so the default path pays nothing, reserve a tighter bound than 50, and initialise unused slots rather than leaving them stale. None of that is done here.

## Why each part is necessary

Applied alone, the first two do nothing at all -- measured, not assumed. Hoisting `write_contact`'s reservation changed neither the contact count nor the final height; adding the fixed trip count on top of it changed nothing either. Both repair code the counting pass never reaches, because the early return has already exited the kernel. Only once that return falls through do the other two matter. A patch shipping either of the first two alone would look like a no-op and be blamed on the wrong thing.

[^13]: `[observed]` — three reversible vendor patches applied together: the EPA early-return turned into a fall-through, the contact loop given a `wp.static(MJ_MAXCONPAIR)` trip count, and `write_contact`'s reservation hoisted above its `detected` branch. box-on-box, `cuda:0`, alias patch applied, 3 repetitions × 40 steps, `deterministic_max_records = 4096`: default mode 2000 contacts / final z 0.598057 / 1 digest; `RUN_TO_RUN` 2000 contacts / final z 0.598057 / 1 digest. The unpatched default-mode baseline is also 0.598057, so the default path is unregressed. Six-pair sweep run through `scripts.collision_pair_probe` under `RUN_TO_RUN`. All patches reverted and the venv verified canonical afterwards.
