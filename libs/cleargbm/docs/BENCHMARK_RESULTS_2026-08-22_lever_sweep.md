# Benchmark 2026-08-22 — researched-lever sweep: two rejections, one constitutional block

Follow-up to the 2026-08-21 session (interleave, scratch reuse, single-pass;
−30% cumulative) and the same-day tech-wiki research sweep. Agent-board task
`d8c3c955`. Three levers from the research were tested or dispositioned; none
landed. The baseline is unchanged and that is the finding.

## Protocol note: wheel-swap A/B

Two early runs of lever 1 failed the anchor tripwire (LightGBM at 0.4907s
± 0.0076 and 0.4877s ± 0.0130 against the 0.4757s ± 0.0035 reference — both
runs started minutes after crate rebuilds). The fix, used for every verdict
below: build both candidate and baseline wheels FIRST, then install and
benchmark each back-to-back with zero compilation between timed runs. Wheel
installation is a file copy; the anchors stayed clean (0.4790s ± 0.0043 and
0.4761s ± 0.0023).

## Lever 1 — bounds-check elision (REJECTED, ≤1%, not separable)

Zipped stream iteration plus `chunks_exact_mut` feature blocks in
`build_node_histograms_single_pass`, replacing indexed access. Bit-identity
held (all quality metrics and leaf counts byte-identical across seeds).

| arm | depth-wise fit | raw ratio |
|---|---|---|
| baseline | 0.6061s ± 0.0094 | 1.265x |
| elision | 0.6001s ± 0.0141 | 1.260x |

−1.0% with overlapping error bars; the leaf-wise arm moved the other way at
4x the variance. Matches the research prediction (perf-book / Shnatsel
cookbook: 1–3% ceiling, and only where checks block vectorization — which
the FP-reduction bar forbids here anyway). The simpler indexed loop stays.

## Lever 2 — 16-byte bins + separate u32 counts (REJECTED, +20%)

`BinAccumulator` shrunk to LightGBM's 16-byte grad/hess pair; counts moved
to a parallel `u32` array (a few KB per node, L1-resident). Bit-identity
held by construction — and the fit time regressed 20%:

| arm | depth-wise fit | raw ratio |
|---|---|---|
| baseline | 0.6061s ± 0.0094 | 1.265x |
| split counts | 0.7276s ± 0.0389 | 1.488x |

The second read-modify-write per bin update (float record + count slot on a
different cache line) costs far more than the 24-byte record's occasional
line straddle saves. This is the fourth measured loss from splitting fused
work in this codebase's history. LightGBM affords 16-byte records only
because it stores no count array at all; relocating ours is strictly worse
than fusing it. The rejection is recorded in the `BinAccumulator` doc
comment alongside the layout rationale.

## Lever 3 — f32 ordered streams (ALREADY TRIED, not re-measured)

LightGBM's `score_t = float` narrow-stream trick — and the wiki records it
was already shipped on 2026-07-21 and reverted on 2026-07-25 at a recorded
"8% slower" (see `wiki/pages/cleargbm-f32-score-narrowing-reverted.md`; the
magnitude is unreproduced but the mechanism matches this session's
independent analysis: both ordered streams are L2-resident at this corpus
size, so there is no bandwidth to save and every element pays a widening
conversion before its accumulate). Restoring it would also re-add the
crate's only cast-lint exemption — the same lint wall that forbids unsafe,
with no allow-exemptions permitted — and would end bit-identity with the
f64 semantics permanently. Per the wiki page's own retry guidance, this
lever only becomes interesting at a genuinely larger data scale; at this
one it is a measured-and-reverted negative. Not retried.

## Where this leaves the ledger

Baseline unchanged: depth-wise 0.5976–0.6061s on this machine's clean runs,
raw ratio ~1.26x, per-leaf ~0.83x. The micro-optimization space around the
current loop shape is now measured to be exhausted in both directions —
tightening (lever 1) and re-layout (lever 2) both fail to beat it. The
remaining documented levers all require a policy decision: leaf-wise
default (operator, −3%), quantized training (quality gate, up to 2x per
Shi 2022), SIMD via wide/pulp (lint-wall + bit-identity re-baseline), f32
streams (lint-wall + bit-identity). The engine, as constituted, is at its
measured optimum.
