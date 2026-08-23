# 2026-08-23 — P2: per-row sample weights; the class weight becomes the special case it always was

Agent-board task `f8d93064` (ClearGBM program charter P2). Training and
evaluation now accept optional per-row weights on both objectives:
gradients and hessians scale by the effective row weight, the base score
is the weighted log-odds / weighted mean, and both evaluation losses are
weight-averaged. Weights are DATA, not configuration — they travel with
the labels through the train call, nothing enters the config or the model
JSON, and every stored artifact remains valid without retraining.

## The factorization that keeps every identity claim honest

The effective row weight is a product: `eff_i = class_term_i * w_i`, where
the class term is `scale_pos_weight` for positives (1 for negatives,
absent under squared error) and `w_i` the optional per-row weight. Two
deliberate consequences:

- **The weightless arm keeps the exact historical expressions** — no
  synthesized `* 1.0` in the gradient loops, and the base score keeps its
  closed-form class multiply (`spw * W_pos`) rather than distributing spw
  into the accumulation. `sample_weight = None` is therefore bit-identical
  to all recorded history, and the all-ones vector is bit-identical to
  `None` (IEEE multiply by 1.0; integer-valued weight sums are exact).
- **`scale_pos_weight` is now provably the derived special case.** The
  gate test trains spw=3.0-through-config against
  w=3.0-for-positives-through-weights (spw=1.0) and asserts bit equality
  of the base score and every prediction. The weight is integer-valued on
  purpose: 3.0 summed n times equals 3.0 × n exactly in f64, so the
  closed-form and per-row routes provably coincide. (For non-integer
  weights the two routes agree mathematically but may differ in final
  ULPs of the base score — accumulation order — which is why the gate
  pins the provable case rather than overclaiming the general one.)

## Semantics, measured at unit scale

- Knob sensitivity (mandatory per the decorative-knob-class page): a
  non-uniform weight vector changes the model, both objectives, at the
  core, through pyo3, and through the Python surface.
- Meaning: rows carrying 50× weight fit visibly closer — on the
  alternating-target fixture the upweighted rows land at |err| ≈ 0.005
  while the downweighted rows sit at 0.12–0.21.
- Validation weights: the early-stopping loss is weight-averaged with its
  own optional `val_sample_weight`; a validation weight without a
  validation split is rejected, as are zero, negative, non-finite, and
  wrong-length weights (each naming the offending index).

## Equivalence gate: PASS, byte-for-byte

The four-arm benchmark under the weight-capable crate reproduces the
2026-08-22 knob-identity manifest exactly — 56/56 cleargbm quality values
and leaf counts across seeds 42–45, LightGBM/XGBoost anchors identical.
Manifest: `BENCHMARK_MANIFEST_2026-08-23_p2_weight_identity.json`.

## Surface

- cleargbm_rs: `train_gradient_boosting` takes `sample_weight:
  Option<&[f64]>`; `ValidationData` carries an optional evaluation
  weight; `ResolvedValidation` replaces the val tuple. Both pyo3 entries
  take the 8-argument layout `(x, y, sample_weight, x_val, y_val,
  val_sample_weight, config, names)` with array-bundle structs inside.
- cleargbm (Python): `sample_weight=` / `val_sample_weight=` keyword-only
  arguments on both train functions, default `None` — a data default that
  cannot silently change semantics, unlike a config default.
- covenant_ml: all backends run unchanged (weights default off; the
  classifier's auto class weight is untouched). Backend-level weight
  exposure lands with the first weighted corpus (P6 science data), where
  the requirement is real rather than speculative.

Gates at land: cleargbm_rs 1412 tests / clippy -D warnings / 100.00%
segment coverage; cleargbm 216 tests / 100.00%; covenant_ml 2431 /
100.00%; covenant-radar-api 2584 / 100.00%. This unblocks GOSS (P5) and
LambdaMART ranking (P4), both of which are weighted-gradient consumers.
