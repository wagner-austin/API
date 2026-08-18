---
title: Digest folding requires a length prefix per element
tags: [determinism, hashing, injectivity, defect]
related: ["[[passing-test-can-miss-its-own-premise]]"]
source_paths:
  - "src/navprobe/digest.py"
  - "src/navprobe/canonical.py"
  - "tests/test_digest.py"
source_git_blobs:
  "src/navprobe/digest.py": "f8406a5a00711f6d5d7896db5f491a28a9808ef4"
  "src/navprobe/canonical.py": "e766888d77755cf4e32080e97de2b16cb45184ba"
  "tests/test_digest.py": "13b3727c5fa8c3732fc7c506c303db6bcca3813d"
fact_checked: 2026-08-13
confidence: high
hubs: [instrument-design]
---

# Digest folding requires a length prefix per element

Folding a list of strings into one digest by concatenating them is not injective, even when the element count is mixed in separately. Two lists of equal length whose elements concatenate to the same bytes produce the same digest.

This was a live defect in `digest_run`, not a hypothetical. Step digests were concatenated raw, and the fold collided.

## The collision

```
digest_run(['aab', 'b'])  -> 15a3a1685b26255d5b86dde7892bab693786cd4c415a5bf36054e6ad56d17319
digest_run(['aa', 'bb'])  -> 15a3a1685b26255d5b86dde7892bab693786cd4c415a5bf36054e6ad56d17319
```

Both lists have two elements, so the element-count prefix matches. Both flatten to `aabb`, so the payload matches. The digests are therefore identical for two different rollouts.[^1]

For a determinism instrument this is the worst available failure: the fold reports two *different* rollouts as *identical*, which is a false negative on the exact question the instrument exists to answer.

## Why the count prefix was not enough

`digest_run` already mixed in the number of step digests, which separates lists of different lengths — `['aabb']` from `['aa','bb']`.[^2] That prefix does nothing for two lists of the *same* length, because the boundary positions inside the concatenation are still unrecoverable.

The general rule: a count prefix distinguishes *how many* elements there were; only a per-element length prefix distinguishes *where each one ended*.

## The fix

Each digest is length-prefixed through `encode_text`, which emits a four-byte little-endian byte count followed by the UTF-8 payload.[^3] The bound check is shared with `encode_row` so both length-prefixed encoders agree on the limit; a second copy would be free to drift.[^4]

The same principle already governed `encode_row`, whose own docstring states that the prefix is what stops two different shapes from colliding. The defect was that the principle had been applied to rows of floats and not to lists of strings.

Two tests pin the fix, both of which fail against the pre-fix fold: the two-element collision above, and a three-element case that shifts a boundary without changing the concatenation.[^5]

## Note on element width

In practice every value passed to `digest_run` is a 64-character hex digest, and fixed-width elements do concatenate injectively. That is not a defence: `digest_run` accepts any `Sequence[str]` and validates no width, so the guarantee would have rested on a caller convention rather than on the function. The instrument's guarantees have to hold at its own signature.

[^1]: tests/test_digest.py::TestDigestRun::test_equal_length_runs_with_the_same_concatenation_do_not_collide, enforcing src/navprobe/digest.py:56 — `[observed]` — run in the package venv: `python -c "from navprobe.digest import digest_run; print(digest_run(['aab','b']) == digest_run(['aa','bb']))"` printed `True` before the fix.
[^2]: `src/navprobe/digest.py` L56-80, `digest_run` — the fold mixes in `encode_row([float(len(step_digests))])` before iterating.
[^3]: `src/navprobe/canonical.py` L155-185, `encode_text` — "Four bytes of little-endian byte count followed by the UTF-8 payload."
[^4]: `src/navprobe/canonical.py` L108-130, `_require_encodable_length` — called by `encode_row` at L149 and by `encode_text` at L174.
[^5]: `tests/test_digest.py` L80 `test_equal_length_runs_with_the_same_concatenation_do_not_collide` and L91 `test_a_boundary_shift_across_three_steps_is_visible`, both in `TestDigestRun` (L53) — both fail against the pre-fix fold.
