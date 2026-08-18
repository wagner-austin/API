---
title: A test can encode the right intent and check the wrong case
tags: [testing, coverage, false-confidence]
related: ["[[digest-fold-requires-length-prefix]]"]
source_paths:
  - "tests/test_digest.py"
  - "src/navprobe/digest.py"
source_git_blobs:
  "tests/test_digest.py": "13b3727c5fa8c3732fc7c506c303db6bcca3813d"
  "src/navprobe/digest.py": "f8406a5a00711f6d5d7896db5f491a28a9808ef4"
fact_checked: 2026-08-13
confidence: high
hubs: [instrument-design]
---

# A test can encode the right intent and check the wrong case

The collision in [[digest-fold-requires-length-prefix]] shipped under a green suite with 100% statement and branch coverage. A test existed whose stated purpose was to rule that collision out. It passed, and the collision was real.

This page is about how that happens, because the mechanism is general and coverage does not detect it.

## The test that gave false confidence

```python
def test_length_is_mixed_in_separately_from_content(self) -> None:
    """A one-step run cannot collide with a two-step run.

    The step count is folded in, so concatenating step digests differently
    cannot produce the same run digest.
    """
    assert digest_run(["aabb"]) != digest_run(["aa", "bb"])
```

The docstring makes the correct claim — concatenating differently must not collide. The assertion checks `["aabb"]` against `["aa","bb"]`: **one** element against **two**. Those are separated by the element-count prefix that was already present, so the assertion passes without exercising the property at all.

The case that actually falsifies the claim needs equal lengths: `["aab","b"]` against `["aa","bb"]`. Same count, same concatenation, different lists.

## Why coverage cannot see this

Both inputs traverse the same statements and the same branches inside `digest_run`. There is no line the weak assertion misses. Coverage answers "was this code executed", and the defect lives in *which values* were passed, not in which lines ran. 100% branch coverage and this defect are entirely compatible.

## The generalisable shape

The failure mode is a test whose **example is separated by a different mechanism than the one under test**. The author reasons about property P, constructs an example, and the example happens to also differ in property Q — which some other part of the system already handles. The test passes because of Q. P is never exercised.

Two questions catch it at review time:

1. *If I deleted the mechanism this test names, would this test fail?* Here, deleting the per-element prefix left the test green, so the answer was no.
2. *What else distinguishes my two inputs?* If the answer is "their lengths", and length is separately prefixed, the example is not testing what it claims.

The second question is the cheap one. Any test asserting "X and Y do not collide" should have its X and Y checked for an accidental second difference.

## What was done

The test was replaced by two that fail against the pre-fix fold, rather than repaired in place — the original assertion is still true and still worth keeping, so it survives under an honest name (`test_step_count_separates_runs_of_different_lengths`) describing what it actually checks.[^1]

The finding was reached by running the suspected collision against the installed code, not by reading it.[^2] Reading the fold would have shown a count prefix and a loop, which is exactly what the original author saw.

[^1]: `tests/test_digest.py` L53-103 `TestDigestRun` now carries `test_step_count_separates_runs_of_different_lengths` (L76, the original assertion, renamed to its real scope), `test_equal_length_runs_with_the_same_concatenation_do_not_collide` (L80), and `test_a_boundary_shift_across_three_steps_is_visible` (L91`.
[^2]: src/navprobe/digest.py:56 `digest_run` — `[observed]` — `python -c "from navprobe.digest import digest_run; print(digest_run(['aab','b']) == digest_run(['aa','bb']))"` printed `True` against the pre-fix code.
