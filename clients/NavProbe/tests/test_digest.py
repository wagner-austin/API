"""Tests for step and run digests."""

from __future__ import annotations

import math

import pytest

from navprobe.canonical import CanonicalEncodingError
from navprobe.digest import DIGEST_SIZE, digest_run, digest_step


class TestDigestStep:
    """Tests for :func:`digest_step`."""

    def test_width_matches_the_pinned_digest_size(self) -> None:
        """The hex digest is exactly twice the pinned byte width."""
        assert len(digest_step(0, [1.0])) == DIGEST_SIZE * 2

    def test_is_lowercase_hexadecimal(self) -> None:
        """The digest is hex, which is what the record format stores."""
        digest = digest_step(0, [1.0])
        assert digest == digest.lower()
        assert all(character in "0123456789abcdef" for character in digest)

    def test_is_stable_across_calls(self) -> None:
        """Equal inputs digest equally, with no per-process salt.

        A salted hash would make the fresh-process comparison meaningless,
        since every restart would report divergence.
        """
        assert digest_step(3, [1.0, 2.0]) == digest_step(3, [1.0, 2.0])

    def test_step_index_is_mixed_in(self) -> None:
        """One observation at two step positions digests differently."""
        assert digest_step(0, [1.0]) != digest_step(1, [1.0])

    def test_observation_changes_the_digest(self) -> None:
        """Different observations at one step digest differently."""
        assert digest_step(0, [1.0]) != digest_step(0, [2.0])

    def test_one_ulp_changes_the_digest(self) -> None:
        """The smallest representable float change is visible."""
        assert digest_step(0, [1.0]) != digest_step(0, [math.nextafter(1.0, 2.0)])

    def test_propagates_nan_rejection(self) -> None:
        """A NaN observation fails rather than digesting."""
        with pytest.raises(CanonicalEncodingError) as caught:
            digest_step(0, [math.nan])
        assert caught.value.code == "NP-CANON-001"


class TestDigestRun:
    """Tests for :func:`digest_run`."""

    def test_width_matches_the_pinned_digest_size(self) -> None:
        """The hex digest is exactly twice the pinned byte width."""
        assert len(digest_run(["aa", "bb"])) == DIGEST_SIZE * 2

    def test_is_stable_across_calls(self) -> None:
        """Equal step sequences fold to equal run digests."""
        assert digest_run(["aa", "bb"]) == digest_run(["aa", "bb"])

    def test_step_order_is_significant(self) -> None:
        """Reordering steps changes the run digest."""
        assert digest_run(["aa", "bb"]) != digest_run(["bb", "aa"])

    def test_empty_run_has_a_defined_digest(self) -> None:
        """A zero-step run digests without special-casing at the call site."""
        assert len(digest_run([])) == DIGEST_SIZE * 2

    def test_empty_runs_agree_with_each_other(self) -> None:
        """Two empty runs compare equal, which is the correct base case."""
        assert digest_run([]) == digest_run([])

    def test_step_count_separates_runs_of_different_lengths(self) -> None:
        """A one-step run cannot collide with a two-step run."""
        assert digest_run(["aabb"]) != digest_run(["aa", "bb"])

    def test_equal_length_runs_with_the_same_concatenation_do_not_collide(self) -> None:
        """Two runs of equal length whose digests concatenate alike stay distinct.

        This is the case the step count does not cover: both lists have two
        elements, and both flatten to ``aabb``. Without a length prefix on each
        element they produce the same run digest, which would report two
        different rollouts as identical — the one error a determinism
        instrument must never make.
        """
        assert digest_run(["aab", "b"]) != digest_run(["aa", "bb"])

    def test_a_boundary_shift_across_three_steps_is_visible(self) -> None:
        """The same property holds past the two-element case."""
        assert digest_run(["a", "ab", "b"]) != digest_run(["aa", "b", "b"])

    def test_run_domain_is_separated_from_step_domain(self) -> None:
        """A run digest never equals a step digest built from the same bytes.

        Without domain separation a single-step run could equal its own step
        digest, and a comparison that mixed the two would report agreement it
        had not established.
        """
        step = digest_step(0, [1.0])
        assert digest_run([step]) != step
