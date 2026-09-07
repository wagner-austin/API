"""The settling policy, over simulated time and against the measured case.

Every test here supplies its own ``now_epoch``, because the policy is pure.
There is no clock to fake and no file to write, which is the point of
keeping it in its own module: the rules are exercised at whatever spacing
the argument needs, including spacings a real cluster would take hours to
produce.
"""

from __future__ import annotations

from typing import Final

import pytest
from hpc3.contracts.closure import Closure
from platform_core.error_codes_tooling import HpcWakeErrorCode
from platform_core.errors import AppError

from hpc_wake.pending import PendingClosure
from hpc_wake.settling import (
    BATCH_CAP,
    MAX_HOLD_SECONDS,
    SETTLE_SECONDS,
    PendingGroup,
    group_pending,
    is_ripe,
    partition_ripe,
)

#: The group every record below belongs to unless a test says otherwise.
KEY: Final[tuple[str, str]] = ("rusted", "opus-scale-arm-0819")

#: The gap this package was changed for: the median spacing between endings
#: in the 24-hour burst that produced 116 posts.
MEASURED_GAP_SECONDS: Final = 180


def record(job_id: str, observed_epoch: int) -> PendingClosure:
    """Build one pending record.

    Args:
        job_id: The job's id.
        observed_epoch: When this bridge first saw it terminal.

    Returns:
        The record.
    """
    return PendingClosure(
        closure=Closure(
            job_id=job_id,
            state="COMPLETED",
            closed_at="2026-09-07T00:00:00+00:00",
            elapsed_seconds=10,
        ),
        observed_epoch=observed_epoch,
    )


def group(*records: PendingClosure) -> PendingGroup:
    """Build one group under the standing key.

    Args:
        records: The waiting endings.

    Returns:
        The group.
    """
    return PendingGroup(key=KEY, records=records)


def keys_for(*job_ids: str) -> dict[str, tuple[str, str]]:
    """Map every id to the standing key.

    Args:
        job_ids: Ids to map.

    Returns:
        The mapping :func:`partition_ripe` needs.
    """
    return dict.fromkeys(job_ids, KEY)


class TestIsRipe:
    """The three rules, each proved to fire and to not fire."""

    def test_a_group_that_has_gone_quiet_is_ripe(self) -> None:
        assert is_ripe(group(record("1", 1000)), 1000 + SETTLE_SECONDS) is True

    def test_a_group_one_second_short_of_quiet_is_held(self) -> None:
        assert is_ripe(group(record("1", 1000)), 1000 + SETTLE_SECONDS - 1) is False

    def test_quiet_is_measured_from_the_newest_member(self) -> None:
        # The oldest member is long past the settle window; the newest is
        # not. Measuring from the oldest would announce a group that is
        # still actively gaining members, which is the whole failure.
        held = group(record("old", 1000), record("new", 1000 + SETTLE_SECONDS))
        assert is_ripe(held, 1000 + SETTLE_SECONDS + 1) is False

    def test_a_full_group_is_ripe_however_recently_it_grew(self) -> None:
        # Every member arrived this instant, so it is neither quiet nor
        # aged. Only the size rule can fire.
        full = group(*(record(str(index), 5000) for index in range(BATCH_CAP)))
        assert is_ripe(full, 5000) is True

    def test_one_short_of_full_and_still_busy_is_held(self) -> None:
        nearly = group(*(record(str(index), 5000) for index in range(BATCH_CAP - 1)))
        assert is_ripe(nearly, 5000) is False

    def test_an_aged_group_is_ripe_even_though_it_never_goes_quiet(self) -> None:
        """THE RULE THAT BOUNDS LATENCY, and the reason quiet is not enough.

        A member arrives every MEASURED_GAP_SECONDS forever. The group is
        never quiet, and it is capped below BATCH_CAP here, so without the
        aged rule this group would wait indefinitely -- the quiet rule
        cannot bound itself.
        """
        arrivals = [
            record(str(index), 1000 + index * MEASURED_GAP_SECONDS)
            for index in range(BATCH_CAP - 1)
        ]
        newest = arrivals[-1]["observed_epoch"]
        now = 1000 + MAX_HOLD_SECONDS
        assert now - newest < SETTLE_SECONDS, "guard: this group must not be quiet"
        assert len(arrivals) < BATCH_CAP, "guard: this group must not be full"
        assert is_ripe(group(*arrivals), now) is True


class TestPartitionRipe:
    """Splitting the waiting set, and never splitting a group."""

    def test_a_trickling_array_coalesces_instead_of_posting_per_job(self) -> None:
        """THE MEASURED DEFECT, as a test.

        Twelve endings arrive 180 seconds apart -- the observed median gap.
        At every arrival the group is checked exactly as a cycle would check
        it. Under the old behaviour this produced twelve posts. It must now
        produce none until the array stops, because 180 < SETTLE_SECONDS.
        """
        keys = keys_for(*(str(index) for index in range(12)))
        waiting: list[PendingClosure] = []
        announcements = 0
        for index in range(12):
            now = 1000 + index * MEASURED_GAP_SECONDS
            waiting.append(record(str(index), now))
            ripe, waiting = partition_ripe(waiting, keys, now)
            announcements += 1 if ripe != [] else 0
        assert announcements == 0
        assert len(waiting) == 12

        # The array stops. One settle window later the whole sweep lands as
        # a single post carrying all twelve.
        quiet_at = 1000 + 11 * MEASURED_GAP_SECONDS + SETTLE_SECONDS
        ripe, holding = partition_ripe(waiting, keys, quiet_at)
        assert len(ripe) == 12
        assert holding == []

    def test_a_ripe_group_and_a_held_group_are_separated(self) -> None:
        quiet = record("quiet", 1000)
        busy = record("busy", 9000)
        keys = {"quiet": ("rusted", "a"), "busy": ("mi", "b")}
        ripe, holding = partition_ripe([quiet, busy], keys, 9000)
        assert [r["closure"]["job_id"] for r in ripe] == ["quiet"]
        assert [r["closure"]["job_id"] for r in holding] == ["busy"]

    def test_a_just_arrived_member_goes_with_its_aged_group(self) -> None:
        """A group is announced whole, including what landed a second ago.

        The oldest member is past MAX_HOLD_SECONDS, so the group is ripe.
        The newest arrived this instant and has settled by no rule of its
        own -- and it is announced anyway, because the alternative is
        posting the sweep without it and then posting it alone later, which
        is the per-job shape this module exists to remove.

        The first version of this test asserted the opposite and failed. It
        was the test that was wrong: it claimed the group would be HELD
        because the newest member was busy, forgetting that ``aged`` reads
        the OLDEST. Recorded rather than quietly corrected, because the
        mistake is the easy one to make about this rule.
        """
        keys = keys_for("old", "new")
        ripe, holding = partition_ripe(
            [record("old", 1000), record("new", 1000 + MAX_HOLD_SECONDS)],
            keys,
            1000 + MAX_HOLD_SECONDS,
        )
        assert [r["closure"]["job_id"] for r in ripe] == ["old", "new"]
        assert holding == []

    def test_a_group_that_is_ripe_by_no_rule_is_held_whole(self) -> None:
        """The other direction: neither member leaves early either."""
        keys = keys_for("a", "b")
        ripe, holding = partition_ripe([record("a", 9000), record("b", 9100)], keys, 9100)
        assert ripe == []
        assert [r["closure"]["job_id"] for r in holding] == ["a", "b"]

    def test_an_empty_waiting_set_yields_nothing(self) -> None:
        assert partition_ripe([], {}, 1000) == ([], [])


class TestGroupPending:
    """Grouping, and the integrity failure it must not swallow."""

    def test_groups_are_ordered_by_key(self) -> None:
        keys = {"z": ("zeta", ""), "a": ("alpha", "")}
        grouped = group_pending([record("z", 1), record("a", 1)], keys)
        assert [g["key"] for g in grouped] == [("alpha", ""), ("zeta", "")]

    def test_a_record_with_no_ledger_entry_raises_rather_than_being_dropped(self) -> None:
        """A dropped record is a post nobody will ever receive.

        This is the one outcome the bridge exists to prevent, so it fails
        the cycle with a traceable code instead of quietly announcing the
        rest and leaving one ending unmentioned forever.
        """
        with pytest.raises(AppError) as raised:
            group_pending([record("orphan", 1)], {})
        # Compared by IDENTITY against the enum member, which is how
        # test_cycle.py already asserts codes. Reading ``.code.value``
        # instead yields an Any expression and mypy refuses it under
        # ``disallow_any_expr`` -- correctly, since the string is the one
        # part of an error code that can drift without anything noticing.
        assert raised.value.code is HpcWakeErrorCode.JOB_UNKNOWN_TO_LEDGER
        assert "orphan" in raised.value.message


def test_the_settle_window_exceeds_the_gap_it_was_chosen_against() -> None:
    """The constant is a measurement, and this is the measurement.

    A settle window at or below the observed median gap would coalesce
    nothing: every arrival would find its group already quiet and post
    immediately, reproducing the defect while looking like a fix. If anyone
    lowers SETTLE_SECONDS to 180 or less, this fails and says why.
    """
    assert SETTLE_SECONDS > MEASURED_GAP_SECONDS


def test_the_hold_ceiling_exceeds_the_settle_window() -> None:
    """Otherwise the aged rule would pre-empt quiet and nothing would batch."""
    assert MAX_HOLD_SECONDS > SETTLE_SECONDS
