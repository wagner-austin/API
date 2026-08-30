"""Whether two runs of one configuration were the same simulation.

The trace lines below are REAL, copied verbatim from
``runs/bracket-ff1-trace.ndjson`` and its ``ff2`` twin -- two of the runs the
2026-08-07 certification was read from. They are embedded rather than read
from ``runs/``, which the repository ignores: a test that depended on an
untracked artifact would pass here and fail on a fresh clone.

Run against the full archived pair, this module's comparison reproduces the
wiki's result exactly -- identical over 250 samples fast-vs-fast, identical
over 150 realtime-vs-realtime.
"""

from __future__ import annotations

import pytest

from rw_bot.harness.replication import (
    NO_FORK,
    TRACE_COLUMNS,
    WORLD_COLUMN,
    ReplicationError,
    compare_pair,
    panel_holds,
    render_verdict,
    world_digests,
)

HEADER = (
    "   frame  army  credits  enemies  extractors  lost  producers  idle  orders"
    "  refused    worth    rival  income  rival_income       world      plan  workers"
)

#: Five real samples of a certified run.
SAMPLES = (
    "       0     0     4004        2           0     0          0     0       0"
    "        0     3500     3500      18            25   761625314  building        1",
    "      75     0     4009        2           0     0          1     1       1"
    "        0     3500     3500      18            25  1492531150  building        1",
    "     150     0     3518        2           0     0          1     0       0"
    "        0     3500     3500      18            25  3872622743  building        1",
    "     225     0     3522        2           0     0          1     0       0"
    "        0     3500     3500      18            25   783148845  building        1",
    "     300     0     3527        2           0     0          1     0       0"
    "        0     3500     3500      18            25  1284636018  building        1",
)

TRACE = (HEADER, *SAMPLES)

#: The same run with one digest moved, which is what a fork looks like: the
#: worlds agree for a while and then one consequential draw lands differently.
#:
#: Written out rather than sliced: a slice of a fixed-length tuple types as
#: ``Any`` under this package's mypy settings, which forbids it outright.
FORKED = (
    HEADER,
    SAMPLES[0],
    SAMPLES[1],
    SAMPLES[2],
    SAMPLES[3].replace("783148845", "783148846"),
    SAMPLES[4],
)

#: The same run cut short, as an interrupted match leaves it.
SHORT = (HEADER, SAMPLES[0], SAMPLES[1], SAMPLES[2])


class TestReadingATrace:
    def test_the_digests_are_read_off_the_world_column(self) -> None:
        assert world_digests(TRACE) == (
            (0, 761625314),
            (75, 1492531150),
            (150, 3872622743),
            (225, 783148845),
            (300, 1284636018),
        )

    def test_the_column_indices_are_the_real_traces(self) -> None:
        """Copied from the archived certification runs, so a header the
        analyser cannot read is a header this catches."""
        assert len(HEADER.split()) == TRACE_COLUMNS
        assert HEADER.split()[WORLD_COLUMN] == "world"

    def test_a_header_naming_other_columns_is_refused(self) -> None:
        """The income pair was once inserted BEFORE the digest rather than
        appended, so this index has moved before. Comparing the wrong column
        would report two runs as identical because they agreed about
        something else."""
        moved = " ".join(["frame"] + [f"c{n}" for n in range(1, TRACE_COLUMNS)])
        with pytest.raises(ReplicationError) as caught:
            world_digests((moved, *SAMPLES))
        assert caught.value.code == "RW-REPLICATE-001"

    def test_a_header_with_too_few_columns_is_refused(self) -> None:
        with pytest.raises(ReplicationError) as caught:
            world_digests(("frame army world", *SAMPLES))
        assert caught.value.code == "RW-REPLICATE-001"

    def test_no_lines_at_all_is_refused(self) -> None:
        with pytest.raises(ReplicationError) as caught:
            world_digests(())
        assert caught.value.code == "RW-REPLICATE-001"

    def test_a_line_that_is_not_a_sample_is_skipped(self) -> None:
        """A killed match leaves its last line half-written, and a trace can
        carry a blank one. Neither is a sample, and treating a truncated line
        as one would compare two runs on a column that is not there."""
        truncated = "     375     0     3531        2           0     0"
        assert world_digests((HEADER, SAMPLES[0], "", truncated, SAMPLES[1])) == (
            (0, 761625314),
            (75, 1492531150),
        )

    def test_a_trace_whose_only_lines_are_unreadable_is_refused(self) -> None:
        """The skip above must not turn into a pass: a trace of nothing but
        half-written lines records no samples at all."""
        with pytest.raises(ReplicationError) as caught:
            world_digests((HEADER, "", "     375     0     3531"))
        assert caught.value.code == "RW-REPLICATE-002"

    def test_a_trace_with_no_samples_is_refused(self) -> None:
        """An interrupted match leaves a header and nothing else, and two of
        those would otherwise compare equal to each other."""
        with pytest.raises(ReplicationError) as caught:
            world_digests((HEADER,))
        assert caught.value.code == "RW-REPLICATE-002"


class TestComparingAPair:
    def test_two_runs_of_one_configuration_are_identical(self) -> None:
        """The property the whole determinism stack exists to give: same seed,
        same match. Certified on a workstation under Java 13; this panel is
        what asks the same question of the cluster's Java 8."""
        verdict = compare_pair(9, world_digests(TRACE), world_digests(TRACE))
        assert verdict["identical"] is True
        assert verdict["forked_at"] == NO_FORK
        assert verdict["samples"] == len(SAMPLES)

    def test_a_fork_names_the_frame_it_happened_at(self) -> None:
        """Divergence is not drift -- it is one consequential draw landing a
        unit over -- so the frame is the finding."""
        verdict = compare_pair(9, world_digests(TRACE), world_digests(FORKED))
        assert verdict["identical"] is False
        assert verdict["forked_at"] == 225
        assert verdict["samples"] == 3

    def test_two_runs_of_different_lengths_are_not_identical(self) -> None:
        """One was cut short, which is a finding in itself. Comparing only the
        overlap would report a truncated match as a replicated one."""
        short = world_digests(SHORT)
        verdict = compare_pair(9, world_digests(TRACE), short)
        assert verdict["identical"] is False
        assert (verdict["left_samples"], verdict["right_samples"]) == (5, 3)
        assert verdict["forked_at"] == 150

    def test_a_pair_sampling_different_frames_forks_there(self) -> None:
        """Two runs that sampled different frames did not run the same
        simulation, whatever their digests say."""
        shifted = tuple(line.replace("     150 ", "     151 ") for line in TRACE)
        verdict = compare_pair(9, world_digests(TRACE), world_digests(shifted))
        assert verdict["identical"] is False
        assert verdict["forked_at"] == 150


class TestTheRendering:
    def test_an_identical_pair_reads_as_one(self) -> None:
        line = render_verdict(compare_pair(9, world_digests(TRACE), world_digests(TRACE)))
        assert "identical over 5 sample(s)" in line

    def test_a_fork_names_the_frame(self) -> None:
        line = render_verdict(compare_pair(9, world_digests(TRACE), world_digests(FORKED)))
        assert "FORKED at frame 225" in line

    def test_a_length_mismatch_says_which_lengths(self) -> None:
        short = world_digests(SHORT)
        line = render_verdict(compare_pair(9, world_digests(TRACE), short))
        assert "DIFFERENT LENGTHS 5 vs 3" in line

    def test_the_seed_leads_so_a_column_sorts_by_it(self) -> None:
        line = render_verdict(compare_pair(12345, world_digests(TRACE), world_digests(TRACE)))
        assert line.startswith("seed ")
        assert "12345" in line


class TestWhetherThePanelHolds:
    def test_every_pair_identical_holds(self) -> None:
        pair = compare_pair(9, world_digests(TRACE), world_digests(TRACE))
        assert panel_holds([pair, pair]) is True

    def test_one_fork_is_enough_to_fail_it(self) -> None:
        good = compare_pair(9, world_digests(TRACE), world_digests(TRACE))
        bad = compare_pair(10, world_digests(TRACE), world_digests(FORKED))
        assert panel_holds([good, bad]) is False

    def test_an_empty_panel_certifies_nothing(self) -> None:
        """A run that compared nothing has not certified anything, and
        reporting it as a pass is how a regime goes unchecked."""
        assert panel_holds([]) is False

    def test_an_empty_tuple_also_certifies_nothing(self) -> None:
        """Tested on the length rather than against ``[]``: a tuple of no
        verdicts is not equal to an empty list, and the first version of this
        would have passed one."""
        assert panel_holds(()) is False
