"""The pump's health record: the surface a five-day outage had none of.

EVERY CASE HERE IS ABOUT BEING READ BY SOMEONE ELSE. The record exists so
the MCPs session-start hook can tell a session that the wake bridges are or
are not publishing, which is the sentence that was missing while all three
sat refused from 2026-09-16 to 2026-09-21. So the shape is pinned, the
failure streak is pinned in both directions, and an unreadable previous
record is pinned as costing the streak its memory and nothing more -- a tick
that refused to publish because its own bookkeeping was malformed would be
the same outage with a different cause.
"""

from __future__ import annotations

import datetime
import pathlib

from scripts.pump_health import (
    HEALTH_VERSION,
    PublisherHealth,
    next_health,
    read_health,
    render_health,
    write_health,
)

FIRST = datetime.datetime(2026, 9, 21, 20, 43, 15, tzinfo=datetime.UTC)
LATER = datetime.datetime(2026, 9, 21, 20, 46, 15, tzinfo=datetime.UTC)


def _entry(
    name: str, *, exit_code: int = 0, failures: int = 0, last_ok: str = ""
) -> PublisherHealth:
    """Build one previous-record entry.

    Args:
        name: The publisher's marker.
        exit_code: Its exit status.
        failures: Its consecutive failure count.
        last_ok: Its last success instant.

    Returns:
        The entry.
    """
    return PublisherHealth(
        name=name, exit_code=exit_code, consecutive_failures=failures, last_ok=last_ok
    )


class TestNextHealth:
    def test_a_first_green_tick_records_the_instant_and_no_failures(self) -> None:
        health = next_health({}, [("ci-wake", 0)], FIRST)

        assert health["version"] == HEALTH_VERSION
        assert health["written"] == "2026-09-21T20:43:15Z"
        assert health["publishers"] == [
            _entry("ci-wake", last_ok="2026-09-21T20:43:15Z"),
        ]

    def test_a_first_red_tick_counts_one_failure_and_no_last_success(self) -> None:
        """A publisher this record has never seen succeed says so with an
        empty instant rather than with the epoch, which would read as a
        success in 1970 and sort like one."""
        health = next_health({}, [("ci-wake", 1)], FIRST)

        assert health["publishers"] == [_entry("ci-wake", exit_code=1, failures=1)]

    def test_the_streak_grows_and_keeps_the_last_success(self) -> None:
        """THE NUMBER IS THE FINDING. 'failing since the last tick' and
        'failing for 2,700 ticks' are different repairs, and the second is
        what nobody could see between 2026-09-16 and 2026-09-21."""
        previous = {
            "ci-wake": _entry("ci-wake", exit_code=1, failures=2, last_ok="2026-09-16T02:52:40Z")
        }

        health = next_health(previous, [("ci-wake", 1)], LATER)

        assert health["publishers"] == [
            _entry("ci-wake", exit_code=1, failures=3, last_ok="2026-09-16T02:52:40Z"),
        ]

    def test_a_recovery_clears_the_streak_and_moves_the_instant(self) -> None:
        previous = {
            "ci-wake": _entry("ci-wake", exit_code=1, failures=9, last_ok="2026-09-16T02:52:40Z")
        }

        health = next_health(previous, [("ci-wake", 0)], LATER)

        assert health["publishers"] == [_entry("ci-wake", last_ok="2026-09-21T20:46:15Z")]

    def test_publishers_are_recorded_in_publication_order(self) -> None:
        health = next_health({}, [("hpc-wake", 0), ("ci-wake", 2), ("lock-wake", 0)], FIRST)

        assert [p["name"] for p in health["publishers"]] == ["hpc-wake", "ci-wake", "lock-wake"]
        assert [p["exit_code"] for p in health["publishers"]] == [0, 2, 0]


class TestReadHealth:
    def test_a_missing_record_reads_as_no_history(self, tmp_path: pathlib.Path) -> None:
        assert read_health(tmp_path / "pump-health.tsv") == {}

    def test_a_written_record_reads_back_by_name(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "pump-health.tsv"
        write_health(path, next_health({}, [("ci-wake", 4), ("lock-wake", 0)], FIRST))

        back = read_health(path)

        assert set(back) == {"ci-wake", "lock-wake"}
        assert back["ci-wake"]["consecutive_failures"] == 1
        assert back["lock-wake"]["last_ok"] == "2026-09-21T20:43:15Z"

    def test_a_file_of_something_else_reads_as_no_history(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "pump-health.tsv"
        path.write_text("", encoding="utf-8")

        assert read_health(path) == {}

    def test_lines_of_the_wrong_shape_are_dropped_and_the_rest_survive(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Per line, not per file: one publisher's row going bad must not
        erase the streak of the ones beside it, and a tick that refused to
        publish because its own bookkeeping was malformed would be the
        outage again with a new cause."""
        path = tmp_path / "pump-health.tsv"
        path.write_text(
            "\n".join(
                [
                    "version\t1",
                    "written\t2026-09-21T20:43:15Z",
                    "publisher\ttoo-few\t0",
                    "publisher\t\t0\t0\t",
                    "publisher\tbad-code\tnought\t0\t",
                    "publisher\tbad-count\t0\tmany\t",
                    "something-else\tci-wake\t9\t9\t",
                    "publisher\tci-wake\t1\t2\t2026-09-16T02:52:40Z",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        back = read_health(path)

        assert set(back) == {"ci-wake"}
        assert back["ci-wake"]["consecutive_failures"] == 2

    def test_a_signal_death_is_read_back_as_the_negative_code_it_was(
        self, tmp_path: pathlib.Path
    ) -> None:
        """A publisher killed by a signal reports a negative status. A
        reader that dropped those would call a killed publisher unknown,
        which reads like no data rather than like a death."""
        path = tmp_path / "pump-health.tsv"
        write_health(path, next_health({}, [("ci-wake", -9)], FIRST))

        assert read_health(path)["ci-wake"]["exit_code"] == -9

    def test_a_field_that_is_only_a_minus_sign_is_not_a_code(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "pump-health.tsv"
        path.write_text("publisher\tci-wake\t-\t0\t\n", encoding="utf-8")

        assert read_health(path) == {}


class TestRenderHealth:
    def test_the_file_names_its_version_and_its_clock_first(self) -> None:
        """A reader that predates a later shape must be able to say so
        rather than read one field as another, and the hook that reads this
        compares the clock against the pump's interval to tell a failing
        publisher from a pump that has stopped ticking."""
        text = render_health(next_health({}, [("ci-wake", 0)], FIRST))

        assert text.splitlines()[0] == f"version\t{HEALTH_VERSION}"
        assert text.splitlines()[1] == "written\t2026-09-21T20:43:15Z"
        assert text.splitlines()[2] == "publisher\tci-wake\t0\t0\t2026-09-21T20:43:15Z"
        assert text.endswith("\n")
