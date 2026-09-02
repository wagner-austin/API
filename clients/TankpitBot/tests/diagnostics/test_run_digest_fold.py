"""The run digest as a resumable fold.

The property that matters is equivalence: folding a run in one pass
and folding the same run in pieces must produce the same digest. The
fleet reads live runs in pieces now, so anything the split changes is
a number the control page would report wrongly.
"""

from __future__ import annotations

import pytest

from tankpit_bot.diagnostics.event_stream import decode_event_lines
from tankpit_bot.diagnostics.run_digest_fold import RunDigestAccumulator
from tankpit_bot.diagnostics.run_digest_types import RunDigestDict
from tankpit_bot.runtime_records import RuntimeEventRecordDict

_RUN = [
    '{"timestamp":"2026-08-06T20:00:00","level":"INFO","logger":"l","mode":"bot",'
    '"channel":"STATE","message":"INITIALIZING"}',
    '{"timestamp":"2026-08-06T20:00:01","level":"INFO","logger":"l","mode":"bot",'
    '"channel":"DIAGNOSTIC","message":"k","diagnostic_kind":"tank_identity","tank_id":601}',
    '{"timestamp":"2026-08-06T20:00:02","level":"INFO","logger":"l","mode":"bot",'
    '"channel":"DIAGNOSTIC","message":"k","diagnostic_kind":"session_room_joined",'
    '"room_id":"World"}',
    '{"timestamp":"2026-08-06T20:00:03","level":"INFO","logger":"l","mode":"bot",'
    '"channel":"DIAGNOSTIC","message":"k","diagnostic_kind":"inventory_sample",'
    '"armor":5,"dual":4,"missile":3,"homing":2,"radar":9}',
    '{"timestamp":"2026-08-06T20:00:04","level":"INFO","logger":"l","mode":"bot",'
    '"channel":"DIAGNOSTIC","message":"k","diagnostic_kind":"radar_dispatch"}',
    '{"timestamp":"2026-08-06T20:00:05","level":"INFO","logger":"l","mode":"bot",'
    '"channel":"WIRE","message":"shoot(231,29)"}',
    '{"timestamp":"2026-08-06T20:00:06","level":"INFO","logger":"l","mode":"bot",'
    '"channel":"DIAGNOSTIC","message":"k","diagnostic_kind":"action_outcome",'
    '"outcome":"hit"}',
    '{"timestamp":"2026-08-06T20:00:07","level":"INFO","logger":"l","mode":"bot",'
    '"channel":"COMBAT","message":"clearance","behavior_reason":"mine_clearance_shot",'
    '"combat_target_x":12,"combat_target_y":34}',
    '{"timestamp":"2026-08-06T20:00:08","level":"INFO","logger":"l","mode":"bot",'
    '"channel":"WIRE","message":"pickup_fuel(12,34)"}',
    '{"timestamp":"2026-08-06T20:00:09","level":"INFO","logger":"l","mode":"bot",'
    '"channel":"DIAGNOSTIC","message":"k","diagnostic_kind":"teleport_displacement",'
    '"requested_x":7,"requested_y":8}',
    '{"timestamp":"2026-08-06T20:00:10","level":"INFO","logger":"l","mode":"bot",'
    '"channel":"WIRE","message":"teleport(7,8)"}',
    '{"timestamp":"2026-08-06T20:00:50","level":"INFO","logger":"l","mode":"bot",'
    '"channel":"WIRE","message":"shoot(9,9)"}',
    '{"timestamp":"2026-08-06T20:00:51","level":"INFO","logger":"l","mode":"bot",'
    '"channel":"DIAGNOSTIC","message":"k","diagnostic_kind":"tank_deactivated",'
    '"victim_id":529,"killer_id":601}',
    '{"timestamp":"2026-08-06T20:00:52","level":"INFO","logger":"l","mode":"bot",'
    '"channel":"DIAGNOSTIC","message":"k","diagnostic_kind":"inventory_sample",'
    '"armor":1,"dual":1,"missile":1,"homing":1,"radar":1}',
]


def _records() -> list[RuntimeEventRecordDict]:
    """Decode the sample run.

    Returns:
        Every record of the run, in file order.
    """
    return decode_event_lines(_RUN)


def _fold_whole() -> RunDigestDict:
    """Fold the sample run in a single pass.

    Returns:
        The digest.
    """
    accumulator = RunDigestAccumulator("sample")
    accumulator.absorb(_records())
    return accumulator.snapshot()


@pytest.mark.parametrize("split", list(range(1, len(_RUN))))
def test_folding_in_two_pieces_matches_folding_in_one(split: int) -> None:
    """Every possible split point produces the identical digest.

    This is the guarantee the live fleet reader depends on: it folds
    whatever bytes arrived since the last poll, and where those
    boundaries fall is decided by the bot's write timing, not by
    anything the reader controls.
    """
    records = _records()
    accumulator = RunDigestAccumulator("sample")
    accumulator.absorb(records[:split])
    accumulator.absorb(records[split:])

    assert accumulator.snapshot() == _fold_whole()


def test_folding_one_record_at_a_time_matches_folding_in_one() -> None:
    """The extreme split: every record its own batch."""
    accumulator = RunDigestAccumulator("sample")
    for record in _records():
        accumulator.absorb([record])

    assert accumulator.snapshot() == _fold_whole()


def test_a_snapshot_does_not_end_the_fold() -> None:
    """Snapshotting mid-run and continuing reaches the same place."""
    records = _records()
    accumulator = RunDigestAccumulator("sample")
    accumulator.absorb(records[:5])
    accumulator.snapshot()
    accumulator.absorb(records[5:])

    assert accumulator.snapshot() == _fold_whole()


def test_an_earlier_snapshot_is_not_changed_by_later_records() -> None:
    """Snapshots are independent, including their nested rows.

    A clearance shot's ``pickup_followed`` flips when its converting
    pickup arrives several records later, so a shallow copy would
    rewrite history inside a digest already handed out.
    """
    records = _records()
    accumulator = RunDigestAccumulator("sample")
    accumulator.absorb(records[:8])
    before = accumulator.snapshot()
    accumulator.absorb(records[8:])
    after = accumulator.snapshot()

    assert before["clearance_shots"][0]["pickup_followed"] is False
    assert after["clearance_shots"][0]["pickup_followed"] is True
    assert before["pickups"] == 0
    assert after["pickups"] == 1


def test_an_open_radar_window_is_charged_to_the_snapshot_not_the_fold() -> None:
    """A still-open radar scan counts as zero-yield once per snapshot.

    Applying that closing step to the running state instead would add
    one more every time the page polled.
    """
    records = decode_event_lines(_RUN[:5])
    accumulator = RunDigestAccumulator("sample")
    accumulator.absorb(records)

    first = accumulator.snapshot()
    second = accumulator.snapshot()

    assert first["zero_yield_radars"] == 1
    assert second["zero_yield_radars"] == 1


def test_the_fold_carries_the_numbers_the_page_reports() -> None:
    """The digest of the sample run, as a whole."""
    digest = _fold_whole()

    assert digest["source"] == "sample"
    assert digest["started_at"] == "2026-08-06T20:00:00"
    assert digest["ended_at"] == "2026-08-06T20:00:52"
    assert digest["duration_s"] == 52
    assert digest["kills"] == 1
    assert digest["shots"] == 2
    assert digest["hits"] == 1
    assert digest["teleports"] == 1
    assert digest["pickups"] == 1
    assert digest["displacements"] == 1
    assert digest["room_id"] == "World"
    assert digest["self_tank_id"] == 601
    assert digest["inventory_first"] == [5, 4, 3, 2, 9]
    assert digest["inventory_last"] == [1, 1, 1, 1, 1]
    assert digest["displacement_top"] == [{"requested_x": 7, "requested_y": 8, "count": 1}]
    assert digest["max_wire_gap_s"] == 40
    assert digest["wire_gaps_over_30s"] == 1


def test_an_empty_fold_describes_an_empty_run() -> None:
    """Absorbing nothing leaves every counter at its unset value."""
    digest = RunDigestAccumulator("sample").snapshot()

    assert digest["started_at"] == ""
    assert digest["duration_s"] == 0
    assert digest["kills"] == 0
    assert digest["self_tank_id"] == -1
    assert digest["timeline"] == []
