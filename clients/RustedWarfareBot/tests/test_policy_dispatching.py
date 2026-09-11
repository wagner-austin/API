"""The dispatch layer's own contracts, exercised without a match running.

The combat dispatch is covered end to end through the campaign loop; what
lives here is the piece the loop cannot cheaply reach -- the press latch
fires past sample 2,000, far beyond a scripted world's economical length,
and its trace code is a contract the analysis side greps for
([[policy-trace]]).
"""

from __future__ import annotations

from rw_bot.policy.dispatching import _note_releases


def test_each_release_signal_lands_its_own_trace_code() -> None:
    """One code per live signal: S the strike window, C the close, P the
    press. The analysis side attributes releases by these letters, so a
    signal firing without its code is a release the record cannot see."""
    events: set[str] = set()
    _note_releases(events, False, False, False)
    assert events == set()
    _note_releases(events, True, False, False)
    assert events == {"S"}
    _note_releases(events, False, True, False)
    assert events == {"S", "C"}
    _note_releases(events, False, False, True)
    assert events == {"S", "C", "P"}
