"""What a worker's copy of the game must contain, as the pure rules it is."""

from __future__ import annotations

import pytest

from rw_bot.harness.clone import (
    REQUIRED_ENTRIES,
    UNUSED_DIRS,
    VOLATILE_DIRS,
    VOLATILE_FILES,
    CloneError,
    clone_name,
    entries_to_copy,
    missing_requirements,
    verify,
)


def test_clones_are_numbered_from_one_so_they_read_like_worker_labels() -> None:
    assert clone_name(".game-w", 0) == ".game-w1"
    assert clone_name(".game-w", 3) == ".game-w4"


def test_a_negative_worker_index_is_refused() -> None:
    with pytest.raises(CloneError) as caught:
        clone_name(".game-w", -1)
    assert caught.value.code == "RW-CLONE-002"


def test_the_trees_the_game_rewrites_are_not_copied() -> None:
    """Copying them risks reading one while a match already in flight is
    part-way through writing it, and the game rebuilds both on boot.
    """
    copied = entries_to_copy(["assets", "saves", "jvm64", "cache", "game-lib.jar"])
    assert copied == ("assets", "jvm64", "game-lib.jar")
    assert set(VOLATILE_DIRS).isdisjoint(copied)


def test_the_thirty_two_bit_jvm_is_not_copied_because_nothing_names_it() -> None:
    """The launcher names jvm64 for the game, for javac and for jar. At 118 MB
    the tree beside it is the largest thing a clone could carry for nothing, and
    it would be carried once per worker.
    """
    copied = entries_to_copy(["jvm", "jvm64", "game-lib.jar"])
    assert copied == ("jvm64", "game-lib.jar")
    assert UNUSED_DIRS == ("jvm",)


def test_the_settings_file_is_named_as_rewritable() -> None:
    """The game rewrites it on every boot, so a clone would otherwise carry the
    previous match's copy into the next one.
    """
    assert VOLATILE_FILES == ("preferences.ini",)


def test_an_entry_the_game_gains_later_is_copied_rather_than_dropped() -> None:
    """Exclusion, not an allow-list: a directory added by a future patch would
    otherwise be silently missing and show up as a match that will not boot.
    """
    assert "someNewTree" in entries_to_copy(["someNewTree", "saves"])


def test_a_complete_clone_has_nothing_missing() -> None:
    assert missing_requirements(REQUIRED_ENTRIES) == ()
    verify(".game-w1", REQUIRED_ENTRIES)


def test_a_truncated_clone_names_every_missing_path_not_just_the_first() -> None:
    """The failure this replaces is "the agent never opened port N" ninety
    seconds later, which reads like a fault in the agent rather than a bad copy.
    """
    present = [name for name in REQUIRED_ENTRIES if name != "game-lib.jar" and name != "libs"]
    with pytest.raises(CloneError) as caught:
        verify(".game-w2", present)
    assert caught.value.code == "RW-CLONE-001"
    assert "game-lib.jar" in str(caught.value)
    assert "libs" in str(caught.value)
    assert ".game-w2" in str(caught.value)


def test_the_jvm_is_required_because_the_launcher_runs_it_from_the_clone() -> None:
    assert "jvm64/bin/java.exe" in REQUIRED_ENTRIES
