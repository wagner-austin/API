"""What a worker's copy of the game must contain, as the pure rules it is."""

from __future__ import annotations

import pytest

from rw_bot.harness.clone import (
    DISPLAY_BASE,
    NO_DISPLAY,
    PLAY_PORT_BASE,
    UNUSED_DIRS,
    VOLATILE_DIRS,
    VOLATILE_FILES,
    CloneError,
    clone_name,
    entries_to_copy,
    leased_display,
    leased_port,
    missing_requirements,
    required_entries,
    verify,
)
from rw_bot.platform_id import WINDOWS

#: A ``sys.platform`` value for the platform the cluster runs, so the rules
#: below are exercised as both rather than as the one the suite happens to be
#: running on.
LINUX = "linux"


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


@pytest.mark.parametrize("platform", [WINDOWS, LINUX])
def test_a_complete_clone_has_nothing_missing(platform: str) -> None:
    complete = required_entries(platform, compiles_agent=True)
    assert missing_requirements(complete, platform, compiles_agent=True) == ()
    verify(".game-w1", complete, platform, compiles_agent=True)


@pytest.mark.parametrize("platform", [WINDOWS, LINUX])
def test_a_truncated_clone_names_every_missing_path_not_just_the_first(platform: str) -> None:
    """The failure this replaces is "the agent never opened port N" ninety
    seconds later, which reads like a fault in the agent rather than a bad copy.
    """
    present = [
        name
        for name in required_entries(platform, compiles_agent=True)
        if name != "game-lib.jar" and name != "libs"
    ]
    with pytest.raises(CloneError) as caught:
        verify(".game-w2", present, platform, compiles_agent=True)
    assert caught.value.code == "RW-CLONE-001"
    assert "game-lib.jar" in str(caught.value)
    assert "libs" in str(caught.value)
    assert ".game-w2" in str(caught.value)


def test_the_jvm_is_required_because_the_launcher_runs_it_from_the_clone() -> None:
    assert "jvm64/bin/java.exe" in required_entries(WINDOWS, compiles_agent=True)
    assert "jvm-linux/bin/java" in required_entries(LINUX, compiles_agent=True)


def test_a_windows_clone_is_not_a_complete_linux_one() -> None:
    """The whole reason the requirement is computed rather than written down:
    a tree carrying java.exe satisfies nothing on a cluster node, and saying so
    here is what turns that into a clone error instead of a launch failure."""
    assert missing_requirements(
        required_entries(WINDOWS, compiles_agent=True), LINUX, compiles_agent=True
    ) == ("jvm-linux/bin/java",)


def test_a_numbered_clone_owns_the_port_its_ordinal_names() -> None:
    """The lease owns the port: random draws collided the first time eight
    matches launched in one instant, and both died on the bind
    (imp-creep12, 2026-08-08)."""
    assert leased_port(".game-w1", ".game-w") == PLAY_PORT_BASE + 1
    assert leased_port(".game-w8", ".game-w") == PLAY_PORT_BASE + 8


def test_a_numbered_clone_owns_the_display_its_ordinal_names() -> None:
    """Same argument as the port: under -nodisplay the engine still opens a
    display, so on a headless node two matches sharing a number race exactly
    as two sharing a port do."""
    assert leased_display(".game-w1", ".game-w") == DISPLAY_BASE + 1
    assert leased_display(".game-w8", ".game-w") == DISPLAY_BASE + 8


def test_no_clone_means_no_server_is_started() -> None:
    """The single-match entry points run wherever the caller already has a
    display, which on a workstation is the desktop."""
    assert leased_display(".game", ".game-w") == NO_DISPLAY
    assert leased_display("elsewhere", ".game-w") == NO_DISPLAY
    assert leased_display(".game-wx", ".game-w") == NO_DISPLAY


def test_the_leased_display_is_never_the_console() -> None:
    """``:0`` is a physical console -- somebody's desktop on a workstation,
    and whatever the last interactive session left on a login node."""
    assert DISPLAY_BASE > 0
    assert leased_display(".game-w1", ".game-w") != 0


def test_the_port_and_the_display_are_leased_from_one_ordinal() -> None:
    """Read from the same clone number rather than parsed twice, so a
    directory cannot be worker 3 for one resource and worker 4 for the
    other."""
    for index in (1, 7, 12):
        name = f".game-w{index}"
        assert leased_port(name, ".game-w") - PLAY_PORT_BASE == index
        assert leased_display(name, ".game-w") - DISPLAY_BASE == index


def test_the_pinned_directory_keeps_the_recipe_draw() -> None:
    """Single-match entry points play in the pinned dir; zero says 'draw'."""
    assert leased_port(".game", ".game-w") == 0


def test_a_directory_outside_the_clone_scheme_keeps_the_recipe_draw() -> None:
    assert leased_port("elsewhere", ".game-w") == 0
    assert leased_port(".game-wx", ".game-w") == 0
