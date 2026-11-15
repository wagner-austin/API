"""nccl_collisions reads a lock's markers, and this package's lock passes it.

The synthetic locks pin what each marker shape means; the last two cases read
the committed ``poetry.lock`` beside this package, once as it is and once with
``nvidia-nccl-cu12``'s marker widened to every Python, the collision CI run
43700695596 met. The widened copy is what proves the assertion on the real
lock can fail.
"""

from __future__ import annotations

from pathlib import Path

from platform_ml.nccl_lock import PYTHON_VERSIONS, nccl_collisions

_LOCK = Path(__file__).resolve().parents[1] / "poetry.lock"

_CU12_ON_311 = 'markers = "python_version == \\"3.11\\" and platform_system == \\"Linux\\""'
_CU13_FROM_312 = 'markers = "python_version >= \\"3.12\\" and platform_system == \\"Linux\\""'
_LINUX_ONLY = 'markers = "platform_system == \\"Linux\\""'


def _package(name: str, marker_line: str | None) -> str:
    """One ``[[package]]`` table as Poetry writes it.

    Args:
        name: The distribution's name.
        marker_line: Its ``markers = ...`` line, or None for an entry with no
            markers, which installs everywhere.

    Returns:
        The table's text.
    """
    marker = f"{marker_line}\n" if marker_line is not None else ""
    return (
        f'[[package]]\nname = "{name}"\nversion = "1.0"\noptional = false\n{marker}files = []\n\n'
    )


def _widen_cu12(lock_text: str) -> str:
    """Rewrite ``nvidia-nccl-cu12``'s marker so every Linux Python installs it.

    Only that table's marker changes: the same marker text sits on other
    nvidia cu12 tables, which keep theirs.

    Args:
        lock_text: A lock holding an ``nvidia-nccl-cu12`` table with markers.

    Returns:
        The lock with that one marker widened.
    """
    head, table_start, rest = lock_text.partition('name = "nvidia-nccl-cu12"')
    table, next_table, tail = rest.partition("[[package]]")
    marker_start = table.index("markers = ")
    marker_end = table.index("\n", marker_start)
    widened = table[:marker_start] + _LINUX_ONLY + table[marker_end:]
    return head + table_start + widened + next_table + tail


def test_markers_kept_apart_collide_nowhere() -> None:
    lock = _package("nvidia-nccl-cu12", _CU12_ON_311) + _package("nvidia-nccl-cu13", _CU13_FROM_312)

    assert nccl_collisions(lock) == {}


def test_a_widened_marker_names_every_python_holding_both() -> None:
    lock = _package("nvidia-nccl-cu12", _LINUX_ONLY) + _package("nvidia-nccl-cu13", _CU13_FROM_312)

    both = ("nvidia-nccl-cu12", "nvidia-nccl-cu13")
    assert nccl_collisions(lock) == dict.fromkeys(PYTHON_VERSIONS[1:], both)


def test_an_entry_without_markers_installs_everywhere() -> None:
    lock = _package("nvidia-nccl-cu12", None) + _package("nvidia-nccl-cu13", _CU12_ON_311)

    assert nccl_collisions(lock) == {"3.11": ("nvidia-nccl-cu12", "nvidia-nccl-cu13")}


def test_one_family_alone_never_collides() -> None:
    lock = _package("nvidia-nccl-cu13", None) + _package("nvidia-nccl-cu13", _CU13_FROM_312)

    assert nccl_collisions(lock) == {}


def test_other_distributions_are_not_read() -> None:
    lock = _package("nvidia-cublas-cu12", None) + _package("nvidia-nccl-cu13", None)

    assert nccl_collisions(lock) == {}


def test_the_versions_read_run_from_311_through_316() -> None:
    assert PYTHON_VERSIONS == ("3.11", "3.12", "3.13", "3.14", "3.15", "3.16")


def test_this_packages_lock_keeps_one_nccl_per_python() -> None:
    assert nccl_collisions(_LOCK.read_text(encoding="utf-8")) == {}


def test_this_packages_lock_widened_collides() -> None:
    widened = nccl_collisions(_widen_cu12(_LOCK.read_text(encoding="utf-8")))

    both = ("nvidia-nccl-cu12", "nvidia-nccl-cu13")
    assert widened == dict.fromkeys(PYTHON_VERSIONS[1:], both)
