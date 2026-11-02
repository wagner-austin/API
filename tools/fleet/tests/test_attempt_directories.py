"""The scratch directory each launch attempt is given (MCPs board task 2e3e69cf).

Fleet job e9009971 was claimed twice on pendragon, and its second attempt's
bundles went in the directory the first had left files in; removing it
raised ENOTEMPTY and ended the node's serving loop. The real seam is asked
here, on the real filesystem, for two directories with one job's prefix.
"""

from __future__ import annotations

import pathlib

from fleet.core import _test_hooks_attempts


def test_two_attempts_of_one_job_are_given_two_new_empty_directories(
    tmp_path: pathlib.Path,
) -> None:
    parent = tmp_path / "fleet-archives"
    first = _test_hooks_attempts.fresh_directory(parent, "e9009971-")
    (first / "left-by-the-first.bundle").write_bytes(b"cut off mid-launch")

    second = _test_hooks_attempts.fresh_directory(parent, "e9009971-")

    assert first != second
    assert (first.parent, second.parent) == (parent, parent)
    assert first.name.startswith("e9009971-")
    assert second.name.startswith("e9009971-")
    assert list(second.iterdir()) == []
