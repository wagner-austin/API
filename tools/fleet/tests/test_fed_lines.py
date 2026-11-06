"""A stream fed through a named pipe fails as either end fails (MCPs board task 5280a25d).

A commit's tree used to land as a tar file and then be extracted from it,
1.18 s and 0.62 s on colossus on 2026-11-06 against 1.17 s streamed, all of
it between a gate job's claim and its first phase. Streamed as
``source | sink`` the script would see the sink's status alone, and GNU tar
exits 0 on a stream that stops between two members. Each case here runs the
rendered lines under the real ``/bin/sh``, which on the package's CI runner
is a dash without ``pipefail``, against a real tar of two members.
"""

from __future__ import annotations

import pathlib
import shlex
import subprocess
import tarfile

from fleet.core import dialect

#: The deadline for the real sh and tar these cases run.
SH_SECONDS = 60

#: Where the first of the two members ends in their tar: a 512-byte header
#: and its 1000 bytes of data padded to 1024.
FIRST_MEMBER_ENDS = 1536


def _two_members(tmp_path: pathlib.Path) -> pathlib.Path:
    """Write a tar of two files, ``a`` and then ``b``.

    Args:
        tmp_path: The case's directory.

    Returns:
        The tar.
    """
    members = tmp_path / "members"
    members.mkdir()
    (members / "a").write_text("a" * 1000, encoding="utf-8")
    (members / "b").write_text("b" * 700, encoding="utf-8")
    archive = tmp_path / "two.tar"
    with tarfile.open(archive, "w", format=tarfile.USTAR_FORMAT) as packed:
        for name in ("a", "b"):
            packed.add(members / name, arcname=name)
    return archive


def _fed(tmp_path: pathlib.Path, source: tuple[str, ...]) -> tuple[int, str, list[str]]:
    """Run a script that feeds ``source`` into tar through a named pipe, then says it ended.

    Args:
        tmp_path: The case's directory.
        source: The command writing the stream.

    Returns:
        The script's status, what it printed, and the files tar extracted.
    """
    out = tmp_path / "out"
    out.mkdir()
    fifo = str(tmp_path / "tree.fifo")
    script = (
        dialect.PROLOGUE
        + dialect.fed_lines(source=source, sink=("tar", "-xmf", fifo, "-C", str(out)), fifo=fifo)
        + dialect.checked_lines((("echo", "reached the end"),))
    )
    done = subprocess.run(
        ("/bin/sh", "-c", script), capture_output=True, text=True, timeout=SH_SECONDS, check=False
    )
    return done.returncode, done.stdout, sorted(path.name for path in out.iterdir())


def test_a_whole_stream_lands_and_the_script_goes_on(tmp_path: pathlib.Path) -> None:
    archive = _two_members(tmp_path)

    status, said, landed = _fed(tmp_path, ("cat", str(archive)))

    assert (status, said, landed) == (0, "reached the end\n", ["a", "b"])


def test_a_source_that_dies_between_two_members_ends_the_script_on_its_own_status(
    tmp_path: pathlib.Path,
) -> None:
    """The case ``source | sink`` passes: tar reads the first member whole,
    meets the end of the stream where the next header would be, and exits
    0 with half the tree."""
    archive = _two_members(tmp_path)
    cut = f"head -c {FIRST_MEMBER_ENDS} {shlex.quote(str(archive))}; exit 3"

    status, said, landed = _fed(tmp_path, ("sh", "-c", cut))

    assert (status, said, landed) == (3, "", ["a"])


def test_a_source_that_fails_at_once_ends_the_script_with_nothing_extracted(
    tmp_path: pathlib.Path,
) -> None:
    status, said, landed = _fed(tmp_path, ("sh", "-c", "exit 5"))

    assert (status, said, landed) == (5, "", [])


def test_a_sink_that_fails_ends_the_script_on_the_sink_s_status(tmp_path: pathlib.Path) -> None:
    status, said, landed = _fed(tmp_path, ("echo", "no tar at all"))

    assert (status, said, landed) == (2, "", [])


def test_the_lines_quote_every_word_and_wait_for_the_sink() -> None:
    lines = dialect.fed_lines(
        source=("git", "archive", "a b"), sink=("tar", "-xf", "/s/p q"), fifo="/s/p q"
    )

    assert lines == (
        "mkfifo '/s/p q'\ntar -xf '/s/p q' &\ngit archive 'a b' > '/s/p q'\nwait \"$!\"\n"
    )
