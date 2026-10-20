"""Running rendered bash under a real bash, each command it asks a stand-in.

Support module, not a test module. A rendered payload is run as written, with
every command a case names replaced by a stand-in script that answers as the
case says, so what a test asserts is what the payload does with those
answers, not the text it was rendered as.
"""

from __future__ import annotations

import pathlib
import subprocess
from collections.abc import Mapping, Sequence

from tests._host_bash import bash_path, host_bash


def run_with_stand_ins(
    tmp_path: pathlib.Path,
    lines: Sequence[str],
    commands: Mapping[str, str],
    *,
    options: Sequence[str],
) -> subprocess.CompletedProcess[str]:
    """Run bash lines with stand-in commands.

    Args:
        tmp_path: The case's directory; the stand-ins land in its
            ``stand-ins`` directory, written into the script as bash spells
            it (:func:`tests._host_bash.bash_path`), and the script runs
            from it.
        lines: The bash lines, as rendered.
        commands: Each command the lines ask, by the name or absolute path
            they spell it with, to the body of the bash script that answers
            in its place. An absolute path is rewritten to the stand-in, and
            a bare name found through ``PATH``, where the stand-ins come
            first.
        options: The options bash runs the script under, e.g. ``-u``.

    Returns:
        The finished run, its output as text.
    """
    stand_ins = tmp_path / "stand-ins"
    stand_ins.mkdir()
    spelled = f"{bash_path(tmp_path)}/stand-ins"
    body = "\n".join(lines)
    for command, answer in commands.items():
        name = command.rsplit("/", 1)[-1]
        stand_in = stand_ins / name
        stand_in.write_bytes(f"#!/usr/bin/env bash\n{answer}\n".encode())
        stand_in.chmod(0o755)
        if command.startswith("/"):
            body = body.replace(command, f"{spelled}/{name}")
    (tmp_path / "script.sh").write_bytes(f'PATH="{spelled}:$PATH"\n{body}\n'.encode())
    return subprocess.run(
        [host_bash(), *options, "script.sh"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )


__all__ = ["run_with_stand_ins"]
