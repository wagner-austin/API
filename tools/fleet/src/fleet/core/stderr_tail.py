"""How much of a failed remote command's standard error a failure carries.

THE END OF STDERR IS WHERE THE REASON IS (MCPs board task ee3d6cdc). ssh
writes its own last word, ``client_loop: send disconnect: Broken pipe`` or
``Connection timed out during banner exchange``, after everything the remote
command wrote, and a failure that carried stderr whole led with whatever came
first. Fleet job af6c4d99 on loki closed ``NODE_UNREACHABLE`` with a detail of
1,101,220 characters, 7778 lines of git's per-file line-ending warnings and
then, last, the broken pipe that ended it; the board stored all of it and
showed the warnings. A failure now carries the last lines only, and says how
many it left out, so a reader is never shown a cut stderr as if it were whole.
"""

from __future__ import annotations

from typing import Final

#: The most lines of stderr a failure carries: its last ones.
TAIL_LINES: Final = 20

#: The most characters those lines may hold, kept from their end, so one
#: very long last line cannot carry the flood back in.
TAIL_CHARS: Final = 4000

#: What a failure says when the command wrote nothing to stderr.
NO_STDERR: Final = "<no stderr>"


def stderr_tail(stderr: str) -> str:
    """The end of a command's standard error, bounded, naming what it left out.

    Args:
        stderr: Everything the command wrote to standard error.

    Returns:
        :data:`NO_STDERR` when it wrote nothing; otherwise its last
        :data:`TAIL_LINES` lines, at most :data:`TAIL_CHARS` characters from
        their end, led by how many earlier lines or characters were left out.
    """
    lines = stderr.strip().splitlines()
    if not lines:
        return NO_STDERR
    kept = "\n".join(lines[-TAIL_LINES:])
    left_lines = max(len(lines) - TAIL_LINES, 0)
    left_chars = max(len(kept) - TAIL_CHARS, 0)
    tail = kept[left_chars:]
    if left_lines == 0 and left_chars == 0:
        return tail
    return f"[{left_lines} earlier line(s) and {left_chars} character(s) of stderr left out] {tail}"


__all__ = ["NO_STDERR", "TAIL_CHARS", "TAIL_LINES", "stderr_tail"]
