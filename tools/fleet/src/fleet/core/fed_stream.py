"""Feeding one command's output to another through a named pipe, in ``sh``.

Lifted out of :mod:`fleet.core.dialect` (MCPs board task 2f8a81b4) when the
reading to the end below took that module past the guard's 600-line ceiling:
this is the one stream a node's scripts run, the commit's tar out of the
node's mirror into the export (:func:`fleet.core.commit_stage.unpack_script`),
and its rules are about the two ends of a pipe rather than about any one
script.
"""

from __future__ import annotations

import shlex

from fleet.core.dialect import checked_lines


def fed_lines(*, source: tuple[str, ...], sink: tuple[str, ...], fifo: str) -> str:
    """Lines that feed one command's output to another through a named pipe, failing as either does.

    A PIPE WITH BOTH STATUSES, IN ANY ``sh`` (MCPs board task 5280a25d). A
    commit's tree used to be written to a tar file and then extracted from
    it: on colossus on 2026-11-06 ``git archive`` to a file took 1.18 s and
    the extract 0.62 s, against 1.17 s streamed, every gate job paying it
    between its claim and its first phase. ``source | sink`` would end the
    script on the sink's status alone, and GNU tar exits 0 on a stream that
    stops between two members, so an archive that died there would leave
    half a tree that the add then committed. ``set -o pipefail`` is not in
    the dash 0.5.12 of the Ubuntu runners the package's CI uses. So the sink
    reads ``fifo`` in the background, the source writes into it, and the
    script waits for the sink: under :data:`fleet.core.dialect.PROLOGUE`'s
    ``set -e`` a source that fails ends it on the source's status and a sink
    that fails on the sink's. The shell opens ``fifo`` for the source before
    the source runs, so a source that fails at once still closes it and the
    sink ends.

    THE STREAM IS READ TO ITS END, WHATEVER THE SINK READS (MCPs board task
    2f8a81b4). GNU tar stops reading at an archive's two zero blocks and
    exits 0, and ``git archive`` pads its tar past them with up to two more
    10240-byte writes of zeros. When the pipe was full as git reached that
    padding, tar could read the end marker and close the pipe before git's
    last write finished, and git died of SIGPIPE, silently: unpack.sh
    exited 141 with nothing on stderr but the clone's empty-repository
    warning, six times in the seven days to 2026-11-19 on colossus, sedona,
    lacrymal and diphtheria, and on CI run 42312210127. So the sink reads
    ``fifo`` as its standard input, and once it ends ``cat`` reads the same
    descriptor to the end, so the source never writes into a closed pipe;
    the background shell then exits with the sink's status, which the
    ``wait`` returns.

    Args:
        source: The command that writes the stream, an argument vector.
        sink: The command that reads the stream on its standard input, an
            argument vector.
        fifo: Where the named pipe is made, which must not exist yet.

    Returns:
        The lines, each word quoted by :func:`shlex.quote`.
    """
    quoted = shlex.quote(fifo)
    return (
        f"mkfifo {quoted}\n"
        f"{{ fed=0; {checked_lines((sink,)).rstrip()} || fed=$?; cat > /dev/null; "
        f'exit "$fed"; }} < {quoted} &\n'
        f"{checked_lines((source,)).rstrip()} > {quoted}\n"
        'wait "$!"\n'
    )


__all__ = ["fed_lines"]
