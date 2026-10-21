"""Where the bundled runtime's ``java`` is and how a classpath is spelled.

Two facts decide whether a launch command is even syntactically correct: where
the game's own JVM sits inside a game directory, and the character a classpath
is joined by. Neither is a preference. A classpath joined with the wrong
character is not a classpath with a cosmetic flaw -- the JVM reads
``game-lib.jar;libs/*`` as one entry named that, finds no such file, and fails
as a missing main class, which reads like a broken jar.

ONE OWNER, BECAUSE THESE TRAVELLED IN A PACK. Before this module the same
facts were spelled out in ``clone.REQUIRED_ENTRIES``, in the ``JAVA`` variable
and the ``AGENT_CP`` line of the Makefile, and inline in every launcher
script. Six copies of one decision is six places to miss, and the failure of
missing one is not a type error -- it is a match that boots and then dies
ninety seconds later as a channel that never opened.

THE GAME RUNS ON THE LINUX DEPOT, and what that depot ships was read off it on
2026-08-29 rather than assumed: ``jvm-linux``, an Oracle JRE 1.8.0_131 with
``java`` and no ``javac`` or ``jar`` at all. Two consequences, each of which
would otherwise be a launch that dies in the first second:

* Java 8 has no module system, so the engine needs no ``--add-opens`` and the
  JVM REJECTS the option as unrecognised and never starts. No launch here
  passes it.
* An agent cannot be COMPILED with the bundled runtime. The host's own JDK
  builds it (:mod:`rw_bot.harness.agent_build`), and a batch carries the jar
  inside its frozen tree, so a cluster node only ever runs one.

The Windows depot's ``jvm64`` (an OpenJDK 13 with a compiler in it) was the
other half of this module until 2026-10-21, when the operator ruled every
Windows path converted to Linux (board task 0eae9619).
"""

from __future__ import annotations

from collections.abc import Sequence

from rw_bot import RwBotError

_NO_RELEASE_VERSION = "RW-JVM-002"

#: What the bundled JVM's directory is called inside a game directory.
JVM_DIR = "jvm-linux"

#: Where the tools live inside the bundled JVM's directory.
JVM_BIN = "bin"

#: The bundled ``java``, relative to a game directory. Forward-slashed so it
#: reads the same in a clone's manifest as in a staged tree's. The only tool
#: the depot ships: it runs the game and the agent's self-test.
JAVA_PATH = f"{JVM_DIR}/{JVM_BIN}/java"

#: The game's own code and the libraries beside it, in classpath order. The
#: wildcard is expanded by the JVM rather than by a shell, so it stays a
#: literal asterisk here and must not be globbed before the launch.
GAME_CLASSPATH_ENTRIES = ("game-lib.jar", "libs/*")

#: How the entries of a path LIST are joined. One constant for both of this
#: package's path lists, because the operating system has one convention and
#: Java honours it: a JVM classpath and a ``PYTHONPATH`` take the same
#: character, and two constants would be two things to keep right.
PATH_LIST_SEPARATOR = ":"

#: The file a JVM distribution states its own identity in, at the root of its
#: directory. Shipped by whoever built the runtime, so unlike the game's
#: version -- which this project maintains by hand on a wiki page -- it is not
#: a label anybody here can get wrong.
JVM_RELEASE_FILE = "release"

#: The key in that file naming the Java version, e.g. ``JAVA_VERSION="1.8.0_131"``.
JVM_VERSION_KEY = "JAVA_VERSION"

#: What the release file wraps its values in.
_RELEASE_QUOTE = '"'
_RELEASE_ASSIGNMENT = "="


class JvmReleaseError(RwBotError):
    """A bundled runtime would not say which Java version it is.

    Args:
        code: Stable machine-readable identifier.
        message: Human-readable description of what the release file lacked.
    """


def release_version(lines: Sequence[str]) -> str:
    """Read the Java version a bundled runtime states in its release file.

    A pure rule over lines the caller read, so it is exercisable without a
    depot on disk -- which is the whole reason this package keeps its
    decisions separate from its I/O.

    Args:
        lines: The release file's lines, in order. Each is ``KEY="value"``.

    Returns:
        The value of :data:`JVM_VERSION_KEY`, unquoted, e.g. ``1.8.0_131``.
        The version comes off the file rather than from a constant because a
        constant is what this harness ASSUMES and this is what the tree in
        hand actually says: a depot that bumped its runtime would move this
        and not that, and a fingerprint carrying the assumption would report
        two different runtimes as one.

    Raises:
        JvmReleaseError: ``RW-JVM-002`` when no line names the key, or it
            names it with an empty value. Refused rather than reported as
            unknown: a runtime that will not identify itself is not a runtime
            two results may be compared across, and recording "unknown" would
            let them be.
    """
    prefix = f"{JVM_VERSION_KEY}{_RELEASE_ASSIGNMENT}"
    for line in lines:
        if not line.startswith(prefix):
            continue
        version = line[len(prefix) :].strip().strip(_RELEASE_QUOTE)
        if version == "":
            raise JvmReleaseError(
                _NO_RELEASE_VERSION,
                f"the runtime's {JVM_RELEASE_FILE} names {JVM_VERSION_KEY} with an empty "
                "value, so it states no version to record a result against",
            )
        return version
    raise JvmReleaseError(
        _NO_RELEASE_VERSION,
        f"the runtime's {JVM_RELEASE_FILE} has no {JVM_VERSION_KEY} line, so it will not "
        "say which Java it is; two batches on different runtimes would record alike",
    )


def classpath(entries: Sequence[str]) -> str:
    """Join classpath entries.

    Args:
        entries: Classpath entries in order. Wildcards are passed through
            untouched for the JVM to expand.

    Returns:
        The joined classpath.

    Raises:
        ValueError: When no entries are given. An empty classpath is not a
            degenerate case to render as an empty string -- the JVM would
            resolve every class against the current directory instead, which
            is a different program, so the caller is wrong rather than
            unlucky.
    """
    if not entries:
        raise ValueError("a classpath needs at least one entry")
    return PATH_LIST_SEPARATOR.join(entries)


#: The classpath a match runs on: :data:`GAME_CLASSPATH_ENTRIES` joined.
#: Relative, because the game process runs with the game directory as its
#: working directory.
GAME_CLASSPATH = classpath(GAME_CLASSPATH_ENTRIES)


__all__ = [
    "GAME_CLASSPATH",
    "GAME_CLASSPATH_ENTRIES",
    "JAVA_PATH",
    "JVM_BIN",
    "JVM_DIR",
    "JVM_RELEASE_FILE",
    "JVM_VERSION_KEY",
    "PATH_LIST_SEPARATOR",
    "JvmReleaseError",
    "classpath",
    "release_version",
]
