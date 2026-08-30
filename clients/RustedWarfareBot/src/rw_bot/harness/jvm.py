"""Where the bundled JDK's tools are and how a classpath is spelled.

Two facts decide whether a launch command is even syntactically correct, and
both of them change with the operating system: the JDK's tools carry ``.exe``
on Windows and nothing anywhere else, and a classpath is joined by the
platform's path-list separator. Neither is a preference. A classpath joined
with the wrong character is not a classpath with a cosmetic flaw -- the JVM
reads ``game-lib.jar:libs/*`` as one entry named that, finds no such file, and
fails as a missing main class, which reads like a broken jar.

THE SEPARATOR ITSELF IS NOT THIS MODULE'S. It belongs to
:func:`~rw_bot.platform_id.path_list_separator`, because the operating system
has one convention for path lists and Java honours it -- a classpath and a
``PYTHONPATH`` take the same character, and defining it twice would be two
things to keep right with one of them eventually wrong.

ONE OWNER, BECAUSE THESE TRAVELLED IN A PACK. Before this module the same
facts were spelled out in ``clone.REQUIRED_ENTRIES``, in the ``JAVAC``/``JAVA``
/``JAR`` variables and the ``AGENT_CP`` line of the Makefile, and inline in
every launcher script. Six copies of one decision is six places to miss, and
the failure of missing one is not a type error -- it is a match that boots and
then dies ninety seconds later as a channel that never opened.

THE TWO PLATFORMS SHIP DIFFERENT RUNTIMES, and this was read off the real
depots on 2026-08-29 rather than assumed. Windows ships ``jvm64``, an OpenJDK
13 with a compiler in it. Linux ships ``jvm-linux``, an Oracle JRE 1.8.0_131
with ``java`` and no ``javac`` or ``jar`` at all. Three consequences, each of
which would otherwise be a launch that dies in the first second of a scheduled
job:

* The directory is named differently, so a path built from one constant is
  wrong on the other platform.
* ``--add-opens`` is a Java 9 option. On the Linux JRE it is not merely
  unnecessary -- the JVM rejects the unrecognised option and never starts.
* An agent cannot be COMPILED on Linux. That is fine and needs no compiler
  there: a batch builds its agent once and carries the jar inside its frozen
  tree, so a cluster node only ever runs one.

An earlier version of this module declared ``jvm64`` for both and called the
name "a contract this harness enforces" -- which would have meant staging
renaming a directory Steam ships under another name, on the strength of an
assumption that turned out to be wrong.
"""

from __future__ import annotations

from collections.abc import Sequence

from rw_bot import RwBotError
from rw_bot.platform_id import is_windows, path_list_separator

_UNKNOWN_TOOL = "RW-JVM-001"
_NO_RELEASE_VERSION = "RW-JVM-002"

#: What the JDK's tools are called on Windows and on nothing else.
WINDOWS_EXECUTABLE_SUFFIX = ".exe"

#: The JDK tools this harness names. ``java`` runs the game and the self-test,
#: ``javac`` and ``jar`` build the agent when a batch is not reusing a frozen
#: tree. Listed so an unknown name is refused rather than composed into a path
#: that will not exist.
JVM_TOOLS = ("java", "javac", "jar")

#: What the bundled JVM's directory is called, per platform.
#:
#: Read off the real depots on 2026-08-29, not assumed: the Windows runtime
#: depot ships ``jvm64`` and the Linux one ships ``jvm-linux``. An earlier
#: version of this module declared ``jvm64`` for both and called it "a
#: contract this harness enforces" -- which would have meant staging renaming
#: a directory Steam ships under another name, for no reason beyond the
#: assumption being wrong.
WINDOWS_JVM_DIR = "jvm64"
POSIX_JVM_DIR = "jvm-linux"

#: The Java major version each platform's bundled JVM is, read off their
#: ``release`` files. Windows ships OpenJDK 13; Linux ships Oracle JRE
#: 1.8.0_131. They are DIFFERENT RUNTIMES, which is why the launch command
#: cannot be one string with the paths swapped.
WINDOWS_JVM_MAJOR = 13
POSIX_JVM_MAJOR = 8

#: The first Java release that enforces module boundaries, and therefore the
#: first that understands ``--add-opens``. Below it the option is not merely
#: unnecessary -- the JVM rejects it as unrecognised and never starts.
MODULE_SYSTEM_MAJOR = 9

#: The JDK tools a bundled runtime carries, per platform.
#:
#: The Linux depot ships a JRE: its ``bin`` holds ``java`` and no ``javac`` or
#: ``jar``. So a match can be PLAYED there and an agent cannot be BUILT there,
#: which is fine -- a batch compiles its agent once and ships the jar inside
#: its frozen tree.
WINDOWS_JVM_TOOLS = ("java", "javac", "jar")
POSIX_JVM_TOOLS = ("java",)

#: Where the tools live inside the bundled JVM's directory.
JVM_BIN = "bin"

#: The game's own code and the libraries beside it, in classpath order. The
#: wildcard is expanded by the JVM rather than by a shell, so it stays a
#: literal asterisk here and must not be globbed before the launch.
GAME_CLASSPATH_ENTRIES = ("game-lib.jar", "libs/*")

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


class JvmToolError(RwBotError):
    """A name was given for a JDK tool this harness does not launch.

    Args:
        code: Stable machine-readable identifier.
        message: Human-readable description of what was asked for.
    """


class JvmReleaseError(RwBotError):
    """A bundled runtime would not say which Java version it is.

    Args:
        code: Stable machine-readable identifier.
        message: Human-readable description of what the release file lacked.
    """


def executable_name(tool: str, platform: str) -> str:
    """Return the filename one JDK tool has on a platform.

    Args:
        tool: One of :data:`JVM_TOOLS`.
        platform: A ``sys.platform`` value.

    Returns:
        The tool's filename, carrying :data:`WINDOWS_EXECUTABLE_SUFFIX` on
        Windows and bare elsewhere.

    Raises:
        JvmToolError: ``RW-JVM-001`` when the tool is not one this harness
            launches. Refused rather than suffixed blindly, because the only
            way a wrong name reaches here is a typo, and a typo composed into
            a path fails as a missing file at launch instead of here.
    """
    if tool not in JVM_TOOLS:
        raise JvmToolError(
            _UNKNOWN_TOOL,
            f"{tool!r} is not a JDK tool this harness launches; "
            f"expected one of {', '.join(JVM_TOOLS)}",
        )
    return f"{tool}{WINDOWS_EXECUTABLE_SUFFIX}" if is_windows(platform) else tool


def jvm_dir(platform: str) -> str:
    """Return the bundled JVM's directory inside a game directory.

    Args:
        platform: A ``sys.platform`` value.

    Returns:
        :data:`WINDOWS_JVM_DIR` or :data:`POSIX_JVM_DIR`.
    """
    return WINDOWS_JVM_DIR if is_windows(platform) else POSIX_JVM_DIR


def jvm_major(platform: str) -> int:
    """Return the Java major version a platform's bundled JVM is.

    Args:
        platform: A ``sys.platform`` value.

    Returns:
        :data:`WINDOWS_JVM_MAJOR` or :data:`POSIX_JVM_MAJOR`.
    """
    return WINDOWS_JVM_MAJOR if is_windows(platform) else POSIX_JVM_MAJOR


def has_module_system(platform: str) -> bool:
    """Report whether a platform's JVM understands ``--add-opens``.

    Args:
        platform: A ``sys.platform`` value.

    Returns:
        True when the bundled JVM is Java 9 or later. On Java 8 the option is
        rejected as unrecognised and the JVM never starts, so this is the
        difference between a match and a launch that dies in a second.
    """
    return jvm_major(platform) >= MODULE_SYSTEM_MAJOR


def release_version(lines: Sequence[str]) -> str:
    """Read the Java version a bundled runtime states in its release file.

    A pure rule over lines the caller read, so both platforms' release files
    are exercisable from either platform -- which is the whole reason this
    package keeps its decisions separate from its I/O.

    Args:
        lines: The release file's lines, in order. Each is ``KEY="value"``.

    Returns:
        The value of :data:`JVM_VERSION_KEY`, unquoted, e.g. ``1.8.0_131``.
        The version comes off the file rather than from
        :func:`jvm_major` because that constant is what this harness ASSUMES
        about a platform and this is what the tree in hand actually says: a
        depot that bumped its runtime would move this and not that, and a
        fingerprint carrying the assumption would report two different
        runtimes as one.

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


def bundled_tools(platform: str) -> tuple[str, ...]:
    """Return the JDK tools a platform's bundled runtime actually carries.

    Args:
        platform: A ``sys.platform`` value.

    Returns:
        :data:`WINDOWS_JVM_TOOLS` or :data:`POSIX_JVM_TOOLS`. The Linux depot
        ships a JRE, so a match can be played there and an agent cannot be
        compiled there.
    """
    return WINDOWS_JVM_TOOLS if is_windows(platform) else POSIX_JVM_TOOLS


def tool_path(tool: str, platform: str) -> str:
    """Return one JDK tool's path inside a game directory.

    Args:
        tool: One of :data:`JVM_TOOLS`.
        platform: A ``sys.platform`` value.

    Returns:
        The path relative to a game directory root, forward-slashed so it
        reads the same in a clone's manifest on either platform.

    Raises:
        JvmToolError: ``RW-JVM-001`` when the tool is unknown.
    """
    return f"{jvm_dir(platform)}/{JVM_BIN}/{executable_name(tool, platform)}"


def classpath(entries: Sequence[str], platform: str) -> str:
    """Join classpath entries for a platform.

    Args:
        entries: Classpath entries in order. Wildcards are passed through
            untouched for the JVM to expand.
        platform: A ``sys.platform`` value.

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
    return path_list_separator(platform).join(entries)


def game_classpath(platform: str) -> str:
    """Return the classpath a headless match runs on.

    Args:
        platform: A ``sys.platform`` value.

    Returns:
        :data:`GAME_CLASSPATH_ENTRIES` joined for the platform. Relative,
        because the game process runs with the game directory as its working
        directory.
    """
    return classpath(GAME_CLASSPATH_ENTRIES, platform)


__all__ = [
    "GAME_CLASSPATH_ENTRIES",
    "JVM_BIN",
    "JVM_RELEASE_FILE",
    "JVM_TOOLS",
    "JVM_VERSION_KEY",
    "MODULE_SYSTEM_MAJOR",
    "POSIX_JVM_DIR",
    "POSIX_JVM_MAJOR",
    "POSIX_JVM_TOOLS",
    "WINDOWS_EXECUTABLE_SUFFIX",
    "WINDOWS_JVM_DIR",
    "WINDOWS_JVM_MAJOR",
    "WINDOWS_JVM_TOOLS",
    "JvmReleaseError",
    "JvmToolError",
    "bundled_tools",
    "classpath",
    "executable_name",
    "game_classpath",
    "has_module_system",
    "jvm_dir",
    "jvm_major",
    "release_version",
    "tool_path",
]
