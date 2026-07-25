"""Resolution and consistency checking of the built javaagent.

The agent is a separate compilation unit in another language, so nothing in the
Python type system can see it. This module is where that gap is closed: it
locates the built jar, reads the manifest the JVM will read, and fails with a
traceable code when the two halves of the client have drifted apart.

Three drifts are reachable and each has its own code. The jar can be missing
because ``make agent`` was never run (``RW-AGENT-002``). The manifest can lack
the ``Premain-Class`` attribute the JVM requires to start an agent at all
(``RW-AGENT-001``). And the attribute can name a class that no longer exists,
which is what happens when the entry point is renamed and the manifest is not
(``RW-AGENT-003``) -- a jar that builds cleanly and then aborts the JVM at
launch with ``Failed to find Premain-Class``.

The checks are deliberately performed here rather than left to the JVM. A
launch that fails inside the game process surfaces as an engine crash log,
which is the hardest place to read a build mistake.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Final, TypedDict

from rw_bot import RwBotError
from rw_bot.harness import _test_hooks
from rw_bot.validation import require_absolute_path, require_non_empty_str

AGENT_SOURCE_DIR_RELATIVE: Final = "agent/src"
"""Java source root; package directories descend from here."""

AGENT_MANIFEST_RELATIVE: Final = "agent/manifest.mf"
"""Jar manifest naming the premain entry point."""

AGENT_JAR_RELATIVE: Final = "agent/build/rw-agent.jar"
"""Built agent jar. Must match ``AGENT_JAR`` in the Makefile, which builds it."""

PREMAIN_ATTRIBUTE: Final = "Premain-Class"
"""Manifest attribute the JVM reads to find the agent entry point."""

_NO_PREMAIN = "RW-AGENT-001"
_JAR_ABSENT = "RW-AGENT-002"
_PREMAIN_SOURCE_ABSENT = "RW-AGENT-003"
_ROOT_NOT_ABSOLUTE = "RW-AGENT-004"


class AgentBuildError(RwBotError):
    """The built agent is missing or inconsistent with its manifest.

    Args:
        code: Stable machine-readable identifier.
        message: Human-readable description of the inconsistency.
    """


class AgentBuild(TypedDict):
    """A built javaagent that has been checked against its own manifest.

    Attributes:
        jar_path: Absolute path to the built jar, ready for ``-javaagent``.
        premain_class: Binary name of the premain entry point, as the JVM will
            resolve it from the manifest.
    """

    jar_path: str
    premain_class: str


def decode_agent_build(payload: Mapping[str, str | int | bool]) -> AgentBuild:
    """Decode an untyped mapping into an :class:`AgentBuild`.

    Args:
        payload: Untyped mapping carrying every field.

    Returns:
        The validated agent build.

    Raises:
        DecodeError: ``RW-DECODE-001`` when a field is absent, ``RW-DECODE-002``
            when mistyped, ``RW-DECODE-003`` when blank, ``RW-DECODE-005`` when
            ``jar_path`` is not absolute.
    """
    return AgentBuild(
        jar_path=require_absolute_path(payload, "jar_path"),
        premain_class=require_non_empty_str(payload, "premain_class"),
    )


def encode_agent_build(build: AgentBuild) -> dict[str, str]:
    """Encode an :class:`AgentBuild` back to a plain mapping.

    Round-trips with :func:`decode_agent_build`.

    Args:
        build: The agent build to encode.

    Returns:
        A plain mapping suitable for run-artifact recording.
    """
    return {
        "jar_path": build["jar_path"],
        "premain_class": build["premain_class"],
    }


def parse_premain_class(manifest_lines: tuple[str, ...]) -> str:
    """Extract the ``Premain-Class`` attribute from manifest lines.

    Attribute names are case-insensitive in the jar manifest specification, so
    the comparison is too; the value is returned exactly as written.

    Args:
        manifest_lines: Manifest contents, one entry per line, newlines removed.

    Returns:
        The binary name of the premain class.

    Raises:
        AgentBuildError: ``RW-AGENT-001`` when the attribute appears zero times
            or more than once. A manifest naming two entry points is ambiguous
            rather than a precedence question.
    """
    prefix = f"{PREMAIN_ATTRIBUTE.lower()}:"
    values = [
        line.split(":", 1)[1].strip() for line in manifest_lines if line.lower().startswith(prefix)
    ]
    if len(values) != 1:
        raise AgentBuildError(
            _NO_PREMAIN,
            f"manifest must declare exactly one {PREMAIN_ATTRIBUTE} attribute, "
            f"found {len(values)}; without it the JVM starts no agent and the "
            "engine dies on its first in-game frame",
        )
    return values[0]


def premain_source_path(client_root: Path, premain_class: str) -> Path:
    """Map a premain binary name to the Java source file that must define it.

    Args:
        client_root: Absolute path to the client root.
        premain_class: Binary class name, dot-separated.

    Returns:
        Path to the ``.java`` file declaring that class.
    """
    relative = premain_class.replace(".", "/")
    return client_root / f"{AGENT_SOURCE_DIR_RELATIVE}/{relative}.java"


def resolve_agent_build(client_root: Path) -> AgentBuild:
    """Locate the built agent and verify it against its manifest.

    Args:
        client_root: Absolute path to the client root, the directory holding
            ``agent/`` and ``pyproject.toml``.

    Returns:
        The validated agent build, carrying an absolute jar path.

    Raises:
        AgentBuildError: ``RW-AGENT-004`` when ``client_root`` is relative,
            ``RW-AGENT-001`` when the manifest declares no single
            ``Premain-Class``, ``RW-AGENT-003`` when that class has no source
            file, ``RW-AGENT-002`` when the jar has not been built.
        OSError: When the manifest cannot be read.
        UnicodeDecodeError: When the manifest is not valid UTF-8.
    """
    if not client_root.is_absolute():
        raise AgentBuildError(
            _ROOT_NOT_ABSOLUTE,
            f"client_root must be absolute, got {str(client_root)!r}: the jar path is "
            "handed to a process whose working directory is the game tree",
        )

    manifest_path = client_root / AGENT_MANIFEST_RELATIVE
    premain_class = parse_premain_class(_test_hooks.read_text_lines(manifest_path))

    source_path = premain_source_path(client_root, premain_class)
    if not _test_hooks.path_exists(source_path):
        raise AgentBuildError(
            _PREMAIN_SOURCE_ABSENT,
            f"manifest names {premain_class!r} as {PREMAIN_ATTRIBUTE} but no source "
            f"exists at {str(source_path)!r}; the jar would build and then abort the "
            "JVM at launch with 'Failed to find Premain-Class'",
        )

    jar_path = client_root / AGENT_JAR_RELATIVE
    if not _test_hooks.path_exists(jar_path):
        raise AgentBuildError(
            _JAR_ABSENT,
            f"agent jar not found at {str(jar_path)!r}; run 'make agent' to build it",
        )

    return decode_agent_build({"jar_path": str(jar_path), "premain_class": premain_class})


__all__ = [
    "AGENT_JAR_RELATIVE",
    "AGENT_MANIFEST_RELATIVE",
    "AGENT_SOURCE_DIR_RELATIVE",
    "PREMAIN_ATTRIBUTE",
    "AgentBuild",
    "AgentBuildError",
    "decode_agent_build",
    "encode_agent_build",
    "parse_premain_class",
    "premain_source_path",
    "resolve_agent_build",
]
