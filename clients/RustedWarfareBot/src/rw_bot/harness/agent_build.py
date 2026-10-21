"""Building the agent jar a match attaches.

Split from :mod:`rw_bot.harness.launch` when that file passed the six-hundred
line ceiling, and the boundary is a real one: building the agent runs a
COMPILER, which the game's depot does not ship -- it is a JRE. The host's own
JDK builds the agent (:data:`HOST_JAVAC`, :data:`HOST_JAR`), and a cluster
node never builds anything: it attaches the jar its batch froze.

The two therefore change for different reasons. A new launch option touches
how a match STARTS; a new agent source touches how it is BUILT; and the
machines that can do the second are a strict subset of those that can do the
first.
"""

from __future__ import annotations

from collections.abc import Sequence

from rw_bot.harness.jvm import JAVA_PATH, classpath
from rw_bot.harness.launch import LaunchConfig

#: Where the agent's Java sources live.
AGENT_SOURCE_DIR = "agent/src/rwbot/agent"

#: Suffix identifying an agent source file.
AGENT_SOURCE_SUFFIX = ".java"

#: The manifest naming the agent's premain class.
AGENT_MANIFEST = "agent/manifest.mf"

#: Where per-match build artifacts go.
AGENT_BUILD_DIR = "agent/build"

#: What a frozen tree's prebuilt agent jar is called inside it.
FROZEN_AGENT_JAR = "rw-agent.jar"

#: The prebuilt agent jar ``make agent`` publishes and a batch freezes, under
#: the same basename it carries inside a frozen tree.
BUILT_AGENT_JAR = f"{AGENT_BUILD_DIR}/{FROZEN_AGENT_JAR}"

#: The host JDK's compiler and archiver, found on ``PATH``. Not the game's:
#: the depot's ``jvm-linux`` is a JRE with neither, and any JDK from 9 on
#: honours ``--release`` (:data:`JAVA_RELEASE`), so the classes it writes
#: load on the depot's Java 8 whichever JDK wrote them.
HOST_JAVAC = "javac"
HOST_JAR = "jar"

#: Bytecode level the agent is compiled to.
#:
#: Eight, because the JVM the agent loads into is the Linux depot's JRE
#: 1.8.0_131 -- and a class file above a JVM's level does not degrade, it
#: fails at load with ``UnsupportedClassVersionError``.
#:
#: It was 11 while only the Windows depot's OpenJDK 13 was in view, which
#: would have loaded on nothing the cluster runs.
JAVA_RELEASE = "8"


def agent_jar(config: LaunchConfig, stamp: str) -> str:
    """Return the agent jar this match attaches.

    A frozen batch reuses the jar built once when its snapshot was taken; a
    single match compiles its own under a per-invocation stamp so concurrent
    matches cannot overwrite each other's.

    Args:
        config: The launch.
        stamp: A per-invocation identifier, ignored for a frozen tree.

    Returns:
        The jar's path, relative to the repository root.
    """
    if config["tree"]:
        return f"{config['tree']}/{FROZEN_AGENT_JAR}"
    return f"{AGENT_BUILD_DIR}/rw-agent-play-{stamp}.jar"


def classes_dir(stamp: str) -> str:
    """Return where a per-match compile puts its class files.

    Args:
        stamp: A per-invocation identifier.

    Returns:
        The directory, relative to the repository root.
    """
    return f"{AGENT_BUILD_DIR}/play-{stamp}"


def agent_sources(names: Sequence[str]) -> tuple[str, ...]:
    """Return the agent source files to compile, from a directory listing.

    Args:
        names: Entry names in :data:`AGENT_SOURCE_DIR`.

    Returns:
        The Java sources, sorted, each as a path under
        :data:`AGENT_SOURCE_DIR`.

    Raises:
        ValueError: When the listing holds no Java source. Compiling nothing
            produces an empty jar, and an empty jar attaches without error and
            silently never opens the channel -- which reads as a hung engine
            ninety seconds later rather than as an empty build.
    """
    sources = sorted(name for name in names if name.endswith(AGENT_SOURCE_SUFFIX))
    if not sources:
        raise ValueError(
            f"no {AGENT_SOURCE_SUFFIX} sources in {AGENT_SOURCE_DIR}: an empty jar "
            "attaches without error and never opens the channel"
        )
    return tuple(f"{AGENT_SOURCE_DIR}/{name}" for name in sources)


def compile_command(javac: str, classes: str, sources: Sequence[str]) -> tuple[str, ...]:
    """Return the command that compiles the agent.

    ``-Werror`` is kept from the original recipe deliberately: the agent is
    loaded into the game's own classloader beside obfuscated classes, and a
    warning there is the first sign of a name that has moved between builds.

    Args:
        javac: Path to the compiler.
        classes: Directory to write class files into.
        sources: The Java sources to compile.

    Returns:
        The argument vector, program first.
    """
    return (javac, "--release", JAVA_RELEASE, "-Xlint:all", "-Werror", "-d", classes, *sources)


#: The agent's self-check entry point: patches every Targets entry in the real
#: jar and lets the JVM's own verifier certify each result.
SELFTEST_CLASS = "rwbot.agent.SelfTest"


def selftest_command(game_dir: str, classes: str) -> tuple[str, ...]:
    """Return the command that runs the agent's self-checks on the real jar.

    Run on the game's OWN runtime rather than the host JDK, because the
    verifier the patch has to satisfy is the one the engine will load it
    under. Every path is relative to the repository root, which is where the
    command runs: the self-test reads the jar, it does not boot the engine.

    Args:
        game_dir: The pinned game directory, relative to the root.
        classes: The freshly compiled agent classes, relative to the root.

    Returns:
        The argument vector, the bundled ``java`` first.
    """
    jar = f"{game_dir}/game-lib.jar"
    return (
        f"{game_dir}/{JAVA_PATH}",
        "-cp",
        classpath((classes, jar, f"{game_dir}/libs/*")),
        SELFTEST_CLASS,
        jar,
    )


def package_command(jar_tool: str, jar_path: str, classes: str) -> tuple[str, ...]:
    """Return the command that packages compiled classes into the agent jar.

    Args:
        jar_tool: Path to the jar tool.
        jar_path: The jar to write.
        classes: Directory holding the class files.

    Returns:
        The argument vector, program first.
    """
    return (jar_tool, "cfm", jar_path, AGENT_MANIFEST, "-C", classes, ".")


__all__ = [
    "AGENT_BUILD_DIR",
    "AGENT_MANIFEST",
    "AGENT_SOURCE_DIR",
    "AGENT_SOURCE_SUFFIX",
    "BUILT_AGENT_JAR",
    "FROZEN_AGENT_JAR",
    "HOST_JAR",
    "HOST_JAVAC",
    "JAVA_RELEASE",
    "SELFTEST_CLASS",
    "agent_jar",
    "agent_sources",
    "classes_dir",
    "compile_command",
    "package_command",
    "selftest_command",
]
