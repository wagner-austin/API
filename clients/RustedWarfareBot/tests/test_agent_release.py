"""The bytecode level the agent is built to, held to one answer.

TWO PLACES BUILD THIS JAR AND ONLY ONE OF THEM SHIPS IT.
:mod:`rw_bot.harness.agent_build` composes the compile command a batch uses;
the Makefile's ``agent`` recipe composes the one ``make agent`` uses, and the
jar on disk -- the jar that gets frozen, staged and attached -- is the
Makefile's.

They disagreed. The recipe said ``--release 11`` while the module said 8, so
the staged agent was class-file version 55 and the Linux depot's JRE 1.8.0_131
refused to load it:

    UnsupportedClassVersionError: rwbot/agent/Premain has been compiled by a
    more recent version of the Java Runtime (class file version 55.0), this
    version of the Java Runtime only recognizes class file versions up to 52.0
    FATAL ERROR in native method: processing of -javaagent failed

Fatal, before the game starts, on every member of every campaign. Nothing on
this workstation could have caught it: the Windows depot ships an OpenJDK 13,
which loads 55 happily.

So this reads the recipe and holds it to the module. A comment saying the two
must agree is a comment; this is the thing that makes them.
"""

from __future__ import annotations

import re
from pathlib import Path

from rw_bot.harness.agent_build import JAVA_RELEASE

#: The repository root, from this file rather than from a working directory.
_ROOT = Path(__file__).resolve().parents[1]

#: What the recipe declares, as ``AGENT_RELEASE := <n>``.
_MAKEFILE_RELEASE = re.compile(r"^AGENT_RELEASE\s*:=\s*(\d+)\s*$", re.MULTILINE)

#: The class-file version each Java release produces, for the two that matter
#: here. Java 8 is 52 and Java 11 is 55; the depot's JRE reads up to 52.
_CLASS_FILE_VERSIONS = {"8": 52, "11": 55}

#: What the Linux depot's JRE 1.8.0_131 will load, read off its own refusal.
_DEPOT_JRE_CEILING = 52


def _declared_release() -> str:
    """Read the release the Makefile's agent recipe compiles to.

    Returns:
        The value of ``AGENT_RELEASE``.

    Raises:
        AssertionError: When the Makefile does not declare one. Absence is a
            failure rather than a skip: the recipe would then be back to
            hardcoding a level nothing checks.
    """
    found = _MAKEFILE_RELEASE.search((_ROOT / "Makefile").read_text(encoding="utf-8"))
    if found is None:
        raise AssertionError(
            "the Makefile declares no AGENT_RELEASE; the agent recipe must name the "
            "bytecode level as a variable so this test can hold it to JAVA_RELEASE"
        )
    return found.group(1)


def test_the_recipe_and_the_module_agree() -> None:
    """The whole point. One rule, two builders, and only one ships the jar."""
    assert _declared_release() == JAVA_RELEASE


def test_the_recipe_passes_it_rather_than_hardcoding_one() -> None:
    """``agent.ps1`` took no release parameter and hardcoded 11, so making the
    Makefile declare a variable would have changed nothing on its own."""
    script = (_ROOT / "scripts" / "make" / "agent.ps1").read_text(encoding="utf-8")
    assert "--release $Release" in script
    assert "--release 11" not in script


def test_the_recipe_forwards_the_variable() -> None:
    """Declared and passed are different things: a variable nothing reads is
    a variable that agrees with anything."""
    recipe = (_ROOT / "Makefile").read_text(encoding="utf-8")
    assert '-Release "$(AGENT_RELEASE)"' in recipe


def test_the_level_is_one_the_depots_jre_can_load() -> None:
    """The reason the answer is 8 and not merely 'consistent'. A class above a
    JVM's ceiling does not degrade -- it aborts the JVM before the game
    starts, which is what the first Linux launch did."""
    assert _CLASS_FILE_VERSIONS[JAVA_RELEASE] <= _DEPOT_JRE_CEILING


def test_eleven_would_not_have_loaded() -> None:
    """The negative half, so the assertion above is measuring something. The
    Windows depot's OpenJDK 13 reads 55 happily, which is why this went
    unnoticed on the workstation."""
    assert _CLASS_FILE_VERSIONS["11"] > _DEPOT_JRE_CEILING
