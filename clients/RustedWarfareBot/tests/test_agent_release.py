"""The bytecode level the agent is built to, held to one answer and one builder.

TWO PLACES USED TO BUILD THIS JAR AND ONLY ONE OF THEM SHIPPED IT.
:mod:`rw_bot.harness.agent_build` composed the compile command a match used;
the Makefile's ``agent`` recipe, through ``scripts/make/agent.ps1``, composed
the one ``make agent`` used, and the jar on disk -- the jar that gets frozen,
staged and attached -- was the recipe's.

They disagreed. The recipe said ``--release 11`` while the module said 8, so
the staged agent was class-file version 55 and the Linux depot's JRE 1.8.0_131
refused to load it:

    UnsupportedClassVersionError: rwbot/agent/Premain has been compiled by a
    more recent version of the Java Runtime (class file version 55.0), this
    version of the Java Runtime only recognizes class file versions up to 52.0
    FATAL ERROR in native method: processing of -javaagent failed

Fatal, before the game starts, on every member of every campaign.

Since 2026-10-21 (board task 0eae9619) the recipe calls
:mod:`scripts.agent`, which builds through the module's own
compile, so there is one builder. These hold it that way: the Makefile names
no compiler and no release of its own, and the one release loads on the depot.
"""

from __future__ import annotations

import re
from pathlib import Path

from rw_bot.harness.agent_build import JAVA_RELEASE, compile_command

#: The repository root, from this file rather than from a working directory.
_ROOT = Path(__file__).resolve().parents[1]

#: The class-file version each Java release produces, for the two that matter
#: here. Java 8 is 52 and Java 11 is 55; the depot's JRE reads up to 52.
_CLASS_FILE_VERSIONS = {"8": 52, "11": 55}

#: What the Linux depot's JRE 1.8.0_131 will load, read off its own refusal.
_DEPOT_JRE_CEILING = 52

#: A second builder's tells: a compiler named, or a release stated.
_SECOND_BUILDER = re.compile(r"javac|--release|AGENT_RELEASE")


def test_the_makefile_composes_no_compile_of_its_own() -> None:
    """A second compile command in the recipe is the second builder that
    shipped class version 55. Comments may tell that history; only the lines
    make runs or assigns are read."""
    lines = (_ROOT / "Makefile").read_text(encoding="utf-8").splitlines()
    live = [line for line in lines if not line.lstrip().startswith("#")]
    assert len(live) > 100
    assert [line for line in live if _SECOND_BUILDER.search(line)] == []


def test_the_agent_recipe_is_the_modules_builder() -> None:
    recipe = (_ROOT / "Makefile").read_text(encoding="utf-8")
    assert "poetry run python -m scripts.agent build" in recipe


def test_the_one_compile_passes_the_release() -> None:
    assert compile_command("javac", "b", ("A.java",))[1:3] == ("--release", JAVA_RELEASE)


def test_the_level_is_one_the_depots_jre_can_load() -> None:
    """The reason the answer is 8 and not merely 'consistent'. A class above a
    JVM's ceiling does not degrade -- it aborts the JVM before the game
    starts, which is what the first Linux launch did."""
    assert _CLASS_FILE_VERSIONS[JAVA_RELEASE] <= _DEPOT_JRE_CEILING


def test_eleven_would_not_have_loaded() -> None:
    """The negative half, so the assertion above is measuring something."""
    assert _CLASS_FILE_VERSIONS["11"] > _DEPOT_JRE_CEILING
