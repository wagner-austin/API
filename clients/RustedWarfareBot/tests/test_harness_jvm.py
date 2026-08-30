"""Where the JDK's tools are and how a classpath is spelled, on both platforms.

These are the two facts that used to be written out in six places. The tests
below are parameterised across platforms for the same reason the module exists:
a rule that is only ever exercised as Windows is a rule nobody has checked.
"""

from __future__ import annotations

import pytest

from rw_bot.harness.jvm import (
    GAME_CLASSPATH_ENTRIES,
    JVM_TOOLS,
    JvmReleaseError,
    JvmToolError,
    bundled_tools,
    classpath,
    executable_name,
    game_classpath,
    has_module_system,
    jvm_dir,
    jvm_major,
    release_version,
    tool_path,
)
from rw_bot.platform_id import WINDOWS

LINUX = "linux"
MACOS = "darwin"


class TestNamingTheTools:
    def test_windows_carries_the_suffix(self) -> None:
        assert executable_name("java", WINDOWS) == "java.exe"

    @pytest.mark.parametrize("platform", [LINUX, MACOS])
    def test_everything_else_does_not(self, platform: str) -> None:
        assert executable_name("java", platform) == "java"

    def test_every_tool_the_harness_launches_can_be_named(self) -> None:
        assert [executable_name(tool, LINUX) for tool in JVM_TOOLS] == [
            "java",
            "javac",
            "jar",
        ]

    def test_a_tool_this_harness_does_not_launch_is_refused(self) -> None:
        """Refused rather than suffixed blindly: the only way a wrong name gets
        here is a typo, and a typo composed into a path fails at launch as a
        missing file instead of here as a bad name."""
        with pytest.raises(JvmToolError) as caught:
            executable_name("jshell", WINDOWS)
        assert caught.value.code == "RW-JVM-001"
        assert "jshell" in str(caught.value)


class TestLocatingThemInAClone:
    def test_the_path_is_relative_to_the_game_directory(self) -> None:
        assert tool_path("java", WINDOWS) == "jvm64/bin/java.exe"
        assert tool_path("java", LINUX) == "jvm-linux/bin/java"

    def test_it_is_forward_slashed_on_both(self) -> None:
        """A clone's manifest is compared against paths this composes, so the
        separator has to be the same on either platform or the comparison
        fails on the string rather than on the file."""
        for platform in (WINDOWS, LINUX):
            assert "\\" not in tool_path("javac", platform)

    def test_an_unknown_tool_is_refused_before_a_path_is_built(self) -> None:
        with pytest.raises(JvmToolError):
            tool_path("keytool", LINUX)


class TestTheTwoBundledRuntimes:
    """Read off the real depots on 2026-08-29. Windows ships jvm64, an
    OpenJDK 13 with a compiler; Linux ships jvm-linux, an Oracle JRE 8 with
    ``java`` and nothing else. They are different runtimes, and every
    assumption that they were not is a launch that dies in its first second."""

    def test_each_platform_names_its_own_jvm_directory(self) -> None:
        assert jvm_dir(WINDOWS) == "jvm64"
        assert jvm_dir(LINUX) == "jvm-linux"

    def test_the_java_path_follows_the_directory(self) -> None:
        assert tool_path("java", WINDOWS) == "jvm64/bin/java.exe"
        assert tool_path("java", LINUX) == "jvm-linux/bin/java"

    def test_the_majors_are_the_ones_the_release_files_state(self) -> None:
        assert jvm_major(WINDOWS) == 13
        assert jvm_major(LINUX) == 8

    def test_only_the_newer_runtime_has_a_module_system(self) -> None:
        """On Java 8 ``--add-opens`` is rejected as unrecognised and the JVM
        never starts, so this is not an optimisation."""
        assert has_module_system(WINDOWS) is True
        assert has_module_system(LINUX) is False

    def test_the_linux_runtime_carries_no_compiler(self) -> None:
        """Its bin holds java and no javac or jar: a match can be played
        there and an agent cannot be built there."""
        assert bundled_tools(WINDOWS) == ("java", "javac", "jar")
        assert bundled_tools(LINUX) == ("java",)


class TestJoiningAClasspath:
    def test_the_game_classpath_differs_by_exactly_the_separator(self) -> None:
        """The failure this prevents: joined wrongly, the JVM reads the whole
        string as ONE entry, finds no such file, and fails as a missing main
        class -- which reads like a broken jar."""
        assert game_classpath(WINDOWS) == "game-lib.jar;libs/*"
        assert game_classpath(LINUX) == "game-lib.jar:libs/*"

    def test_the_wildcard_survives_because_the_jvm_expands_it(self) -> None:
        assert GAME_CLASSPATH_ENTRIES[-1] == "libs/*"
        assert game_classpath(LINUX).endswith("libs/*")

    def test_a_single_entry_needs_no_separator(self) -> None:
        assert classpath(("only.jar",), WINDOWS) == "only.jar"

    def test_an_empty_classpath_is_refused(self) -> None:
        """Rendering it as an empty string would leave the JVM resolving every
        class against the current directory, which is a different program."""
        with pytest.raises(ValueError, match="at least one entry"):
            classpath((), LINUX)


#: The head of the real Linux depot's ``jvm-linux/release``, transcribed on
#: 2026-08-29. Kept verbatim rather than reduced to the one line under test,
#: because what the parser has to survive is the file as shipped: the version
#: is not first, and one of the later values holds spaces, quotes and colons.
LINUX_RELEASE = (
    'JAVA_VERSION="1.8.0_131"',
    'OS_NAME="Linux"',
    'OS_VERSION="2.6"',
    'OS_ARCH="amd64"',
    'SOURCE=" .:94b119876028 corba:2b88cb53e31f deploy:28196a8e62a4"',
    'BUILD_TYPE="commercial"',
)

#: The Windows depot's, which states a different major version entirely.
WINDOWS_RELEASE = (
    'IMPLEMENTOR="AdoptOpenJDK"',
    'JAVA_VERSION="13.0.1"',
    'OS_ARCH="x86_64"',
)


class TestReadingWhatARuntimeSaysAboutItself:
    """The version is read off the tree in hand rather than taken from
    :func:`jvm_major`, which is what this harness ASSUMES about a platform. A
    depot that bumped its bundled JRE moves one and not the other."""

    def test_the_linux_depot_states_java_eight(self) -> None:
        assert release_version(LINUX_RELEASE) == "1.8.0_131"

    def test_the_windows_depot_states_java_thirteen(self) -> None:
        assert release_version(WINDOWS_RELEASE) == "13.0.1"

    def test_the_version_comes_first_or_not_at_all_indifferently(self) -> None:
        """It does not come first in the Windows file, which leads with
        IMPLEMENTOR, and it does in the Linux one."""
        assert release_version(('JAVA_VERSION="13.0.1"', 'OS_ARCH="x86_64"')) == "13.0.1"
        assert release_version(('OS_ARCH="x86_64"', 'JAVA_VERSION="13.0.1"')) == "13.0.1"

    def test_a_later_line_holding_quotes_and_colons_is_not_confused_for_it(self) -> None:
        """SOURCE carries both, and a parser that split on the first quote or
        the first colon anywhere would read it as the version."""
        assert release_version(LINUX_RELEASE) == "1.8.0_131"

    def test_the_stated_major_agrees_with_what_this_harness_assumes(self) -> None:
        """The two are separate on purpose and must still agree today: if a
        depot moves and this fails, the constant is what is now wrong."""
        assert release_version(LINUX_RELEASE).startswith("1.8.")
        assert jvm_major(LINUX) == 8
        assert release_version(WINDOWS_RELEASE).startswith(f"{jvm_major(WINDOWS)}.")

    def test_a_file_naming_no_version_is_refused(self) -> None:
        """A runtime that will not identify itself is not one two results may
        be compared across, and recording "unknown" would let them be."""
        with pytest.raises(JvmReleaseError) as caught:
            release_version(('OS_ARCH="amd64"',))
        assert caught.value.code == "RW-JVM-002"

    def test_an_empty_file_is_refused(self) -> None:
        with pytest.raises(JvmReleaseError) as caught:
            release_version(())
        assert caught.value.code == "RW-JVM-002"

    def test_a_version_stated_as_empty_is_refused(self) -> None:
        with pytest.raises(JvmReleaseError) as caught:
            release_version(('JAVA_VERSION=""',))
        assert caught.value.code == "RW-JVM-002"
