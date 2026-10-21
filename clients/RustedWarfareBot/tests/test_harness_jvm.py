"""Where the bundled runtime's ``java`` is and how a classpath is spelled.

These are the two facts that used to be written out in six places, pinned to
the Linux depot the game runs on.
"""

from __future__ import annotations

import pytest

from rw_bot.harness.jvm import (
    GAME_CLASSPATH,
    GAME_CLASSPATH_ENTRIES,
    JAVA_PATH,
    PATH_LIST_SEPARATOR,
    JvmReleaseError,
    classpath,
    release_version,
)


class TestLocatingJavaInAClone:
    def test_the_path_is_the_linux_depots_runtime(self) -> None:
        """Read off the real depot, not assumed: it ships ``jvm-linux``, and a
        path built from any other name is a launch that cannot find java."""
        assert JAVA_PATH == "jvm-linux/bin/java"

    def test_it_is_relative_and_forward_slashed(self) -> None:
        assert not JAVA_PATH.startswith("/")
        assert "\\" not in JAVA_PATH


class TestSpellingAClasspath:
    def test_the_game_classpath_joins_both_entries_with_a_colon(self) -> None:
        assert GAME_CLASSPATH == "game-lib.jar:libs/*"
        assert GAME_CLASSPATH.split(PATH_LIST_SEPARATOR) == list(GAME_CLASSPATH_ENTRIES)

    def test_the_wildcard_survives_because_the_jvm_expands_it(self) -> None:
        assert GAME_CLASSPATH.endswith("libs/*")

    def test_a_single_entry_needs_no_separator(self) -> None:
        assert classpath(("only.jar",)) == "only.jar"

    def test_an_empty_classpath_is_refused(self) -> None:
        """Rendering it as an empty string would leave the JVM resolving every
        class against the current directory, which is a different program."""
        with pytest.raises(ValueError, match="at least one entry"):
            classpath(())


#: The head of the real Linux depot's ``jvm-linux/release``, transcribed on
#: 2026-08-29. Kept verbatim rather than reduced to the one line under test,
#: because what the parser has to survive is the file as shipped: the version
#: is not alone, and one of the later values holds spaces, quotes and colons.
LINUX_RELEASE = (
    'JAVA_VERSION="1.8.0_131"',
    'OS_NAME="Linux"',
    'OS_VERSION="2.6"',
    'OS_ARCH="amd64"',
    'SOURCE=" .:94b119876028 corba:2b88cb53e31f deploy:28196a8e62a4"',
    'BUILD_TYPE="commercial"',
)


class TestReadingWhatARuntimeSaysAboutItself:
    """The version is read off the tree in hand rather than assumed. A depot
    that bumped its bundled JRE moves this and nothing else."""

    def test_the_linux_depot_states_java_eight(self) -> None:
        assert release_version(LINUX_RELEASE) == "1.8.0_131"

    def test_the_version_need_not_come_first(self) -> None:
        assert release_version(('OS_ARCH="amd64"', 'JAVA_VERSION="1.8.0_131"')) == "1.8.0_131"

    def test_a_later_line_holding_quotes_and_colons_is_not_confused_for_it(self) -> None:
        """SOURCE carries both, and a parser that split on the first quote or
        the first colon anywhere would read it as the version."""
        reordered = (
            'SOURCE=" .:94b119876028 corba:2b88cb53e31f deploy:28196a8e62a4"',
            'JAVA_VERSION="1.8.0_131"',
        )
        assert release_version(reordered) == "1.8.0_131"

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
