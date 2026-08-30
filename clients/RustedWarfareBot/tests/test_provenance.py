"""What a sweep's numbers were produced under, and the record carrying them.

This project needed the shared record last and needs it most: every verdict it
has produced is a comparison, and none of those numbers carried anything
saying whether they were produced under the same conditions. It already knew
that mattered -- the wiki pins ``game_version`` on every page "because the jar
is obfuscated and class names change silently between releases" -- but had no
way to say it about a RESULT.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.comparability import IMAGE_DIGEST_ENV_VAR, NO_VALUE
from platform_core.determinism_env import SINGLE_THREAD
from platform_core.determinism_record import determinism_record
from platform_core.run_record import (
    NO_PAYLOAD,
    compare_run_records,
    decode_run_record,
    encode_run_record,
)
from platform_core.testing import SAMPLE_HOST, FakeHostProbe

from rw_bot.harness.jvm import JVM_RELEASE_FILE, JvmReleaseError, jvm_dir
from rw_bot.harness.scorecards import MatchRow
from rw_bot.provenance import (
    ASSETS_DISTRIBUTION,
    DIGEST_LENGTH,
    GAME_DISTRIBUTION,
    JVM_DISTRIBUTION,
    SWEEP_EXPERIMENT,
    VERSION_DIGEST_SEPARATOR,
    ArmSummary,
    arm_label,
    arm_observations,
    arm_run_record,
    asset_tree,
    bundled_runtime,
    game_build,
    game_packages,
    summarize_arm,
    sweep_fingerprint,
)
from rw_bot.tree_identity import TreeIdentityError
from tests.sample_game import LINUX, SAMPLE_JAVA_VERSION, write_sample_game

#: The other platform, so the runtime axis is exercised as both from whichever
#: one the suite happens to be running on.
WINDOWS = "win32"

_PINNED = determinism_record("cpu", {"OMP_NUM_THREADS": SINGLE_THREAD})

_NO_ENV: dict[str, str] = {}


def _probe() -> FakeHostProbe:
    """Build a probe reporting the stated machine.

    Returns:
        A probe reporting :data:`platform_core.testing.SAMPLE_HOST`.
    """
    return FakeHostProbe(
        platform=SAMPLE_HOST["platform"],
        machine=SAMPLE_HOST["machine"],
        logical_cores=SAMPLE_HOST["logical_cores"],
    )


def _jar(tmp_path: Path, contents: bytes = b"pretend this is a game") -> Path:
    """Write a stand-in for the game jar.

    Args:
        tmp_path: Directory to write into.
        contents: Bytes to digest.

    Returns:
        The jar path.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    jar = tmp_path / "game-lib.jar"
    jar.write_bytes(contents)
    return jar


def _row(
    arm: str,
    verdict: str,
    dropped: int = 0,
    worth_end: int = 1000,
    targets_end: int = 0,
    engageable: int = 0,
    intercepted: int = 0,
) -> MatchRow:
    """Build one match row the way a batch reader produces it.

    Typed rather than a scratch dictionary: every consumer used to narrow each
    field itself, so a misspelled key read as a missing figure instead of an
    error.

    Args:
        arm: Which arm played it.
        verdict: How it ended.
        dropped: Extractors lost between peak and end.
        worth_end: Total worth at the end.
        targets_end: Enemies visible at the end.
        engageable: How many of those could be fought.
        intercepted: Interceptions made.

    Returns:
        The row.
    """
    return MatchRow(
        arm=arm,
        seed=12345,
        verdict=verdict,
        extr_end=0,
        peak=0,
        dropped=dropped,
        worth_end=worth_end,
        rival_end=0,
        dip=0,
        targets_end=targets_end,
        engageable=engageable,
        intercepted=intercepted,
        income="18/s",
    )


class TestTheGameBuildIsAnAxis:
    """The jar is the subject, so the jar is what a record must pin."""

    def test_it_is_named_as_a_distribution(self, tmp_path: Path) -> None:
        """A boosting benchmark records lightgbm's version because a bump
        moves the comparison; here the equivalent is the game build."""
        assert game_build(_jar(tmp_path))["name"] == GAME_DISTRIBUTION

    def test_two_builds_are_distinguishable(self, tmp_path: Path) -> None:
        """The failure obfuscation creates: class names change silently, so a
        hand-maintained label is the last thing to notice."""
        first = game_build(_jar(tmp_path / "a", b"build 28"))
        second = game_build(_jar(tmp_path / "b", b"build 29"))
        assert first["version"] != second["version"]

    def test_the_same_bytes_give_the_same_build(self, tmp_path: Path) -> None:
        left = game_build(_jar(tmp_path / "a", b"build 28"))
        right = game_build(_jar(tmp_path / "b", b"build 28"))
        assert left["version"] == right["version"]

    def test_the_version_is_a_digest_prefix(self, tmp_path: Path) -> None:
        version = game_build(_jar(tmp_path))["version"]
        assert len(version) == DIGEST_LENGTH
        assert set(version) <= set("0123456789abcdef")

    def test_an_absent_jar_is_refused_rather_than_recorded_as_unknown(self, tmp_path: Path) -> None:
        """A record that says "some build" about an obfuscated binary says
        nothing, and every claim here is valid for one build only."""
        with pytest.raises(FileNotFoundError):
            game_build(tmp_path / "not-here.jar")


class TestTheBundledRuntimeIsAnAxis:
    """Linux runs Java 8 and Windows runs Java 13, so the runtime that
    executed the simulation is not a formality."""

    def test_it_is_named_as_a_distribution(self, tmp_path: Path) -> None:
        runtime = bundled_runtime(write_sample_game(tmp_path), LINUX)
        assert runtime["name"] == JVM_DISTRIBUTION

    def test_the_version_leads_with_what_the_runtime_says_about_itself(
        self, tmp_path: Path
    ) -> None:
        """Unlike the game's version -- which this project maintains by hand
        on a wiki page -- ``JAVA_VERSION`` is vendor-shipped, so a reader
        asking why two batches differ is better served by ``1.8.0_131`` than
        by two hex strings that merely differ."""
        runtime = bundled_runtime(write_sample_game(tmp_path), LINUX)
        version, _, digest = runtime["version"].partition(VERSION_DIGEST_SEPARATOR)
        assert version == SAMPLE_JAVA_VERSION
        assert len(digest) == DIGEST_LENGTH

    def test_a_repackaged_runtime_under_one_version_string_still_differs(
        self, tmp_path: Path
    ) -> None:
        """Why the digest is there at all: the label alone would call these
        two the same runtime."""
        left = bundled_runtime(write_sample_game(tmp_path / "a", runtime=b"build one"), LINUX)
        right = bundled_runtime(write_sample_game(tmp_path / "b", runtime=b"build two"), LINUX)
        assert left["version"] != right["version"]
        assert left["version"].split(VERSION_DIGEST_SEPARATOR)[0] == SAMPLE_JAVA_VERSION
        assert right["version"].split(VERSION_DIGEST_SEPARATOR)[0] == SAMPLE_JAVA_VERSION

    def test_two_java_versions_differ(self, tmp_path: Path) -> None:
        left = bundled_runtime(write_sample_game(tmp_path / "a", java_version="1.8.0_131"), LINUX)
        right = bundled_runtime(write_sample_game(tmp_path / "b", java_version="13.0.1"), LINUX)
        assert left["version"] != right["version"]

    def test_the_directory_read_is_the_platform_s_own(self, tmp_path: Path) -> None:
        """A Windows tree carries ``jvm64`` and a Linux one ``jvm-linux``, so
        reading the wrong name finds no runtime at all rather than the wrong
        one."""
        game = write_sample_game(tmp_path, platform=WINDOWS)
        assert bundled_runtime(game, WINDOWS)["name"] == JVM_DISTRIBUTION
        with pytest.raises(FileNotFoundError):
            bundled_runtime(game, LINUX)

    def test_a_runtime_that_states_no_version_is_refused(self, tmp_path: Path) -> None:
        """A runtime that will not identify itself is not one two results may
        be compared across, and recording "unknown" would let them be."""
        game = write_sample_game(tmp_path)
        (game / jvm_dir(LINUX) / JVM_RELEASE_FILE).write_text('OS_ARCH="amd64"\n', encoding="utf-8")
        with pytest.raises(JvmReleaseError) as caught:
            bundled_runtime(game, LINUX)
        assert caught.value.code == "RW-JVM-002"

    def test_a_release_naming_an_empty_version_is_refused(self, tmp_path: Path) -> None:
        game = write_sample_game(tmp_path, java_version="")
        with pytest.raises(JvmReleaseError) as caught:
            bundled_runtime(game, LINUX)
        assert caught.value.code == "RW-JVM-002"


class TestTheAssetsAreAnAxis:
    """A map missing from a clone sent the engine to its boot sandbox and
    voided a batch family, with the jar digest matching throughout."""

    def test_it_is_named_as_a_distribution(self, tmp_path: Path) -> None:
        assert asset_tree(write_sample_game(tmp_path))["name"] == ASSETS_DISTRIBUTION

    def test_the_version_is_a_bare_digest_prefix(self, tmp_path: Path) -> None:
        """Nothing in ``assets/`` states a version, which is the point: a mod
        folder edited between two batches announces itself nowhere else."""
        version = asset_tree(write_sample_game(tmp_path))["version"]
        assert len(version) == DIGEST_LENGTH
        assert set(version) <= set("0123456789abcdef")

    def test_an_edited_map_moves_it(self, tmp_path: Path) -> None:
        left = asset_tree(write_sample_game(tmp_path / "a", assets=b"eight extractors"))
        right = asset_tree(write_sample_game(tmp_path / "b", assets=b"nine extractors"))
        assert left["version"] != right["version"]

    def test_an_absent_asset_tree_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(TreeIdentityError) as caught:
            asset_tree(tmp_path / "no-such-game")
        assert caught.value.code == "RW-TREE-001"


class TestTheThreePackages:
    def test_they_are_the_code_the_runtime_and_the_data(self, tmp_path: Path) -> None:
        packages = game_packages(write_sample_game(tmp_path), LINUX)
        assert {package["name"] for package in packages} == {
            GAME_DISTRIBUTION,
            JVM_DISTRIBUTION,
            ASSETS_DISTRIBUTION,
        }

    def test_they_come_back_in_the_order_the_shared_decoder_will_put_them_in(
        self, tmp_path: Path
    ) -> None:
        """Not the order they are read in. The axis canonicalises by name, so
        a tuple assembled engine-outwards does not equal its own round trip --
        which is how this was found, by the codec test below failing."""
        packages = game_packages(write_sample_game(tmp_path), LINUX)
        assert [package["name"] for package in packages] == sorted(
            [GAME_DISTRIBUTION, JVM_DISTRIBUTION, ASSETS_DISTRIBUTION]
        )


class TestTheFingerprint:
    def test_it_carries_the_code_the_runtime_and_the_data(self, tmp_path: Path) -> None:
        fingerprint = sweep_fingerprint(
            _PINNED, _NO_ENV.get, _probe(), write_sample_game(tmp_path), LINUX
        )
        assert {p["name"] for p in fingerprint["packages"]} == {
            GAME_DISTRIBUTION,
            JVM_DISTRIBUTION,
            ASSETS_DISTRIBUTION,
        }

    def test_it_carries_the_machine(self, tmp_path: Path) -> None:
        """A match is a wall-clock simulation; the box it ran on is not
        incidental to how far it got."""
        fingerprint = sweep_fingerprint(
            _PINNED, _NO_ENV.get, _probe(), write_sample_game(tmp_path), LINUX
        )
        assert fingerprint["host"] == SAMPLE_HOST

    def test_a_run_outside_any_image_states_no_card_and_no_digest(self, tmp_path: Path) -> None:
        """The ordinary case: a workstation, headless, no card. Empty differs
        from every real value rather than matching all of them."""
        fingerprint = sweep_fingerprint(
            _PINNED, _NO_ENV.get, _probe(), write_sample_game(tmp_path), LINUX
        )
        assert fingerprint["image_digest"] == NO_VALUE
        assert fingerprint["gpu_model"] == NO_VALUE
        assert fingerprint["driver_version"] == NO_VALUE

    def test_a_launcher_that_named_an_image_is_believed(self, tmp_path: Path) -> None:
        digest = "sha256:" + "ab" * 32
        fingerprint = sweep_fingerprint(
            _PINNED,
            {IMAGE_DIGEST_ENV_VAR: digest}.get,
            _probe(),
            write_sample_game(tmp_path),
            LINUX,
        )
        assert fingerprint["image_digest"] == digest

    def test_two_game_builds_do_not_compare_equal(self, tmp_path: Path) -> None:
        """The whole reason the jar is on the axis."""
        left = sweep_fingerprint(
            _PINNED, _NO_ENV.get, _probe(), write_sample_game(tmp_path / "a", jar=b"28"), LINUX
        )
        right = sweep_fingerprint(
            _PINNED, _NO_ENV.get, _probe(), write_sample_game(tmp_path / "b", jar=b"29"), LINUX
        )
        assert left != right

    def test_two_bundled_runtimes_do_not_compare_equal(self, tmp_path: Path) -> None:
        """The gap this closed. Today the host axis separates Windows from
        Linux by accident; two Linux runs either side of a depot that bumped
        its bundled JRE used to fingerprint identically."""
        left = sweep_fingerprint(
            _PINNED,
            _NO_ENV.get,
            _probe(),
            write_sample_game(tmp_path / "a", java_version="1.8.0_131"),
            LINUX,
        )
        right = sweep_fingerprint(
            _PINNED,
            _NO_ENV.get,
            _probe(),
            write_sample_game(tmp_path / "b", java_version="1.8.0_202"),
            LINUX,
        )
        assert left != right

    def test_two_asset_trees_do_not_compare_equal(self, tmp_path: Path) -> None:
        """The other gap. An edited map changes the simulation and used to
        change nothing in the record."""
        left = sweep_fingerprint(
            _PINNED, _NO_ENV.get, _probe(), write_sample_game(tmp_path / "a", assets=b"m1"), LINUX
        )
        right = sweep_fingerprint(
            _PINNED, _NO_ENV.get, _probe(), write_sample_game(tmp_path / "b", assets=b"m2"), LINUX
        )
        assert left != right


class TestSummarisingAnArm:
    def test_it_counts_wins_and_losses_separately(self) -> None:
        """Not `matches - wins`: a match can end in neither, and folding
        those into losses reports a rate nobody measured."""
        rows = [
            _row("attack", "won"),
            _row("attack", "defeated"),
            _row("attack", "draw"),
        ]
        summary = summarize_arm(rows, "attack")
        assert (summary["matches"], summary["wins"], summary["losses"]) == (3, 1, 1)

    def test_it_sums_the_figures_a_verdict_rests_on(self) -> None:
        rows = [
            _row("attack", "won", dropped=2, targets_end=5, engageable=3, intercepted=1),
            _row("attack", "won", dropped=3, targets_end=4, engageable=4, intercepted=2),
        ]
        summary = summarize_arm(rows, "attack")
        assert summary["drops"] == 5
        assert summary["unengageable"] == 2
        assert summary["intercepts"] == 3

    def test_it_ignores_other_arms(self) -> None:
        rows = [_row("attack", "won"), _row("defend", "defeated")]
        assert summarize_arm(rows, "attack")["matches"] == 1

    def test_an_arm_with_no_matches_is_refused(self) -> None:
        """The caller derives arm names from the rows, so an empty one means
        the two disagree -- and a median over nothing has no answer."""
        with pytest.raises(ValueError, match="no matches to summarise"):
            summarize_arm([_row("attack", "won")], "defend")


class TestTheRecord:
    def _summary(self, wins: int = 3, matches: int = 4) -> ArmSummary:
        """Build an arm aggregate.

        Args:
            wins: Matches won.
            matches: Matches played.

        Returns:
            The aggregate.
        """
        return ArmSummary(
            arm="attack",
            matches=matches,
            wins=wins,
            losses=matches - wins,
            drops=5,
            median_worth=1200,
            unengageable=2,
            intercepts=3,
        )

    def test_every_batch_is_one_experiment(self, tmp_path: Path) -> None:
        """The question here is longitudinal -- how the champion's rate moved
        across five batches -- and two records naming different experiments
        are not comparable at all."""
        record = arm_run_record(
            "aggression",
            self._summary(),
            sweep_fingerprint(_PINNED, _NO_ENV.get, _probe(), write_sample_game(tmp_path), LINUX),
            NO_PAYLOAD,
        )
        assert record["experiment"] == SWEEP_EXPERIMENT

    def test_the_label_carries_the_batch_and_the_arm(self) -> None:
        assert arm_label("aggression", "attack") == "aggression/attack"

    def test_the_win_rate_is_recorded_not_left_to_the_reader(self) -> None:
        """It is the number the stated goal is written in, and a rate
        recomputed at each reading is one two readers can disagree about."""
        names = {o["name"]: o["value"] for o in arm_observations(self._summary())}
        assert names["win_rate"] == pytest.approx(0.75)

    def test_the_counts_survive_beside_the_rate(self) -> None:
        """Three wins from three and thirty from thirty are both 1.0, and
        only one of them is evidence."""
        names = {o["name"]: o["value"] for o in arm_observations(self._summary())}
        assert names["matches"] == 4.0
        assert names["wins"] == 3.0

    def test_it_round_trips_through_the_shared_codec(self, tmp_path: Path) -> None:
        record = arm_run_record(
            "aggression",
            self._summary(),
            sweep_fingerprint(_PINNED, _NO_ENV.get, _probe(), write_sample_game(tmp_path), LINUX),
            NO_PAYLOAD,
        )
        assert decode_run_record(encode_run_record(record)) == record

    def test_two_arms_on_one_build_are_subtractable(self, tmp_path: Path) -> None:
        """The comparison this project makes constantly and could not state."""
        fingerprint = sweep_fingerprint(
            _PINNED, _NO_ENV.get, _probe(), write_sample_game(tmp_path), LINUX
        )
        left = arm_run_record("aggression", self._summary(wins=3), fingerprint, NO_PAYLOAD)
        right = arm_run_record(
            "aggression",
            ArmSummary(
                arm="defend",
                matches=4,
                wins=1,
                losses=3,
                drops=1,
                median_worth=900,
                unengageable=0,
                intercepts=0,
            ),
            fingerprint,
            NO_PAYLOAD,
        )
        comparison = compare_run_records(left, right, ())
        if comparison["kind"] != "compared":
            raise AssertionError(f"one build should compare: {comparison}")
        deltas = {delta["name"]: delta["difference"] for delta in comparison["deltas"]}
        assert deltas["win_rate"] == pytest.approx(-0.5)

    def test_two_game_builds_are_refused(self, tmp_path: Path) -> None:
        """The failure this whole module exists to prevent: an arm measured
        against build 28 subtracted from one measured against build 29."""
        left = arm_run_record(
            "aggression",
            self._summary(),
            sweep_fingerprint(
                _PINNED, _NO_ENV.get, _probe(), write_sample_game(tmp_path / "a", jar=b"28"), LINUX
            ),
            NO_PAYLOAD,
        )
        right = arm_run_record(
            "aggression",
            self._summary(wins=1),
            sweep_fingerprint(
                _PINNED, _NO_ENV.get, _probe(), write_sample_game(tmp_path / "b", jar=b"29"), LINUX
            ),
            NO_PAYLOAD,
        )
        assert compare_run_records(left, right, ())["kind"] == "uncalibrated"
