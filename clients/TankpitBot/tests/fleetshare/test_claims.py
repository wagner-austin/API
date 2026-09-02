"""Tests for the authoritative container claim protocol.

Every arm of the mutex: winning the create, refreshing an own claim,
denial by a fresh sibling, reaping a stale one, the mid-create and
released-between windows, and the owner-only release law. All I/O
rides the fake filesystem hooks; the real O_CREAT|O_EXCL primitive is
proven in ``tests/test_test_hooks.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.json_utils import JSONTypeError, dump_json_str, load_json_str

from tankpit_bot.fleetshare.claims import (
    CLAIM_TTL_MS,
    ContainerClaimDict,
    acquire_container_claim,
    claim_path,
    decode_container_claim,
    encode_container_claim,
    release_container_claim,
)
from tests.conftest import FakeFileSystem

_NOW = 500_000


def _claim(instance: str, *, claimed_ms: int = _NOW - 1_000) -> ContainerClaimDict:
    """Build a claim record."""
    return ContainerClaimDict(instance=instance, tank_id=1301, claimed_ms=claimed_ms)


def _plant(fs: FakeFileSystem, room: str, x: int, y: int, claim: ContainerClaimDict) -> None:
    """Write an existing claim file into the fake filesystem."""
    fs.write_text(claim_path(room, x, y), dump_json_str(encode_container_claim(claim)))


def _read_planted(fs: FakeFileSystem, room: str, x: int, y: int) -> ContainerClaimDict:
    """Read the claim file back through the codec."""
    parsed = load_json_str(fs.read_text(claim_path(room, x, y)))
    assert isinstance(parsed, dict)
    return decode_container_claim(parsed)


class TestClaimCodec:
    """The claim record's typed encode/decode pair."""

    def test_roundtrip(self) -> None:
        """Encode then decode reproduces the claim exactly."""
        claim = _claim("artax")
        assert decode_container_claim(encode_container_claim(claim)) == claim

    def test_decode_rejects_a_missing_field(self) -> None:
        """A claim without its stamp fails validation loudly."""
        with pytest.raises(JSONTypeError):
            decode_container_claim({"instance": "artax", "tank_id": 1301})


class TestClaimPath:
    """The per-room claim namespace."""

    def test_path_is_room_scoped_under_the_claims_directory(self) -> None:
        """One file per tile, inside runs/bot/_claims/<room>."""
        path = claim_path("6", 100, 136)
        assert path.parts[-4:] == ("bot", "_claims", "6", "100_136.claim")

    def test_a_room_with_path_characters_is_refused(self) -> None:
        """Separators must never reach the filesystem layer."""
        with pytest.raises(ValueError, match="not a valid claim namespace"):
            claim_path("../evil", 1, 2)


class TestAcquire:
    """Acquisition: win, refresh, deny, reap, and the race windows."""

    def test_an_unclaimed_container_is_won(self, fake_fs: FakeFileSystem) -> None:
        """The exclusive create lands this bot's claim record."""
        assert acquire_container_claim("6", 10, 20, instance="artax", tank_id=7, now_ms=_NOW)

        written = _read_planted(fake_fs, "6", 10, 20)
        assert written == ContainerClaimDict(instance="artax", tank_id=7, claimed_ms=_NOW)

    def test_an_own_claim_refreshes_its_stamp(self, fake_fs: FakeFileSystem) -> None:
        """The holder's later acquire rewrites the stamp, never loses."""
        _plant(fake_fs, "6", 10, 20, _claim("artax", claimed_ms=_NOW - 2_000))

        assert acquire_container_claim("6", 10, 20, instance="artax", tank_id=7, now_ms=_NOW)

        assert _read_planted(fake_fs, "6", 10, 20)["claimed_ms"] == _NOW

    def test_a_fresh_foreign_claim_denies(self, fake_fs: FakeFileSystem) -> None:
        """A sibling's live claim stands; the loser's content never lands."""
        _plant(fake_fs, "6", 10, 20, _claim("yuppler", claimed_ms=_NOW - CLAIM_TTL_MS))

        assert not acquire_container_claim("6", 10, 20, instance="artax", tank_id=7, now_ms=_NOW)

        assert _read_planted(fake_fs, "6", 10, 20)["instance"] == "yuppler"

    def test_a_stale_foreign_claim_is_reaped_and_won(self, fake_fs: FakeFileSystem) -> None:
        """A crashed holder's leftover past the TTL frees the container."""
        _plant(fake_fs, "6", 10, 20, _claim("yuppler", claimed_ms=_NOW - CLAIM_TTL_MS - 1))

        assert acquire_container_claim("6", 10, 20, instance="artax", tank_id=7, now_ms=_NOW)

        assert _read_planted(fake_fs, "6", 10, 20)["instance"] == "artax"

    def test_losing_the_reap_recreate_race_denies(self, fake_fs: FakeFileSystem) -> None:
        """Two reapers race a stale claim: the recreate loser yields.

        The fake's remove drops the file and a concurrent winner's
        recreate is simulated by the create hook landing the rival's
        file first — the loser's second exclusive create must refuse.
        """
        from tankpit_bot import _test_hooks

        _plant(fake_fs, "6", 10, 20, _claim("yuppler", claimed_ms=_NOW - CLAIM_TTL_MS - 1))
        real_create = _test_hooks.create_text_exclusive
        calls = {"n": 0}

        def rival_wins_recreate(path: Path, content: str) -> bool:
            calls["n"] += 1
            if calls["n"] == 2:
                # The rival's exclusive create landed between this
                # bot's reap-unlink and its retry create.
                _plant(fake_fs, "6", 10, 20, _claim("malignant", claimed_ms=_NOW))
            return fake_fs.create_text_exclusive(path, content)

        _test_hooks.create_text_exclusive = rival_wins_recreate
        try:
            won = acquire_container_claim("6", 10, 20, instance="artax", tank_id=7, now_ms=_NOW)
        finally:
            _test_hooks.create_text_exclusive = real_create

        assert not won
        assert calls["n"] == 2
        assert _read_planted(fake_fs, "6", 10, 20)["instance"] == "malignant"

    def test_a_claim_released_between_fail_and_read_is_won_on_retry(
        self, fake_fs: FakeFileSystem
    ) -> None:
        """The holder released in the window: one retry create wins it."""
        from tankpit_bot import _test_hooks

        _plant(fake_fs, "6", 10, 20, _claim("yuppler"))
        real_read = _test_hooks.read_text

        def release_then_read(path: Path) -> str:
            # The holder's release lands between this bot's failed
            # create and its read of the standing claim.
            fake_fs.remove(path)
            return fake_fs.read_text(path)

        _test_hooks.read_text = release_then_read
        try:
            won = acquire_container_claim("6", 10, 20, instance="artax", tank_id=7, now_ms=_NOW)
        finally:
            _test_hooks.read_text = real_read

        assert won
        assert _read_planted(fake_fs, "6", 10, 20)["instance"] == "artax"

    def test_the_mid_create_window_denies_for_one_beat(self, fake_fs: FakeFileSystem) -> None:
        """A claim whose content has not landed reads as held.

        Existence is the lock and content is metadata: the winner's
        exclusive create is atomic but its write is not, so a rival
        catching the empty file cannot judge ownership — it loses the
        beat, never the file.
        """
        fake_fs.write_text(claim_path("6", 10, 20), "")

        assert not acquire_container_claim("6", 10, 20, instance="artax", tank_id=7, now_ms=_NOW)

        assert fake_fs.read_text(claim_path("6", 10, 20)) == ""

    def test_a_non_object_claim_body_denies_for_one_beat(self, fake_fs: FakeFileSystem) -> None:
        """Valid JSON that is not an object is still the unreadable window."""
        fake_fs.write_text(claim_path("6", 10, 20), "[1, 2]")

        assert not acquire_container_claim("6", 10, 20, instance="artax", tank_id=7, now_ms=_NOW)

    def test_a_mistyped_claim_body_denies_for_one_beat(self, fake_fs: FakeFileSystem) -> None:
        """An object missing the claim fields is still the window."""
        fake_fs.write_text(claim_path("6", 10, 20), dump_json_str({"instance": "artax"}))

        assert not acquire_container_claim("6", 10, 20, instance="artax", tank_id=7, now_ms=_NOW)


class TestRelease:
    """Release: owner-only deletion."""

    def test_an_own_claim_is_deleted(self, fake_fs: FakeFileSystem) -> None:
        """The owner's release removes the file and reports it."""
        _plant(fake_fs, "6", 10, 20, _claim("artax"))

        assert release_container_claim("6", 10, 20, instance="artax")

        assert not fake_fs.path_exists(claim_path("6", 10, 20))

    def test_a_foreign_claim_is_never_deleted(self, fake_fs: FakeFileSystem) -> None:
        """A sibling's claim survives another bot's release call."""
        _plant(fake_fs, "6", 10, 20, _claim("yuppler"))

        assert not release_container_claim("6", 10, 20, instance="artax")

        assert _read_planted(fake_fs, "6", 10, 20)["instance"] == "yuppler"

    def test_a_missing_claim_is_nothing_to_release(self, fake_fs: FakeFileSystem) -> None:
        """Releasing an unclaimed tile reports False and touches nothing."""
        assert not release_container_claim("6", 10, 20, instance="artax")
