"""The distribution carries its own data, and one module says where it is.

These tests run against the REAL shipped files rather than a fake tree,
because the property under test is precisely that the files ship. A fake
filesystem would pass whether or not the wheel contains anything, which is
the failure that produced this module: two consumers each rebuilt the data
environment by hand and both looked correct until an install without a
checkout beside it ([[packaged-data-assets]]).

The absence branches use the filesystem hook, because an install missing its
own data cannot be produced any other way.
"""

from __future__ import annotations

import pytest

from tankpit_bot.resources import (
    DATA_PACKAGE,
    FIELD_GIF_SUFFIX,
    STATIC_KEY_NAME,
    BundledAssetMissingError,
    data_directory,
    field_gif_path,
    require_asset,
    static_key_file_path,
)

_STATIC_KEY_LENGTH = 1000
_SHIPPED_MINIMAP_COUNT = 44


def test_the_data_directory_is_inside_the_installed_package() -> None:
    """The assets travel with the package rather than beside a checkout.

    The old lookup read four parents above the module, which is the repo
    root from a checkout and site-packages from an install. Anchoring on the
    package itself is what makes the answer the same in both.
    """
    directory = data_directory()

    assert directory.is_dir()
    assert directory.name == "data"
    assert directory.parent.name == "tankpit_bot"


def test_the_static_key_ships_and_is_the_length_the_cipher_expects() -> None:
    """Without this file no session decodes a single wire byte.

    Length is asserted rather than mere existence: a truncated key builds a
    table that decodes to plausible garbage, which is the failure the cipher
    refuses to risk.
    """
    path = static_key_file_path()

    assert path.name == STATIC_KEY_NAME
    assert path.is_file()
    assert len(path.read_text(encoding="utf-8").strip()) == _STATIC_KEY_LENGTH


def test_the_static_key_path_is_inside_the_data_package() -> None:
    """One owner for the location, so a consumer cannot invent a second."""
    assert static_key_file_path().parent == data_directory()


def test_a_field_the_server_names_resolves_to_its_shipped_minimap() -> None:
    """The server names the display image; the terrain is the _r variant."""
    resolved = field_gif_path("field01.gif")

    assert resolved == data_directory() / "field01_r.gif"
    assert (data_directory() / "field01_r.gif").is_file()


def test_every_shipped_minimap_is_reachable_by_its_server_name() -> None:
    """The whole set resolves, and carries exactly one spelling.

    A room the bot joins is whichever field the server chose, so a gap in
    this mapping is a session that silently runs without terrain.

    The single-spelling assertion is the half that keeps a duplicate from
    coming back. ``field42-r.gif`` shipped for eight months beside an
    identical ``field42_r.gif`` -- unreachable past its underscore sibling,
    so it was bytes in the wheel that no lookup could ever load. This test
    used to SKIP such a file; now it fails on it.
    """
    directory = data_directory()
    shipped = sorted(p.name for p in directory.glob("field*.gif"))
    assert len(shipped) == _SHIPPED_MINIMAP_COUNT

    for name in shipped:
        if not name.endswith(FIELD_GIF_SUFFIX):
            raise AssertionError(f"{name} ships but no lookup can reach it")
        stem = name.removesuffix(FIELD_GIF_SUFFIX)
        resolved = field_gif_path(f"{stem}.gif")
        if resolved is None:
            raise AssertionError(f"{stem}.gif resolves to no shipped minimap")
        assert resolved == directory / name
        assert resolved.is_file()


def test_an_unknown_field_resolves_to_none_rather_than_raising() -> None:
    """A room whose field ships no minimap is survivable.

    The session runs without terrain, which is a real state the world
    service already handles, so this answers None where the key's absence
    would raise.
    """
    assert field_gif_path("field9999.gif") is None


def test_a_named_asset_resolves_to_its_shipped_file() -> None:
    """The sim names its own terrain, and gets a path into the package."""
    resolved = require_asset("field01_r.gif")

    assert resolved == data_directory() / "field01_r.gif"
    assert resolved.is_file()


def test_a_named_asset_that_does_not_ship_is_refused() -> None:
    """An install missing a file it should carry is broken, not degraded.

    Distinct from an unknown field: the caller already knows this name, so
    the only way it is absent is that the distribution is incomplete.
    """
    with pytest.raises(BundledAssetMissingError) as excinfo:
        require_asset("field9999_r.gif")

    message = str(excinfo.value)
    assert "field9999_r.gif" in message
    assert DATA_PACKAGE in message


def test_the_missing_asset_error_is_a_file_not_found_error() -> None:
    """Callers that already narrow on FileNotFoundError keep working."""
    assert issubclass(BundledAssetMissingError, FileNotFoundError)
