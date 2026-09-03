"""Where a bundled data file lives, answered from the installed package.

THE ONE OWNER of "where is this asset". Two families of data ship inside
:mod:`tankpit_bot.data` -- the static XOR key and the field minimap GIFs --
and every consumer asks here rather than composing a path of its own.

WHY THIS MODULE EXISTS AT ALL. Both families used to be reached by paths
that only resolve in a source checkout. The key was read four parents above
its module, which is the repository root from a checkout and site-packages
from an install; the GIFs were bare relative filenames, so they resolved
against whatever directory the process happened to start in. Neither
survives ``pip install``, and the cost was paid twice over: the fleet
container copied the key in and pointed at it with ``TANKPIT_XOR_KEY_FILE``,
and a cluster job staged forty-six files beside itself, ran from that
directory, and passed the same variable on every submission. Two workarounds
for one distribution that did not carry its own data
([[packaged-data-assets]]).

Addressed through :func:`importlib.resources.files`, the asset travels with
the wheel and the question has one answer under a checkout, a pip install, a
container and a cluster image alike. There is no environment override and no
search path, deliberately: an override is a second answer to a question that
must have one, and it is exactly what let the packaging defect survive.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

from tankpit_bot import _test_hooks

DATA_PACKAGE = "tankpit_bot.data"
"""The package the bundled data files live in."""

STATIC_KEY_NAME = "xor_static_key.txt"
"""The static XOR key's filename inside :data:`DATA_PACKAGE`."""

FIELD_GIF_SUFFIX = "_r.gif"
"""What a field's terrain minimap is named, given the server's display name.

ONE spelling, and the set is held to it. A minimap named any other way is
unreachable -- no lookup here can reach past this suffix -- so it would ship
as bytes nothing can ever load, which is what
``test_every_shipped_minimap_is_reachable_by_its_server_name`` refuses by
asserting the whole set answers to this one
([[packaged-data-assets]]).
"""


class BundledAssetMissingError(FileNotFoundError):
    """Raised when a data file this distribution should carry is absent.

    A :class:`FileNotFoundError` because that is what an absent file is, and
    the callers that already narrow on it keep working. Distinct from a
    caller passing a path that was never expected to exist: this names an
    asset the wheel is supposed to contain, so it fires on a broken install
    rather than on bad input, and the message says which asset and which
    package.
    """


def data_directory() -> Path:
    """Locate the directory the bundled data files were installed into.

    Returns:
        The filesystem directory of :data:`DATA_PACKAGE`. A real directory
        on every install shape this project supports -- a checkout, an
        editable install, a wheel unpacked into site-packages, and a
        container image -- because the package is plain files rather than a
        zip import.
    """
    return Path(str(files(DATA_PACKAGE)))


def static_key_file_path() -> Path:
    """Resolve the static XOR key that every session's cipher is built from.

    Returns:
        The key's path inside the installed package. The path is returned
        whether or not the file is present; callers that must have the key
        check for it, because a missing key and an unreadable one are the
        same refusal to them (see
        :func:`tankpit_bot.capture.xor.require_static_key`).
    """
    return data_directory() / STATIC_KEY_NAME


def field_gif_path(field_image: str) -> Path | None:
    """Resolve the local minimap GIF for a field the server named.

    Args:
        field_image: The field image filename as the server reports it,
            e.g. ``"field42.gif"``. The server names the display image; the
            terrain data lives in the :data:`FIELD_GIF_SUFFIX` variant
            beside it.

    Returns:
        The path to the shipped variant, or ``None`` when this distribution
        carries no minimap for that field. ``None`` rather than an error
        because a room whose field is unknown is a real and survivable
        state -- the session runs without terrain -- while an absent key is
        not.
    """
    candidate = data_directory() / f"{field_image.removesuffix('.gif')}{FIELD_GIF_SUFFIX}"
    if _test_hooks.path_exists(candidate):
        return candidate
    return None


def require_asset(name: str) -> Path:
    """Resolve a bundled file the caller names directly and cannot run without.

    Distinct from :func:`field_gif_path`, which answers "which minimap
    serves this field the SERVER named" and may honestly answer none. This
    answers "where is the file I already know the name of", so an absent one
    is a broken install rather than an unknown room.

    Args:
        name: The file's name inside :data:`DATA_PACKAGE`, e.g. a sim
            scenario's own terrain asset.

    Returns:
        The path to the bundled file.

    Raises:
        BundledAssetMissingError: When this distribution carries no such
            file. Used by callers whose whole run is defined by it -- a sim
            scenario without its map is not a degraded run but a different
            one.
    """
    candidate = data_directory() / name
    if not _test_hooks.path_exists(candidate):
        raise BundledAssetMissingError(f"{name!r} does not ship in {DATA_PACKAGE}")
    return candidate


__all__ = [
    "DATA_PACKAGE",
    "FIELD_GIF_SUFFIX",
    "STATIC_KEY_NAME",
    "BundledAssetMissingError",
    "data_directory",
    "field_gif_path",
    "require_asset",
    "static_key_file_path",
]
