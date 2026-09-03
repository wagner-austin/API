"""Turning a live environment into an image spec.

The first spec in this package was written by hand: run ``pip list``, delete
the first-party lines, paste the rest into JSON, and remember to exclude pip
itself. That is fine once and wrong the second time -- unrepeatable, silently
missing whatever the author forgot, and saying nothing about which
environment it came from. A project adopting an image should not have to do
it.

This module is the repeatable half, and it works from what
:func:`~hpc3.core.env_probe.parse_installed` already returns rather than
re-parsing the probe's output: one parser for what an environment reports,
so a spec and a preflight cannot disagree about what is installed.

Two rules a hand transcription gets wrong:

* **pip, setuptools and wheel are excluded.** The build installs them before
  reading the requirements file, so pinning them there makes it install a
  version and then immediately replace it -- a contradiction resolved
  silently, in an order the spec never declared.
* **First-party distributions become wheels, not requirements.** They are
  built from the repository at a known commit, and pinning them by version
  would let the build resolve ``0.1.0`` from a public index instead --
  installing someone else's package under our name.

Nothing here touches a filesystem or a network. The installed mapping arrives
as an argument, which is what lets these rules be tested without a cluster.
"""

from __future__ import annotations

from collections.abc import Mapping

from platform_core.errors import AppError, Hpc3ErrorCode

from hpc3.contracts.pins import normalise_name
from hpc3.core.env_probe import InstalledDistribution

BUILD_PROVIDED = frozenset({"pip", "setuptools", "wheel"})
"""Distributions the image installs before reading the requirements file.

Pinning these as requirements makes the build install a version and then
replace it, resolving in an order the spec does not declare.
"""

#: Removed on 2026-09-03: the tag was the constant "py3-none-any", assumed
#: because every first-party distribution then in play was pure Python. Its
#: own docstring named the bound -- "a project shipping a compiled extension
#: would need its real tag" -- and cleargbm is that project: `cleargbm_rs` is
#: compiled Rust, so its wheel is `cp311-cp311-linux_x86_64` and the spec
#: named a file that does not exist. The tag is now read from each
#: distribution's WHEEL metadata by the probe. See
#: :class:`~hpc3.core.env_probe.InstalledDistribution`.


def wheel_filename(distribution: str, version: str, wheel_tag: str) -> str:
    """Build the wheel filename poetry emits for a pure-Python distribution.

    Args:
        distribution: Distribution name in any spelling.
        version: Exact version.
        wheel_tag: The distribution's own PEP 425 tag, as the environment
            reported it. Passed in rather than assumed: a compiled extension's
            tag names an interpreter and a platform, and guessing
            ``py3-none-any`` for one produces a filename nothing will find.

    Returns:
        The filename, with hyphens folded to underscores as the wheel format
        requires -- pip reports ``platform-core`` while the wheel on disk is
        ``platform_core``, and a spec naming the reported spelling would name
        a file that does not exist.
    """
    return f"{normalise_name(distribution).replace('-', '_')}-{version}-{wheel_tag}.whl"


def _first_party_wheel(distribution: str, reported: InstalledDistribution) -> str:
    """Name the wheel a first-party distribution will be installed from.

    Args:
        distribution: Normalised distribution name.
        reported: What the probe read about it.

    Returns:
        The wheel filename.

    Raises:
        AppError: With ``WHEEL_TAG_UNKNOWN`` when the environment reports no
            PEP 425 tag for it. A distribution with no ``WHEEL`` metadata was
            not installed from a wheel -- conda-installed, or an editable
            checkout -- so there is no filename to name, and assuming one puts
            a file into the spec that the staging step will not find. Refusing
            here says which distribution and why; the build would say only
            that a path was missing.
    """
    if reported["wheel_tag"] == "":
        raise AppError(
            Hpc3ErrorCode.WHEEL_TAG_UNKNOWN,
            f"The environment reports no wheel tag for first-party distribution "
            f"{distribution!r}, so its wheel filename cannot be named. It was not "
            "installed from a wheel -- a conda package and an editable install both "
            "look like this. Install it from a built wheel in the environment being "
            "captured, or drop it from --first-party if it is not one.",
        )
    return wheel_filename(distribution, reported["version"], reported["wheel_tag"])


def third_party_versions(
    installed: Mapping[str, InstalledDistribution], first_party: frozenset[str]
) -> dict[str, str]:
    """Distribution to version for everything the IMAGE must install.

    The one definition of "third-party" in this package: what the environment
    reported, minus the first-party distributions (which ship as wheels) and
    minus what the base image already provides. :func:`capture_layers` renders
    the same set as ``==`` lines, and both read it from here so a change to
    what counts cannot reach one and not the other.

    Args:
        installed: Distribution name to version, as
            :func:`~hpc3.core.env_probe.parse_installed` returns it.
        first_party: Distributions built from the repository, in any spelling.

    Returns:
        The third-party distributions and their exact versions.
    """
    wanted = frozenset(normalise_name(name) for name in first_party)
    return {
        distribution: reported["version"]
        for distribution, reported in installed.items()
        if distribution not in wanted and distribution not in BUILD_PROVIDED
    }


def capture_layers(
    installed: Mapping[str, InstalledDistribution], first_party: frozenset[str]
) -> tuple[list[str], list[str]]:
    """Split what an environment reports into requirements and wheel names.

    Args:
        installed: Distribution name to version, as
            :func:`~hpc3.core.env_probe.parse_installed` returns it. Keys are
            already normalised.
        first_party: Distributions built from the repository, in any
            spelling; normalised here so either spelling matches.

    Returns:
        The third-party requirement lines, sorted, and the first-party wheel
        filenames, sorted. Sorted rather than in reported order because the
        spec is a document people diff: a capture that reordered its
        requirements between runs would show changes where nothing changed.

    Raises:
        AppError: With ``ENV_PROBE_UNREADABLE`` if the environment reports
            none of the named first-party distributions. A typo'd name would
            otherwise pass silently and be captured as an ordinary
            requirement, and the build would resolve it from a public index
            under someone else's ownership.
    """
    wanted = frozenset(normalise_name(name) for name in first_party)
    requirements = [
        f"{distribution}=={version}"
        for distribution, version in third_party_versions(installed, first_party).items()
    ]
    wheels: list[str] = []
    seen: set[str] = set()

    for distribution, reported in installed.items():
        if distribution in wanted:
            seen.add(distribution)
            wheels.append(_first_party_wheel(distribution, reported))

    missing = sorted(wanted - seen)
    if missing:
        raise AppError(
            Hpc3ErrorCode.ENV_PROBE_UNREADABLE,
            f"The environment does not contain first-party distribution(s): "
            f"{', '.join(missing)}. A name that matches nothing is captured as "
            "an ordinary requirement, and the build would resolve it from a "
            "public index under someone else's ownership.",
        )
    return sorted(requirements), sorted(wheels)


__all__ = [
    "BUILD_PROVIDED",
    "capture_layers",
    "third_party_versions",
    "wheel_filename",
]
