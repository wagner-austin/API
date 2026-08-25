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

BUILD_PROVIDED = frozenset({"pip", "setuptools", "wheel"})
"""Distributions the image installs before reading the requirements file.

Pinning these as requirements makes the build install a version and then
replace it, resolving in an order the spec does not declare.
"""

WHEEL_TAG = "py3-none-any"
"""Compatibility tag assumed for first-party wheels.

Every first-party distribution here is pure Python, so poetry emits this tag.
A project shipping a compiled extension would need its real tag; the build
would then fail on a missing file rather than install a wrong one, which is
the right failure but a bound worth stating rather than discovering.
"""


def wheel_filename(distribution: str, version: str) -> str:
    """Build the wheel filename poetry emits for a pure-Python distribution.

    Args:
        distribution: Distribution name in any spelling.
        version: Exact version.

    Returns:
        The filename, with hyphens folded to underscores as the wheel format
        requires -- pip reports ``platform-core`` while the wheel on disk is
        ``platform_core``, and a spec naming the reported spelling would name
        a file that does not exist.
    """
    return f"{normalise_name(distribution).replace('-', '_')}-{version}-{WHEEL_TAG}.whl"


def capture_layers(
    installed: Mapping[str, str], first_party: frozenset[str]
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
    requirements: list[str] = []
    wheels: list[str] = []
    seen: set[str] = set()

    for distribution, version in installed.items():
        if distribution in wanted:
            seen.add(distribution)
            wheels.append(wheel_filename(distribution, version))
            continue
        if distribution in BUILD_PROVIDED:
            continue
        requirements.append(f"{distribution}=={version}")

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


__all__ = ["BUILD_PROVIDED", "WHEEL_TAG", "capture_layers", "wheel_filename"]
