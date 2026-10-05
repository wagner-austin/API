"""Reading ``hpc3-register``'s flags into the values a declaration holds.

Every flag here is a decision a person makes about a project -- its partition,
its card, its caps, what it pins -- and none of them has a default. A defaulted
resource is a guess about a project nobody has asked the registrant about,
and the registry's own contract already refuses each unasked question
(``pinned_packages`` and ``certified_inputs`` are required even when the answer
is "none" or "no"). So the flags spell those answers out: ``none`` for no GPU
or no pins, ``free`` for a project that bills nothing.

Format errors are ValueErrors naming the flag, raised before anything touches
the filesystem or the cluster.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final

from platform_core.json_utils import JSONValue

from hpc3.contracts.budget import Budget, decode_budget
from hpc3.contracts.cluster import ClusterFacts, GpuRequest, decode_gpu_request
from hpc3.contracts.pins import require_pinned_packages

#: How a flag says "nothing": no GPU, no pins.
NONE_VALUE: Final[str] = "none"

#: How ``--billing`` says the project spends nothing and bills no account.
FREE_BILLING: Final[str] = "free"

_YES: Final[str] = "yes"
_NO: Final[str] = "no"
_DIGITS: Final[frozenset[str]] = frozenset("0123456789")


def require_every_flag(parsed: Mapping[str, str], flags: Sequence[str]) -> None:
    """Refuse a command line that leaves any flag out, naming all of them.

    Args:
        parsed: Flags already read from the command line.
        flags: Every flag the command requires.

    Raises:
        ValueError: Naming every missing flag at once. One at a time would be
            the surprise-failure procedure this command replaces.
    """
    missing = [flag for flag in flags if flag not in parsed]
    if missing:
        raise ValueError(
            f"registration needs every one of {list(flags)}; missing {len(missing)}: {missing}"
        )


def positive_int(flag: str, value: str) -> int:
    """Read a whole number of at least one.

    Args:
        flag: The flag, for the message.
        value: Its value.

    Returns:
        The number.

    Raises:
        ValueError: If the value is not all digits or is zero.
    """
    if value == "" or not set(value) <= _DIGITS or int(value) < 1:
        raise ValueError(f"{flag} must be a whole number of at least 1, got {value!r}")
    return int(value)


def non_negative_number(flag: str, value: str) -> float:
    """Read a decimal number of zero or more, such as a GPU-hour cap.

    Args:
        flag: The flag, for the message.
        value: Its value.

    Returns:
        The number.

    Raises:
        ValueError: If the value is not digits with at most one point.
    """
    whole, _, fraction = value.partition(".")
    if whole == "" or not set(whole + fraction) <= _DIGITS:
        raise ValueError(f"{flag} must be a number such as 0 or 12.5, got {value!r}")
    return float(value)


def yes_or_no(flag: str, value: str) -> bool:
    """Read a yes-or-no answer.

    Args:
        flag: The flag, for the message.
        value: Its value.

    Returns:
        True for ``yes``, False for ``no``.

    Raises:
        ValueError: For anything else. ``true`` and ``1`` are refused rather
            than accepted, so there is one spelling to read back.
    """
    if value not in (_YES, _NO):
        raise ValueError(f"{flag} must be {_YES!r} or {_NO!r}, got {value!r}")
    return value == _YES


def gpu_request(cluster: ClusterFacts, flag: str, value: str) -> GpuRequest | None:
    """Read a GPU request as ``MODEL:COUNT``, or ``none`` for CPU-only work.

    Args:
        cluster: The cluster whose GPU models the request must name.
        flag: The flag, for the message.
        value: Its value, e.g. ``A100:1``.

    Returns:
        The request, or None.

    Raises:
        ValueError: If the value is neither ``none`` nor ``MODEL:COUNT``.
        AppError: With ``GPU_TYPE_UNPINNED`` if the cluster has no such model.
    """
    if value == NONE_VALUE:
        return None
    model, separator, count = value.partition(":")
    if separator == "" or model == "":
        raise ValueError(
            f"{flag} must be {NONE_VALUE!r} or MODEL:COUNT such as A100:1, got {value!r}"
        )
    request: dict[str, JSONValue] = {"model": model, "count": positive_int(flag, count)}
    return decode_gpu_request(cluster, request, flag)


def pinned_packages(flag: str, value: str) -> dict[str, str]:
    """Read pins as ``name==version,name==version``, or ``none``.

    Args:
        flag: The flag, for the message.
        value: Its value.

    Returns:
        Required versions keyed by normalised name, through the same
        validator a workspace document's pins pass.

    Raises:
        ValueError: If an entry is not ``name==version``.
    """
    if value == NONE_VALUE:
        return {}
    pins: dict[str, JSONValue] = {}
    for entry in value.split(","):
        name, separator, version = entry.partition("==")
        if separator == "":
            raise ValueError(f"{flag} entries must be name==version, got {entry!r} in {value!r}")
        pins[name] = version
    return require_pinned_packages({flag: pins}, flag)


def budget(gpu_hours: float, flag: str, billing: str) -> Budget:
    """Build a project's caps from its GPU-hour cap and its billing answer.

    Args:
        gpu_hours: The self-imposed GPU-hour cap.
        flag: The billing flag, for the message.
        billing: ``free``, or ``ACCOUNT:UNITS`` for a project that may spend
            up to that many service units from that Slurm account.

    Returns:
        The budget, through the same decoder a workspace document's passes.

    Raises:
        ValueError: If the billing answer is neither form.
    """
    if billing == FREE_BILLING:
        return decode_budget(
            {"self_imposed_gpu_hours": gpu_hours, "max_service_units": 0.0, "charge_account": ""}
        )
    account, separator, units = billing.partition(":")
    if separator == "" or account == "":
        raise ValueError(
            f"{flag} must be {FREE_BILLING!r} or ACCOUNT:UNITS such as mylab:500, got {billing!r}"
        )
    return decode_budget(
        {
            "self_imposed_gpu_hours": gpu_hours,
            "max_service_units": non_negative_number(flag, units),
            "charge_account": account,
        }
    )


__all__ = [
    "FREE_BILLING",
    "NONE_VALUE",
    "budget",
    "gpu_request",
    "non_negative_number",
    "pinned_packages",
    "positive_int",
    "require_every_flag",
    "yes_or_no",
]
