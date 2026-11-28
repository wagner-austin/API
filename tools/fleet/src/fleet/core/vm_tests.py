"""Which NixOS VM tests a check ran, and which of their drivers it had to realise.

MCPs board task d6b03ba3. MCPs/fleet-nixos's checks are VM tests, each built
by nix as a ``vm-test-run-<name>`` derivation whose driver ends with ``test
script finished in <s>s``, a line nix prefixes with the derivation's name.
Only a test the run built prints it: one already in the store, or
substituted, is not run. Those lines are spread through a transcript of some
40,000 lines, past the tail the verdict reads, so the audit of that task's
closure (7de4c7a5) found which tests a cold run built, and how long each
took, in a session's grep of the log and in no record.

So a check's build reads its own transcript once the recipe has ended and
prints one :data:`VM_TEST_MARKER` line per finished test and a
:data:`VM_TESTS_MARKER` line counting them, which the verdict
(:func:`read_vm_tests`, in :func:`fleet.core.verdict.judge`) copies onto the
queue row and the thread as ``vmtests=<count>:<name>=<s>s,...``.

WHAT THE STORE LACKED (MCPs board task 7b62e759). A VM test boots its
``nixos-test-driver-<name>``, and a run whose store already holds it neither
copies nor builds it. nix writes ``copying path '<store path>' from '<cache>'...``
for one it substitutes and ``building '<derivation>'...`` for one it builds,
unprefixed, as lavender's fleet job 6742e77b transcript holds them. The same
read prints one :data:`VM_DRIVER_MARKER` line per such driver and a
:data:`VM_DRIVERS_MARKER` count, which the verdict copies as
``drivers=<count>:<name>=<fetched|built>,...``: a program-written record of
which drivers the node lacked before the run, where a session's note was the
only one (review cfbb5b91 on MCPs board task 97e2c1ee).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Final

#: One finished test's line.
VM_TEST_MARKER: Final = "FLEET_VM_TEST"

#: The count's line, printed whether or not any test ran.
VM_TESTS_MARKER: Final = "FLEET_VM_TESTS"

#: One realised driver's line.
VM_DRIVER_MARKER: Final = "FLEET_VM_DRIVER"

#: The drivers' count line, printed whether or not the run realised any.
VM_DRIVERS_MARKER: Final = "FLEET_VM_DRIVERS"

#: nix's prefix and the NixOS test driver's last line, as ``sed -E`` reads them.
FINISHED_SED: Final = r"^vm-test-run-([^>[:space:]]+)> test script finished in ([0-9.]+)s$"

#: A driver nix substituted, as ``sed -E`` reads it. Each quote is matched as
#: any one character, since the program is single-quoted in ``sh``; a name
#: starts with a letter, so the generic ``nixos-test-driver-1.1`` package is
#: no test's driver.
FETCHED_SED: Final = (
    r"^copying path ./nix/store/[0-9a-z]{32}-nixos-test-driver-([a-z][0-9a-z-]*). from .*$"
)

#: A driver nix built, as ``sed -E`` reads it.
BUILT_SED: Final = (
    r"^building ./nix/store/[0-9a-z]{32}-nixos-test-driver-([a-z][0-9a-z-]*)\.drv.\.\.\.$"
)

#: A finished test's line, as the build prints it.
VM_TEST_LINE: Final[re.Pattern[str]] = re.compile(
    rf"^{VM_TEST_MARKER}: (\S+) (\d+(?:\.\d+)?)$", re.MULTILINE
)

#: The count's line.
VM_TESTS_LINE: Final[re.Pattern[str]] = re.compile(
    rf"^{VM_TESTS_MARKER}: (\d+) finished$", re.MULTILINE
)

#: A realised driver's line.
VM_DRIVER_LINE: Final[re.Pattern[str]] = re.compile(
    rf"^{VM_DRIVER_MARKER}: (\S+) (fetched|built)$", re.MULTILINE
)

#: The drivers' count line.
VM_DRIVERS_LINE: Final[re.Pattern[str]] = re.compile(
    rf"^{VM_DRIVERS_MARKER}: (\d+) realised$", re.MULTILINE
)


def sh_vm_test_lines(log: str) -> list[str]:
    """The ``sh`` lines that print the transcript's finished tests and realised drivers into it.

    The transcript is read whole, for both, before anything is appended, so
    the lines never read their own output.

    Args:
        log: The transcript's path, read and appended to.

    Returns:
        The lines, run after the recipe's phase and before the status is written.
    """
    return [
        f"vm_tests=\"$(sed -nE 's/{FINISHED_SED}/{VM_TEST_MARKER}: \\1 \\2/p' '{log}')\"",
        f"vm_drivers=\"$(sed -nE -e 's#{FETCHED_SED}#{VM_DRIVER_MARKER}: \\1 fetched#p' "
        f"-e 's#{BUILT_SED}#{VM_DRIVER_MARKER}: \\1 built#p' '{log}')\"",
        f"if [ -n \"$vm_tests\" ]; then printf '%s\\n' \"$vm_tests\" >> '{log}'; fi",
        f"printf '{VM_TESTS_MARKER}: %s finished\\n' "
        f"\"$(printf '%s' \"$vm_tests\" | grep -c '^{VM_TEST_MARKER}: ' || true)\" >> '{log}'",
        f"if [ -n \"$vm_drivers\" ]; then printf '%s\\n' \"$vm_drivers\" >> '{log}'; fi",
        f"printf '{VM_DRIVERS_MARKER}: %s realised\\n' "
        f"\"$(printf '%s' \"$vm_drivers\" | grep -c '^{VM_DRIVER_MARKER}: ' || true)\" >> '{log}'",
    ]


def _read_counted(
    tail: str,
    counts: re.Pattern[str],
    items: re.Pattern[str],
    render: Callable[[str, str], str],
) -> str:
    """Read the items a count line closes off a transcript tail.

    Args:
        tail: The transcript's last lines.
        counts: The count's line, its first group the count.
        items: One item's line, its two groups what ``render`` is given.
        render: One item as the field writes it.

    Returns:
        ``<count>:<item>,...``, the items sorted by their groups, from the
        last count line and the item lines between it and the count line
        before it; the count alone when none is there; else ``unread`` (a
        build from before the lines, or one stopped before them).
    """
    # Split at each count line, its count kept: the text before the first
    # count, then each count and the text after it, so the last count is
    # parts[-2] and the text it closes is parts[-3].
    if counts.search(tail) is None:
        return "unread"
    parts = counts.split(tail)
    read = sorted((match.group(1), match.group(2)) for match in items.finditer(parts[-3]))
    if not read:
        return parts[-2]
    return f"{parts[-2]}:" + ",".join(render(first, second) for first, second in read)


def read_vm_tests(tail: str) -> str:
    """Read the VM tests a check ran off its transcript tail.

    Args:
        tail: The transcript's last lines.

    Returns:
        ``<count>:<name>=<s>s,...``, the tests by name, ``0`` when it ran
        none, else ``unread``, as :func:`_read_counted` reads them.
    """
    return _read_counted(tail, VM_TESTS_LINE, VM_TEST_LINE, lambda name, s: f"{name}={s}s")


def read_vm_drivers(tail: str) -> str:
    """Read the VM test drivers a check had to realise off its transcript tail.

    Args:
        tail: The transcript's last lines.

    Returns:
        ``<count>:<name>=<fetched|built>,...``, the drivers by name, ``0``
        when it realised none, else ``unread``, as :func:`_read_counted`
        reads them.
    """
    return _read_counted(tail, VM_DRIVERS_LINE, VM_DRIVER_LINE, lambda name, how: f"{name}={how}")


__all__ = [
    "BUILT_SED",
    "FETCHED_SED",
    "FINISHED_SED",
    "VM_DRIVERS_LINE",
    "VM_DRIVERS_MARKER",
    "VM_DRIVER_LINE",
    "VM_DRIVER_MARKER",
    "VM_TESTS_LINE",
    "VM_TESTS_MARKER",
    "VM_TEST_LINE",
    "VM_TEST_MARKER",
    "read_vm_drivers",
    "read_vm_tests",
    "sh_vm_test_lines",
]
