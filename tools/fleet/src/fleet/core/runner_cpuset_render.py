"""The CPUs a host's runners may run on: whole cores, as many as its roster reserves.

MCPs board task 349e25ee. lavender's roster file reserves 8 of its 16
logical cores for the GitHub runners (``dispatch.budget.reserved_cores``),
and dispatch places fleet checks in the other 8 (MCPs
``scripts/host/lib/cpu-shares.sh`` renders that half as
``system-dispatch.slice``'s ``CPUQuota``). Nothing held the runners to their
half: :data:`~fleet.contracts.runner_slice.CI_SLICE_NAME` carried a
``CPUWeight`` and no CPU bound, so CI took every core whenever it wanted
them. MCPs/execution-deploy, whose Keycloak twin is timing-bound, ran 308 s
of its 300 s budget on lavender (fleet job 5d96ec2e) while CI's torch jobs
ran, and its twin took 115 to 139 s under CI load against 61 s on colossus.

A ``CPUQuota`` would bound the runners' time and not their place: eight
cores' worth of CI threads still land on all eight physical cores, each
beside a check's thread on its hyperthread sibling. So the slice gets an
``AllowedCPUs`` set of whole physical cores, the highest-numbered first,
until it holds the reserved count, which leaves the lower cores to
everything else. Measured on lavender (MCPs board task 349e25ee), all at
MCPs 043245cb3: with the runners on cores 4 to 7 and 16 synthetic busy
threads added to their slice, the suite ran 235 s (fleet job 367c99ef,
twin 110 s); with the same load and no set, lavender's runner refused to
place it at all (``NODE_OWNER_RESERVED``, CPU pressure 49.1%).

The set is read from the host's own topology (``lscpu -p=CPU,CORE``),
because which CPUs share a core differs per machine and only the host can
say. ``systemctl set-property`` writes it into systemd's control drop-in
and applies it to the live slice at once, so jobs already running move
onto the set with nothing restarted, and the lines then read the slice's
``EffectiveCPUs`` back and fail the provision unless it is exactly the set.
"""

from __future__ import annotations

from typing import Final

from fleet.contracts.runner_slice import CI_SLICE_NAME

#: The awk program that picks the set from ``lscpu -p=CPU,CORE``: whole
#: cores from the highest-numbered down until the set holds ``want`` CPUs,
#: printed space-separated. It exits 1, printing nothing, when the host has
#: too few CPUs or the set would be every CPU it has.
SELECT_PROGRAM: Final = (
    "/^#/ { next } "
    '{ cpus[$2] = ($2 in cpus) ? cpus[$2] " " $1 : $1; total++; if ($2 + 0 > top) top = $2 + 0 } '
    "END { for (core = top; core >= 0 && taken < want; core--) "
    'if (core in cpus) { picked = picked " " cpus[core]; taken += split(cpus[core], each, " ") } '
    "if (taken < want || taken >= total) exit 1; print picked }"
)

#: The awk program that expands systemd's ``EffectiveCPUs`` ranges, such as
#: ``4-7 12-15``, into one comma-separated ascending list.
EXPAND_PROGRAM: Final = (
    '{ for (field = 1; field <= NF; field++) { n = split($field, ends, "-"); '
    'for (cpu = ends[1]; cpu <= ends[n]; cpu++) printf "%s%d", (out++ ? "," : ""), cpu } } '
    'END { print "" }'
)


def render_ci_cpuset_lines(reserved_cores: int) -> list[str]:
    """Bash lines that hold the CI slice to whole cores, as many as the roster reserves.

    Run after the slice unit is written and started
    (:func:`fleet.core.runner_slice_render.render_slice_unit_lines`).

    Args:
        reserved_cores: The host's ``dispatch.budget.reserved_cores``, the
            logical cores its roster file reserves for its runners.

    Returns:
        The lines. They print the set and the slice's measured
        ``EffectiveCPUs``, and exit 1 naming why when the host cannot give
        the set or the slice does not take it.

    Raises:
        ValueError: When ``reserved_cores`` is below 1: a runner host whose
            roster reserves no cores would hold its runners to none.
    """
    if reserved_cores < 1:
        raise ValueError(
            f"reserved_cores is {reserved_cores}; a runner host's roster file must reserve its "
            "runners at least one core in dispatch.budget.reserved_cores"
        )
    return [
        f"ci_cpus=$(lscpu -p=CPU,CORE | awk -F, -v want={reserved_cores} '{SELECT_PROGRAM}' "
        "| tr ' ' '\\n' | grep . | sort -n | paste -sd, -) || {",
        f"    echo 'ci cpus refused: lscpu shows no {reserved_cores} CPUs on whole cores that "
        "leave the host any other CPU' >&2",
        "    exit 1",
        "}",
        f'systemctl set-property {CI_SLICE_NAME} "AllowedCPUs=$ci_cpus"',
        f"effective=$(systemctl show {CI_SLICE_NAME} -p EffectiveCPUs --value "
        f"| awk '{EXPAND_PROGRAM}')",
        'if [ "$effective" != "$ci_cpus" ]; then',
        f'    echo "ci cpus refused: {CI_SLICE_NAME} reports EffectiveCPUs=$effective after '
        'AllowedCPUs=$ci_cpus was set" >&2',
        "    exit 1",
        "fi",
        f'echo "ci cpus set: {CI_SLICE_NAME} AllowedCPUs=$ci_cpus, '
        f"measured EffectiveCPUs=$effective, for the roster's reserved_cores {reserved_cores}\"",
    ]


__all__ = ["EXPAND_PROGRAM", "SELECT_PROGRAM", "render_ci_cpuset_lines"]
