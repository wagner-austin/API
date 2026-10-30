"""The CI slice's CPU set: whole cores, as many as the roster reserves (MCPs board task 349e25ee).

Each case runs the rendered lines under a real bash with the provision's own
options, ``lscpu`` answering a topology and ``systemctl`` recording every call
and answering the slice's ``EffectiveCPUs``, so what is asserted is the set
the lines choose and what they do with systemd's answer, not their text.
"""

from __future__ import annotations

import pathlib
import subprocess

import pytest

from fleet.contracts.runner_slice import CI_SLICE_NAME
from fleet.core.runner_cpuset_render import render_ci_cpu_select_lines, render_ci_cpuset_lines
from tests._runner_fixtures import RESERVED_CORES
from tests._stand_ins import run_with_stand_ins

#: ``lscpu -p=CPU,CORE`` on lavender, measured 2026-10-30: 16 logical CPUs,
#: CPU n and n+8 the two threads of core n.
LAVENDER = "# The following is the parsable format\n# CPU,Core\n" + "".join(
    f"{cpu},{cpu % 8}\n" for cpu in range(16)
)

#: A host of 8 cores with one thread each.
NO_SMT = "# CPU,Core\n" + "".join(f"{cpu},{cpu}\n" for cpu in range(8))

#: lavender's four highest cores, both threads of each.
LAVENDER_SET = "4,5,6,7,12,13,14,15"


def _run(
    tmp_path: pathlib.Path, reserved: int, topology: str, effective: str
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    """Run the lines for ``reserved`` cores on a host.

    Args:
        tmp_path: The case's directory.
        reserved: The roster's reservation.
        topology: What ``lscpu -p=CPU,CORE`` prints.
        effective: What ``systemctl show -p EffectiveCPUs --value`` prints.

    Returns:
        The finished run, and the systemctl calls it made.
    """
    log = tmp_path / "systemctl.log"
    log.write_bytes(b"")
    ran = run_with_stand_ins(
        tmp_path,
        render_ci_cpuset_lines(reserved),
        {
            "lscpu": f"printf '%s' '{topology}'",
            "systemctl": (
                'echo "$*" >> "$(dirname "$0")/../systemctl.log"\n'
                f'if [ "$1" = show ]; then echo "{effective}"; fi'
            ),
        },
        options=["-euo", "pipefail"],
    )
    return ran, log.read_text(encoding="utf-8").splitlines()


class TestTheSet:
    """Which CPUs the lines give the runners."""

    def test_lavender_s_eight_reserved_cores_are_its_four_highest_whole_cores(
        self, tmp_path: pathlib.Path
    ) -> None:
        """The set measured on lavender: the suite ran 235 s of its 300 under
        a saturating CI load with the runners there (fleet job 367c99ef)."""
        ran, calls = _run(tmp_path, RESERVED_CORES, LAVENDER, "4-7 12-15")
        assert ran.returncode == 0, ran.stderr
        assert calls == [
            f"set-property {CI_SLICE_NAME} AllowedCPUs={LAVENDER_SET}",
            f"show {CI_SLICE_NAME} -p EffectiveCPUs --value",
        ]
        assert ran.stdout == (
            f"ci cpus set: {CI_SLICE_NAME} AllowedCPUs={LAVENDER_SET}, measured "
            f"EffectiveCPUs={LAVENDER_SET}, for the roster's reserved_cores 8\n"
        )

    def test_an_odd_reservation_takes_the_whole_core_it_reaches(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Three CPUs would split core 6 between CI and a check, the sharing
        the set exists to end, so the core is taken whole."""
        ran, calls = _run(tmp_path, 3, LAVENDER, "6-7 14-15")
        assert ran.returncode == 0, ran.stderr
        assert calls[0] == f"set-property {CI_SLICE_NAME} AllowedCPUs=6,7,14,15"

    def test_a_host_of_single_thread_cores_gives_one_cpu_per_core(
        self, tmp_path: pathlib.Path
    ) -> None:
        ran, calls = _run(tmp_path, 2, NO_SMT, "6-7")
        assert ran.returncode == 0, ran.stderr
        assert calls[0] == f"set-property {CI_SLICE_NAME} AllowedCPUs=6,7"

    @pytest.mark.parametrize("reserved", [16, 20])
    def test_a_reservation_of_the_whole_host_is_refused_before_systemd_is_asked(
        self, tmp_path: pathlib.Path, reserved: int
    ) -> None:
        ran, calls = _run(tmp_path, reserved, LAVENDER, "")
        assert ran.returncode == 1
        assert ran.stderr == (
            f"ci cpus refused: lscpu shows no {reserved} CPUs on whole cores that leave the "
            "host any other CPU\n"
        )
        assert calls == []

    def test_the_selection_alone_leaves_the_set_and_asks_systemd_nothing(
        self, tmp_path: pathlib.Path
    ) -> None:
        """What tools/fleet-execution-runners runs on the runner host to say
        which set its live slice must hold, and the provision's first lines."""
        selection = render_ci_cpu_select_lines(RESERVED_CORES)
        assert render_ci_cpuset_lines(RESERVED_CORES)[: len(selection)] == selection
        ran = run_with_stand_ins(
            tmp_path,
            [*selection, 'printf "%s" "$ci_cpus"'],
            {"lscpu": f"printf '%s' '{LAVENDER}'", "systemctl": "exit 9"},
            options=["-euo", "pipefail"],
        )
        assert (ran.returncode, ran.stdout, ran.stderr) == (0, LAVENDER_SET, "")

    def test_no_reservation_is_refused_at_render(self) -> None:
        with pytest.raises(ValueError) as refused:
            render_ci_cpuset_lines(0)
        assert str(refused.value) == (
            "reserved_cores is 0; a runner host's roster file must reserve its runners at least "
            "one core in dispatch.budget.reserved_cores"
        )


class TestTheMeasurement:
    """The slice's EffectiveCPUs, read back after the set."""

    @pytest.mark.parametrize(
        ("answer", "read"),
        [("0-15", ",".join(str(cpu) for cpu in range(16))), ("", ""), ("4-7", "4,5,6,7")],
    )
    def test_a_slice_that_did_not_take_the_set_fails_the_provision_naming_both(
        self, tmp_path: pathlib.Path, answer: str, read: str
    ) -> None:
        """Every CPU (the slice before the set), nothing (an unloaded
        slice), and half the set all fail rather than pass on the write."""
        ran, calls = _run(tmp_path, RESERVED_CORES, LAVENDER, answer)
        assert ran.returncode == 1
        assert ran.stderr == (
            f"ci cpus refused: {CI_SLICE_NAME} reports EffectiveCPUs={read} after "
            f"AllowedCPUs={LAVENDER_SET} was set\n"
        )
        assert ran.stdout == ""
        assert len(calls) == 2
