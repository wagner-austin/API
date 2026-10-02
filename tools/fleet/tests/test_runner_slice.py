"""The runners' share of a CI host: the budget's contract and its bash (MCPs board task 45a4f22b).

The move is executed by a real bash against a ``systemctl`` stand-in on
``PATH`` that answers the unit's control group and records every call, so
what is asserted is the decision the rendered lines make, not their text.
"""

from __future__ import annotations

import pathlib
import subprocess

import pytest
from platform_core.json_utils import JSONTypeError, JSONValue

from fleet.contracts.runner_slice import (
    CI_SLICE_NAME,
    CiSlice,
    decode_ci_slice,
    encode_ci_slice,
)
from fleet.contracts.runners import HostRunnerSpec, RunnerInstall
from fleet.core import runner_render, runner_slice_render
from tests._host_bash import host_bash
from tests._runner_fixtures import a_base, a_ci_slice, ci_slice_json

#: The unit every case places.
UNIT = "actions.runner.wagner-austin-MCPs.lavender-wsl.service"


def _systemctl_stand_in(cgroup: str) -> str:
    """A systemctl that answers ``show -p ControlGroup`` with ``cgroup``
    and appends every invocation to ``systemctl.log`` beside it.

    Args:
        cgroup: The control group it reports.

    Returns:
        The script text.
    """
    return (
        "#!/usr/bin/env bash\n"
        'echo "$*" >> "$(dirname "$0")/../systemctl.log"\n'
        f'if [ "$1" = show ]; then echo "{cgroup}"; fi\n'
    )


def _install() -> RunnerInstall:
    """The WSL install every render case places.

    Returns:
        The install.
    """
    return RunnerInstall(
        repo="wagner-austin/MCPs",
        runner_name="lavender-wsl",
        side="wsl",
        service=UNIT,
        workdir="/home/gharunner/actions-runner/_work",
        labels=["lavender-wsl"],
        python_toolcache=[],
    )


def _budget(**overrides: JSONValue) -> dict[str, JSONValue]:
    """Lavender's raw budget with fields overridden.

    Args:
        overrides: Field values replacing lavender's.

    Returns:
        The raw mapping.
    """
    raw = ci_slice_json()
    raw.update(overrides)
    return raw


def _move(tmp_path: pathlib.Path, cgroup: str) -> tuple[str, list[str]]:
    """Run the rendered move for one unit whose live control group is ``cgroup``.

    Only the lines from the control-group read onward run: the drop-in
    before them writes under /etc/systemd, which a test host must not touch.

    Args:
        tmp_path: The case's directory.
        cgroup: What the stand-in reports as the unit's ControlGroup.

    Returns:
        The move's standard output, and the systemctl calls it made.
    """
    lines = runner_slice_render.render_runner_slice_lines(_install())
    first = next(index for index, line in enumerate(lines) if line.startswith("cgroup="))
    (tmp_path / "move.sh").write_bytes(("\n".join(lines[first:]) + "\n").encode())
    stand_ins = tmp_path / "bin"
    stand_ins.mkdir()
    (stand_ins / "systemctl").write_bytes(_systemctl_stand_in(cgroup).encode())
    log = tmp_path / "systemctl.log"
    log.write_bytes(b"")
    ran = subprocess.run(
        [host_bash(), "-c", 'chmod +x bin/systemctl && PATH="$PWD/bin:$PATH" bash move.sh'],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
    )
    assert ran.returncode == 0, ran.stderr
    return ran.stdout, log.read_text(encoding="utf-8").splitlines()


class TestContract:
    """decode_ci_slice's refusals and the round trip."""

    def test_lavender_s_budget_decodes_and_round_trips(self) -> None:
        decoded: CiSlice = decode_ci_slice(ci_slice_json(), vm_memory_gb=26)
        assert decoded == a_ci_slice()
        assert encode_ci_slice(decoded) == ci_slice_json()

    def test_a_host_with_the_default_vm_size_is_checked_only_against_itself(self) -> None:
        assert decode_ci_slice(_budget(memory_max_gb=64), vm_memory_gb=None)["memory_max_gb"] == 64

    @pytest.mark.parametrize("field", ["memory_high_gb", "memory_max_gb", "cpu_weight"])
    def test_a_missing_field_is_refused(self, field: str) -> None:
        raw = _budget()
        del raw[field]
        with pytest.raises(JSONTypeError, match=field):
            decode_ci_slice(raw, vm_memory_gb=26)

    def test_a_throttle_point_of_nothing_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="memory_high_gb must be positive, got 0"):
            decode_ci_slice(_budget(memory_high_gb=0), vm_memory_gb=26)

    def test_a_ceiling_below_the_throttle_point_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="kill jobs before systemd ever throttled"):
            decode_ci_slice(_budget(memory_max_gb=15), vm_memory_gb=26)

    @pytest.mark.parametrize("weight", [0, 10001])
    def test_a_weight_outside_systemd_s_range_is_refused(self, weight: int) -> None:
        with pytest.raises(JSONTypeError, match=f"between 1 and 10000, got {weight}"):
            decode_ci_slice(_budget(cpu_weight=weight), vm_memory_gb=26)

    @pytest.mark.parametrize("vm_memory_gb", [18, 12])
    def test_a_ceiling_that_leaves_the_vm_nothing_is_refused(self, vm_memory_gb: int) -> None:
        with pytest.raises(JSONTypeError, match="starvation this ceiling exists to prevent"):
            decode_ci_slice(ci_slice_json(), vm_memory_gb=vm_memory_gb)


class TestSliceUnit:
    """The slice unit and the lines that keep it current."""

    def test_the_unit_carries_the_budget(self) -> None:
        unit = runner_slice_render.render_slice_unit(a_ci_slice())
        assert "[Slice]\n" in unit
        assert "MemoryHigh=16G\nMemoryMax=18G\n" in unit
        assert unit.endswith("CPUAccounting=yes\nCPUWeight=20\n")

    def test_the_unit_is_rewritten_only_when_it_differs_and_always_started(self) -> None:
        path = runner_slice_render.SLICE_UNIT_PATH
        body = runner_slice_render.render_slice_unit(a_ci_slice()).rstrip("\n")
        assert runner_slice_render.render_slice_unit_lines(a_ci_slice()) == [
            f'if [ "$(cat {path} 2>/dev/null)" != "$(cat <<\'SLICE_EOF\'',
            body,
            "SLICE_EOF",
            ')" ]; then',
            f"    cat > {path} <<'SLICE_EOF'",
            body,
            "SLICE_EOF",
            "    systemctl daemon-reload",
            "    echo 'ci budget set: runners.slice MemoryHigh=16G MemoryMax=18G CPUWeight=20'",
            "fi",
            "systemctl start runners.slice",
        ]

    def test_the_slice_is_laid_before_any_runner_is_placed_in_it(self) -> None:
        script = runner_render.render_provision(_roster_host())["linux_script"]
        assert script.index(f"systemctl start {CI_SLICE_NAME}") < script.index(
            runner_slice_render.SLICE_DROP_IN_NAME
        )

    def test_the_whole_linux_provision_parses_as_bash(self, tmp_path: pathlib.Path) -> None:
        script = runner_render.render_provision(_roster_host())["linux_script"]
        (tmp_path / "provision.sh").write_bytes(script.encode())
        parsed = subprocess.run(
            [host_bash(), "-n", "provision.sh"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
        )
        assert parsed.returncode == 0, parsed.stderr


class TestRunnerPlacement:
    """Each runner's drop-in and the move into the slice."""

    def test_the_drop_in_names_the_slice_and_is_written_only_when_it_differs(self) -> None:
        lines = runner_slice_render.render_runner_slice_lines(_install())
        path = f"/etc/systemd/system/{UNIT}.d/fleet-slice.conf"
        assert runner_slice_render.SLICE_DROP_IN == "[Service]\nSlice=runners.slice\n"
        assert lines[0] == (
            f'if [ "$(cat {path} 2>/dev/null)" != '
            "\"$(printf '[Service]\\nSlice=runners.slice\\n')\" ]; then"
        )
        assert lines[2] == f"    printf '[Service]\\nSlice=runners.slice\\n' > {path}"

    def test_a_runner_already_in_the_slice_is_left_running(self, tmp_path: pathlib.Path) -> None:
        out, calls = _move(tmp_path, f"/{CI_SLICE_NAME}/{UNIT}")
        assert out == ""
        assert calls == [f"show -p ControlGroup --value {UNIT}"]

    def test_an_idle_runner_outside_the_slice_is_restarted_into_it(
        self, tmp_path: pathlib.Path
    ) -> None:
        """No process is listed for the unit here, which is an idle runner's
        shape between jobs; the /dev/null guard keeps grep off stdin."""
        out, calls = _move(tmp_path, f"/system.slice/{UNIT}")
        assert out == f"ci budget applied: {UNIT} moved into {CI_SLICE_NAME}\n"
        assert calls == [f"show -p ControlGroup --value {UNIT}", f"restart {UNIT}"]

    def test_a_runner_running_a_job_is_named_and_never_restarted(self) -> None:
        lines = runner_slice_render.render_runner_slice_lines(_install())
        busy = next(index for index, line in enumerate(lines) if "Runner.Worker" in line)
        assert lines[busy + 1] == (
            f"            echo 'ci budget pending: {UNIT} is running a job; "
            "the next provision moves it'"
        )
        assert lines[busy + 2] == "        else"

    def test_an_unscriptable_unit_name_is_refused(self) -> None:
        install = _install()
        install["service"] = "bad'unit.service"
        with pytest.raises(ValueError, match="service"):
            runner_slice_render.render_runner_slice_lines(install)


def _roster_host() -> HostRunnerSpec:
    """A one-install host carrying lavender's budget.

    Returns:
        The spec.
    """
    return HostRunnerSpec(
        name="lavender",
        host="lavender",
        wsl_distro="Ubuntu",
        keepalive_task="wsl-keepalive",
        wslconfig_min_memory_gb=26,
        scratch_dir="C:/fleet/stage",
        gpu_required=False,
        systemd_timers=["ci-clean.timer"],
        job_timeout_minutes=360,
        installs=[_install()],
        assets=[],
        base=a_base(),
        ci_slice=a_ci_slice(),
    )
