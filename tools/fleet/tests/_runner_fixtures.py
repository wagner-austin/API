"""The runner-host base every roster fixture in these tests carries.

One definition, shaped like lavender's real entry in ``runners.json``, so a
test that builds a host spec does not restate the base it is not about, and
a change to the base contract lands in one fixture rather than in every
test file's own copy.
"""

from __future__ import annotations

from collections.abc import Sequence

from platform_core.json_utils import JSONValue

from fleet.contracts.runner_base import DiskCeiling, HostBase
from fleet.contracts.runner_slice import CiSlice
from fleet.contracts.runner_windows import WindowsRunnerHost, decode_windows_runner_host
from fleet.contracts.runners import FileAsset, HostRunnerSpec, RunnerInstall
from fleet.core import _test_hooks
from tests.conftest import ok


def an_install(
    repo: str = "wagner-austin/API", runner_name: str = "lavender", *, slot: int = 1
) -> RunnerInstall:
    """One install named as the roster names lavender's.

    Args:
        repo: Its repository, ``owner/name``.
        runner_name: Its runner's name.
        slot: Which of the repository's installs on the host it is, the
            number in its install directory.

    Returns:
        The install: unit ``actions.runner.<owner>-<name>.<runner>.service``,
        work tree ``/home/gharunner/actions-runner-<name>-<slot>/_work``, and
        the host's label.
    """
    owner, name = repo.split("/")
    return RunnerInstall(
        repo=repo,
        runner_name=runner_name,
        service=f"actions.runner.{owner}-{name}.{runner_name}.service",
        workdir=f"/home/gharunner/actions-runner-{name.lower()}-{slot}/_work",
        labels=["lavender"],
    )


def a_host(
    installs: Sequence[RunnerInstall] = (),
    *,
    gpu_required: bool = False,
    systemd_timers: Sequence[str] = (),
    assets: Sequence[FileAsset] = (),
) -> HostRunnerSpec:
    """A lavender-shaped host.

    Args:
        installs: Its installs; none given is one :func:`an_install`.
        gpu_required: Whether its jobs digest a GPU.
        systemd_timers: The timers it must keep enabled.
        assets: The paths its jobs need.

    Returns:
        The spec: 31 GB, payloads under ``/home/corvis/fleet/runner-payloads``,
        a six-hour job timeout, :func:`a_base` and :func:`a_ci_slice`.
    """
    return HostRunnerSpec(
        name="lavender",
        host="lavender",
        memory_gb=31,
        payload_dir="/home/corvis/fleet/runner-payloads",
        gpu_required=gpu_required,
        systemd_timers=list(systemd_timers),
        job_timeout_minutes=360,
        installs=list(installs) if installs else [an_install()],
        assets=list(assets),
        base=a_base(),
        ci_slice=a_ci_slice(),
    )


def a_base() -> HostBase:
    """A lavender-shaped host base.

    Returns:
        The base: ffmpeg and golang-go as the packages a fleet worker lacks,
        and a 150 GB ceiling over an 18 GB baseline.
    """
    return HostBase(
        apt_packages=["ffmpeg", "golang-go"],
        disk=DiskCeiling(
            ceiling_gb=150,
            baseline_gb=18,
            baseline_measured="2026-10-20",
            cache_path="/home/gharunner/.cache",
            cache_ceiling_gb=60,
            work_ceiling_gb=15,
        ),
    )


def base_json() -> dict[str, JSONValue]:
    """The same base as the roster's JSON spells it.

    Returns:
        The encoded mapping, built by hand rather than by the encoder, so a
        decoder test is not a round trip through the code it is testing.
    """
    return {
        "apt_packages": ["ffmpeg", "golang-go"],
        "disk": {
            "ceiling_gb": 150,
            "baseline_gb": 18,
            "baseline_measured": "2026-10-20",
            "cache_path": "/home/gharunner/.cache",
            "cache_ceiling_gb": 60,
            "work_ceiling_gb": 15,
        },
    }


def no_dropped_jobs(spec: HostRunnerSpec) -> list[_test_hooks.CommandResult]:
    """The dropped-job row's reads on a host where nothing is dropped (MCPs task 715b9ec5).

    Args:
        spec: The audited host.

    Returns:
        One ``gh`` runner list per repository of its installs, listing none,
        then the job scope helper's ``--workers`` answer naming each unit
        with no Worker.
    """
    installs = spec["installs"]
    return [
        *(ok("") for _ in sorted({install["repo"] for install in installs})),
        ok("".join(f"{install['service']} -\n" for install in installs)),
    ]


def raw_windows_install(repo: str = "wagner-austin/chat") -> dict[str, JSONValue]:
    """One of loki's runners as the roster's JSON spells it.

    Args:
        repo: The repository; its name names the install's directory.

    Returns:
        The raw install.
    """
    labels: list[JSONValue] = ["loki"]
    return {
        "repo": repo,
        "runner_name": "loki",
        "service": f"actions.runner.{repo.replace('/', '-')}.loki",
        "root": f"C:/actions-runner/{repo.split('/')[1]}",
        "labels": labels,
    }


def raw_windows_host() -> dict[str, JSONValue]:
    """loki with one runner, as the roster's JSON spells it.

    Returns:
        The raw host.
    """
    installs: list[JSONValue] = [raw_windows_install()]
    return {
        "name": "loki",
        "host": "loki",
        "payload_dir": "C:/fleet/runner-payloads",
        "installs": installs,
    }


def a_windows_host() -> WindowsRunnerHost:
    """loki with one runner, decoded.

    Returns:
        The host.
    """
    return decode_windows_runner_host(raw_windows_host())


def all_declared(hosts: Sequence[HostRunnerSpec]) -> list[_test_hooks.CommandResult]:
    """The registrations rows' reads where GitHub lists exactly what the hosts declare.

    Args:
        hosts: The audited hosts.

    Returns:
        One ``gh`` runner list per repository of their installs, in sorted
        order, listing each install of those hosts with an id, online and
        idle.
    """
    installs = [install for host in hosts for install in host["installs"]]
    repos = sorted({install["repo"] for install in installs})
    return [
        ok(
            "".join(
                f"{index} {install['runner_name']} online false\n"
                for index, install in enumerate(installs, start=1)
                if install["repo"] == repo
            )
        )
        for repo in repos
    ]


def quiet_rebuild_answers(
    transcript: str, spec: HostRunnerSpec, *, repos: int
) -> list[_test_hooks.CommandResult]:
    """Every command a rebuild of an already-built host runs, in order.

    Written out rather than generated from the code, because the sequence is
    the claim: each stage sends its payload and runs it.

    Args:
        transcript: What the closing audit prints.
        spec: The rebuilt host, whose installs the audit's dropped-job row
            reads.
        repos: How many repositories the roster names, one token each.

    Returns:
        The replies: the base (send, run), one ``gh`` mint per repository,
        the provision (send, run), the audit (send, run) and its dropped-job
        reads (:func:`no_dropped_jobs`).
    """
    return [
        ok(""),
        ok("base ready: packages installed, docker reachable as gharunner\n"),
        *(ok(f"TOKEN{index}\n") for index in range(repos)),
        ok(""),
        ok(""),
        ok(""),
        ok(transcript),
        *no_dropped_jobs(spec),
    ]


def a_ci_slice() -> CiSlice:
    """Lavender's CI budget, as ``runners.json`` declares it.

    Returns:
        16 GB throttle, 18 GB ceiling, 2 GB of swap, weight 20, and root-disk
        reads of 200 MB/s and 5000 per second, under a 31 GB host.
    """
    return CiSlice(
        memory_high_gb=16,
        memory_max_gb=18,
        swap_max_gb=2,
        cpu_weight=20,
        io_read_mb_per_s=200,
        io_read_iops=5000,
    )


def ci_slice_json() -> dict[str, JSONValue]:
    """The same budget as the roster's JSON spells it, built by hand.

    Returns:
        The mapping.
    """
    return {
        "memory_high_gb": 16,
        "memory_max_gb": 18,
        "swap_max_gb": 2,
        "cpu_weight": 20,
        "io_read_mb_per_s": 200,
        "io_read_iops": 5000,
    }


__all__ = [
    "a_base",
    "a_ci_slice",
    "a_host",
    "a_windows_host",
    "all_declared",
    "an_install",
    "base_json",
    "ci_slice_json",
    "no_dropped_jobs",
    "quiet_rebuild_answers",
    "raw_windows_host",
    "raw_windows_install",
]
