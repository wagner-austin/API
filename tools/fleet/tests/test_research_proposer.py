"""Closed proposal-process policy and lifecycle; host twins exercise the daemon."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, JSONValue, dump_json_str

from fleet.core import _test_hooks
from fleet.core.research_proposer import MAX_BYTES, run_proposer
from tests.conftest import FakeRun, failed, ok, timed_out

SOCKET = "/run/user/1001/docker.sock"
IMAGE = "sha256:" + "b" * 64
CID = "a" * 64
INFO: JSONObject = {
    "SecurityOptions": ["name=rootless"],
    "MemoryLimit": True,
    "CpuCfsQuota": True,
    "PidsLimit": True,
    "CgroupVersion": "2",
}


def replies(output: str = "{}") -> list[_test_hooks.CommandResult]:
    """A successful rootless daemon lifecycle, with configurable proposal output."""
    return [
        ok(value) for value in (dump_json_str(INFO), "null", CID, CID, "0", output, CID, CID, "")
    ]


def test_success_retains_closed_runtime_policy_and_verifies_cleanup() -> None:
    fake = FakeRun(replies('{"partition":"validation"}'))
    _test_hooks.run = fake
    assert run_proposer(SOCKET, IMAGE, "print(1)", {"budget": 2}, timeout=10) == {
        "partition": "validation"
    }
    assert MAX_BYTES == 8192
    create = fake.calls[2]
    for flag in (
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--pull=never",
        "--security-opt=no-new-privileges",
        "--user=65534:65534",
        "--memory=256m",
        "--memory-swap=256m",
        "--pids-limit=64",
        "--cpus=1",
        "--log-driver=local",
        "--log-opt=max-size=16k",
        "--log-opt=max-file=1",
    ):
        assert flag in create
    assert create[-7:] == (
        "-i",
        "PATH=/usr/local/bin:/usr/bin:/bin",
        "/usr/local/bin/python",
        "-I",
        "-c",
        "print(1)",
        '{"budget":2}',
    )
    assert not any(arg.startswith(("--mount", "--volume", "--env")) for arg in create)
    assert create[:2] == ("/usr/bin/env", "-i")
    label = create[create.index("--label") + 1]
    assert label.startswith("corvis.research.proposer=")
    assert fake.calls[6][-1] == fake.calls[8][-1] == "label=" + label
    assert fake.calls[7][-4:] == ("container", "rm", "--force", CID)
    assert fake.timeouts == [30, 30, 30, 30, 10, 30, 30, 30, 30]
    assert fake.stdin == [None] * 9
    assert fake.set_env == [()] * 9


@pytest.mark.parametrize(
    ("program", "context", "timeout"),
    [
        ("", {}, 10),
        ("x", {}, 0),
        ("x", {}, 121),
        ("x" * 8193, {}, 10),
        ("x", {"context": "é" * 8192}, 10),
    ],
)
def test_bounds_refuse_before_any_daemon_call(
    program: str, context: JSONObject, timeout: int
) -> None:
    fake = FakeRun([])
    _test_hooks.run = fake
    with pytest.raises(ValueError, match="RESEARCH_PROPOSER_BOUNDS"):
        run_proposer(SOCKET, IMAGE, program, context, timeout=timeout)
    assert fake.calls == []


@pytest.mark.parametrize(
    ("socket", "image", "error"),
    [
        ("/var/run/docker.sock", IMAGE, "PROPOSER_SOCKET"),
        (SOCKET, "b" * 64, "PROPOSER_IMAGE"),
        (SOCKET, "python:latest", "REFERENCE_DIGEST"),
    ],
)
def test_unsafe_endpoint_or_mutable_image_is_refused(socket: str, image: str, error: str) -> None:
    _test_hooks.run = FakeRun([])
    with pytest.raises(ValueError, match=error):
        run_proposer(socket, image, "print(1)", {}, timeout=10)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("SecurityOptions", ["name=seccomp"]),
        ("MemoryLimit", False),
        ("CpuCfsQuota", False),
        ("PidsLimit", False),
        ("CgroupVersion", "1"),
    ],
)
def test_daemon_must_enforce_rootless_resource_limits(key: str, value: JSONValue) -> None:
    info: JSONObject = {**INFO, key: value}
    fake = FakeRun([ok(dump_json_str(info))])
    _test_hooks.run = fake
    with pytest.raises(ValueError, match="PROPOSER_RUNTIME"):
        run_proposer(SOCKET, IMAGE, "print(1)", {}, timeout=10)
    assert len(fake.calls) == 1


@pytest.mark.parametrize("spelling", ["CPUCfsQuota", "cpuCfsQuota", "CpuCFSQuota", "cpu_cfs_quota"])
def test_only_dockers_json_cpu_quota_key_is_accepted(spelling: str) -> None:
    info: JSONObject = {key: value for key, value in INFO.items() if key != "CpuCfsQuota"}
    info[spelling] = True
    fake = FakeRun([ok(dump_json_str(info))])
    _test_hooks.run = fake
    with pytest.raises(ValueError, match="PROPOSER_RUNTIME"):
        run_proposer(SOCKET, IMAGE, "print(1)", {}, timeout=10)
    assert len(fake.calls) == 1


@pytest.mark.parametrize("volume", ['{"/data":{}}', "[]"])
def test_image_cannot_add_implicit_mounts(volume: str) -> None:
    _test_hooks.run = FakeRun([ok(dump_json_str(INFO)), ok(volume)])
    with pytest.raises(ValueError, match="PROPOSER_IMAGE"):
        run_proposer(SOCKET, IMAGE, "print(1)", {}, timeout=10)


@pytest.mark.parametrize("output", ["[1]", "not json", "x" * 8193])
def test_invalid_output_is_refused_and_removed(output: str) -> None:
    fake = FakeRun(replies(output))
    _test_hooks.run = fake
    with pytest.raises((ValueError, TypeError)):
        run_proposer(SOCKET, IMAGE, "print(1)", {}, timeout=10)
    assert fake.calls[-2][-4:] == ("container", "rm", "--force", CID)


@pytest.mark.parametrize("case", ["create", "wait", "exit", "identity", "cleanup"])
def test_lifecycle_failures_refuse_and_cleanup_is_verified(case: str) -> None:
    values = replies()
    error = "PROPOSER_PROCESS"
    if case == "create":
        values = [*values[:2], failed(1, "creation failed"), ok(""), ok("")]
    elif case == "wait":
        values = [*values[:4], timed_out(10), *values[6:]]
    elif case == "exit":
        values = [*values[:4], ok("7"), *values[6:]]
        error = "PROPOSER_EXIT"
    elif case == "identity":
        values = [*values[:2], ok("malformed"), *values[6:]]
        error = "REFERENCE_DIGEST"
    else:
        values[-1] = ok(CID)
        error = "PROPOSER_CLEANUP"
    fake = FakeRun(values)
    _test_hooks.run = fake
    with pytest.raises(ValueError, match=error):
        run_proposer(SOCKET, IMAGE, "print(1)", {}, timeout=10)
    assert fake.calls[-1][-2] == "--filter"
