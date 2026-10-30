"""The runner roster's contracts: every decode refusal, and the round trip.

The shipped ``runners.json`` is decoded here through the same functions
production uses, so the document in the repo cannot drift from the contract
that reads it -- the same discipline as every other document this package
owns.
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.json_utils import JSONTypeError, JSONValue, load_json_str

from fleet.contracts.runner_base import disk_bounds_gb
from fleet.contracts.runners import (
    FileAsset,
    HostRunnerSpec,
    RunnerInstall,
    decode_file_asset,
    decode_host_runner_spec,
    decode_runner_install,
    decode_runner_spec,
    encode_host_runner_spec,
    encode_runner_spec,
)
from fleet.contracts.tagged_tools import CHROME_LINUX_PATH, TAGGED_TOOLS
from fleet.contracts.workspace import decode_fleet_workspace
from tests._registry import shipped_document
from tests._runner_fixtures import base_json, ci_slice_json

#: The shipped roster, resolved from this file so the test runs from any cwd.
SHIPPED_ROSTER = pathlib.Path(__file__).resolve().parent.parent / "runners.json"


def _install(**overrides: JSONValue) -> dict[str, JSONValue]:
    """A valid raw install, with fields overridden per test.

    Args:
        overrides: Field values replacing the valid defaults.

    Returns:
        The raw mapping.
    """
    raw: dict[str, JSONValue] = {
        "repo": "wagner-austin/API",
        "runner_name": "lavender",
        "service": "actions.runner.wagner-austin-API.lavender.service",
        "workdir": "/home/gharunner/actions-runner-api-1/_work",
        "labels": ["lavender"],
    }
    raw.update(overrides)
    return raw


def _asset(**overrides: JSONValue) -> dict[str, JSONValue]:
    """A valid raw asset, with fields overridden per test.

    Args:
        overrides: Field values replacing the valid defaults.

    Returns:
        The raw mapping.
    """
    raw: dict[str, JSONValue] = {
        "path": "/data",
        "sha256": None,
        "writable": True,
        "reason": "checkpoint tests write under it",
        "manual": False,
        "provision_command": None,
    }
    raw.update(overrides)
    return raw


def _host(**overrides: JSONValue) -> dict[str, JSONValue]:
    """A valid raw host, with fields overridden per test.

    Args:
        overrides: Field values replacing the valid defaults.

    Returns:
        The raw mapping.
    """
    raw: dict[str, JSONValue] = {
        "name": "lavender",
        "host": "lavender",
        "memory_gb": 31,
        "payload_dir": "/home/corvis/fleet/runner-payloads",
        "gpu_required": True,
        "systemd_timers": ["ci-clean.timer"],
        "job_timeout_minutes": 360,
        "installs": [_install()],
        "assets": [_asset()],
        "base": base_json(),
        "ci_slice": ci_slice_json(),
    }
    raw.update(overrides)
    return raw


class TestRunnerInstall:
    """decode_runner_install's contract."""

    def test_a_valid_install_round_trips(self) -> None:
        decoded: RunnerInstall = decode_runner_install(_install())
        assert decoded == RunnerInstall(
            repo="wagner-austin/API",
            runner_name="lavender",
            service="actions.runner.wagner-austin-API.lavender.service",
            workdir="/home/gharunner/actions-runner-api-1/_work",
            labels=["lavender"],
        )

    def test_a_non_object_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="must be a JSON object"):
            decode_runner_install("not an object")

    @pytest.mark.parametrize("repo", ["no-slash", "a/b/c", "/name", "owner/"])
    def test_a_repo_that_is_not_owner_name_is_refused(self, repo: str) -> None:
        with pytest.raises(JSONTypeError, match="owner/name"):
            decode_runner_install(_install(repo=repo))

    def test_empty_labels_are_refused_because_nothing_could_target_the_runner(self) -> None:
        with pytest.raises(JSONTypeError, match="unreachable"):
            decode_runner_install(_install(labels=[]))

    @pytest.mark.parametrize("workdir", ["C:/actions-runner/_work", "actions-runner/_work", ""])
    def test_a_workdir_that_is_not_an_absolute_posix_path_is_refused(self, workdir: str) -> None:
        with pytest.raises(JSONTypeError) as refused:
            decode_runner_install(_install(workdir=workdir))
        assert str(refused.value) == (
            f"an install's workdir must be an absolute POSIX path, got {workdir!r}"
        )

    @pytest.mark.parametrize("field", ["runner_name", "service"])
    def test_a_missing_name_or_service_is_refused(self, field: str) -> None:
        raw = _install()
        del raw[field]
        with pytest.raises(JSONTypeError, match=field):
            decode_runner_install(raw)


class TestFileAsset:
    """decode_file_asset's contract."""

    def test_a_valid_writable_asset_round_trips(self) -> None:
        decoded: FileAsset = decode_file_asset(_asset())
        assert decoded["writable"] is True
        assert decoded["sha256"] is None

    def test_a_valid_pinned_asset_keeps_its_pin(self) -> None:
        pin = "8a550a37e2d8a5430866090d4e7d5892f9010b47f52a5a09350fc66c620deec9"
        decoded = decode_file_asset(
            _asset(sha256=pin, writable=False, manual=True, provision_command=None)
        )
        assert decoded["sha256"] == pin

    def test_a_non_object_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="must be a JSON object"):
            decode_file_asset(17)

    def test_an_absent_sha256_key_is_refused(self) -> None:
        raw = _asset()
        del raw["sha256"]
        with pytest.raises(JSONTypeError, match="must declare 'sha256'"):
            decode_file_asset(raw)

    @pytest.mark.parametrize("pin", ["abc", "G" * 64, "A" * 64])
    def test_a_malformed_pin_is_refused(self, pin: str) -> None:
        with pytest.raises(JSONTypeError, match="64 lowercase hex"):
            decode_file_asset(_asset(sha256=pin))

    def test_a_non_string_pin_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="string or null"):
            decode_file_asset(_asset(sha256=64))

    def test_a_provision_command_on_a_manual_asset_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="must be null on a manual or writable"):
            decode_file_asset(
                _asset(manual=True, writable=False, provision_command="curl something")
            )

    def test_a_provision_command_on_a_writable_asset_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="must be null on a manual or writable"):
            decode_file_asset(_asset(provision_command="mkdir -p /data"))

    def test_a_fetchable_asset_without_a_command_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="must carry provision_command"):
            decode_file_asset(_asset(writable=False, provision_command=None))

    def test_a_fetchable_asset_with_a_command_is_accepted(self) -> None:
        decoded = decode_file_asset(_asset(writable=False, provision_command="git clone x /opt/x"))
        assert decoded["provision_command"] == "git clone x /opt/x"

    def test_an_absent_provision_command_key_is_refused(self) -> None:
        raw = _asset()
        del raw["provision_command"]
        with pytest.raises(JSONTypeError, match="must declare 'provision_command'"):
            decode_file_asset(raw)

    def test_a_mistyped_provision_command_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="string or null"):
            decode_file_asset(_asset(provision_command=3))


class TestHostRunnerSpec:
    """decode_host_runner_spec's contract."""

    def test_a_valid_host_round_trips(self) -> None:
        decoded: HostRunnerSpec = decode_host_runner_spec(_host())
        assert (decoded["name"], decoded["memory_gb"], decoded["payload_dir"]) == (
            "lavender",
            31,
            "/home/corvis/fleet/runner-payloads",
        )
        assert decode_host_runner_spec(dict(encode_host_runner_spec(decoded))) == decoded

    def test_a_non_object_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="must be a JSON object"):
            decode_host_runner_spec([])

    @pytest.mark.parametrize("memory", [0, -4])
    def test_a_memory_that_is_not_positive_is_refused(self, memory: int) -> None:
        with pytest.raises(JSONTypeError) as refused:
            decode_host_runner_spec(_host(memory_gb=memory))
        assert str(refused.value) == f"memory_gb must be positive, got {memory}"

    def test_a_ci_slice_reaching_the_host_s_memory_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="leaves nothing of the host's 18 GB"):
            decode_host_runner_spec(_host(memory_gb=18))

    @pytest.mark.parametrize("payload_dir", ["C:/fleet/stage", "fleet/payloads", ""])
    def test_a_payload_dir_that_is_not_absolute_is_refused(self, payload_dir: str) -> None:
        with pytest.raises(JSONTypeError) as refused:
            decode_host_runner_spec(_host(payload_dir=payload_dir))
        assert str(refused.value) == (
            f"payload_dir must be an absolute POSIX path, got {payload_dir!r}"
        )

    @pytest.mark.parametrize("field", ["memory_gb", "payload_dir"])
    def test_a_missing_memory_or_payload_dir_is_refused(self, field: str) -> None:
        raw = _host()
        del raw[field]
        with pytest.raises(JSONTypeError, match=field):
            decode_host_runner_spec(raw)

    def test_a_host_with_no_installs_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="not a CI host"):
            decode_host_runner_spec(_host(installs=[]))

    def test_the_job_timeout_round_trips(self) -> None:
        decoded = decode_host_runner_spec(_host(job_timeout_minutes=60))
        assert decoded["job_timeout_minutes"] == 60
        assert encode_host_runner_spec(decoded)["job_timeout_minutes"] == 60

    def test_an_absent_job_timeout_is_refused(self) -> None:
        raw = _host()
        del raw["job_timeout_minutes"]
        with pytest.raises(JSONTypeError, match="job_timeout_minutes"):
            decode_host_runner_spec(raw)

    @pytest.mark.parametrize("minutes", [0, -60])
    def test_a_non_positive_job_timeout_is_refused(self, minutes: int) -> None:
        with pytest.raises(JSONTypeError, match="job_timeout_minutes must be positive"):
            decode_host_runner_spec(_host(job_timeout_minutes=minutes))

    def test_a_host_without_a_base_is_refused(self) -> None:
        raw = _host()
        del raw["base"]
        with pytest.raises(JSONTypeError, match="base"):
            decode_host_runner_spec(raw)

    def test_the_base_decodes_with_the_host(self) -> None:
        decoded = decode_host_runner_spec(_host())
        assert decoded["base"]["apt_packages"] == ["ffmpeg", "golang-go"]
        assert decoded["base"]["disk"]["ceiling_gb"] == 150

    def test_a_ceiling_under_the_host_s_own_bounds_is_accepted(self) -> None:
        # The operator holds lavender's root under 150 GB ('Get it under
        # 150', board task 621902c3) although its areas' bounds add to 193:
        # the audit's root row is its own requirement, so a host whose every
        # area is inside its bound can still fail it, and should.
        base = base_json()
        base["disk"] = {
            "ceiling_gb": 100,
            "baseline_gb": 18,
            "baseline_measured": "2026-10-20",
            "cache_path": "/home/gharunner/.cache",
            "cache_ceiling_gb": 60,
            "work_ceiling_gb": 15,
        }
        decoded = decode_host_runner_spec(_host(base=base))
        bounds = disk_bounds_gb(decoded["base"]["disk"], len(decoded["installs"]))
        assert (decoded["base"]["disk"]["ceiling_gb"], bounds) == (100, 103)


class TestRunnerSpec:
    """decode_runner_spec's contract, and the shipped document."""

    def test_a_non_object_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="must be a JSON object"):
            decode_runner_spec(None)

    def test_an_empty_roster_is_refused_because_it_reads_as_healthy(self) -> None:
        with pytest.raises(JSONTypeError, match="reads exactly like a healthy fleet"):
            decode_runner_spec({"hosts": []})

    def test_duplicate_host_names_are_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="must be unique, got duplicates: lavender"):
            decode_runner_spec({"hosts": [_host(), _host()]})

    def test_encode_then_decode_is_the_identity(self) -> None:
        spec = decode_runner_spec({"hosts": [_host()]})
        assert dict(encode_runner_spec(spec)).keys() == {"hosts"}
        assert decode_runner_spec(dict(encode_runner_spec(spec))) == spec

    def test_the_shipped_roster_decodes_and_round_trips(self) -> None:
        """The runners.json in the repo satisfies its own contract.

        This is the integration claim of the module: the document operators
        edit is validated by the exact decoder the CLI runs, so a bad edit
        fails here before it fails in the field.
        """
        spec = decode_runner_spec(load_json_str(SHIPPED_ROSTER.read_text(encoding="utf-8")))
        assert decode_runner_spec(dict(encode_runner_spec(spec))) == spec
        lavender = spec["hosts"][0]
        assert lavender["name"] == "lavender"
        # The Linux runners lavender's WSL guest carried until its reimage to
        # Ubuntu on 2026-10-19 (MCPs board task 1e9ab738), now on the host
        # itself: 2 API, MCPs, corvis-stick and tree-bot, every one
        # answering the host's own label, and chat's since its CI left loki
        # with the rest of the Windows runners (board task e705f08c).
        # corvis-stick was portable-claude until its rename, and registering
        # against the old URL 404s. MCPs went from 4 to 2 when four of its
        # check jobs at once oversubscribed the CI slice's eight threads
        # (board task 49d1a916).
        per_repo = [install["repo"] for install in lavender["installs"]]
        assert {repo: per_repo.count(repo) for repo in per_repo} == {
            "wagner-austin/API": 2,
            "wagner-austin/MCPs": 2,
            "wagner-austin/corvis-stick": 1,
            "wagner-austin/tree-bot": 1,
            "wagner-austin/chat": 1,
        }
        assert len(spec["hosts"]) == 1
        assert {label for install in lavender["installs"] for label in install["labels"]} == {
            "lavender"
        }
        # Every asset is one the provision places. The one placed by hand,
        # RustedWarfareBot's licensed game jar, had no reader once its CI left
        # lavender (board task 4f228608) and was dropped (MCPs board task
        # faec98a5).
        assert [asset["path"] for asset in lavender["assets"] if asset["manual"]] == []
        assert lavender["base"]["disk"]["baseline_gb"] < lavender["base"]["disk"]["ceiling_gb"]

    def test_a_runner_host_installs_every_tagged_tool_apt_supplies(self) -> None:
        """A rebuild never takes a capability tag away from a fleet node.

        A CI runner host here is a fleet machine of registry/machines too,
        whose runner claims with the tags its probe finds each tick: lavender
        was through its WSL guest lavender-wsl, and is itself since its
        reimage to Ubuntu on 2026-10-19 (MCPs board task 1e9ab738). A tagged
        tool installed by hand there would be lost on the next ``--rebuild``,
        and every job needing its tag would silently wait for another node:
        on 2026-10-05 no node had go, and MCPs rcs-bridge's check had nowhere
        to run (MCPs board task 1da15750). So the roster's base installs each
        tagged tool's apt-get package, and carries as an asset the one tagged
        tool no apt repository supplies: Chrome, the .deb Google ships, whose
        asset path is the file the probe's chrome line asks for (MCPs board
        task 2f596185). poppler-utils joined the base when pdftotext became a
        tag (MCPs board task 6dd73ada): lavender-wsl had it on 2026-10-11,
        but from no line here.
        """
        spec = decode_runner_spec(load_json_str(SHIPPED_ROSTER.read_text(encoding="utf-8")))
        machines = set(decode_fleet_workspace(shipped_document())["machines"])
        apt_installs = {
            tool["name"]: tool["install"]["apt-get"]
            for tool in TAGGED_TOOLS
            if "apt-get" in tool["install"]
        }
        apt_supplied = {
            install.split()[-1]
            for install in apt_installs.values()
            if install.startswith("sudo apt-get install -y ")
        }
        assert apt_supplied == {"ffmpeg", "golang-go", "poppler-utils"}
        assert set(apt_installs) - {"ffmpeg", "go", "pdftotext"} == {"chrome"}
        assert apt_installs["chrome"].startswith("curl -fsSL -o ")
        assert [host["name"] for host in spec["hosts"] if host["name"] in machines] == ["lavender"]
        for host in spec["hosts"]:
            assert apt_supplied <= set(host["base"]["apt_packages"]), host["name"]
            assert CHROME_LINUX_PATH in {asset["path"] for asset in host["assets"]}, host["name"]
