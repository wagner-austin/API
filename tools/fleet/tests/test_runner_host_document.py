"""Every runner host as its NixOS generation carries it (MCPs board task 9aba868d).

MCPs ``fleet-reach render`` copies the committed document into the flake
that builds lavender, so a runner body or a roster change published without
``make render-documents`` would build the old host: the first class fails
until the copy is the render of the live roster. The rest hold the document
to its contract: every body the runner modules render, byte for byte, laid
out by path and mode, and each asset where its kind says it goes.
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.json_utils import (
    load_json_str,
    narrow_json_to_dict,
    require_dict,
    require_list,
    require_str,
)

from fleet.cli.render_documents import RENDERED_DIRECTORY
from fleet.cli.runners import load_reserved_cores, load_runner_spec
from fleet.contracts.runners import FileAsset, RunnerSpec
from fleet.core.ci_clean import CI_CLEAN_SCRIPT, CI_CLEAN_SERVICE, CI_CLEAN_TIMER
from fleet.core.runner_action_cache import CACHE_DIR, CACHE_DROP_IN, FILL_SCRIPT
from fleet.core.runner_cpuset_render import CPUSET_SERVICE, render_ci_cpuset_script
from fleet.core.runner_host_document import (
    RUNNER_HOSTS_DOCUMENT,
    DocumentInstall,
    HostDirectory,
    HostFile,
    ManualAsset,
    host_document,
    host_files,
    render_runner_hosts_document,
)
from fleet.core.runner_install import LOCAL_BIN, RUNNER_SHA256, RUNNER_VERSION, TOKEN_DIRECTORY
from fleet.core.runner_job_scope_render import HOOK_DROP_IN, SUDOERS_RULE
from fleet.core.runner_reaper_render import KILL_DROP_IN, REAPER_SCRIPT, render_reaper_service
from fleet.core.runner_recovery import SYSTEMD_DROP_IN
from fleet.core.runner_slice_render import LISTENER_SLICE_UNIT, SLICE_DROP_IN, render_slice_unit
from tests._runner_fixtures import RESERVED_CORES, a_host, an_install

#: tools/fleet.
_PACKAGE = pathlib.Path(__file__).resolve().parent.parent

#: The committed copy.
_COMMITTED = _PACKAGE / RENDERED_DIRECTORY / RUNNER_HOSTS_DOCUMENT

#: The unit directory, for the drop-in paths below.
_UNITS = "/etc/systemd/system"


def _asset(path: str, *, writable: bool = False, manual: bool = False) -> FileAsset:
    """An asset of one kind: fetched, writable or manual."""
    return FileAsset(
        path=path,
        sha256=None,
        writable=writable,
        reason=f"the job class that needs {path}",
        manual=manual,
    )


class TestTheCommittedCopy:
    def test_the_committed_copy_is_the_render_of_the_live_roster(self) -> None:
        spec = load_runner_spec(str(_PACKAGE / "runners.json"))
        registry = str(_PACKAGE / "registry")
        reserved = {
            host["name"]: load_reserved_cores(registry, host["name"]) for host in spec["hosts"]
        }
        assert _COMMITTED.read_text(encoding="utf-8") == render_runner_hosts_document(
            spec, reserved
        ), "run `make render-documents` in tools/fleet"

    def test_the_copy_carries_lavender_s_seven_installs(self) -> None:
        document = narrow_json_to_dict(load_json_str(_COMMITTED.read_text(encoding="utf-8")))
        installs = require_list(require_dict(document, "lavender"), "installs")
        assert [require_str(narrow_json_to_dict(install), "service") for install in installs] == [
            "actions.runner.wagner-austin-API.lavender.service",
            "actions.runner.wagner-austin-API.lavender-2.service",
            "actions.runner.wagner-austin-MCPs.lavender.service",
            "actions.runner.wagner-austin-MCPs.lavender-2.service",
            "actions.runner.wagner-austin-corvis-stick.lavender.service",
            "actions.runner.wagner-austin-tree-bot.lavender.service",
            "actions.runner.wagner-austin-chat.lavender.service",
        ]


class TestTheFiles:
    def test_every_body_is_the_runner_module_s_own_by_path_and_mode(self) -> None:
        spec = a_host([an_install()])
        files = {file["path"]: (file["mode"], file["text"]) for file in host_files(spec, 8)}
        assert files["/usr/local/bin/ci-clean"] == ("0755", CI_CLEAN_SCRIPT)
        assert files["/usr/local/sbin/fleet-ci-cpuset"] == ("0755", render_ci_cpuset_script(8))
        assert files["/usr/local/sbin/fleet-runner-reaper"] == ("0755", REAPER_SCRIPT)
        assert files["/usr/local/sbin/fleet-action-cache-fill"] == ("0755", FILL_SCRIPT)
        assert files["/etc/sudoers.d/fleet-runner-job-scope"] == ("0440", SUDOERS_RULE)
        assert files[f"{_UNITS}/ci-clean.service"] == ("0644", CI_CLEAN_SERVICE)
        assert files[f"{_UNITS}/ci-clean.timer"] == ("0644", CI_CLEAN_TIMER)
        assert files[f"{_UNITS}/runners.slice"] == ("0644", render_slice_unit(spec["ci_slice"]))
        assert files[f"{_UNITS}/listeners.slice"] == ("0644", LISTENER_SLICE_UNIT)
        assert files[f"{_UNITS}/fleet-ci-cpuset.service"] == ("0644", CPUSET_SERVICE)
        assert files[f"{_UNITS}/fleet-runner-reaper.service"] == (
            "0644",
            render_reaper_service(spec),
        )

    def test_scripts_come_first_then_units_then_the_drop_ins(self) -> None:
        paths = [file["path"] for file in host_files(a_host([an_install()]), RESERVED_CORES)]
        assert len(paths) == len(set(paths))
        kinds = [
            "drop-in" if ".service.d/" in path else "unit" if path.startswith(_UNITS) else "script"
            for path in paths
        ]
        assert kinds == sorted(kinds, key=["script", "unit", "drop-in"].index)
        assert kinds.count("script") == 10
        assert kinds.count("unit") == 11

    def test_each_install_gets_its_five_drop_ins_in_the_provision_s_order(self) -> None:
        api = an_install()
        mcps = an_install("wagner-austin/MCPs")
        files = host_files(a_host([api, mcps]), RESERVED_CORES)
        drop_ins = [file for file in files if ".service.d/" in file["path"]]
        expected = [
            ("fleet-restart.conf", SYSTEMD_DROP_IN),
            ("fleet-kill.conf", KILL_DROP_IN),
            ("fleet-job-scope.conf", HOOK_DROP_IN),
            ("fleet-action-cache.conf", CACHE_DROP_IN),
            ("fleet-slice.conf", SLICE_DROP_IN),
        ]
        assert drop_ins == [
            HostFile(path=f"{_UNITS}/{install['service']}.d/{name}", mode="0644", text=body)
            for install in (api, mcps)
            for name, body in expected
        ]

    def test_a_reservation_of_none_is_refused(self) -> None:
        with pytest.raises(ValueError, match="reserved_cores is 0"):
            host_files(a_host(), 0)


class TestTheEntry:
    def test_it_names_the_account_the_pinned_release_and_each_install(self) -> None:
        install = an_install("wagner-austin/corvis-stick", slot=1)
        entry = host_document(a_host([install]), RESERVED_CORES)
        assert entry["account"] == "gharunner"
        assert entry["runner"] == {"version": RUNNER_VERSION, "sha256": RUNNER_SHA256}
        assert entry["localBin"] == LOCAL_BIN
        assert entry["tokenDirectory"] == TOKEN_DIRECTORY == "/var/lib/fleet-runner-tokens"
        assert entry["packages"] == ["ffmpeg", "go"]
        assert entry["installs"] == [
            DocumentInstall(
                service="actions.runner.wagner-austin-corvis-stick.lavender.service",
                root="/home/gharunner/actions-runner-corvis-stick-1",
                repo="wagner-austin/corvis-stick",
                name="lavender",
                labels=["lavender"],
                tokenVariable="RUNNER_TOKEN_CORVIS_STICK",
            )
        ]

    def test_it_starts_every_timer_and_the_two_boot_passes(self) -> None:
        entry = host_document(a_host(), RESERVED_CORES)
        assert entry["timers"] == [
            "ci-clean.timer",
            "fleet-runner-reaper.timer",
            "fleet-action-cache-fill.timer",
            "fleet-action-cache-adopt.timer",
        ]
        assert entry["services"] == ["fleet-ci-cpuset.service", "ci-clean.service"]
        units = {file["path"] for file in entry["files"]}
        assert {f"{_UNITS}/{name}" for name in [*entry["timers"], *entry["services"]]} <= units

    def test_each_asset_goes_where_its_kind_says(self) -> None:
        """A writable asset is a directory the runners own, a fetched one a
        path the generation must provide, and a manual one is named alone."""
        spec = a_host(
            assets=[
                _asset("/opt/tool/convert.py"),
                _asset("/data", writable=True),
                _asset("/opt/licensed/game.jar", manual=True),
            ]
        )
        entry = host_document(spec, RESERVED_CORES)
        assert entry["directories"] == [
            HostDirectory(path=CACHE_DIR, mode="0755", owner="root"),
            HostDirectory(path="/data", mode="0755", owner="gharunner"),
        ]
        assert entry["assets"] == ["/opt/tool/convert.py"]
        assert entry["manualAssets"] == [
            ManualAsset(
                path="/opt/licensed/game.jar",
                reason="the job class that needs /opt/licensed/game.jar",
            )
        ]


class TestTheDocument:
    def test_it_keys_each_host_by_its_name(self) -> None:
        spec = RunnerSpec(hosts=[a_host()])
        document = load_json_str(render_runner_hosts_document(spec, {"lavender": RESERVED_CORES}))
        assert list(narrow_json_to_dict(document)) == ["lavender"]

    def test_a_host_with_no_reserved_figure_is_refused(self) -> None:
        with pytest.raises(KeyError, match="lavender"):
            render_runner_hosts_document(RunnerSpec(hosts=[a_host()]), {})
