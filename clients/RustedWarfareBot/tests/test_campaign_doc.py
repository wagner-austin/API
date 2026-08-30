"""Writing the campaign document a batch becomes on the cluster.

The document carries only what a batch can say for itself; ``hpc3`` merges it
with the project's declared defaults from its own workspace. These check that
what is emitted is exactly that set, that hpc3's own decoder accepts every
member of it, and that the one cluster fact the document cannot avoid carrying
-- the root every path hangs off -- is READ from the workspace rather than
typed here, where a second copy of it would eventually be the stale one.
"""

from __future__ import annotations

import runpy

import pytest
from hpc3.contracts.run import SWEEP_IDENTITY_FIELDS
from hpc3.contracts.sweep import decode_sweep_member
from platform_core.errors import AppError
from platform_core.json_utils import (
    JSONValue,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    require_list,
    require_str,
)
from scripts.campaign_doc import (
    EXIT_OK,
    PROJECT,
    campaign_document,
    experiment_of,
    interpreter_of,
    main,
)

from rw_bot.harness.match import MatchConfig
from rw_bot.harness.sweep import SweepJob
from tests.harness_fakes import FakeHost

_CONFIG = "runs/hpc3.json"
_JOBS = "sweeps/demo.txt"
_OUT = "sweeps/demo-campaign.json"
_BATCH = "demo"
_LOCKSTEP = 75
_ROOT = "/pub/wagnera3"
_ENV_PATH = "/opt/env"

#: The match every member of this batch plays. The map every sweep in this
#: project has ever used, so a cluster batch measures the same simulation a
#: workstation batch does.
_MAP = "maps/skirmish/[p2]duel_lake.tmx"
_DIFFICULTY = -2
_MATCH = MatchConfig(map_path=_MAP, opponents=1, difficulty=_DIFFICULTY)
_LINES = (
    "attack|1|doctrines/a.doctrine|1500",
    "attack|2|doctrines/a.doctrine|1500",
    "defend|1|doctrines/b.doctrine|1500",
)


def _workspace(
    *, root: str = _ROOT, env_path: str = _ENV_PATH, project: str = PROJECT
) -> dict[str, JSONValue]:
    """Build a workspace shaped exactly like the committed one.

    Decoded by hpc3's real decoder rather than stubbed: the point of reading
    the root and the environment from here is that the two sides cannot
    disagree, and a fake that skipped the decode would not be testing that.

    Args:
        root: The cluster root it declares.
        env_path: The environment the project declares.
        project: The project it declares, so a workspace that does not carry
            this batch's project can be built.

    Returns:
        The document, ready to serialise.
    """
    return {
        "cluster": "hpc3",
        "host": "hpc3",
        "root": root,
        "ledger": "ledger.jsonl",
        "quiet_seconds": 1800,
        "projects": {
            project: {
                "partition": "free",
                "gpu": None,
                "cpus": 1,
                "mem_gb": 2,
                "minutes": 45,
                "requeue": False,
                "checkpoint_steps": 0,
                "env_path": env_path,
                "pinned_packages": {},
                "deterministic": False,
                "budget": {
                    "self_imposed_gpu_hours": 0.0,
                    "max_service_units": 0.0,
                    "charge_account": "",
                },
                "repo": "../../../clients/RustedWarfareBot",
            }
        },
    }


def _planted(workspace: dict[str, JSONValue] | None = None) -> FakeHost:
    """Build a host holding the workspace and the batch's job file.

    Args:
        workspace: The workspace document to plant. ``None`` plants the
            committed shape unchanged.

    Returns:
        The host.
    """
    host = FakeHost()
    document = _workspace() if workspace is None else workspace
    host.files[_CONFIG] = tuple(dump_json_str(document).splitlines())
    host.files[_JOBS] = _LINES
    return host


def _argv(out: str = _OUT) -> list[str]:
    """Build the command line the emitter is run with.

    Args:
        out: Where the document goes.

    Returns:
        The arguments after the program name.
    """
    return [
        "--config",
        _CONFIG,
        "--jobs",
        _JOBS,
        "--batch",
        _BATCH,
        "--map",
        _MAP,
        "--difficulty",
        str(_DIFFICULTY),
        "--out",
        out,
    ]


def _document(jobs: list[SweepJob] | None = None) -> dict[str, JSONValue]:
    """Build the document for the sample batch.

    Args:
        jobs: The matches it describes. ``None`` uses the sample batch.

    Returns:
        The document.
    """
    return campaign_document(
        _ROOT, _ENV_PATH, _JOBS, _BATCH, _jobs() if jobs is None else jobs, _LOCKSTEP, _MATCH
    )


def _jobs() -> list[SweepJob]:
    """Build the batch the lines above describe.

    Returns:
        The jobs.
    """
    return [
        SweepJob(label="attack", seed=1, doctrine="doctrines/a.doctrine", samples=1500),
        SweepJob(label="attack", seed=2, doctrine="doctrines/a.doctrine", samples=1500),
        SweepJob(label="defend", seed=1, doctrine="doctrines/b.doctrine", samples=1500),
    ]


class TestWhatTheCampaignIs:
    def test_it_records_the_facts_that_distinguish_one_batch(self) -> None:
        """A job id and a name say which row in squeue a member was and
        nothing about which experiment it belonged to."""
        assert experiment_of(_BATCH, _jobs(), _LOCKSTEP, _MATCH) == {
            "batch": "demo",
            "matches": "3",
            "arms": "attack,defend",
            "lockstep": "75",
            "map": _MAP,
            "difficulty": "-2",
        }

    def test_the_map_is_recorded_because_it_decides_the_opponent_count(self) -> None:
        """Two batches on different maps are not comparable, and the ledger
        is where a reader finds out which was which."""
        assert experiment_of(_BATCH, _jobs(), _LOCKSTEP, _MATCH)["map"] == _MAP

    def test_every_value_is_a_string_because_that_is_what_the_ledger_stores(self) -> None:
        values = experiment_of(_BATCH, _jobs(), _LOCKSTEP, _MATCH).values()
        assert [type(value) for value in values] == [str] * len(values)


class TestTheDocument:
    def test_it_carries_only_a_sweeps_own_identity_fields(self) -> None:
        """The partition, the wall clock and the image come from hpc3's
        workspace. Writing them here would be this package guessing at
        another's configuration."""
        document = _document()
        assert set(document) <= set(SWEEP_IDENTITY_FIELDS)

    def test_it_names_the_project_and_the_batch(self) -> None:
        document = _document()
        assert document["project"] == PROJECT
        assert document["name"] == _BATCH

    def test_one_member_per_match(self) -> None:
        document = _document()
        assert len(require_list(document, "members")) == 3

    def test_every_member_declares_an_absolute_artifact(self) -> None:
        """A relative one is refused by nothing, tests ABSENT against ``$HOME``
        over SSH, and makes the campaign resubmit the batch on every pass."""
        document = _document()
        for member in require_list(document, "members"):
            assert require_str(narrow_json_to_dict(member), "artifact").startswith(f"{_ROOT}/")

    def test_every_member_survives_hpc3s_own_decoder(self) -> None:
        """The whole point of depending on hpc3 rather than describing its
        format again: a document only this side had checked would be refused
        at submission, after the game tree had been staged."""
        document = _document()
        for member in require_list(document, "members"):
            decode_sweep_member(member)

    def test_an_empty_batch_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one member"):
            _document(jobs=[])


class TestWritingIt:
    def test_the_document_is_written_where_asked(self) -> None:
        with _planted() as host:
            assert main(_argv()) == EXIT_OK
            assert _OUT in host.files

    def test_what_lands_on_disk_is_loadable_json(self) -> None:
        with _planted() as host:
            main(_argv())
            written = load_json_str("\n".join(host.files[_OUT]))
        assert len(require_list(narrow_json_to_dict(written), "members")) == 3

    def test_the_interpreter_comes_from_the_projects_declared_environment(self) -> None:
        """A second copy of it here would point at where the environment used
        to be, on the day it moved."""
        assert interpreter_of(_ENV_PATH) == "/opt/env/bin/python"
        with _planted(_workspace(env_path="/opt/rw")) as host:
            main(_argv())
            written = "\n".join(host.files[_OUT])
        assert "/opt/rw/bin/python -m " in written
        assert "/opt/env/bin/python" not in written

    def test_the_root_comes_from_the_workspace_rather_than_from_here(self) -> None:
        """The one cluster fact the document must carry. Typed in two places
        it would be wrong the day the workspace moved, so it is read -- and
        moving it in the workspace moves every path in the document."""
        with _planted(_workspace(root="/dfs6b/pub/elsewhere")) as host:
            main(_argv())
            written = "\n".join(host.files[_OUT])
        assert "/dfs6b/pub/elsewhere/rusted/runs/sweeps/demo/" in written
        assert _ROOT not in written

    def test_a_workspace_that_does_not_declare_this_project_is_refused(self) -> None:
        """Refused here rather than at submission: a document naming a project
        the workspace has never heard of cannot be resolved into a job at
        all, and finding that out after staging is the expensive order."""
        with _planted(_workspace(project="cleargbm")), pytest.raises(AppError):
            main(_argv())

    def test_it_reports_how_many_members_it_wrote(self) -> None:
        with _planted() as host:
            main(_argv())
            assert any("3 member(s)" in line for line in host.printed)

    def test_a_missing_flag_is_refused(self) -> None:
        with _planted(), pytest.raises(ValueError, match="--out is required"):
            main(_argv()[:-2])

    def test_it_reads_the_process_arguments_when_given_none(self) -> None:
        with _planted() as host:
            host.argv = _argv()
            assert main(None) == EXIT_OK

    def test_the_module_guard_runs_main(self) -> None:
        with _planted() as host:
            host.argv = _argv()
            with pytest.raises(SystemExit) as caught:
                runpy.run_module("scripts.campaign_doc", run_name="__main__")
            assert caught.value.code == EXIT_OK
