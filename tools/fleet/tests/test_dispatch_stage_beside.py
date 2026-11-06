"""A launch stages its tree, its registry and its companions at once (MCPs board task 5280a25d).

Timed from lacrymal against colossus on 2026-11-06, the registry's seven ssh
connections took 0.88 s, every one after the 3.05 s the node spent unpacking
the commit, and all of it between a gate job's claim and its first phase.
Each case binds the real stage pool and holds the tree's stage until the
node has run the registry's extract: staged one after the other, the tree
would be waiting for a registry that comes only after it, and the case
fails at that wait rather than hanging the suite. Only the node is faked,
answering each command by what it is rather than by its place in a script.
"""

from __future__ import annotations

import pathlib
import threading
from collections.abc import Sequence

import pytest
from platform_core.errors import AppError, FleetErrorCode

from fleet.contracts.feed import FeedKind
from fleet.contracts.lease import Lease
from fleet.core import _test_hooks, dialect, dispatch, names, records, staging
from fleet.core.recipe import Recipe
from tests._contract_fixtures import node
from tests._thread_fakes import await_event
from tests.conftest import ok

#: The run every case launches, on lavender's stage root.
RUN_ID = "MCPs-execution-linux-lavender-1793970467"
STAGE_ROOT = node()["stage_root"]
TARGET = names.dispatch_directory(STAGE_ROOT, RUN_ID)

#: Where the registry's archive lands and is digested and extracted.
CARRIED = f"{names.staging_directory(TARGET)}/{names.REGISTRY_ARCHIVE_DIRECTORY}"


class _Node:
    """Lavender, answering each command a launch sends by what the command is.

    Satisfies :class:`~fleet.core._test_hooks.RunProtocol`: the registry's
    digest with the archive's digest, its extract by marking the registry
    landed, and every other command with nothing.
    """

    def __init__(self, registry_digest: str) -> None:
        """Bind the digest the registry's archive has.

        Args:
            registry_digest: What the node reports having received.
        """
        self.landed = threading.Event()
        self.calls: list[tuple[str, ...]] = []
        self._calls = threading.Lock()
        self._digest = registry_digest

    def __call__(
        self,
        argv: Sequence[str],
        *,
        timeout_seconds: int,
        stdin_bytes: bytes | None = None,
        unset_env: Sequence[str] = (),
        set_env: Sequence[tuple[str, str]] = (),
    ) -> _test_hooks.CommandResult:
        """Answer one command.

        Args:
            argv: The command.
            timeout_seconds: Its deadline.
            stdin_bytes: Its standard input, or None.
            unset_env: The variables withheld from it.
            set_env: The variables set for it.

        Returns:
            The digest for the registry's digest script, else nothing.
        """
        with self._calls:
            self.calls.append(tuple(argv))
        if argv[-1] == dialect.script_path(CARRIED, names.DIGEST_STEM):
            return ok(self._digest)
        if argv[-1] == dialect.script_path(CARRIED, names.EXTRACT_STEM):
            self.landed.set()
        return ok("")


def _registry(tmp_path: pathlib.Path) -> pathlib.Path:
    """A registry of one file, as the runner loaded it.

    Args:
        tmp_path: The case's directory.

    Returns:
        The registry directory.
    """
    registry = tmp_path / "registry"
    registry.mkdir(parents=True)
    (registry / "settings.json").write_text("{}\n", encoding="utf-8")
    return registry


def _tree_once_the_registry_landed(lavender: _Node, *, fails: bool) -> dispatch.TreeStager:
    """A tree's stage that ends only once the registry beside it has landed.

    Args:
        lavender: The node, which marks the registry landed.
        fails: Whether the tree's stage then fails, as a refused bundle does.

    Returns:
        The stager.
    """

    def stage(host: str, *, stage_root: str, run_id: str) -> dispatch.StagedTree:
        await_event(lavender.landed, what="the registry to land while the tree was staged")
        if fails:
            raise AppError(FleetErrorCode.EXPORT_FAILED, f"no bundle of {run_id} for {host}")
        return dispatch.StagedTree(
            target=names.dispatch_directory(stage_root, run_id), detail="the tree, from its mirror"
        )

    return stage


def _launch(tmp_path: pathlib.Path, stage_tree: dispatch.TreeStager) -> None:
    """Launch :data:`RUN_ID` on lavender under a held lease.

    Args:
        tmp_path: The case's directory, holding its records.
        stage_tree: How the tree is staged.
    """
    dispatch.launch(
        tmp_path / "ledger.jsonl",
        tmp_path / "feed.jsonl",
        lease=Lease(
            node="lavender",
            project="MCPs/execution-linux",
            run_id=RUN_ID,
            agent="opus-fleet-w2-1106",
            session_id="e89979e1-0813-40fb-8186-b58081e3a91b",
            acquired_unix=1_793_970_467,
            expires_unix=1_793_974_067,
            resources=(),
        ),
        node=node(),
        workers=4,
        stage_tree=stage_tree,
        companions=(),
        recipe=Recipe(
            path="execution", install=(), isolated_docker=False, loopback_only=False, artifact=None
        ),
        registry=_registry(tmp_path),
        registry_commit="7973682" + "0" * 33,
        scratch=tmp_path / "archives",
    )


def test_the_registry_lands_while_the_tree_is_staged_and_the_launch_follows_both(
    tmp_path: pathlib.Path,
) -> None:
    lavender = _Node(staging.digest(staging.registry_payload(_registry(tmp_path / "loaded"))))
    _test_hooks.run = lavender
    _test_hooks.executor = _test_hooks._default_executor

    _launch(tmp_path, _tree_once_the_registry_landed(lavender, fails=False))

    scripts = [call[-1] for call in lavender.calls]
    extracted = scripts.index(dialect.script_path(CARRIED, names.EXTRACT_STEM))
    built = scripts.index(dialect.write_command(dialect.script_path(TARGET, names.BUILD_STEM)))
    assert extracted < built
    feed = records.read_feed(tmp_path / "feed.jsonl")
    assert [(row["kind"], row["detail"]) for row in feed][-2:] == [
        (FeedKind.STAGED, "the tree, from its mirror"),
        (FeedKind.STARTED, "make check at 4 worker(s)"),
    ]
    assert [row["run_id"] for row in records.read_ledger(tmp_path / "ledger.jsonl")] == [RUN_ID]
    assert not (tmp_path / "archives" / f"{RUN_ID}-{names.REGISTRY_DIRECTORY}.tar.gz").exists()


def test_a_tree_that_fails_is_raised_once_the_registry_beside_it_has_ended(
    tmp_path: pathlib.Path,
) -> None:
    lavender = _Node(staging.digest(staging.registry_payload(_registry(tmp_path / "loaded"))))
    _test_hooks.run = lavender
    _test_hooks.executor = _test_hooks._default_executor

    with pytest.raises(AppError) as raised:
        _launch(tmp_path, _tree_once_the_registry_landed(lavender, fails=True))

    assert raised.value.code is FleetErrorCode.EXPORT_FAILED
    assert raised.value.message == f"no bundle of {RUN_ID} for lavender"
    assert [call[-1] for call in lavender.calls][-1] == dialect.script_path(
        CARRIED, names.EXTRACT_STEM
    )
    assert records.read_feed(tmp_path / "feed.jsonl") == ()
    assert records.read_ledger(tmp_path / "ledger.jsonl") == ()
