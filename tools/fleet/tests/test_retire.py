"""Retiring a settled dispatch's directory (MCPs board task bfca20e6).

:func:`fleet.core.retire.retire_on_node` is asked, per platform, which script
it sends where; the ``sh`` rendering is then executed for real under this
host's bash against a stage root laid out under ``tmp_path``, run twice to
show a retire that already happened is not an error. The PowerShell rendering
is executed by ``tests/pester/rendered-dialect-state.Tests.ps1`` from its
committed copy.
"""

from __future__ import annotations

import pathlib
import subprocess

import pytest

from fleet.contracts.node import NodeConfig, NodePlatform
from fleet.core import _test_hooks, dialect, names, retire
from fleet.core.dialect_linux import PROLOGUE
from tests._host_bash import host_bash
from tests.conftest import DEMO_RUN_ID, FakeRun, retire_replies

#: Each platform's stage root, as the roster declares them.
STAGE_ROOTS = {
    NodePlatform.WINDOWS: "C:/fleet/stage",
    NodePlatform.LINUX: "/home/corvis/fleet/stage",
}


def _node(platform: NodePlatform) -> NodeConfig:
    """A node of the given platform.

    Args:
        platform: Its declared platform.

    Returns:
        Its declaration.
    """
    return NodeConfig(
        host="node-1",
        platform=platform,
        stage_root=STAGE_ROOTS[platform],
        logical_cores=8,
        ram_gb=16.0,
        gpu=None,
        enabled=True,
        test_database=False,
        rust=None,
        cxx=None,
        docker=None,
        budget={
            "reserved_cores": 2,
            "reserved_ram_gb": 4.0,
            "worker_ram_gb": 1.1,
            "max_concurrent_runs": 2,
            "max_disk_gb": 20.0,
        },
    )


def test_the_names_are_one_spelling() -> None:
    assert names.retire_stem("libs-demo-1") == "retire-libs-demo-1"
    assert names.retained_log_path("/s", "libs-demo-1") == "/s/logs/libs-demo-1.log"
    assert names.root_script_stems("libs-demo-1") == (
        "mkdir-libs-demo-1",
        "stop-libs-demo-1",
        "retire-libs-demo-1",
    )


@pytest.mark.parametrize("platform", list(NodePlatform))
def test_it_sends_the_retire_to_the_stage_root_runs_it_and_names_the_kept_transcript(
    platform: NodePlatform,
) -> None:
    node = _node(platform)
    spoken = dialect.for_platform(platform)
    root = STAGE_ROOTS[platform]
    runner = FakeRun(retire_replies())
    _test_hooks.run = runner

    kept = retire.retire_on_node(node, run_id=DEMO_RUN_ID)

    assert kept == f"{root}/logs/{DEMO_RUN_ID}.log"
    script_path = spoken.script_path(root, f"retire-{DEMO_RUN_ID}")
    assert runner.stdin[0] == spoken.retire_script(
        target=f"{root}/{DEMO_RUN_ID}",
        retained=kept,
        scripts=(
            spoken.script_path(root, f"mkdir-{DEMO_RUN_ID}"),
            spoken.script_path(root, f"stop-{DEMO_RUN_ID}"),
            script_path,
        ),
    ).encode("utf-8")
    assert script_path in " ".join(runner.calls[0])
    assert runner.calls[1][-1].endswith(script_path)


class TestTheShScriptRunsForReal:
    """The ``sh`` retire against a real stage root, under this host's bash."""

    def lay_out(self, root: pathlib.Path, *, transcript: bool) -> pathlib.Path:
        """A finished run's directory and its root scripts, as a node leaves them.

        Args:
            root: The stage root.
            transcript: Whether the build wrote its transcript.

        Returns:
            The run's directory.
        """
        target = root / DEMO_RUN_ID
        objects = target / ".git" / "objects"
        objects.mkdir(parents=True)
        loose = objects / "ab"
        loose.write_text("loose object", encoding="utf-8")
        loose.chmod(0o444)
        (target / names.RESULT_NAME).write_text("0\n", encoding="utf-8")
        if transcript:
            (target / f"{names.RESULT_NAME}.log").write_text("887 passed\n", encoding="utf-8")
        for stem in names.root_script_stems(DEMO_RUN_ID):
            (root / f"{stem}.sh").write_text(PROLOGUE, encoding="utf-8")
        (root / "cache").mkdir()
        (root / "libs-other-1757000001").mkdir()
        return target

    def retire(self, root: pathlib.Path) -> None:
        """Render the retire for ``root`` and run it by its own path, as a node does.

        Args:
            root: The stage root.
        """
        stage_root = root.as_posix()
        spoken = dialect.for_platform(NodePlatform.LINUX)
        script = pathlib.Path(root / f"{names.retire_stem(DEMO_RUN_ID)}.sh")
        script.write_text(
            spoken.retire_script(
                target=names.dispatch_directory(stage_root, DEMO_RUN_ID),
                retained=names.retained_log_path(stage_root, DEMO_RUN_ID),
                scripts=tuple(
                    spoken.script_path(stage_root, stem)
                    for stem in names.root_script_stems(DEMO_RUN_ID)
                ),
            ),
            encoding="utf-8",
        )
        ran = subprocess.run(
            [host_bash(), script.as_posix()],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        assert ran.returncode == 0, ran.stderr

    def test_it_keeps_the_transcript_and_removes_the_tree_and_the_run_s_root_scripts(
        self, tmp_path: pathlib.Path
    ) -> None:
        self.lay_out(tmp_path, transcript=True)

        self.retire(tmp_path)

        assert (tmp_path / "logs" / f"{DEMO_RUN_ID}.log").read_text(encoding="utf-8") == (
            "887 passed\n"
        )
        # The cache and another run's directory are not this run's to remove.
        assert sorted(entry.name for entry in tmp_path.iterdir()) == [
            "cache",
            "libs-other-1757000001",
            "logs",
        ]

    def test_a_run_that_wrote_no_transcript_is_retired_all_the_same_and_twice(
        self, tmp_path: pathlib.Path
    ) -> None:
        """A run cancelled before its build began wrote no transcript, and a
        retire that failed part way is run again by the next tick."""
        self.lay_out(tmp_path, transcript=False)

        self.retire(tmp_path)
        self.retire(tmp_path)

        assert list((tmp_path / "logs").iterdir()) == []
        assert not (tmp_path / DEMO_RUN_ID).exists()
