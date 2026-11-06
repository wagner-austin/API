"""What a node's mirror says it holds before anything is sent to it.

Moved out of ``tests/test_commit_stage.py`` with the probe itself (MCPs board
task a48b5236). The node's script runs under real ``/bin/sh`` against real
repositories, as a node runs it; the base branch tip is answered on its own
line, because ``git bundle create`` leaves out any ref its exclusions reach
and the node must fetch exactly the refs a bundle carries.
"""

from __future__ import annotations

import pathlib

from fleet.core import node_mirror_probe
from tests.test_commit_stage import SHA, _commit, _git, _node_mirror_at, _sh


def _asked(
    tmp_path: pathlib.Path, node: pathlib.Path, sha: str, base_tip: str | None
) -> node_mirror_probe.NodeMirror:
    """Run the prepare script against a node mirror and read its answer.

    Args:
        tmp_path: The test's directory, where the export's directories go.
        node: The node's mirror.
        sha: The commit asked about.
        base_tip: The base branch tip asked about, or None.

    Returns:
        What the script said, read.
    """
    script = node_mirror_probe.prepare_script(
        target=str(tmp_path / "t"),
        staged=str(tmp_path / "s"),
        mirror=str(node),
        sha=sha,
        base_tip=base_tip,
    )
    return node_mirror_probe.read_node_mirror(_sh(script))


class TestTheNodeMirrorSaysWhatItHolds:
    def test_a_node_without_the_mirror_gets_its_directories_and_says_nothing(
        self, tmp_path: pathlib.Path
    ) -> None:
        target = tmp_path / "run" / "run"
        said = _sh(
            node_mirror_probe.prepare_script(
                target=str(target),
                staged=f"{target}.stage",
                mirror=str(tmp_path / "mirrors" / "repo.git"),
                sha=SHA,
                base_tip=SHA,
            )
        )

        assert said == ""
        assert target.is_dir()
        assert pathlib.Path(f"{target}.stage").is_dir()
        assert node_mirror_probe.read_node_mirror(said) == node_mirror_probe.NodeMirror(
            tip=None, held=False, base_held=False
        )

    def test_a_mirror_names_its_tip_and_whether_it_holds_the_commit(
        self, tmp_path: pathlib.Path
    ) -> None:
        hub = tmp_path / "hub"
        first = _commit(hub, {"a.txt": "one\n"})
        node = tmp_path / "node" / "repo.git"
        node.parent.mkdir()
        _node_mirror_at(hub, node, "run-1", None)
        second = _commit(hub, {"a.txt": "two\n"})

        assert _asked(tmp_path, node, first, None) == node_mirror_probe.NodeMirror(
            tip=first, held=True, base_held=False
        )
        assert _asked(tmp_path, node, second, None) == node_mirror_probe.NodeMirror(
            tip=first, held=False, base_held=False
        )

    def test_a_base_tip_is_answered_apart_from_the_commit(self, tmp_path: pathlib.Path) -> None:
        """The tip the node reaches and the commit it lacks are two answers,
        so a bundle carries the commit alone; and the other way round."""
        hub = tmp_path / "hub"
        first = _commit(hub, {"a.txt": "one\n"})
        node = tmp_path / "node" / "repo.git"
        node.parent.mkdir()
        _node_mirror_at(hub, node, "run-1", None)
        second = _commit(hub, {"a.txt": "two\n"})

        assert _asked(tmp_path, node, second, first) == node_mirror_probe.NodeMirror(
            tip=first, held=False, base_held=True
        )
        assert _asked(tmp_path, node, first, second) == node_mirror_probe.NodeMirror(
            tip=first, held=True, base_held=False
        )
        # A tip beyond the staged commit is held once the base ref reaches it.
        _git(node, "fetch", "--quiet", str(hub), f"+HEAD:{node_mirror_probe.NODE_BASE_REF}")
        assert _asked(tmp_path, node, first, second) == node_mirror_probe.NodeMirror(
            tip=first, held=True, base_held=True
        )

    def test_words_that_are_no_commit_are_no_tip(self) -> None:
        said = f"fatal: something\n{node_mirror_probe.HELD_ANSWER}\n"

        assert node_mirror_probe.read_node_mirror(said) == node_mirror_probe.NodeMirror(
            tip=None, held=True, base_held=False
        )

    def test_the_base_answer_is_its_own_word(self) -> None:
        said = f"{SHA}\n{node_mirror_probe.BASE_HELD_ANSWER}\n"

        assert node_mirror_probe.read_node_mirror(said) == node_mirror_probe.NodeMirror(
            tip=SHA, held=False, base_held=True
        )
