"""Generating one arm of a continuation sweep from the command line.

Two things here are the design rather than the plumbing.

**Determinism is pinned before the weights load**, because loading them
creates the cuBLAS handle and ``CUBLAS_WORKSPACE_CONFIG`` is read exactly
then. A test that only checked the record's contents would pass on a version
that pinned last and recorded a posture the run did not have.

**Resume is at batch granularity.** Dropping finished items out of a batch
would repack the survivors with different neighbours, and padding is what a
neighbour changes -- so a resumed arm would not be numerically the arm the
other arm is compared against.
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.continuation_task import generated_path, manifest_path

from model_trainer.cli import continuations
from tests._continuations_support import (
    LONG_SOURCE,
    Recorder,
    command_line,
    install,
    restore_hooks,
    spec_file,
)

__all__ = ["restore_hooks"]


@pytest.mark.usefixtures("restore_hooks")
def test_determinism_is_pinned_before_the_weights_load(tmp_path: pathlib.Path) -> None:
    # The whole point. Loading weights creates the cuBLAS handle, and the
    # workspace variable is read exactly then -- a pin afterwards is accepted
    # in silence and the record would claim a posture the run did not have.
    recorder = Recorder()
    install(recorder)

    assert continuations.main(command_line(tmp_path, spec_file(tmp_path))) == 0
    assert recorder.order[0] == "pin"
    assert recorder.order[1] == "load:candidate"


@pytest.mark.usefixtures("restore_hooks")
def test_both_determinism_controls_are_applied(tmp_path: pathlib.Path) -> None:
    # Matching the scoring path exactly. A generation posture that differed
    # from the scoring one would make the two disagree in the last bits for a
    # reason nobody would look for.
    recorder = Recorder()
    install(recorder)

    _ = continuations.main(command_line(tmp_path, spec_file(tmp_path)))

    assert recorder.postures == [(True, True)]


@pytest.mark.usefixtures("restore_hooks")
def test_the_base_arm_loads_the_base_of_the_same_artifact(tmp_path: pathlib.Path) -> None:
    # Not a fresh model from the hub. The control for a QLoRA adapter is that
    # adapter's own base under that adapter's own quantization.
    recorder = Recorder()
    install(recorder)

    _ = continuations.main(command_line(tmp_path, spec_file(tmp_path, arm="base")))

    assert "load:base" in recorder.order


@pytest.mark.usefixtures("restore_hooks")
def test_a_file_is_written_where_the_scorer_looks_for_it(tmp_path: pathlib.Path) -> None:
    recorder = Recorder()
    install(recorder)

    _ = continuations.main(command_line(tmp_path, spec_file(tmp_path, paths=("src/a.py",))))

    out_dir = tmp_path / "generated" / "candidate"
    assert generated_path(out_dir, "src/a.py").is_file()


@pytest.mark.usefixtures("restore_hooks")
def test_the_written_file_is_the_prompt_plus_the_continuation(tmp_path: pathlib.Path) -> None:
    # Scoring the continuation alone would fail every checker on imports the
    # prompt already supplied.
    recorder = Recorder()
    install(recorder)

    _ = continuations.main(command_line(tmp_path, spec_file(tmp_path, paths=("src/a.py",))))

    out_dir = tmp_path / "generated" / "candidate"
    written = generated_path(out_dir, "src/a.py").read_text(encoding="utf-8")
    assert written == LONG_SOURCE[: len("".join(f"line{i}\n" for i in range(20)))] + (
        "# written for src/a.py\n"
    )


@pytest.mark.usefixtures("restore_hooks")
def test_no_partial_file_survives_a_completed_run(tmp_path: pathlib.Path) -> None:
    # The resume check reads existence as completion, so a leftover staging
    # file would be trusted by every later run.
    recorder = Recorder()
    install(recorder)

    _ = continuations.main(command_line(tmp_path, spec_file(tmp_path)))

    out_dir = tmp_path / "generated" / "candidate"
    leftovers = sorted(out_dir.rglob("*" + continuations.PARTIAL_SUFFIX))
    assert leftovers == []


@pytest.mark.usefixtures("restore_hooks")
def test_prompts_are_grouped_into_batches_of_the_declared_size(
    tmp_path: pathlib.Path,
) -> None:
    recorder = Recorder()
    install(recorder)

    _ = continuations.main(
        command_line(tmp_path, spec_file(tmp_path, paths=("src/a.py", "src/b.py", "src/c.py")))
    )

    assert [len(batch) for batch in recorder.batches] == [2, 1]


@pytest.mark.usefixtures("restore_hooks")
def test_the_manifest_records_one_row_per_item(tmp_path: pathlib.Path) -> None:
    recorder = Recorder(finished_ids=("src/a.py",))
    install(recorder)

    _ = continuations.main(
        command_line(tmp_path, spec_file(tmp_path, paths=("src/a.py", "src/b.py")))
    )

    out_dir = tmp_path / "generated" / "candidate"
    rows = manifest_path(out_dir).read_text(encoding="utf-8").splitlines()
    assert len(rows) == 2


@pytest.mark.usefixtures("restore_hooks")
def test_the_manifest_says_which_completions_ended_on_their_own(
    tmp_path: pathlib.Path,
) -> None:
    # Recorded rather than inferred from the text later: a file ending on a
    # plausible line and one ending because the budget did are
    # indistinguishable once the tokens are gone.
    recorder = Recorder(finished_ids=("src/a.py",))
    install(recorder)

    _ = continuations.main(
        command_line(tmp_path, spec_file(tmp_path, paths=("src/a.py", "src/b.py")))
    )

    out_dir = tmp_path / "generated" / "candidate"
    assert continuations.read_manifest(out_dir) == {"src/a.py": True, "src/b.py": False}


@pytest.mark.usefixtures("restore_hooks")
def test_the_manifest_is_a_sibling_of_the_generated_directory(
    tmp_path: pathlib.Path,
) -> None:
    # Inside it, every reader walking the tree would have to know to skip it.
    recorder = Recorder()
    install(recorder)

    _ = continuations.main(command_line(tmp_path, spec_file(tmp_path, paths=("src/a.py",))))

    out_dir = tmp_path / "generated" / "candidate"
    assert manifest_path(out_dir).parent == out_dir.parent


@pytest.mark.usefixtures("restore_hooks")
def test_a_second_run_regenerates_nothing(tmp_path: pathlib.Path) -> None:
    recorder = Recorder()
    install(recorder)
    tokens = command_line(tmp_path, spec_file(tmp_path))

    _ = continuations.main(tokens)
    first = len(recorder.batches)
    _ = continuations.main(tokens)

    assert len(recorder.batches) == first


@pytest.mark.usefixtures("restore_hooks")
def test_a_resumed_run_redoes_a_partial_batch_whole(tmp_path: pathlib.Path) -> None:
    # Dropping the finished item out of the batch would repack the survivors
    # with different neighbours, and padding is what a neighbour changes.
    recorder = Recorder()
    install(recorder)
    tokens = command_line(tmp_path, spec_file(tmp_path, paths=("src/a.py", "src/b.py")))

    _ = continuations.main(tokens)
    out_dir = tmp_path / "generated" / "candidate"
    generated_path(out_dir, "src/b.py").unlink()
    recorder.batches.clear()

    _ = continuations.main(tokens)

    assert recorder.batches == [["src/a.py", "src/b.py"]]
