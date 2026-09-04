"""What a finished sweep records about itself, and the seams that produced it.

**The record describes the arm, not the invocation.** A resumed run holds only
the batches it redid, so the finish counts and the payload digest are read
back off the manifest rather than remembered.

The production defaults at the end are the two seams every test in the sibling
file substitutes. Without them the lines that decide WHICH WEIGHTS ANSWER
would be the only ones in the command with no coverage.
"""

from __future__ import annotations

import pathlib
import runpy
import sys
from typing import ClassVar

import pytest
from platform_core.continuation_task import EvalPrompt, generated_path, manifest_path
from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
)
from platform_core.run_record import decode_run_record

from model_trainer.cli import _test_hooks as cli_hooks
from model_trainer.cli import continuations
from model_trainer.core.contracts.continuation_sweep import Completion
from tests._continuations_support import (
    Recorder,
    command_line,
    fake_full_model_loader,
    fake_model_loader,
    fake_tokenizer_loader,
    generating_model,
    install,
    read_record,
    restore_hooks,
    spec_file,
)

__all__ = ["restore_hooks"]


@pytest.mark.usefixtures("restore_hooks")
def test_a_resumed_runs_record_still_describes_the_whole_arm(
    tmp_path: pathlib.Path,
) -> None:
    # The record has to describe what is on disk, not what the last process
    # happened to redo.
    recorder = Recorder(finished_ids=("src/a.py", "src/b.py", "src/c.py"))
    install(recorder)
    tokens = command_line(tmp_path, spec_file(tmp_path))

    _ = continuations.main(tokens)
    out_dir = tmp_path / "generated" / "candidate"
    generated_path(out_dir, "src/c.py").unlink()

    _ = continuations.main(tokens)

    observations = read_record(tmp_path)
    assert observations["items_in_scope"] == 3.0
    assert observations["items_reused"] == 2.0
    assert observations["items_generated"] == 1.0
    assert observations["completions_finished"] == 3.0


@pytest.mark.usefixtures("restore_hooks")
def test_the_record_carries_the_finish_rate_beside_its_counts(
    tmp_path: pathlib.Path,
) -> None:
    # Three from three and thirty from thirty are both 1.0, and only one is
    # evidence.
    recorder = Recorder(finished_ids=("src/a.py",))
    install(recorder)

    _ = continuations.main(
        command_line(tmp_path, spec_file(tmp_path, paths=("src/a.py", "src/b.py")))
    )

    observations = read_record(tmp_path)
    assert observations["completions_finished"] == 1.0
    assert observations["items_in_scope"] == 2.0
    assert observations["finished_rate"] == 0.5


@pytest.mark.usefixtures("restore_hooks")
def test_the_records_payload_digest_covers_which_items_finished(
    tmp_path: pathlib.Path,
) -> None:
    # Two arms agreeing on a finish rate can still disagree on which items ran
    # out of budget, and the rate cannot tell you.
    recorder = Recorder(finished_ids=("src/a.py",))
    install(recorder)
    _ = continuations.main(
        command_line(tmp_path, spec_file(tmp_path, paths=("src/a.py", "src/b.py")))
    )
    first = decode_run_record(
        load_json_str((tmp_path / "out" / "record.json").read_text(encoding="utf-8"))
    )["payload_digest"]

    other = tmp_path / "other"
    other.mkdir()
    recorder_b = Recorder(finished_ids=("src/b.py",))
    install(recorder_b)
    _ = continuations.main(command_line(other, spec_file(other, paths=("src/a.py", "src/b.py"))))
    second = decode_run_record(
        load_json_str((other / "out" / "record.json").read_text(encoding="utf-8"))
    )["payload_digest"]

    assert first != second


@pytest.mark.usefixtures("restore_hooks")
def test_the_record_is_named_for_the_experiment_not_the_arm(
    tmp_path: pathlib.Path,
) -> None:
    # The experiment is what makes two sweeps comparable; the arm is the label.
    recorder = Recorder()
    install(recorder)

    _ = continuations.main(command_line(tmp_path, spec_file(tmp_path)))

    record = decode_run_record(
        load_json_str((tmp_path / "out" / "record.json").read_text(encoding="utf-8"))
    )
    assert record["experiment"] == continuations.CONTINUATION_EXPERIMENT
    assert record["label"] == "qwen-qlora-v1-candidate"


@pytest.mark.usefixtures("restore_hooks")
def test_the_package_axis_names_the_libraries_the_contrast_depends_on(
    tmp_path: pathlib.Path,
) -> None:
    # Read from the ARTIFACT's metadata, so both arms name the same set. A
    # set read off the loaded model would omit peft on the base side, and the
    # two records would then differ on an axis every single sweep.
    recorder = Recorder()
    install(recorder)

    _ = continuations.main(command_line(tmp_path, spec_file(tmp_path, arm="base")))

    record = decode_run_record(
        load_json_str((tmp_path / "out" / "record.json").read_text(encoding="utf-8"))
    )
    named = [package["name"] for package in record["fingerprint"]["packages"]]
    assert sorted(named) == sorted(["numpy", "torch", "transformers", "peft", "bitsandbytes"])


@pytest.mark.usefixtures("restore_hooks")
def test_an_unquantized_artifact_does_not_name_bitsandbytes(
    tmp_path: pathlib.Path,
) -> None:
    # A fingerprint over libraries the run never used differs between two runs
    # over a bump that cannot reach the arithmetic.
    recorder = Recorder()
    install(recorder)

    _ = continuations.main(command_line(tmp_path, spec_file(tmp_path, quantized=False)))

    record = decode_run_record(
        load_json_str((tmp_path / "out" / "record.json").read_text(encoding="utf-8"))
    )
    named = [package["name"] for package in record["fingerprint"]["packages"]]
    assert "bitsandbytes" not in named


@pytest.mark.usefixtures("restore_hooks")
def test_an_arm_with_nothing_in_scope_is_refused(tmp_path: pathlib.Path) -> None:
    # Generating nothing and reporting success is indistinguishable, in the
    # outcome file, from a crashed generation.
    recorder = Recorder()
    install(recorder)

    with pytest.raises(ValueError, match="nothing would be generated"):
        _ = continuations.main(command_line(tmp_path, spec_file(tmp_path, max_new_tokens=1)))


@pytest.mark.usefixtures("restore_hooks")
def test_a_directory_and_manifest_that_disagree_are_refused(
    tmp_path: pathlib.Path,
) -> None:
    # A file generated without being recorded would make the digest describe a
    # different set of items than the directory holds.
    recorder = Recorder()
    install(recorder)
    tokens = command_line(tmp_path, spec_file(tmp_path))
    _ = continuations.main(tokens)
    out_dir = tmp_path / "generated" / "candidate"
    manifest_path(out_dir).unlink()

    with pytest.raises(JSONTypeError, match="no manifest row"):
        _ = continuations.main(tokens)


def test_reading_a_manifest_that_does_not_exist_yet_finds_nothing(
    tmp_path: pathlib.Path,
) -> None:
    # The first batch of the first run reads it before anything has written it.
    assert continuations.read_manifest(tmp_path / "generated" / "candidate") == {}


def test_a_blank_manifest_line_is_skipped(tmp_path: pathlib.Path) -> None:
    # Framing is a property of the file, not a row.
    out_dir = tmp_path / "generated" / "candidate"
    out_dir.mkdir(parents=True)
    manifest_path(out_dir).write_text(
        dump_json_str({"item_id": "src/a.py", "finished": True}) + "\n\n",
        encoding="utf-8",
    )

    assert continuations.read_manifest(out_dir) == {"src/a.py": True}


def test_a_later_manifest_row_replaces_an_earlier_one(tmp_path: pathlib.Path) -> None:
    # That is what makes a redone batch correct the row it wrote the first time.
    out_dir = tmp_path / "generated" / "candidate"
    out_dir.mkdir(parents=True)
    manifest_path(out_dir).write_text(
        dump_json_str({"item_id": "src/a.py", "finished": False})
        + "\n"
        + dump_json_str({"item_id": "src/a.py", "finished": True})
        + "\n",
        encoding="utf-8",
    )

    assert continuations.read_manifest(out_dir) == {"src/a.py": True}


def test_a_spec_that_is_not_an_object_is_refused(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "spec.json"
    path.write_text(dump_json_str(["candidate"]), encoding="utf-8")

    with pytest.raises(JSONTypeError):
        _ = continuations.load_spec(path)


def test_a_spec_document_round_trips_through_the_loader(tmp_path: pathlib.Path) -> None:
    spec = continuations.load_spec(spec_file(tmp_path))

    assert spec["arm"] == "candidate"
    assert spec["batch_size"] == 2


@pytest.mark.usefixtures("restore_hooks")
def test_the_entry_point_carries_the_exit_code(tmp_path: pathlib.Path) -> None:
    recorder = Recorder()
    install(recorder)
    tokens = command_line(tmp_path, spec_file(tmp_path))
    saved = sys.argv
    sys.argv = ["modeltrainer-continuations", *tokens]
    try:
        with pytest.raises(SystemExit) as excinfo:
            continuations.entrypoint()
    finally:
        sys.argv = saved

    assert excinfo.value.code == 0


@pytest.mark.usefixtures("restore_hooks")
def test_running_it_as_a_module_actually_generates(tmp_path: pathlib.Path) -> None:
    # A module that imports, defines an entry point and exits 0 is worse than
    # one that crashes: the console script works, so the two forms disagree.
    recorder = Recorder()
    install(recorder)
    tokens = command_line(tmp_path, spec_file(tmp_path, paths=("src/a.py",)))
    saved = sys.argv
    sys.argv = ["modeltrainer-continuations", *tokens]
    try:
        with pytest.raises(SystemExit):
            runpy.run_module("model_trainer.cli.continuations", run_name="__main__")
    finally:
        sys.argv = saved

    out_dir = tmp_path / "generated" / "candidate"
    assert generated_path(out_dir, "src/a.py").is_file()


def test_the_manifest_rows_are_readable_json_objects(tmp_path: pathlib.Path) -> None:
    # The scorer's tooling reads these; a row it cannot decode is a row that
    # silently drops an item out of every later count.
    out_dir = tmp_path / "generated" / "candidate"
    out_dir.mkdir(parents=True)
    continuations.append_manifest(
        out_dir,
        [Completion(item_id="src/a.py", text="x", finished=True)],
    )

    line = manifest_path(out_dir).read_text(encoding="utf-8").splitlines()[0]
    assert narrow_json_to_dict(load_json_str(line)) == {"item_id": "src/a.py", "finished": True}


class TestTheProductionDefaults:
    """The hooks the cluster actually runs, exercised through fakes below them.

    Substituted in every test above, so without this the two seams that carry
    the whole sweep would be the only lines in the command with no coverage --
    and they are the ones that decide which weights answer.
    """

    _METADATA: ClassVar[JSONObject] = {
        "strategy_name": "full",
        "hub_model_id": "test/base-model",
        "tokenizer_id": "test-tok",
        "is_peft": False,
        "quantization": None,
    }

    def _saved_run(self, tmp_path: pathlib.Path) -> str:
        """Write a saved run the strategy registry can reload.

        Args:
            tmp_path: The test's temporary directory.

        Returns:
            The artifact directory, as a string.
        """
        (tmp_path / "hf_lm_metadata.json").write_text(
            dump_json_str(self._METADATA), encoding="utf-8"
        )
        return str(tmp_path)

    def test_the_base_arm_default_attaches_nothing(self, tmp_path: pathlib.Path) -> None:
        from model_trainer.core.services.model.backends.hf_lm import _test_hooks as hf_hooks

        saved = (hf_hooks.Hooks.load_hf_model, hf_hooks.Hooks.load_hf_tokenizer)
        hf_hooks.Hooks.load_hf_model = fake_model_loader
        hf_hooks.Hooks.load_hf_tokenizer = fake_tokenizer_loader
        try:
            prepared = cli_hooks._default_load_continuation_arm(self._saved_run(tmp_path), "base")
        finally:
            (hf_hooks.Hooks.load_hf_model, hf_hooks.Hooks.load_hf_tokenizer) = saved

        assert prepared.is_peft is False
        assert prepared.strategy_name is None

    def test_the_candidate_arm_default_reapplies_the_saved_strategy(
        self, tmp_path: pathlib.Path
    ) -> None:
        from model_trainer.core.services.finetuning.strategies import _test_hooks as ft_hooks
        from model_trainer.core.services.model.backends.hf_lm import _test_hooks as hf_hooks

        saved = (hf_hooks.Hooks.load_hf_model, hf_hooks.Hooks.load_hf_tokenizer)
        saved_full = ft_hooks.Hooks.load_full_model
        hf_hooks.Hooks.load_hf_model = fake_model_loader
        hf_hooks.Hooks.load_hf_tokenizer = fake_tokenizer_loader
        ft_hooks.Hooks.load_full_model = fake_full_model_loader
        try:
            prepared = cli_hooks._default_load_continuation_arm(
                self._saved_run(tmp_path), "candidate"
            )
        finally:
            (hf_hooks.Hooks.load_hf_model, hf_hooks.Hooks.load_hf_tokenizer) = saved
            ft_hooks.Hooks.load_full_model = saved_full

        assert prepared.strategy_name == "full"

    def test_the_decoder_default_reaches_the_batched_generator(self) -> None:
        # The only reason this default is not the generator itself: the entry
        # holds keyword arguments and the generator takes them by name.
        prompts = [EvalPrompt(item_id="src/a.py", prompt="abc", reference="rest")]

        completions = cli_hooks._default_generate_continuation_batch(
            model=generating_model(),
            prompts=prompts,
            max_new_tokens=4,
            max_prompt_tokens=8,
            repetition_penalty=1.1,
            device="cpu",
            seed=0,
        )

        assert [c["item_id"] for c in completions] == ["src/a.py"]
