"""Tests for scoring a TRAINED run's artifact from the command line.

Two properties carry the design. The ordering one is inherited from
``score_baseline`` and holds for the same reason: determinism must be pinned
before the model loads, because loading weights creates the cuBLAS handle and
a pin afterwards is accepted in silence.

The one that is this command's own reason to exist is that it loads through
the ARTIFACT loader rather than the hub loader. For a ``full`` finetune both
would give the right weights; for an adapter arm the hub loader would score
the untrained base and report the number as the arm's. The test that pins
this asserts WHICH loader was called, because a test on the record's contents
would pass on a version that scored the wrong model.
"""

from __future__ import annotations

import pathlib
import runpy
import sys
from collections.abc import Generator

import pytest
from platform_core.determinism_record import TRUE, DeterminismRecord, determinism_record
from platform_core.json_utils import load_json_str, narrow_json_to_dict
from platform_core.run_record import decode_run_record

from model_trainer.cli import _test_hooks as cli_hooks
from model_trainer.cli import score_run
from model_trainer.core import _test_hooks as core_hooks
from model_trainer.core.contracts.cloze import ClozeEvalResult, ClozeItem, ClozeItemOutcome
from model_trainer.core.contracts.model import PreparedLMModel
from model_trainer.core.services.model.deterministic_gemm import CUBLAS_ARM
from tests.core.services.model.backends.hf_lm.testing import (
    FakeEncoder,
    FakeHFModel,
    FakeHFTokenizer,
)

PINNED = determinism_record("torch", {"cudnn_deterministic": TRUE})

_ITEMS_JSONL = (
    '{"item_id": "a::0", "template": "The tower is <<BLANK>> metres.", '
    '"answer": "324", "distractors": ["12", "99", "700"]}\n'
    '{"item_id": "a::1", "template": "It opened in <<BLANK>>.", '
    '"answer": "1889", "distractors": ["1789", "1989", "1689"]}\n'
)

_PREPARED: PreparedLMModel = PreparedLMModel(
    model=FakeHFModel(),
    tokenizer_id=None,
    eos_id=0,
    pad_id=0,
    max_seq_len=512,
    tok_for_dataset=FakeEncoder(),
)


class _Recorder:
    """Records the order of the seams and which loader was asked."""

    def __init__(self) -> None:
        self.order: list[str] = []
        self.scored_with: dict[str, str] = {}
        self.postures: list[tuple[bool, bool]] = []
        self.hub_calls: list[str] = []

    def apply_determinism(self, *, remove_split_k: bool, math_attention: bool) -> DeterminismRecord:
        self.order.append("pin")
        self.postures.append((remove_split_k, math_attention))
        return PINNED

    def load_run_model(self, artifact_path: str, /) -> PreparedLMModel:
        self.order.append(f"load-artifact:{artifact_path}")
        return _PREPARED

    def load_hub_model(self, hub_model_id: str, /) -> PreparedLMModel:
        """Stands in for the loader this command must NEVER use.

        Recorded rather than raising, so the test that pins the choice reports
        "the hub loader was called" instead of an exception from somewhere
        inside the scorer.
        """
        self.hub_calls.append(hub_model_id)
        return _PREPARED

    def score_cloze(
        self,
        *,
        items: list[ClozeItem],
        model: PreparedLMModel,
        device: str,
        max_seq_len: int,
    ) -> ClozeEvalResult:
        self.order.append("score")
        self.scored_with = {"device": device, "max_seq_len": str(max_seq_len)}
        return ClozeEvalResult(
            total=len(items),
            correct=1,
            accuracy=0.5,
            chance=0.25,
            outcomes=[
                ClozeItemOutcome(item_id=i["item_id"], correct=n == 0, scores=[1.0, 2.0, 3.0, 4.0])
                for n, i in enumerate(items)
            ],
        )


def _install(recorder: _Recorder) -> None:
    cli_hooks.apply_determinism_hook = recorder.apply_determinism
    cli_hooks.load_run_model = recorder.load_run_model
    cli_hooks.load_hub_model = recorder.load_hub_model
    cli_hooks.score_cloze = recorder.score_cloze


def _restore_hooks() -> Generator[None, None, None]:
    """Put the module-global hooks back after each test.

    Yields:
        None, for the duration of one test.
    """
    saved = (
        cli_hooks.apply_determinism_hook,
        cli_hooks.load_run_model,
        cli_hooks.load_hub_model,
        cli_hooks.score_cloze,
    )
    git = core_hooks.env_git_commit
    name = core_hooks.cuda_device_name
    driver = core_hooks.cuda_driver_version
    yield
    (
        cli_hooks.apply_determinism_hook,
        cli_hooks.load_run_model,
        cli_hooks.load_hub_model,
        cli_hooks.score_cloze,
    ) = saved
    core_hooks.env_git_commit = git
    core_hooks.cuda_device_name = name
    core_hooks.cuda_driver_version = driver


restore_hooks = pytest.fixture(_restore_hooks)


def _items_file(tmp_path: pathlib.Path) -> pathlib.Path:
    path = tmp_path / "items.jsonl"
    path.write_text(_ITEMS_JSONL, encoding="utf-8")
    return path


def _artifact_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    path = tmp_path / "artifacts" / "armB-s42"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _record_path(tmp_path: pathlib.Path) -> pathlib.Path:
    return tmp_path / "out" / "record.json"


def _outcomes_path(tmp_path: pathlib.Path) -> pathlib.Path:
    return tmp_path / "out" / "outcomes.json"


def _cpu_argv(tmp_path: pathlib.Path) -> list[str]:
    return [
        "--artifact-dir",
        str(_artifact_dir(tmp_path)),
        "--items",
        str(_items_file(tmp_path)),
        "--device",
        "cpu",
        "--max-seq-len",
        "512",
        "--experiment",
        "extraction-eval-redo",
        "--label",
        "armB-s42",
        "--out",
        str(_record_path(tmp_path)),
        "--outcomes",
        str(_outcomes_path(tmp_path)),
        "--kernel",
        CUBLAS_ARM,
    ]


@pytest.mark.usefixtures("restore_hooks")
def test_it_loads_through_the_artifact_loader_never_the_hub_loader(
    tmp_path: pathlib.Path,
) -> None:
    """This command's reason to exist.

    The hub loader reapplies no finetuning strategy. On an adapter arm it
    would score the untrained base and report the number as the arm's, with
    nothing in the record to notice it by.
    """
    recorder = _Recorder()
    _install(recorder)
    artifact = _artifact_dir(tmp_path)

    _ = score_run.score_run_with_outcomes(
        artifact_dir=artifact,
        items_path=_items_file(tmp_path),
        device="cpu",
        max_seq_len=512,
        experiment="e",
        label="l",
        kernel=CUBLAS_ARM,
    )

    assert recorder.hub_calls == []
    assert f"load-artifact:{artifact}" in recorder.order


@pytest.mark.usefixtures("restore_hooks")
def test_determinism_is_pinned_before_the_artifact_loads(tmp_path: pathlib.Path) -> None:
    recorder = _Recorder()
    _install(recorder)
    artifact = _artifact_dir(tmp_path)

    _ = score_run.score_run_with_outcomes(
        artifact_dir=artifact,
        items_path=_items_file(tmp_path),
        device="cpu",
        max_seq_len=512,
        experiment="e",
        label="l",
        kernel=CUBLAS_ARM,
    )

    assert recorder.order == ["pin", f"load-artifact:{artifact}", "score"]


@pytest.mark.usefixtures("restore_hooks")
def test_the_posture_matches_the_baseline_command(tmp_path: pathlib.Path) -> None:
    """An arm and a floor are subtracted from each other.

    A posture that differed between the two commands would land in every
    lift computed from the pair.
    """
    recorder = _Recorder()
    _install(recorder)

    _ = score_run.score_run_with_outcomes(
        artifact_dir=_artifact_dir(tmp_path),
        items_path=_items_file(tmp_path),
        device="cpu",
        max_seq_len=512,
        experiment="e",
        label="l",
        kernel=CUBLAS_ARM,
    )

    assert recorder.postures == [(True, True)]


@pytest.mark.usefixtures("restore_hooks")
def test_the_record_carries_the_four_numbers_and_the_configuration(
    tmp_path: pathlib.Path,
) -> None:
    _install(_Recorder())

    record, _ = score_run.score_run_with_outcomes(
        artifact_dir=_artifact_dir(tmp_path),
        items_path=_items_file(tmp_path),
        device="cpu",
        max_seq_len=512,
        experiment="extraction-eval-redo",
        label="armB-s42",
        kernel=CUBLAS_ARM,
    )

    assert record["experiment"] == "extraction-eval-redo"
    assert record["label"] == "armB-s42"
    assert [(o["name"], o["value"]) for o in record["observations"]] == [
        ("cloze_accuracy", 0.5),
        ("cloze_chance", 0.25),
        ("cloze_correct", 1.0),
        ("cloze_total", 2.0),
    ]
    assert record["fingerprint"]["determinism"] == PINNED
    assert record["payload_digest"].startswith("sha256:")


@pytest.mark.usefixtures("restore_hooks")
def test_main_writes_both_files_and_the_record_decodes(tmp_path: pathlib.Path) -> None:
    _install(_Recorder())

    assert score_run.main(_cpu_argv(tmp_path)) == 0

    written = narrow_json_to_dict(load_json_str(_record_path(tmp_path).read_text(encoding="utf-8")))
    decoded = decode_run_record(written)
    assert decoded["label"] == "armB-s42"
    assert _outcomes_path(tmp_path).read_text(encoding="utf-8").startswith("[")


@pytest.mark.usefixtures("restore_hooks")
def test_the_scorer_is_given_the_device_and_budget_it_was_asked_for(
    tmp_path: pathlib.Path,
) -> None:
    recorder = _Recorder()
    _install(recorder)

    score_run.main(_cpu_argv(tmp_path))

    assert recorder.scored_with == {"device": "cpu", "max_seq_len": "512"}


@pytest.mark.usefixtures("restore_hooks")
def test_a_non_numeric_sequence_length_scores_nothing(tmp_path: pathlib.Path) -> None:
    recorder = _Recorder()
    _install(recorder)
    argv = _cpu_argv(tmp_path)
    argv[argv.index("--max-seq-len") + 1] = "many"

    with pytest.raises(ValueError, match="positive integer"):
        score_run.main(argv)

    assert recorder.order == []


@pytest.mark.usefixtures("restore_hooks")
def test_a_zero_sequence_length_scores_nothing(tmp_path: pathlib.Path) -> None:
    recorder = _Recorder()
    _install(recorder)
    argv = _cpu_argv(tmp_path)
    argv[argv.index("--max-seq-len") + 1] = "0"

    with pytest.raises(ValueError, match="positive integer"):
        score_run.main(argv)

    assert recorder.order == []


@pytest.mark.usefixtures("restore_hooks")
def test_a_missing_required_flag_scores_nothing(tmp_path: pathlib.Path) -> None:
    recorder = _Recorder()
    _install(recorder)
    argv = _cpu_argv(tmp_path)
    del argv[argv.index("--kernel") : argv.index("--kernel") + 2]

    with pytest.raises(ValueError):
        score_run.main(argv)

    assert recorder.order == []


@pytest.mark.usefixtures("restore_hooks")
def test_a_cpu_run_records_no_card_rather_than_a_wrong_one(tmp_path: pathlib.Path) -> None:
    _install(_Recorder())

    record, _ = score_run.score_run_with_outcomes(
        artifact_dir=_artifact_dir(tmp_path),
        items_path=_items_file(tmp_path),
        device="cpu",
        max_seq_len=512,
        experiment="e",
        label="l",
        kernel=CUBLAS_ARM,
    )

    assert record["fingerprint"]["gpu_model"] == ""
    assert record["fingerprint"]["driver_version"] == ""


def test_the_default_loader_reapplies_the_strategy_rather_than_nothing(
    tmp_path: pathlib.Path,
) -> None:
    """The real hook, with fakes only at the torch/transformers boundary.

    This is the contrast that justifies the whole command. The baseline
    loader's test asserts ``strategy_name is None`` -- *a baseline is defined
    by having nothing applied to it*. This asserts the opposite, on the same
    kind of artifact, because an arm is defined by having something applied
    and a scorer that quietly applied nothing would report the base model's
    accuracy under the arm's name.
    """
    from platform_core.json_utils import JSONObject, dump_json_str

    from model_trainer.core.contracts.model import QuantizationConfig
    from model_trainer.core.services.finetuning.strategies._test_hooks import Hooks as FtHooks
    from model_trainer.core.services.model.backends.hf_lm._test_hooks import Hooks as HfLmHooks
    from model_trainer.core.types import LMModelProto

    loaded: list[str] = []

    def _load_model(model_id_or_path: str, quantization: QuantizationConfig | None) -> LMModelProto:
        loaded.append(model_id_or_path)
        assert quantization is None
        return FakeHFModel(model_id_or_path)

    def _load_full(model_path: str) -> LMModelProto:
        return FakeHFModel(f"adapted-{model_path}")

    def _load_tokenizer(model_id_or_path: str) -> FakeHFTokenizer:
        del model_id_or_path
        return FakeHFTokenizer()

    metadata: JSONObject = {
        "strategy_name": "full",
        "hub_model_id": "test/base-model",
        "tokenizer_id": "test-tok",
        "is_peft": False,
        "quantization": None,
    }
    artifact = _artifact_dir(tmp_path)
    (artifact / "hf_lm_metadata.json").write_text(dump_json_str(metadata), encoding="utf-8")

    saved = (HfLmHooks.load_hf_model, HfLmHooks.load_hf_tokenizer, FtHooks.load_full_model)
    HfLmHooks.load_hf_model = _load_model
    HfLmHooks.load_hf_tokenizer = _load_tokenizer
    FtHooks.load_full_model = _load_full
    try:
        prepared = cli_hooks._default_load_run_model(str(artifact))
    finally:
        (HfLmHooks.load_hf_model, HfLmHooks.load_hf_tokenizer, FtHooks.load_full_model) = saved

    assert loaded == ["test/base-model"]
    assert prepared.strategy_name == "full"
    assert prepared.hub_model_id == "test/base-model"


@pytest.mark.usefixtures("restore_hooks")
def test_the_entrypoint_reads_the_process_arguments(tmp_path: pathlib.Path) -> None:
    _install(_Recorder())
    saved = sys.argv
    sys.argv = ["modeltrainer-score-run", *_cpu_argv(tmp_path)]
    try:
        with pytest.raises(SystemExit) as excinfo:
            score_run.entrypoint()
    finally:
        sys.argv = saved

    assert excinfo.value.code == 0


@pytest.mark.usefixtures("restore_hooks")
def test_running_it_as_a_module_actually_runs(tmp_path: pathlib.Path) -> None:
    """``python -m model_trainer.cli.score_run`` must DO something.

    Without the guard the module imports, defines its functions and exits 0
    having scored nothing -- indistinguishable from a run that legitimately
    produced no file. The assertion is that the record EXISTS, not merely
    that the exit code was zero, because zero is what the broken shape
    returns.
    """
    _install(_Recorder())
    saved = sys.argv
    sys.argv = ["score_run", *_cpu_argv(tmp_path)]
    try:
        with pytest.raises(SystemExit) as excinfo:
            runpy.run_module("model_trainer.cli.score_run", run_name="__main__")
    finally:
        sys.argv = saved

    assert excinfo.value.code == 0
    assert _record_path(tmp_path).is_file()
