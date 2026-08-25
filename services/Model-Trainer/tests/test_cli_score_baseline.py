"""Tests for scoring a baseline from the command line.

The case that carries the design is the ordering one: determinism must be
pinned BEFORE the model loads, because loading weights is enough to create
the cuBLAS handle and a pin after that is accepted in silence. A test that
only checked the record's contents would pass on a version that pinned last
and recorded a posture the run did not have.
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Callable, Generator

import pytest
from platform_core.determinism_record import TRUE, DeterminismRecord, determinism_record
from platform_core.json_utils import load_json_str, narrow_json_to_dict
from platform_core.run_record import decode_run_record

from model_trainer.cli import _test_hooks as cli_hooks
from model_trainer.cli import score_baseline
from model_trainer.core import _test_hooks as core_hooks
from model_trainer.core.contracts.cloze import ClozeEvalResult, ClozeItem, ClozeItemOutcome
from model_trainer.core.contracts.model import PreparedLMModel
from model_trainer.core.services.model.backends.hf_lm._test_hooks import Hooks as HfLmHooks
from model_trainer.core.types import LMModelProto
from model_trainer.worker.cloze_job import parse_items
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


class _Recorder:
    """Records the order of the two seams that must not be transposed."""

    def __init__(self) -> None:
        self.order: list[str] = []
        self.scored_with: dict[str, str] = {}

    def apply_determinism(self) -> DeterminismRecord:
        self.order.append("pin")
        return PINNED

    def load_hub_model(self, hub_model_id: str, /) -> PreparedLMModel:
        self.order.append(f"load:{hub_model_id}")
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


# The prepared model is never exercised by these tests -- the scorer is
# behind a hook -- so this only has to be a real object the loader can hand
# back. The two doubles are lifted from the hf_lm helpers rather than
# rewritten, because a second FakeHFModel would drift from the protocol the
# first one already tracks.
_PREPARED: PreparedLMModel = PreparedLMModel(
    model=FakeHFModel(),
    tokenizer_id=None,
    eos_id=0,
    pad_id=0,
    max_seq_len=512,
    tok_for_dataset=FakeEncoder(),
)


def _install(recorder: _Recorder) -> None:
    cli_hooks.apply_determinism_hook = recorder.apply_determinism
    cli_hooks.load_hub_model = recorder.load_hub_model
    cli_hooks.score_cloze = recorder.score_cloze


def _restore_hooks() -> Generator[None, None, None]:
    """Put the module-global hooks back after each test.

    Left swapped they would answer for every later test in the same worker,
    which is the failure mode a shared hook has and a fixture is the whole of
    the fix.

    Yields:
        None, for the duration of one test.
    """
    saved = (cli_hooks.apply_determinism_hook, cli_hooks.load_hub_model, cli_hooks.score_cloze)
    git = core_hooks.env_git_commit
    name = core_hooks.cuda_device_name
    driver = core_hooks.cuda_driver_version
    yield
    (cli_hooks.apply_determinism_hook, cli_hooks.load_hub_model, cli_hooks.score_cloze) = saved
    core_hooks.env_git_commit = git
    core_hooks.cuda_device_name = name
    core_hooks.cuda_driver_version = driver


restore_hooks = pytest.fixture(_restore_hooks)


def _items_file(tmp_path: pathlib.Path) -> pathlib.Path:
    path = tmp_path / "items.jsonl"
    path.write_text(_ITEMS_JSONL, encoding="utf-8")
    return path


def _cpu_argv(tmp_path: pathlib.Path) -> list[str]:
    return [
        "--model",
        "gpt2",
        "--items",
        str(_items_file(tmp_path)),
        "--device",
        "cpu",
        "--max-seq-len",
        "512",
        "--experiment",
        "wiki-corpus-extraction-ablation",
        "--label",
        "gpt2-baseline",
        "--out",
        str(tmp_path / "out" / "record.json"),
    ]


@pytest.mark.usefixtures("restore_hooks")
def test_determinism_is_pinned_before_the_model_loads(tmp_path: pathlib.Path) -> None:
    # The whole point. Loading weights creates the cuBLAS handle, and
    # CUBLAS_WORKSPACE_CONFIG is read exactly then -- so a pin afterwards is
    # accepted without error and does nothing, and the record would claim a
    # posture the run did not have.
    recorder = _Recorder()
    _install(recorder)

    score_baseline.score(
        hub_model_id="gpt2",
        items_path=_items_file(tmp_path),
        device="cpu",
        max_seq_len=512,
        experiment="e",
        label="l",
    )

    assert recorder.order == ["pin", "load:gpt2", "score"]


@pytest.mark.usefixtures("restore_hooks")
def test_the_record_carries_the_four_numbers_and_the_configuration(
    tmp_path: pathlib.Path,
) -> None:
    _install(_Recorder())

    record = score_baseline.score(
        hub_model_id="gpt2",
        items_path=_items_file(tmp_path),
        device="cpu",
        max_seq_len=512,
        experiment="wiki-corpus-extraction-ablation",
        label="gpt2-baseline",
    )

    assert record["experiment"] == "wiki-corpus-extraction-ablation"
    assert record["label"] == "gpt2-baseline"
    assert [(o["name"], o["value"]) for o in record["observations"]] == [
        ("cloze_accuracy", 0.5),
        ("cloze_chance", 0.25),
        ("cloze_correct", 1.0),
        ("cloze_total", 2.0),
    ]
    assert record["fingerprint"]["determinism"] == PINNED
    assert record["payload_digest"].startswith("sha256:")


@pytest.mark.usefixtures("restore_hooks")
def test_a_cpu_run_records_no_card_rather_than_a_wrong_one(tmp_path: pathlib.Path) -> None:
    _install(_Recorder())

    record = score_baseline.score(
        hub_model_id="gpt2",
        items_path=_items_file(tmp_path),
        device="cpu",
        max_seq_len=512,
        experiment="e",
        label="l",
    )

    assert record["fingerprint"]["gpu_model"] == ""
    assert record["fingerprint"]["driver_version"] == ""


def test_the_digest_changes_when_an_item_outcome_changes() -> None:
    # Two runs can agree on an accuracy and disagree on WHICH items they got
    # right. The digest is what catches that without the comparing layer
    # understanding a single item.
    left = [ClozeItemOutcome(item_id="a", correct=True, scores=[1.0, 2.0])]
    right = [ClozeItemOutcome(item_id="a", correct=False, scores=[1.0, 2.0])]

    assert score_baseline.outcomes_digest(left) != score_baseline.outcomes_digest(right)


def test_the_digest_is_stable_for_the_same_outcomes() -> None:
    outcomes = [ClozeItemOutcome(item_id="a", correct=True, scores=[1.0, 2.0])]

    assert score_baseline.outcomes_digest(outcomes) == score_baseline.outcomes_digest(outcomes)


@pytest.mark.usefixtures("restore_hooks")
def test_main_writes_a_record_that_decodes(tmp_path: pathlib.Path) -> None:
    _install(_Recorder())
    argv = _cpu_argv(tmp_path)

    assert score_baseline.main(argv) == 0

    written = pathlib.Path(argv[-1])
    decoded = decode_run_record(narrow_json_to_dict(load_json_str(written.read_text("utf-8"))))
    assert decoded["label"] == "gpt2-baseline"
    assert decoded["observations"][0]["name"] == "cloze_accuracy"


@pytest.mark.usefixtures("restore_hooks")
def test_main_creates_the_output_directory(tmp_path: pathlib.Path) -> None:
    # The out path names a directory a staged job has not created yet.
    _install(_Recorder())
    argv = _cpu_argv(tmp_path)

    score_baseline.main(argv)

    assert pathlib.Path(argv[-1]).parent.is_dir()


@pytest.mark.usefixtures("restore_hooks")
def test_the_scorer_gets_the_device_and_budget_it_was_given(tmp_path: pathlib.Path) -> None:
    recorder = _Recorder()
    _install(recorder)
    argv = _cpu_argv(tmp_path)
    argv[argv.index("--max-seq-len") + 1] = "128"

    score_baseline.main(argv)

    assert recorder.scored_with == {"device": "cpu", "max_seq_len": "128"}


def _add_unknown_flag(argv: list[str]) -> list[str]:
    """Append a flag the command does not accept."""
    return [*argv, "--nonsense", "x"]


def _repeat_a_flag(argv: list[str]) -> list[str]:
    """Give --model twice, so the intent is ambiguous."""
    return [*argv, "--model", "gpt2"]


def _drop_the_last_value(argv: list[str]) -> list[str]:
    """Leave the final flag with no value after it."""
    return argv[:-1]


def _drop_a_required_flag(argv: list[str]) -> list[str]:
    """Remove --model entirely."""
    return argv[2:]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (_add_unknown_flag, "unknown argument"),
        (_repeat_a_flag, "more than once"),
        (_drop_the_last_value, "requires a value"),
        (_drop_a_required_flag, "--model is required"),
    ],
    ids=["unknown-flag", "repeated-flag", "flag-without-value", "missing-required"],
)
@pytest.mark.usefixtures("restore_hooks")
def test_a_command_line_that_was_not_understood_scores_nothing(
    tmp_path: pathlib.Path,
    mutate: Callable[[list[str]], list[str]],
    message: str,
) -> None:
    # A run under a mistyped flag is a different run, and it would otherwise
    # write a record claiming to be this one.
    recorder = _Recorder()
    _install(recorder)

    with pytest.raises(ValueError, match=message):
        score_baseline.main(mutate(_cpu_argv(tmp_path)))

    assert recorder.order == []


class TestTheProductionHooks:
    """The default hooks are real code, so they are exercised, not skipped.

    Fakes sit only at the torch/transformers boundary -- the same seam
    ``test_baseline_cloze_job`` fakes when it runs the real
    ``load_prepared_hf_lm_from_hub``. The function under test is the real one
    in every case here.
    """

    def test_the_hub_loader_asks_the_hub_for_the_model_it_was_given(self) -> None:
        loaded: list[str] = []

        def _load_model(model_id_or_path: str) -> LMModelProto:
            loaded.append(model_id_or_path)
            return FakeHFModel(model_id_or_path)

        def _load_tokenizer(model_id_or_path: str) -> FakeHFTokenizer:
            return FakeHFTokenizer()

        saved = (HfLmHooks.load_hf_model, HfLmHooks.load_hf_tokenizer)
        HfLmHooks.load_hf_model = _load_model
        HfLmHooks.load_hf_tokenizer = _load_tokenizer
        try:
            prepared = cli_hooks._default_load_hub_model("gpt2-medium")
        finally:
            (HfLmHooks.load_hf_model, HfLmHooks.load_hf_tokenizer) = saved

        assert loaded == ["gpt2-medium"]
        assert prepared.hub_model_id == "gpt2-medium"
        # A baseline is defined by having nothing applied to it.
        assert prepared.strategy_name is None

    def test_the_scorer_default_unpacks_the_prepared_model_and_scores(
        self, tmp_path: pathlib.Path
    ) -> None:
        # The only reason this default is not the scorer itself: the entry
        # holds a PreparedLMModel and score_cloze_items takes the two halves.
        items = parse_items(_ITEMS_JSONL)

        result = cli_hooks._default_score_cloze(
            items=items, model=_PREPARED, device="cpu", max_seq_len=512
        )

        assert result["total"] == 2
        assert result["chance"] == pytest.approx(0.25)

    @pytest.mark.usefixtures("restore_hooks")
    def test_the_determinism_default_delegates_to_the_one_the_workers_use(self) -> None:
        # A second spelling here would be a second posture nobody noticed
        # diverging, so this must reach the core hook rather than pin again.
        calls: list[str] = []

        def _core_pin() -> DeterminismRecord:
            calls.append("core")
            return PINNED

        core_hooks.apply_determinism_hook = _core_pin
        try:
            record = cli_hooks._default_apply_determinism()
        finally:
            core_hooks.apply_determinism_hook = core_hooks._default_apply_determinism

        assert calls == ["core"]
        assert record == PINNED

    def test_the_console_entry_point_carries_the_exit_code(self, tmp_path: pathlib.Path) -> None:
        recorder = _Recorder()
        _install(recorder)
        saved = sys.argv
        sys.argv = ["modeltrainer-score-baseline", *_cpu_argv(tmp_path)]
        try:
            with pytest.raises(SystemExit) as excinfo:
                score_baseline.entrypoint()
        finally:
            sys.argv = saved
            cli_hooks.apply_determinism_hook = cli_hooks._default_apply_determinism
            cli_hooks.load_hub_model = cli_hooks._default_load_hub_model
            cli_hooks.score_cloze = cli_hooks._default_score_cloze

        assert excinfo.value.code == 0


@pytest.mark.parametrize(
    "bad", ["0", "-1", "abc", "5.5"], ids=["zero", "negative", "word", "float"]
)
@pytest.mark.usefixtures("restore_hooks")
def test_a_token_budget_that_is_not_a_positive_integer_is_refused(
    tmp_path: pathlib.Path, bad: str
) -> None:
    recorder = _Recorder()
    _install(recorder)
    argv = _cpu_argv(tmp_path)
    argv[argv.index("--max-seq-len") + 1] = bad

    with pytest.raises(ValueError, match="positive integer"):
        score_baseline.main(argv)

    assert recorder.order == []
