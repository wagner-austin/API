"""Tests for scoring a baseline from the command line.

The case that carries the design is the ordering one: determinism must be
pinned BEFORE the model loads, because loading weights is enough to create
the cuBLAS handle and a pin after that is accepted in silence. A test that
only checked the record's contents would pass on a version that pinned last
and recorded a posture the run did not have.
"""

from __future__ import annotations

import pathlib
import runpy
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
        self.postures: list[tuple[bool, bool]] = []

    def apply_determinism(self, *, remove_split_k: bool, math_attention: bool) -> DeterminismRecord:
        self.order.append("pin")
        self.postures.append((remove_split_k, math_attention))
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


def _record_path(tmp_path: pathlib.Path) -> pathlib.Path:
    """Where a run's record lands.

    Named rather than read back off the end of the argv list: two output
    flags now follow each other, and a positional read would silently start
    naming the other one.

    Args:
        tmp_path: The test's temporary directory.

    Returns:
        The record path.
    """
    return tmp_path / "out" / "record.json"


def _outcomes_path(tmp_path: pathlib.Path) -> pathlib.Path:
    """Where a run's per-item outcomes land.

    Args:
        tmp_path: The test's temporary directory.

    Returns:
        The outcomes path.
    """
    return tmp_path / "out" / "outcomes.json"


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
        str(_record_path(tmp_path)),
        "--outcomes",
        str(_outcomes_path(tmp_path)),
    ]


@pytest.mark.usefixtures("restore_hooks")
def test_determinism_is_pinned_before_the_model_loads(tmp_path: pathlib.Path) -> None:
    # The whole point. Loading weights creates the cuBLAS handle, and
    # CUBLAS_WORKSPACE_CONFIG is read exactly then -- so a pin afterwards is
    # accepted without error and does nothing, and the record would claim a
    # posture the run did not have.
    recorder = _Recorder()
    _install(recorder)

    _ = score_baseline.score_with_outcomes(
        hub_model_id="gpt2",
        items_path=_items_file(tmp_path),
        device="cpu",
        max_seq_len=512,
        experiment="e",
        label="l",
    )

    assert recorder.order == ["pin", "load:gpt2", "score"]
    # Split-K removed, matching `baseline_cloze_job`. The two produce the same
    # floor by two routes -- one from the queue, one from the command line --
    # and a posture that differed between them would make the two disagree in
    # the last bits, which is exactly where a cloze tie is decided.
    assert recorder.postures == [(True, True)]


@pytest.mark.usefixtures("restore_hooks")
def test_the_record_carries_the_four_numbers_and_the_configuration(
    tmp_path: pathlib.Path,
) -> None:
    _install(_Recorder())

    record, _ = score_baseline.score_with_outcomes(
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

    record, _ = score_baseline.score_with_outcomes(
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


class TestTheDigestIdentifiesTheDecisionsNotTheArithmetic:
    """Measured 2026-08-25, first cross-card comparison this project ran.

    gpt2 on the same 2,627 items scored 1374 correct on both a 3090 Ti and an
    A100 80GB -- accuracy identical to all fifteen digits -- and the digests
    differed, because `scores` were in them. They are raw negative
    log-likelihoods, so they differ in their low bits between two cards
    whatever the answers were. A digest that always differs across hardware
    cannot distinguish "these runs disagreed about an item" from "these runs
    agreed completely on different cards", so it answers nothing.
    """

    def test_the_same_decisions_digest_the_same_on_different_arithmetic(self) -> None:
        card = [ClozeItemOutcome(item_id="a", correct=True, scores=[1.0, 2.0])]
        other_card = [ClozeItemOutcome(item_id="a", correct=True, scores=[1.0000000000000002, 2.0])]

        assert score_baseline.outcomes_digest(card) == score_baseline.outcomes_digest(other_card)

    def test_a_different_item_being_right_still_changes_it(self) -> None:
        """The property the digest exists for survives dropping the scores."""
        left = [
            ClozeItemOutcome(item_id="a", correct=True, scores=[1.0, 2.0]),
            ClozeItemOutcome(item_id="b", correct=False, scores=[2.0, 1.0]),
        ]
        right = [
            ClozeItemOutcome(item_id="a", correct=False, scores=[1.0, 2.0]),
            ClozeItemOutcome(item_id="b", correct=True, scores=[2.0, 1.0]),
        ]

        assert left[0]["correct"] is not right[0]["correct"]
        assert score_baseline.outcomes_digest(left) != score_baseline.outcomes_digest(right)

    def test_the_same_decisions_in_a_different_order_are_a_difference(self) -> None:
        """One item set is scored in one order; a reordering is itself news."""
        forward = [
            ClozeItemOutcome(item_id="a", correct=True, scores=[1.0]),
            ClozeItemOutcome(item_id="b", correct=False, scores=[1.0]),
        ]
        backward = list(reversed(forward))

        assert score_baseline.outcomes_digest(forward) != score_baseline.outcomes_digest(backward)

    def test_the_scores_are_kept_where_a_diagnosis_can_reach_them(self) -> None:
        """Excluded from the digest is not discarded.

        ClozeItemOutcome says it is carried "so that two arms scored on the
        same item set can be compared item by item", and the scorer reduced
        it to one digest and dropped it -- so when the A100 digest differed
        there was no way to ask which items moved.
        """
        outcomes = [ClozeItemOutcome(item_id="a", correct=True, scores=[1.5, 2.5])]

        assert load_json_str(score_baseline.encode_outcomes(outcomes)) == [
            {"item_id": "a", "correct": True, "scores": [1.5, 2.5]}
        ]


@pytest.mark.usefixtures("restore_hooks")
def test_main_writes_a_record_that_decodes(tmp_path: pathlib.Path) -> None:
    _install(_Recorder())
    argv = _cpu_argv(tmp_path)

    assert score_baseline.main(argv) == 0

    written = _record_path(tmp_path)
    decoded = decode_run_record(narrow_json_to_dict(load_json_str(written.read_text("utf-8"))))
    assert decoded["label"] == "gpt2-baseline"
    assert decoded["observations"][0]["name"] == "cloze_accuracy"


@pytest.mark.usefixtures("restore_hooks")
def test_main_creates_the_output_directory(tmp_path: pathlib.Path) -> None:
    # The out path names a directory a staged job has not created yet.
    _install(_Recorder())
    argv = _cpu_argv(tmp_path)

    score_baseline.main(argv)

    assert _record_path(tmp_path).parent.is_dir()


@pytest.mark.usefixtures("restore_hooks")
def test_main_writes_every_per_item_outcome_beside_the_record(
    tmp_path: pathlib.Path,
) -> None:
    """The record says WHETHER two runs agreed; this says about what.

    Required rather than optional: a flag nobody remembers to pass is not
    there on the run that turns out to need it, which is what happened to the
    first A100 floor -- its digest differed from the 3090 Ti's and the
    outcomes to explain it had already been thrown away.
    """
    _install(_Recorder())

    assert score_baseline.main(_cpu_argv(tmp_path)) == 0

    written = load_json_str(_outcomes_path(tmp_path).read_text(encoding="utf-8"))
    assert written == [
        {"item_id": "a::0", "correct": True, "scores": [1.0, 2.0, 3.0, 4.0]},
        {"item_id": "a::1", "correct": False, "scores": [1.0, 2.0, 3.0, 4.0]},
    ]


@pytest.mark.usefixtures("restore_hooks")
def test_main_creates_the_outcomes_directory_too(tmp_path: pathlib.Path) -> None:
    """A staged job has created neither path."""
    _install(_Recorder())

    score_baseline.main(_cpu_argv(tmp_path))

    assert _outcomes_path(tmp_path).parent.is_dir()


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
        postures: list[tuple[bool, bool]] = []

        def _core_pin(*, remove_split_k: bool, math_attention: bool) -> DeterminismRecord:
            calls.append("core")
            postures.append((remove_split_k, math_attention))
            return PINNED

        core_hooks.apply_determinism_hook = _core_pin
        try:
            record = cli_hooks._default_apply_determinism(remove_split_k=True, math_attention=True)
            declined = cli_hooks._default_apply_determinism(
                remove_split_k=False, math_attention=False
            )
        finally:
            core_hooks.apply_determinism_hook = core_hooks._default_apply_determinism

        assert calls == ["core", "core"]
        # Forwarded rather than decided here. Both values are exercised
        # because a delegate that hardcoded either one would still satisfy a
        # single-value assertion, and the whole point of the CLI tier is that
        # a scoring command and a measurement command pass different postures
        # through the same hook.
        assert postures == [(True, True), (False, False)]
        assert record == PINNED
        assert declined == PINNED

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

    def test_running_it_as_a_module_actually_scores(self, tmp_path: pathlib.Path) -> None:
        """The guard that was missing until 2026-08-27.

        Without ``if __name__ == "__main__"`` this module imported, ran
        nothing and exited 0 -- while the console script above worked, so the
        two invocation forms disagreed and the broken one looked exactly like
        a scoring run that legitimately produced no file. It cost real time
        during the known-answer re-registration.

        Args:
            tmp_path: The test's temporary directory.
        """
        recorder = _Recorder()
        _install(recorder)
        module_name = "model_trainer.cli.score_baseline"
        saved_argv = sys.argv
        saved_module = sys.modules.pop(module_name, None)
        sys.argv = ["x", *_cpu_argv(tmp_path)]
        try:
            with pytest.raises(SystemExit) as raised:
                runpy.run_module(module_name, run_name="__main__", alter_sys=False)
        finally:
            sys.argv = saved_argv
            if saved_module is not None:
                sys.modules[module_name] = saved_module
            cli_hooks.apply_determinism_hook = cli_hooks._default_apply_determinism
            cli_hooks.load_hub_model = cli_hooks._default_load_hub_model
            cli_hooks.score_cloze = cli_hooks._default_score_cloze

        assert raised.value.code == 0
        assert _record_path(tmp_path).is_file()


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
