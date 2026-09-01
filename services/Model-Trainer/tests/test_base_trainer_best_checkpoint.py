"""The reported test score must describe the weights that ship.

``_save_best_checkpoint`` overwrites the run's output directory on every
validation improvement, and ``train`` skips the final save while such a
checkpoint exists. So a run WITH a holdout ends with the best epoch on disk
and the LAST epoch in memory. Scoring the live model then publishes a number
for weights nobody receives.

That is not hypothetical. ``mi-kk-armB-realsplit-seed42`` (HPC3 job 55570784,
2026-08-25) saved 15 best checkpoints, the last at epoch 17, ran 20 epochs,
and shipped a 462 MB artifact holding epoch 17 alongside a manifest reporting
``test_loss`` measured on epoch 20 -- both numbers side by side with nothing
recording that they describe different models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch

from model_trainer.core import _test_hooks
from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.model import (
    EarlyStoppingState,
    ModelTrainConfig,
    PreparedLMModel,
)
from model_trainer.core.contracts.tokenizer import TokenizerTrainConfig
from model_trainer.core.infra.paths import model_dir
from model_trainer.core.services.dataset.local_text_builder import LocalTextDatasetBuilder
from model_trainer.core.services.model.backend_factory import create_char_lstm_backend
from model_trainer.core.services.model.backends.char_lstm.model import CharLSTMModel
from model_trainer.core.services.tokenizer.char_backend import CharBackend
from model_trainer.core.services.training import base_trainer as bt
from model_trainer.core.types import LMModelProto
from tests.core.services.model.backends.char_lstm._train_branches_support import (
    _LM,
    UNPINNED,
    _make_cfg,
    _make_prepared,
    _make_settings,
)

_WEIGHT_FILE = "weights.pt"


class _DiskBackedLM(_LM):
    """A model whose one weight round-trips through a real file.

    ``_restore_best_checkpoint`` is specified to score the bytes on disk
    rather than an in-memory snapshot, so the fake has to actually read
    them back for the assertion to mean anything.
    """

    def __init__(self: _DiskBackedLM, value: float = 0.0) -> None:
        super().__init__()
        values: list[float] = [value]
        self._w = torch.tensor(values, dtype=torch.float32)

    def save_pretrained(self: _DiskBackedLM, out_dir: str) -> None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        torch.save(self._w, Path(out_dir) / _WEIGHT_FILE)

    def read_back(self: _DiskBackedLM, path: str) -> None:
        """Load this fake's weight file into itself, in place.

        Named ``read_back`` rather than ``from_pretrained`` deliberately.
        The old name made the fake advertise a one-argument classmethod
        loader, which is the signature ``PeftModel`` does NOT have, and a
        fake carrying that shape is what let the real defect through: the
        line under test was covered while the behaviour was broken for
        every PEFT model. See ``services/training/reload.py``.
        """
        self._w = torch.load(Path(path) / _WEIGHT_FILE, weights_only=True)

    def state_dict(self: _DiskBackedLM) -> dict[str, torch.Tensor]:
        return {"w": self._w}

    def load_state_dict(self: _DiskBackedLM, state_dict: dict[str, torch.Tensor]) -> _DiskBackedLM:
        self._w = state_dict["w"]
        return self

    def weight(self: _DiskBackedLM) -> float:
        return float(self._w.item())


def _install_disk_reader(model: _DiskBackedLM) -> None:
    """Point the reload hook at this fake's own reader.

    ``_restore_best_checkpoint`` delegates the read to the hook, so a test
    of its contract supplies the reader. What is under test here is that the
    trainer reads from DISK and replaces the live weights; which reader a
    real artifact needs is covered in ``test_reload_shipped_weights.py``.

    Args:
        model: The fake whose file should be read back.
    """

    def _reader(
        prepared: PreparedLMModel,
        model_family: Literal["gpt2", "llama", "qwen", "char_lstm", "hf_lm"],
        path: str,
    ) -> None:
        model.read_back(path)

    _test_hooks.reload_shipped_weights = _reader


def _trainer_for(model: _DiskBackedLM) -> bt.BaseTrainer:
    """A trainer holding ``model``, ready for ``_restore_best_checkpoint``."""
    base = _make_prepared()
    prepared = PreparedLMModel(
        model=model,
        tokenizer_id=base.tokenizer_id,
        eos_id=base.eos_id,
        pad_id=base.pad_id,
        max_seq_len=base.max_seq_len,
        tok_for_dataset=base.tok_for_dataset,
    )
    trainer = bt.BaseTrainer(
        prepared,
        _make_cfg(),
        _make_settings(),
        run_id="test-best-checkpoint",
        redis_hb=lambda _: None,
        cancelled=lambda: False,
        resume=False,
        progress=None,
        service_name="char-lstm-train",
        determinism=UNPINNED,
    )
    trainer._device = torch.device("cpu")
    trainer._es_state = EarlyStoppingState(best_val_loss=1.0586, epochs_no_improve=3)
    return trainer


def test_restore_loads_the_shipped_weights_over_the_live_ones(tmp_path: Path) -> None:
    """The epoch on disk replaces the epoch in memory before scoring."""
    model = _DiskBackedLM(1.0)
    best_dir = tmp_path / "best"
    model.save_pretrained(str(best_dir))

    # Training continued past the best epoch: memory has moved on, disk has not.
    advanced: list[float] = [2.0]
    _ = model.load_state_dict({"w": torch.tensor(advanced, dtype=torch.float32)})
    if model.weight() != 2.0:
        raise AssertionError("fixture failed to advance the live weights")

    _install_disk_reader(model)
    trainer = _trainer_for(model)
    trainer._best_checkpoint_path = best_dir
    trainer._restore_best_checkpoint()

    assert model.weight() == 1.0


def test_restore_without_a_holdout_leaves_the_live_model_untouched(
    tmp_path: Path,
) -> None:
    """No best checkpoint means the live model already IS the artifact."""
    model = _DiskBackedLM(2.0)
    # A stale directory exists but was never registered as the best path.
    _DiskBackedLM(1.0).save_pretrained(str(tmp_path / "unused"))

    _install_disk_reader(model)
    trainer = _trainer_for(model)
    trainer._best_checkpoint_path = None
    trainer._restore_best_checkpoint()

    assert model.weight() == 2.0


def _write_corpus(root: Path) -> str:
    """A corpus whose holdout is adversarial to its training portion.

    ``split_corpus`` assigns lines in corpus order -- train, then validation,
    then test -- so putting a different alphabet in the tail 40% guarantees
    that fitting the training text makes validation WORSE. Without that, a
    uniform corpus drives validation down monotonically, the best epoch is
    the last one, and disk and memory agree whether or not the restore ran:
    the test would pass against the very defect it exists to catch.
    """
    out_dir = root / "corpus"
    out_dir.mkdir(parents=True, exist_ok=True)
    train_lines = ["abbbba", "aabbaa", "bbaaab", "aaabbb", "bbbaaa", "abababab"] * 2
    holdout_lines = ["cdcdcd", "ddccdd", "cddddc", "dccddc", "cdddcc", "dcdcdcdc"]
    body = "\n".join([*train_lines, *holdout_lines])
    (out_dir / "tiny.txt").write_text(body + "\n", encoding="utf-8")
    return str(out_dir)


def _train_tokenizer(root: Path, corpus_path: str) -> str:
    cfg = TokenizerTrainConfig(
        method="char",
        vocab_size=0,
        min_frequency=1,
        corpus_path=corpus_path,
        holdout_fraction=0.05,
        seed=42,
        out_dir=str(root / "artifacts" / "tokenizers" / "tokbest"),
    )
    _ = CharBackend().train(cfg)
    return "tokbest"


def test_a_holdout_run_ships_the_weights_it_scored(
    settings_with_paths: Settings, tmp_path: Path
) -> None:
    """End to end: after ``train``, the artifact and the live model agree.

    This is the wiring assertion the unit tests above cannot make -- that
    ``train`` performs the restore BEFORE its test evaluation rather than
    after it, or not at all.
    """
    corpus_path = _write_corpus(tmp_path)
    tokenizer_id = _train_tokenizer(tmp_path, corpus_path)
    cfg: ModelTrainConfig = {
        **_make_cfg(),
        "num_epochs": 30,
        "batch_size": 2,
        "max_seq_len": 16,
        "learning_rate": 5e-3,
        "corpus_path": corpus_path,
        "corpus_format": "lines",
        "tokenizer_id": tokenizer_id,
        "holdout_fraction": 0.2,
        "test_split_ratio": 0.2,
        "early_stopping_patience": 0,
    }

    backend = create_char_lstm_backend(LocalTextDatasetBuilder())
    tok_dir = Path(settings_with_paths["app"]["artifacts_root"]) / "tokenizers" / tokenizer_id
    handle = CharBackend().load(str(tok_dir / "tokenizer.json"))
    torch.manual_seed(cfg["seed"])
    prepared = backend.prepare(cfg, settings_with_paths, tokenizer=handle)

    val_by_epoch: dict[int, float] = {}
    train_by_epoch: dict[int, float] = {}

    def _capture(
        step: int,
        epoch: int,
        train_loss: float,
        train_ppl: float,
        grad_norm: float,
        samples_per_sec: float,
        val_loss: float | None,
        val_ppl: float | None,
    ) -> None:
        _ = (step, train_ppl, grad_norm, samples_per_sec, val_ppl)
        if val_loss is not None:
            val_by_epoch[epoch] = val_loss
            train_by_epoch[epoch] = train_loss

    outcome = backend.train(
        cfg,
        settings_with_paths,
        run_id="best-ckpt-e2e",
        heartbeat=lambda _: None,
        cancelled=lambda: False,
        resume=False,
        prepared=prepared,
        progress=_capture,
        determinism=UNPINNED,
    )

    # The premise: a holdout ran, so a best checkpoint was tracked.
    if outcome["best_val_loss"] is None:
        raise AssertionError("fixture ran no validation, so it cannot exercise the restore")
    if outcome["test_loss"] is None:
        raise AssertionError("fixture ran no test evaluation, so nothing was scored")

    # The premise that makes this test able to FAIL: the best epoch must not
    # be the last one. When they coincide, disk and memory agree whether or
    # not the restore happens, and the assertion below is vacuous.
    epochs = sorted(val_by_epoch)

    # The fixture only produces a validation turning point because the model
    # MEMORISES its twelve training lines. If it never learned them, the best
    # epoch below would be an artifact of noise rather than of overfitting.
    initial_loss = train_by_epoch[epochs[0]]
    final_loss = train_by_epoch[epochs[-1]]
    assert final_loss < initial_loss

    def _val_at(epoch_index: int) -> float:
        return val_by_epoch[epoch_index]

    best_epoch = min(epochs, key=_val_at)
    if best_epoch == epochs[-1]:
        raise AssertionError(
            f"fixture's best epoch is its last ({best_epoch}); "
            f"val curve {[(e, round(val_by_epoch[e], 4)) for e in epochs]} "
            "cannot distinguish a restored model from an unrestored one"
        )

    # Read the artifact back with the class that actually owns the format.
    # `prepared.model.from_pretrained` used to work here by accident: the
    # model protocol advertised a one-argument classmethod loader that a
    # PeftModel does not have, which is the defect this run's restore path
    # now avoids. A char-LSTM run ships a char-LSTM artifact, so the reader
    # is named rather than reached through the instance.
    reloaded: LMModelProto = CharLSTMModel.from_pretrained(
        str(model_dir(settings_with_paths, "best-ckpt-e2e"))
    )
    shipped: dict[str, torch.Tensor] = reloaded.state_dict()
    scored: dict[str, torch.Tensor] = prepared.model.state_dict()

    assert sorted(shipped.keys()) == sorted(scored.keys())
    for name in sorted(shipped.keys()):
        assert torch.equal(shipped[name], scored[name]), f"{name} differs between disk and memory"
