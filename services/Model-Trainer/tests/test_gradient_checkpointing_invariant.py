"""Every model about to be trained is put into its checkpointing posture.

THE DEFECT THESE PIN. Gradient checkpointing was enabled inside each
strategy's ``apply``, which runs only when a model is built fresh. A run
continuing from ``pretrained_run_id`` is rebuilt by ``load_adapted`` instead,
which never enabled it -- so continuations trained with every layer's
activations retained. Measured 2026-09-04 on HPC3 job 55764567: gpt2 (124M)
at batch 8, seq 512 took 22.84 GiB and died on a 24 GB A30, in the second
stage of a schedule whose first stage had trained on that same card without
trouble.

``load_adapted`` is deliberately not where this was fixed. The same loader
backs ``modeltrainer-score-run``, which reads a trained artifact for
inference; enabling checkpointing there would force ``use_cache=False`` on
the scorer and put a training concern in a reload path.

THE SECOND DEFECT, which the first hid: ``supports_gradient_checkpointing``
was declared by all four strategies and read by nothing outside their own
tests. Each strategy hard-coded the hook call instead, and ``cartridge``
expressed its "no" by omitting the call. The capability was documentation.
Reading it is what makes it load-bearing.
"""

from __future__ import annotations

from pathlib import Path

from model_trainer.core import _test_hooks
from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.model import ModelTrainConfig
from model_trainer.core.contracts.strategy_names import StrategyName
from model_trainer.core.contracts.tokenizer import TokenizerTrainConfig
from model_trainer.core.services.dataset.local_text_builder import LocalTextDatasetBuilder
from model_trainer.core.services.model.backend_factory import create_char_lstm_backend
from model_trainer.core.services.tokenizer.char_backend import CharBackend
from model_trainer.core.services.training import base_trainer as bt
from model_trainer.core.services.training.trainer_grad_utils import (
    _enable_gradient_checkpointing_if_supported,
)
from model_trainer.core.types import LMModelProto
from tests.core.services.finetuning.testing import FakeModel
from tests.core.services.model.backends.char_lstm._train_branches_support import (
    UNPINNED,
    _make_cfg,
)


class _RecordingModel(FakeModel):
    """The shared fake, plus the one observation these tests need.

    ``FakeModel.gradient_checkpointing_enable`` is a no-op that records
    nothing, so "was it called" is unanswerable through it. Subclassing keeps
    the rest of the fake's behaviour identical rather than forking a second
    model fake.
    """

    def __init__(self) -> None:
        """Initialise the fake with a zeroed call counter."""
        super().__init__("recording")
        self.checkpointing_calls = 0

    def gradient_checkpointing_enable(self) -> None:
        """Record the call the production path is expected to make."""
        self.checkpointing_calls += 1


class TestCapabilityIsConsulted:
    """The strategy's declared capability decides, not the call site."""

    def test_full_strategy_is_checkpointed(self) -> None:
        """A full finetune supports checkpointing, so it is enabled."""
        model = _RecordingModel()

        enabled = _enable_gradient_checkpointing_if_supported(model, "full")

        assert enabled is True
        assert model.checkpointing_calls == 1

    def test_lora_strategy_is_checkpointed(self) -> None:
        """LoRA declares support, so the trainer enables it."""
        model = _RecordingModel()

        assert _enable_gradient_checkpointing_if_supported(model, "lora") is True
        assert model.checkpointing_calls == 1

    def test_qlora_strategy_is_checkpointed(self) -> None:
        """QLoRA declares support, so the trainer enables it."""
        model = _RecordingModel()

        assert _enable_gradient_checkpointing_if_supported(model, "qlora") is True
        assert model.checkpointing_calls == 1

    def test_cartridge_strategy_is_not_checkpointed(self) -> None:
        """Cartridge declares no support, and the model is left untouched.

        Not a style preference. A checkpointed model discards the key-value
        cache it is handed (transformers 4.46.3), so the trained prefix never
        reaches attention -- the memory saving would be bought by not
        training the thing the run exists to train.
        """
        model = _RecordingModel()

        enabled = _enable_gradient_checkpointing_if_supported(model, "cartridge")

        assert enabled is False
        assert model.checkpointing_calls == 0

    def test_every_registered_strategy_is_decided(self) -> None:
        """No strategy falls through undecided.

        A new strategy that forgot to declare the capability would raise from
        the registry rather than silently training uncheckpointed, which is
        the failure mode this whole file exists to close.
        """
        for name in ("full", "lora", "qlora", "cartridge"):
            strategy: StrategyName = name
            model = _RecordingModel()
            enabled = _enable_gradient_checkpointing_if_supported(model, strategy)
            assert enabled == (model.checkpointing_calls == 1)


class TestDefaultHookDelegates:
    """The production hook is the helper, not a second implementation."""

    def test_default_hook_enables_for_a_supporting_strategy(self) -> None:
        """The bound default reaches the real helper and reports its answer."""
        from model_trainer.core._hook_defaults import _default_enable_gradient_checkpointing

        model = _RecordingModel()

        assert _default_enable_gradient_checkpointing(model, "full") is True
        assert model.checkpointing_calls == 1

    def test_default_hook_declines_for_cartridge(self) -> None:
        """The bound default carries the capability answer through unchanged."""
        from model_trainer.core._hook_defaults import _default_enable_gradient_checkpointing

        model = _RecordingModel()

        assert _default_enable_gradient_checkpointing(model, "cartridge") is False
        assert model.checkpointing_calls == 0

    def test_hook_is_bound_to_the_default(self) -> None:
        """The module-level hook is wired to the production implementation."""
        from model_trainer.core._hook_defaults import _default_enable_gradient_checkpointing

        assert _test_hooks.enable_gradient_checkpointing is _default_enable_gradient_checkpointing


class TestTrainerAppliesItOnEveryPath:
    """``BaseTrainer.train`` is the single site, so it must actually call."""

    def test_train_enables_checkpointing_with_the_configured_strategy(
        self,
        settings_with_paths: Settings,
        tmp_path: Path,
    ) -> None:
        """A real training run puts its model into checkpointing posture.

        This drives ``train`` end to end rather than asserting on the source,
        because the defect being closed was one of REACHABILITY: the old call
        existed and was correct, on a path half the runs did not take. Only a
        run can show that this one is reached.

        It asserts the call happens AND that it carries the run's own
        strategy. Passing the wrong strategy would still enable checkpointing
        for three of the four, so a test that only counted calls would pass
        while cartridge silently got checkpointed.
        """
        seen: list[tuple[LMModelProto, StrategyName]] = []
        losses: list[float] = []

        def fake_enable(model: LMModelProto, strategy: StrategyName) -> bool:
            seen.append((model, strategy))
            return True

        def capture(
            step: int,
            epoch: int,
            train_loss: float,
            train_ppl: float,
            grad_norm: float,
            samples_per_sec: float,
            val_loss: float | None,
            val_ppl: float | None,
        ) -> None:
            """Record each epoch's training loss so the run can be shown real."""
            losses.append(train_loss)

        corpus_dir = tmp_path / "corpus"
        corpus_dir.mkdir(parents=True, exist_ok=True)
        (corpus_dir / "tiny.txt").write_text("ababab\nbabababa\nabbaab\n" * 4, encoding="utf-8")
        tok_cfg = TokenizerTrainConfig(
            method="char",
            vocab_size=0,
            min_frequency=1,
            corpus_path=str(corpus_dir),
            holdout_fraction=0.05,
            seed=42,
            out_dir=str(tmp_path / "artifacts" / "tokenizers" / "tokgc"),
        )
        _ = CharBackend().train(tok_cfg)
        cfg: ModelTrainConfig = {
            **_make_cfg(),
            "corpus_path": str(corpus_dir),
            "tokenizer_id": "tokgc",
            "num_epochs": 12,
            "batch_size": 2,
            "max_seq_len": 8,
            "learning_rate": 5e-3,
        }
        tok_dir = Path(settings_with_paths["app"]["artifacts_root"]) / "tokenizers" / "tokgc"
        handle = CharBackend().load(str(tok_dir / "tokenizer.json"))
        backend = create_char_lstm_backend(LocalTextDatasetBuilder())
        prepared = backend.prepare(cfg, settings_with_paths, tokenizer=handle)

        original = _test_hooks.enable_gradient_checkpointing
        _test_hooks.enable_gradient_checkpointing = fake_enable
        try:
            trainer = bt.BaseTrainer(
                prepared,
                cfg,
                settings_with_paths,
                run_id="grad-ckpt-invariant",
                redis_hb=lambda _: None,
                cancelled=lambda: False,
                resume=False,
                progress=capture,
                service_name="char-lstm-train",
                determinism=UNPINNED,
            )
            trainer.train()
        finally:
            _test_hooks.enable_gradient_checkpointing = original

        assert len(seen) == 1
        called_model, called_strategy = seen[0]
        assert called_model is prepared.model
        assert called_strategy == cfg["finetuning_strategy"]

        # The run has to be a real one for the assertion above to mean
        # anything: a `train` that returned before reaching the loop would
        # also record exactly one hook call.
        assert len(losses) >= 2
        initial_loss = losses[0]
        final_loss = losses[-1]
        assert final_loss < initial_loss
