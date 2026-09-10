"""The solo-seeds CLI, exercised on a real tiny model over a fake corpus.

Same split as the headroom suite it sits beside: real tiny GPT-2, real
cartridge training and scoring, real window arithmetic; faked hub
loaders, corpus reader and plan table. The assertions concentrate on
what a reliability measurement must guarantee: one gain row per seed,
the mean and spread derived from exactly those rows, the recorded-knob
threading, the placement call, and the production seed tuple whose
first three are the original recorded draw.
"""

from __future__ import annotations

import pathlib
import runpy
import sys
from collections.abc import Sequence

import pytest
import torch
from platform_core.json_utils import load_json_str
from platform_core.run_record import decode_run_record

from model_trainer.cli import cartridge_solo_seeds as solo
from model_trainer.cli.cartridge_lora_policy import quantization_for
from model_trainer.core.contracts.model import QuantizationConfig, StoredBf16Precision
from model_trainer.core.services.finetuning.strategies.cartridge import require_cache_capable
from model_trainer.core.services.finetuning.strategies.cartridge_model import CartridgeModel
from model_trainer.core.services.model.backends.hf_lm import _test_hooks as hf_hooks
from model_trainer.core.services.model.cartridge_corpus import build_windows, split_by_stride
from model_trainer.core.services.model.cartridge_measurement import (
    held_out_gain,
    train_cartridge,
)
from model_trainer.core.services.model.known_answer_probe import probe_model_and_input
from model_trainer.core.services.model.probe_shapes import PROBE_SHAPES
from model_trainer.core.types import (
    CacheCapableLMProto,
    CacheCarryingOutProto,
    ConfigLike,
    EmbeddingModuleProto,
    ForwardOutProto,
    KVCacheProto,
    LMModelProto,
    LoadStateDictResultProto,
    NamedParameter,
    ParameterLike,
)
from tests._solo_seeds_harness import documents, fake_model, fake_tokenizer, staged, wired

#: Imported for the autouse fixture's side effect; pytest collects it from
#: this module's namespace, and naming it in ``__all__`` is what makes that
#: deliberate rather than an unused-import that a later cleanup deletes.
__all__ = ["wired"]


class _PlacementRecordingBase:
    """A real base behind a typed shim, recording where it is placed.

    Training needs a real model, so the headroom suite's fake cannot
    carry this assertion here; every member delegates and only ``to``
    observes.
    """

    def __init__(self, inner: CacheCapableLMProto, placed: list[str]) -> None:
        """Wrap the inner model.

        Args:
            inner: The real model everything delegates to.
            placed: Where ``to`` calls are recorded.
        """
        self._inner = inner
        self._placed = placed

    @classmethod
    def from_pretrained(cls, path: str) -> LMModelProto:
        """Refuse: a recording wrapper wraps, it does not load.

        Args:
            path: Ignored.

        Raises:
            ValueError: Always.
        """
        raise ValueError(f"a recording wrapper is not loadable (got {path!r})")

    def to(self, device: str) -> LMModelProto:
        """Record the placement and pass it through.

        Args:
            device: Target device.

        Returns:
            Self.
        """
        self._placed.append(device)
        self._inner.to(device)
        return self

    def train(self) -> None:
        """Delegate."""
        self._inner.train()

    def eval(self) -> None:
        """Delegate."""
        self._inner.eval()

    def forward(self, *, input_ids: torch.Tensor, labels: torch.Tensor) -> ForwardOutProto:
        """Delegate.

        Args:
            input_ids: Token ids.
            labels: Targets.

        Returns:
            The inner model's output.
        """
        return self._inner.forward(input_ids=input_ids, labels=labels)

    def __call__(
        self,
        *,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        past_key_values: KVCacheProto | None = None,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> CacheCarryingOutProto:
        """Delegate the cached-forward call.

        Args:
            input_ids: Token ids.
            labels: Targets, or None.
            past_key_values: Prefix cache, or None.
            attention_mask: Mask, or None.
            use_cache: Whether a cache is asked for.

        Returns:
            The inner model's output.
        """
        return self._inner(
            input_ids=input_ids,
            labels=labels,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            use_cache=use_cache,
        )

    def parameters(self) -> Sequence[ParameterLike]:
        """Delegate.

        Returns:
            The inner model's parameters.
        """
        return self._inner.parameters()

    def named_parameters(self) -> Sequence[tuple[str, NamedParameter]]:
        """Delegate.

        Returns:
            The inner model's named parameters.
        """
        return self._inner.named_parameters()

    def save_pretrained(self, out_dir: str) -> None:
        """Delegate.

        Args:
            out_dir: Output directory.
        """
        self._inner.save_pretrained(out_dir)

    def gradient_checkpointing_enable(self) -> None:
        """Delegate."""
        self._inner.gradient_checkpointing_enable()

    @property
    def config(self) -> ConfigLike:
        """Delegate.

        Returns:
            The inner model's config.
        """
        return self._inner.config

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Delegate.

        Returns:
            The inner model's state dict.
        """
        return self._inner.state_dict()

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> LoadStateDictResultProto:
        """Delegate.

        Args:
            state_dict: The state to load.

        Returns:
            The inner model's result.
        """
        return self._inner.load_state_dict(state_dict)

    def get_input_embeddings(self) -> EmbeddingModuleProto:
        """Delegate.

        Returns:
            The inner model's embedding module.
        """
        return self._inner.get_input_embeddings()


class TestMeasureSoloSeeds:
    def test_one_gain_row_per_seed_and_the_summary_rows_derive_from_them(
        self, tmp_path: pathlib.Path
    ) -> None:
        alpha = staged(tmp_path, "alpha")

        observations, _digest = solo.measure_solo_seeds(
            alpha,
            model_id="gpt2",
            load_precision=None,
            seeds=(7, 8),
            device="cpu",
            checkpoints=tmp_path / "ckpt",
        )

        recorded = {o["name"]: o["value"] for o in observations}
        names = [o["name"] for o in observations]
        assert len(names) == len(set(names))
        gains = [recorded["solo-gpt2-seed7_gain"], recorded["solo-gpt2-seed8_gain"]]
        assert recorded["solo-gpt2_gain_mean"] == sum(gains) / 2
        assert recorded["solo-gpt2_gain_spread"] == max(gains) - min(gains)
        assert recorded["solo-gpt2_characters"] == float(
            sum(len(document) for document in documents("a"))
        )
        assert recorded["solo-gpt2_train_windows"] > 0
        assert recorded["solo-gpt2_held_out_windows"] > 0

    def test_each_gain_equals_an_independent_recomputation(self, tmp_path: pathlib.Path) -> None:
        alpha = staged(tmp_path, "alpha")

        observations, _digest = solo.measure_solo_seeds(
            alpha,
            model_id="gpt2",
            load_precision=None,
            seeds=(7,),
            device="cpu",
            checkpoints=tmp_path / "ckpt",
        )
        recorded = {o["name"]: o["value"] for o in observations}

        tokenizer = fake_tokenizer("gpt2")
        encoded = [tokenizer.encode(document) for document in documents("a")]
        train, held_out = split_by_stride(
            build_windows(encoded, window=8, device="cpu"), held_out_stride=3
        )
        base = require_cache_capable(fake_model("gpt2", None))
        slots = train_cartridge(base, train, num_slots=2, seed=7, epochs=1, learning_rate=0.05)
        expected = held_out_gain(CartridgeModel(base=base, slots=slots), held_out)

        assert recorded["solo-gpt2-seed7_gain"] == expected

    def test_the_measurement_reproduces_itself(self, tmp_path: pathlib.Path) -> None:
        alpha = staged(tmp_path, "alpha")

        # SEPARATE CHECKPOINT DIRECTORIES, and this is load-bearing. Sharing
        # one would let the second call RESUME the first's cells and return
        # them verbatim, so the test would pass without re-measuring anything
        # -- it would assert that a dict equals itself. Distinct directories
        # force the second run to retrain and rescore, which is the property
        # the test is named for.
        first, _ = solo.measure_solo_seeds(
            alpha,
            model_id="gpt2",
            load_precision=None,
            seeds=(7, 8),
            device="cpu",
            checkpoints=tmp_path / "ckpt-first",
        )
        second, _ = solo.measure_solo_seeds(
            alpha,
            model_id="gpt2",
            load_precision=None,
            seeds=(7, 8),
            device="cpu",
            checkpoints=tmp_path / "ckpt-second",
        )

        assert first == second

    def test_no_seeds_is_refused(self, tmp_path: pathlib.Path) -> None:
        alpha = staged(tmp_path, "alpha")
        with pytest.raises(ValueError, match="no seeds named"):
            solo.measure_solo_seeds(
                alpha,
                model_id="gpt2",
                load_precision=None,
                seeds=(),
                device="cpu",
                checkpoints=tmp_path / "ckpt",
            )

    def test_the_loaded_model_is_moved_to_the_measurement_device(
        self, tmp_path: pathlib.Path
    ) -> None:
        alpha = staged(tmp_path, "alpha")
        placed: list[str] = []

        def _recording_loader(
            model_id_or_path: str, quantization: QuantizationConfig | StoredBf16Precision | None
        ) -> LMModelProto:
            inner = require_cache_capable(fake_model(model_id_or_path, quantization))
            return _PlacementRecordingBase(inner, placed)

        hf_hooks.Hooks.load_hf_model = _recording_loader

        solo.measure_solo_seeds(
            alpha,
            model_id="gpt2",
            load_precision=None,
            seeds=(7,),
            device="cpu",
            checkpoints=tmp_path / "ckpt",
        )

        # The measurement chain may place the base again downstream; the
        # load-bearing claim is that the FIRST placement is the requested
        # device and nothing ever places it anywhere else.
        assert placed[0] == "cpu"
        assert set(placed) == {"cpu"}


class TestSeedsPin:
    def test_the_production_seeds_extend_the_original_draw(self) -> None:
        # The full-tuple pin carries the subset property: the first three
        # ARE the recorded draw, per the gpt2-wiki-9seed precedent.
        assert solo.SOLO_SEEDS == (7, 8, 9, 10, 11, 12, 13, 14, 15)

    def test_the_label_names_the_base_and_the_draw_count(self) -> None:
        label = solo.solo_seeds_label(
            model_id="EleutherAI/pythia-6.9b",
            seeds=solo.SOLO_SEEDS,
            precision_token="",
            digest="0" * 12,
        )
        assert label == "cartridge-solo-seeds-pythia-6.9b-n9-000000000000"

    def test_the_stored_bf16_token_unpairs_the_label(self) -> None:
        label = solo.solo_seeds_label(
            model_id="EleutherAI/pythia-6.9b",
            seeds=solo.SOLO_SEEDS,
            precision_token="-storedbf16",
            digest="0" * 12,
        )
        assert label == "cartridge-solo-seeds-pythia-6.9b-storedbf16-n9-000000000000"


class TestResolvePrecision:
    def test_policy_mode_is_the_policy_value_and_an_empty_token(self) -> None:
        assert solo.resolve_precision("gpt2", "policy") == (None, "")
        load, token = solo.resolve_precision("EleutherAI/pythia-6.9b", "policy")
        assert load == quantization_for("EleutherAI/pythia-6.9b")
        assert token == ""

    def test_stored_bf16_is_the_declared_load_and_its_token(self) -> None:
        assert solo.resolve_precision("EleutherAI/pythia-6.9b", "stored-bf16") == (
            {"torch_dtype": "bfloat16"},
            "-storedbf16",
        )

    def test_stored_bf16_refuses_an_undeclared_base(self) -> None:
        with pytest.raises(ValueError, match="no stored-bf16 load is declared"):
            solo.resolve_precision("gpt2", "stored-bf16")

    def test_an_unknown_selector_is_refused(self) -> None:
        with pytest.raises(ValueError, match="unknown precision selector"):
            solo.resolve_precision("gpt2", "fp8")


class TestMain:
    def test_main_writes_the_record(self, tmp_path: pathlib.Path) -> None:
        alpha = staged(tmp_path, "alpha")
        out = tmp_path / "record" / "solo.json"

        code = solo.main(
            [
                "--model-id",
                "gpt2",
                "--precision",
                "policy",
                "--corpus",
                str(alpha),
                "--device",
                "cpu",
                "--out",
                str(out),
            ]
        )

        assert code == 0
        record = decode_run_record(load_json_str(out.read_text(encoding="utf-8")))
        assert record["experiment"] == solo.SOLO_SEEDS_EXPERIMENT
        assert record["label"].startswith("cartridge-solo-seeds-gpt2-n9-")
        names = {o["name"] for o in record["observations"]}
        for seed in solo.SOLO_SEEDS:
            assert f"solo-gpt2-seed{seed}_gain" in names

    def test_stored_bf16_reaches_the_loader_and_the_label(self, tmp_path: pathlib.Path) -> None:
        alpha = staged(tmp_path, "alpha")
        out = tmp_path / "bf16" / "solo.json"
        received: list[QuantizationConfig | StoredBf16Precision | None] = []

        def _bf16_loader(
            model_id_or_path: str,
            quantization: QuantizationConfig | StoredBf16Precision | None,
        ) -> LMModelProto:
            assert model_id_or_path == "EleutherAI/pythia-6.9b"
            received.append(quantization)
            model, _ids = probe_model_and_input("cpu", PROBE_SHAPES["tiny"])
            return model

        hf_hooks.Hooks.load_hf_model = _bf16_loader

        code = solo.main(
            [
                "--model-id",
                "EleutherAI/pythia-6.9b",
                "--precision",
                "stored-bf16",
                "--corpus",
                str(alpha),
                "--device",
                "cpu",
                "--out",
                str(out),
            ]
        )

        assert code == 0
        assert received == [{"torch_dtype": "bfloat16"}]
        record = decode_run_record(load_json_str(out.read_text(encoding="utf-8")))
        assert record["label"].startswith("cartridge-solo-seeds-pythia-6.9b-storedbf16-n9-")

    def test_entrypoint_reads_process_argv_and_exits_with_mains_code(
        self, tmp_path: pathlib.Path
    ) -> None:
        alpha = staged(tmp_path, "alpha")
        out = tmp_path / "solo.json"
        saved = sys.argv
        sys.argv = [
            "prog",
            "--model-id",
            "gpt2",
            "--precision",
            "policy",
            "--corpus",
            str(alpha),
            "--device",
            "cpu",
            "--out",
            str(out),
        ]
        try:
            with pytest.raises(SystemExit) as caught:
                solo.entrypoint()
        finally:
            sys.argv = saved

        assert caught.value.code == 0
        assert out.exists()

    def test_running_it_as_a_module_actually_measures(self, tmp_path: pathlib.Path) -> None:
        alpha = staged(tmp_path, "alpha")
        out = tmp_path / "module" / "solo.json"
        module_name = "model_trainer.cli.cartridge_solo_seeds"
        saved_argv = sys.argv
        saved_module = sys.modules.pop(module_name, None)
        sys.argv = [
            "x",
            "--model-id",
            "gpt2",
            "--precision",
            "policy",
            "--corpus",
            str(alpha),
            "--device",
            "cpu",
            "--out",
            str(out),
        ]
        try:
            with pytest.raises(SystemExit) as raised:
                runpy.run_module(module_name, run_name="__main__", alter_sys=False)
        finally:
            sys.argv = saved_argv
            if saved_module is not None:
                sys.modules[module_name] = saved_module

        assert raised.value.code == 0
        assert out.is_file()
