"""The headroom CLI, exercised on a real tiny model over fake corpora.

Same split as the sweep suites it sits beside: real tiny GPT-2, real
window/split arithmetic, real scoring; faked hub loaders and corpus
reader. What the assertions concentrate on is what this record exists
for: the base-loss rows subtract against the recorded gains, so the mean
must equal an independent recomputation over the same split, the counts
must be exact, and the production base list must resolve through the
loading policy that every run document relies on.
"""

from __future__ import annotations

import pathlib
import runpy
import sys
from collections.abc import Generator

import pytest
import torch
from platform_core.json_utils import load_json_str
from platform_core.run_record import decode_run_record

from model_trainer.cli import _measurement_hooks as measurement_hooks
from model_trainer.cli import _test_hooks as cli_hooks
from model_trainer.cli import cartridge_headroom as headroom
from model_trainer.cli.cartridge_lora_policy import quantization_for
from model_trainer.core.contracts.model import QuantizationConfig
from model_trainer.core.services.finetuning.strategies.cartridge import require_cache_capable
from model_trainer.core.services.model.backends.hf_lm import _test_hooks as hf_hooks
from model_trainer.core.services.model.backends.hf_lm._hook_protocols import HFTokenizerProto
from model_trainer.core.services.model.cartridge_corpus import build_windows, split_by_stride
from model_trainer.core.services.model.cartridge_pool_plans import (
    BASE_LORA_SWEEP_PLANS,
    BaseLoraSweepPlan,
)
from model_trainer.core.services.model.cartridge_scoring import base_loss
from model_trainer.core.services.model.known_answer_probe import probe_model_and_input
from model_trainer.core.services.model.probe_shapes import PROBE_SHAPES
from model_trainer.core.types import LMModelProto

_VOCAB = PROBE_SHAPES["tiny"]["vocab_size"]

#: Small enough to run in a test, large enough that every corpus fills
#: windows at both the tiny geometry and the production one (one fake
#: token per character).
_DOCUMENT_CHARS = 520


def _fake_tokenizer(model_id_or_path: str) -> HFTokenizerProto:
    """Stand in for the hub tokenizer loader, for any measured base."""
    assert model_id_or_path in headroom.MEASURED_BASES
    from tests.core.services.model.backends.hf_lm.testing import FakeHFTokenizer

    return FakeHFTokenizer(vocab_size=_VOCAB)


def _fake_model(model_id_or_path: str, quantization: QuantizationConfig | None) -> LMModelProto:
    """Stand in for the hub loader, asserting the policy value threads.

    The loader is handed exactly what :func:`quantization_for` declares
    for the id -- None for the gpt2 family, the NF4 block for the 7B --
    and returns the same real tiny GPT-2 either way, because what this
    suite tests is the measurement around the model, not quantization.
    """
    assert model_id_or_path in headroom.MEASURED_BASES
    assert quantization == quantization_for(model_id_or_path)
    model, _ids = probe_model_and_input("cpu", PROBE_SHAPES["tiny"])
    return model


def _documents(marker: str) -> tuple[str, ...]:
    """Two documents per corpus, distinct by marker character."""
    return tuple(f"{marker}{index}" * (_DOCUMENT_CHARS // 2) for index in range(2))


def _fake_corpus_reader(corpus_dir: pathlib.Path, /) -> tuple[str, ...]:
    """Stand in for the corpus reader, keyed on the directory's name."""
    return _documents(corpus_dir.name[0])


#: The geometry the record-building path reads through the plan hook,
#: shrunk so the tiny probe model's position table can host a window.
#: Only ``window`` and ``held_out_stride`` are read by the headroom path;
#: the rest keeps the row a valid plan.
TINY_GEOMETRY_PLAN: BaseLoraSweepPlan = {
    "model_id": "gpt2",
    "window": 8,
    "held_out_stride": 3,
    "compartment_counts": (2, 3),
    "slots": 2,
    "probability": 0.5,
    "max_companions": 2,
    "lora_rank": 2,
    "lora_alpha": 4,
    "lora_epochs": 1,
    "lora_learning_rate": 0.05,
    "max_drawn": 2,
    "pool_members_per_corpus": 1,
    "seeds": (7, 8, 9),
    "epochs": 1,
    "learning_rate": 0.05,
}


def _fake_plans() -> dict[str, BaseLoraSweepPlan]:
    """Stand in for the production plan table, geometry row included."""
    return {headroom.GEOMETRY_PLAN_NAME: TINY_GEOMETRY_PLAN}


@pytest.fixture(name="wired", autouse=True)
def _wired() -> Generator[None, None, None]:
    """Install the fakes, and put the real hooks back afterwards."""
    measurement_hooks.base_lora_sweep_plans = _fake_plans
    cli_hooks.read_corpus_documents = _fake_corpus_reader
    hf_hooks.Hooks.load_hf_tokenizer = _fake_tokenizer
    hf_hooks.Hooks.load_hf_model = _fake_model
    yield None
    measurement_hooks.base_lora_sweep_plans = measurement_hooks._default_base_lora_sweep_plans
    cli_hooks.read_corpus_documents = cli_hooks._default_read_corpus_documents
    hf_hooks.Hooks.reset()


def _staged(tmp_path: pathlib.Path, names: tuple[str, ...]) -> list[pathlib.Path]:
    """Create one directory per corpus name."""
    created: list[pathlib.Path] = []
    for name in names:
        path = tmp_path / name
        path.mkdir()
        created.append(path)
    return created


class TestMeasureHeadroom:
    def test_every_row_is_named_once_and_the_grid_is_complete(self, tmp_path: pathlib.Path) -> None:
        alpha, beta = _staged(tmp_path, ("alpha", "beta"))

        observations, _digest = headroom.measure_headroom(
            [alpha, beta],
            bases=("gpt2", "gpt2-medium"),
            window=8,
            held_out_stride=3,
            device="cpu",
        )

        names = [observation["name"] for observation in observations]
        assert len(names) == len(set(names))
        for corpus in ("alpha", "beta"):
            assert f"headroom-{corpus}_characters" in names
            assert f"headroom-{corpus}_documents" in names
            for short in ("gpt2", "gpt2-medium"):
                assert f"headroom-{short}-{corpus}_base_loss_mean" in names
                assert f"headroom-{short}-{corpus}_held_out_windows" in names
                assert f"headroom-{short}-{corpus}_held_out_tokens" in names
        assert len(names) == 2 * 2 + 2 * 2 * 3

    def test_the_mean_equals_an_independent_recomputation(self, tmp_path: pathlib.Path) -> None:
        (alpha,) = _staged(tmp_path, ("alpha",))

        observations, _digest = headroom.measure_headroom(
            [alpha], bases=("gpt2",), window=8, held_out_stride=3, device="cpu"
        )
        recorded = {o["name"]: o["value"] for o in observations}

        tokenizer = _fake_tokenizer("gpt2")
        encoded = [tokenizer.encode(document) for document in _documents("a")]
        _train, held_out = split_by_stride(
            build_windows(encoded, window=8, device="cpu"), held_out_stride=3
        )
        model = require_cache_capable(_fake_model("gpt2", None))
        model.eval()
        expected = sum(base_loss(model, item) for item in held_out) / len(held_out)

        assert recorded["headroom-gpt2-alpha_base_loss_mean"] == expected
        assert recorded["headroom-gpt2-alpha_held_out_windows"] == float(len(held_out))
        assert recorded["headroom-gpt2-alpha_held_out_tokens"] == float(len(held_out) * 8)
        assert recorded["headroom-alpha_characters"] == float(
            sum(len(document) for document in _documents("a"))
        )
        assert recorded["headroom-alpha_documents"] == 2.0

    def test_no_corpora_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no corpora named"):
            headroom.measure_headroom(
                [], bases=("gpt2",), window=8, held_out_stride=3, device="cpu"
            )

    def test_no_bases_is_refused(self, tmp_path: pathlib.Path) -> None:
        (alpha,) = _staged(tmp_path, ("alpha",))
        with pytest.raises(ValueError, match="no bases named"):
            headroom.measure_headroom([alpha], bases=(), window=8, held_out_stride=3, device="cpu")


class TestPolicyPin:
    def test_the_measured_bases_are_the_recorded_ladder(self) -> None:
        assert headroom.MEASURED_BASES == (
            "gpt2",
            "gpt2-medium",
            "gpt2-xl",
            "EleutherAI/pythia-6.9b",
        )

    def test_every_measured_base_resolves_through_the_loading_policy(self) -> None:
        for model_id in headroom.MEASURED_BASES:
            quantization_for(model_id)

    def test_base_short_names_the_row_segment(self) -> None:
        assert headroom.base_short("gpt2-xl") == "gpt2-xl"
        assert headroom.base_short("EleutherAI/pythia-6.9b") == "pythia-6.9b"

    def test_the_production_geometry_row_is_the_recorded_one(self) -> None:
        row = BASE_LORA_SWEEP_PLANS[headroom.GEOMETRY_PLAN_NAME]
        assert row["window"] == 256
        assert row["held_out_stride"] == 4


class TestLabel:
    def test_the_label_carries_the_geometry_and_the_counts(self) -> None:
        label = headroom.headroom_label(
            bases=("gpt2", "gpt2-xl"),
            corpora=[pathlib.Path("a"), pathlib.Path("b"), pathlib.Path("c")],
            window=256,
            held_out_stride=4,
            digest="0" * 12,
        )
        assert label == "cartridge-headroom-w256-s4-b2-c3-000000000000"


class TestMain:
    def test_main_measures_every_base_and_writes_the_record(self, tmp_path: pathlib.Path) -> None:
        alpha, beta = _staged(tmp_path, ("alpha", "beta"))
        out = tmp_path / "record" / "headroom.json"

        code = headroom.main(
            [
                "--corpora",
                f"{alpha},{beta}",
                "--device",
                "cpu",
                "--out",
                str(out),
            ]
        )

        assert code == 0
        record = decode_run_record(load_json_str(out.read_text(encoding="utf-8")))
        assert record["experiment"] == headroom.HEADROOM_EXPERIMENT
        assert record["label"].startswith("cartridge-headroom-w8-s3-b4-c2-")
        names = {o["name"] for o in record["observations"]}
        for model_id in headroom.MEASURED_BASES:
            short = headroom.base_short(model_id)
            assert f"headroom-{short}-alpha_base_loss_mean" in names
            assert f"headroom-{short}-beta_base_loss_mean" in names

    def test_entrypoint_reads_process_argv_and_exits_with_mains_code(
        self, tmp_path: pathlib.Path
    ) -> None:
        (alpha,) = _staged(tmp_path, ("alpha",))
        out = tmp_path / "headroom.json"
        saved = sys.argv
        sys.argv = ["prog", "--corpora", str(alpha), "--device", "cpu", "--out", str(out)]
        try:
            with pytest.raises(SystemExit) as caught:
                headroom.entrypoint()
        finally:
            sys.argv = saved

        assert caught.value.code == 0
        assert out.exists()

    def test_running_it_as_a_module_actually_measures(self, tmp_path: pathlib.Path) -> None:
        (alpha,) = _staged(tmp_path, ("alpha",))
        out = tmp_path / "module" / "headroom.json"
        module_name = "model_trainer.cli.cartridge_headroom"
        saved_argv = sys.argv
        saved_module = sys.modules.pop(module_name, None)
        sys.argv = ["x", "--corpora", str(alpha), "--device", "cpu", "--out", str(out)]
        try:
            with pytest.raises(SystemExit) as raised:
                runpy.run_module(module_name, run_name="__main__", alter_sys=False)
        finally:
            sys.argv = saved_argv
            if saved_module is not None:
                sys.modules[module_name] = saved_module

        assert raised.value.code == 0
        assert out.is_file()


class TestBaseLoss:
    def test_base_loss_is_the_models_own_forward_loss(self) -> None:
        raw, ids = probe_model_and_input("cpu", PROBE_SHAPES["tiny"])
        model = require_cache_capable(raw)
        model.eval()
        with torch.no_grad():
            expected = float(model(input_ids=ids, labels=ids).loss.item())
        assert base_loss(model, ids) == expected
