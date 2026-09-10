"""The fakes the solo-seeds suites share, in one place rather than two.

EXTRACTED WHEN THE SUITE SPLIT, not copied into the second file. The suite
went over the 600-line ceiling once the eviction tests arrived and split by
role -- what the sweep MEASURES in one module, how it SURVIVES AN EVICTION in
the other -- and both need the same tiny plan, the same fake loaders and the
same corpus reader.

Copying them would have been the cheaper edit and the wrong one: these fakes
encode what the production path reads (the plan row's geometry, the policy's
precision value, the corpus the digest is taken over), and two copies drifting
would let one suite pass against a world the other no longer describes. The
resume tests in particular assert on a digest computed from ``_documents``, so
that function being the SAME function the corpus reader serves is load-bearing.

Not a ``conftest.py``: these are imported by name, so a reader of either suite
can see where its world comes from.
"""

from __future__ import annotations

import pathlib
from collections.abc import Generator

import pytest

from model_trainer.cli import _measurement_hooks as measurement_hooks
from model_trainer.cli import _test_hooks as cli_hooks
from model_trainer.cli.cartridge_headroom import GEOMETRY_PLAN_NAME
from model_trainer.cli.cartridge_lora_policy import quantization_for
from model_trainer.core.contracts.model import QuantizationConfig, StoredBf16Precision
from model_trainer.core.services.model.backends.hf_lm import _test_hooks as hf_hooks
from model_trainer.core.services.model.backends.hf_lm._hook_protocols import HFTokenizerProto
from model_trainer.core.services.model.cartridge_pool_plans import BaseLoraSweepPlan
from model_trainer.core.services.model.known_answer_probe import probe_model_and_input
from model_trainer.core.services.model.probe_shapes import PROBE_SHAPES
from model_trainer.core.types import LMModelProto
from tests.core.services.model.backends.hf_lm.testing import FakeHFTokenizer

VOCAB = PROBE_SHAPES["tiny"]["vocab_size"]

DOCUMENT_CHARS = 96

#: Tiny knobs behind the same plan-hook seam the production path reads;
#: only window/stride/slots/epochs/learning_rate are consumed here.
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


def fake_plans() -> dict[str, BaseLoraSweepPlan]:
    """Stand in for the production plan table, geometry row included.

    Returns:
        One plan, under the name the production path resolves.
    """
    return {GEOMETRY_PLAN_NAME: TINY_GEOMETRY_PLAN}


def fake_tokenizer(model_id_or_path: str) -> HFTokenizerProto:
    """Stand in for the hub tokenizer loader.

    Args:
        model_id_or_path: The base being loaded.

    Returns:
        A fake tokenizer over the tiny vocabulary.
    """
    assert model_id_or_path in ("gpt2", "gpt2-xl", "EleutherAI/pythia-6.9b")
    return FakeHFTokenizer(vocab_size=VOCAB)


def fake_model(
    model_id_or_path: str, quantization: QuantizationConfig | StoredBf16Precision | None
) -> LMModelProto:
    """Stand in for the hub loader, asserting the policy value threads.

    Args:
        model_id_or_path: The base being loaded.
        quantization: What the CLI handed the loader.

    Returns:
        A real tiny GPT-2.
    """
    assert quantization == quantization_for(model_id_or_path)
    model, _ids = probe_model_and_input("cpu", PROBE_SHAPES["tiny"])
    return model


def documents(marker: str) -> tuple[str, ...]:
    """Two documents, distinct by marker character.

    Args:
        marker: Character that makes this corpus different from another.

    Returns:
        The corpus bodies, in the order they will be windowed.
    """
    return tuple(f"{marker}{index}" * (DOCUMENT_CHARS // 2) for index in range(2))


def fake_corpus_reader(corpus_dir: pathlib.Path, /) -> tuple[str, ...]:
    """Stand in for the corpus reader, keyed on the directory's name.

    Args:
        corpus_dir: The directory a CLI was pointed at.

    Returns:
        The documents that directory stands for.
    """
    return documents(corpus_dir.name[0])


@pytest.fixture(name="wired", autouse=True)
def wired() -> Generator[None, None, None]:
    """Install the fakes, and put the real hooks back afterwards.

    Yields:
        None, once the hooks are the fakes.
    """
    measurement_hooks.base_lora_sweep_plans = fake_plans
    cli_hooks.read_corpus_documents = fake_corpus_reader
    hf_hooks.Hooks.load_hf_tokenizer = fake_tokenizer
    hf_hooks.Hooks.load_hf_model = fake_model
    yield None
    measurement_hooks.base_lora_sweep_plans = measurement_hooks._default_base_lora_sweep_plans
    cli_hooks.read_corpus_documents = cli_hooks._default_read_corpus_documents
    hf_hooks.Hooks.reset()


def staged(tmp_path: pathlib.Path, name: str) -> pathlib.Path:
    """Create one corpus directory.

    Args:
        tmp_path: The test's temporary directory.
        name: Directory name; the fake reader keys corpora on it.

    Returns:
        The created path.
    """
    path = tmp_path / name
    path.mkdir()
    return path


__all__ = [
    "DOCUMENT_CHARS",
    "TINY_GEOMETRY_PLAN",
    "VOCAB",
    "documents",
    "fake_corpus_reader",
    "fake_model",
    "fake_plans",
    "fake_tokenizer",
    "staged",
    "wired",
]
