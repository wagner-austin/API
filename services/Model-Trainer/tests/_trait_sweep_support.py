"""Shared fakes for the trait-composition sweep's tests.

Extracted when the solo precondition became a recorded verdict and its tests
moved to `test_cartridge_trait_precondition.py`, which would have pushed the
sweep's test module past the 600-line ceiling. A pytest fixture is not
visible across modules, so the install/restore pair lives here and each
module wraps it in its own autouse fixture -- the pattern
`_qa_benchmark_support.py` set for the question-set benchmark.

WHAT IS FAKED AND WHY. The arms are REAL -- a real tiny GPT-2, real
cartridges trained and composed, real contrastive directions read off real
activations -- and the faked seams are the hub loaders, the trait reader and
the plan table, because the production plan is GPU-minutes of cartridges over
a base these suites must not download.
"""

from __future__ import annotations

import pathlib
from collections.abc import Mapping, Sequence

from platform_core.power_distributions import McNemarTest

from model_trainer.cli import _measurement_hooks, _trait_hooks
from model_trainer.core.contracts.model import QuantizationConfig, StoredBf16Precision
from model_trainer.core.contracts.trait_corpus import TraitCorpus, TraitPairSpec
from model_trainer.core.contracts.trait_plan import TraitPlan
from model_trainer.core.services.model.backends.hf_lm import _test_hooks as hf_hooks
from model_trainer.core.services.model.backends.hf_lm._hook_protocols import HFTokenizerProto
from model_trainer.core.services.model.known_answer_probe import probe_model_and_input
from model_trainer.core.services.model.probe_shapes import PROBE_SHAPES
from model_trainer.core.types import LMModelProto
from tests.core.services.model.backends.hf_lm.testing import FakeHFTokenizer

#: The site the tiny probe model actually has, tensor-valued so a direction
#: can be both read and applied there.
SITE = "transformer.h.0.mlp.c_proj"

#: Three traits so an n3 cell exists, two counts so the step verdicts have a
#: pair to compare, and three seeds because fewer is refused. Twelve pairs at
#: stride two hold out six, which is exactly the fewest that can ever reject
#: at alpha 0.05 under the exact test -- so the pair floor is 1.0 and the
#: per-pair gate passes by the narrowest margin it can. The pilot spread is
#: chosen so three seeds resolve the derived SEI: at an sd of 0.01 the
#: three-seed MDE is 0.0248 against an SEI of 0.0526.
TINY_TRAIT_PLAN: TraitPlan = {
    "model_id": "gpt2",  # a real policy id (the fakes return a tiny GPT-2 anyway)
    "traits": ("bullets", "formal-tone", "step-by-step"),
    "held_out_stride": 2,
    "max_seq_len": 32,
    "slots": 2,
    "seeds": (7, 8, 9),
    "epochs": 1,
    "learning_rate": 0.05,
    "compartment_counts": (2, 3),
    "pair_test_floor": 1.0,
    "acted_on_retention": 0.0526,
    "pilot_alone_gain": 1.0,
    "pilot_paired_differences": (0.01, 0.02, 0.03),
    "alpha": 0.05,
    "mcnemar_test": McNemarTest.EXACT,
    "steering_module": SITE,
    "steering_strengths": (1.0, 10.0),
    "steering_coherence_bar": 1000.0,
}

_VOCAB = PROBE_SHAPES["tiny"]["vocab_size"]

#: The character that makes each trait's text unlike the others'.
_MARKERS = {"bullets": "a", "formal-tone": "b", "step-by-step": "c"}


def trait_corpus(trait: str) -> TraitCorpus:
    """Author one trait's pairs, short enough for the declared budget.

    Args:
        trait: The trait to declare; one of the three the tiny plan names.

    Returns:
        The corpus: twelve pairs, so the stride holds out six.
    """
    marker = _MARKERS[trait]
    return TraitCorpus(
        trait=trait,
        pairs=[
            TraitPairSpec(
                prompt=f"p{marker}{index} ",
                expressing=f"{marker * 4}{index}",
                neutral=f"{chr(ord(marker) + 9) * 4}{index}",
            )
            for index in range(12)
        ],
    )


def fake_trait_reader(
    corpus_dir: pathlib.Path, traits: Sequence[str], /
) -> tuple[TraitCorpus, ...]:
    """Stand in for the trait reader, returning the roster in order.

    Args:
        corpus_dir: Unused; the corpora are authored here.
        traits: The roster requested.

    Returns:
        One corpus per requested trait, in roster order.
    """
    return tuple(trait_corpus(trait) for trait in traits)


def _fake_tokenizer(model_id_or_path: str) -> HFTokenizerProto:
    """Stand in for the hub tokenizer loader.

    Args:
        model_id_or_path: The id the plan declares.

    Returns:
        The fake tokenizer.
    """
    assert model_id_or_path == TINY_TRAIT_PLAN["model_id"]
    return FakeHFTokenizer(vocab_size=_VOCAB)


def _fake_model(
    model_id_or_path: str, quantization: QuantizationConfig | StoredBf16Precision | None
) -> LMModelProto:
    """Stand in for the hub model loader, returning a real tiny GPT-2.

    Args:
        model_id_or_path: The id the plan declares.
        quantization: Must be None for a gpt2-class id.

    Returns:
        The model.
    """
    assert model_id_or_path == TINY_TRAIT_PLAN["model_id"]
    assert quantization is None
    model, _ids = probe_model_and_input("cpu", PROBE_SHAPES["tiny"])
    return model


def _fake_plans() -> Mapping[str, TraitPlan]:
    """Stand in for the production plan table.

    Returns:
        One runnable plan.
    """
    return {"tiny": TINY_TRAIT_PLAN}


def install_fakes() -> None:
    """Point every seam the sweep reads at its fake."""
    _measurement_hooks.trait_sweep_plans = _fake_plans
    _trait_hooks.read_trait_corpora = fake_trait_reader
    hf_hooks.Hooks.load_hf_tokenizer = _fake_tokenizer
    hf_hooks.Hooks.load_hf_model = _fake_model


def restore_fakes() -> None:
    """Put every seam back on its production implementation."""
    _measurement_hooks.trait_sweep_plans = _measurement_hooks._default_trait_sweep_plans
    _trait_hooks.read_trait_corpora = _trait_hooks._default_read_trait_corpora
    hf_hooks.Hooks.reset()


__all__ = [
    "SITE",
    "TINY_TRAIT_PLAN",
    "fake_trait_reader",
    "install_fakes",
    "restore_fakes",
    "trait_corpus",
]
