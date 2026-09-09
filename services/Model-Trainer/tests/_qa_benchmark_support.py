"""Shared fakes for the question-set benchmark's tests.

Extracted when the benchmark's test module passed the 600-line ceiling and
its arm tests moved to `test_cartridge_qa_arms.py`. A pytest fixture is not
visible across modules, so the install/restore pair lives here and each
module wraps it in its own three-line autouse fixture.

WHAT IS FAKED AND WHY, since the list has grown:

* The hub loader returns a REAL tiny GPT-2 from the deterministic probe
  init. Only its provenance is faked, so the arms run real attention.
* The tokenizer is word-level and reversible, which the 64-position tiny
  rung requires -- a character tokenizer turns one qualifying sentence into
  more tokens than the whole window.
* The corpus reader stands in for a wiki checkout.
* The embedder is hashed rather than real, so these tests do not depend on
  what one particular gte model thinks two sentences mean. The dense arm's
  real weights are exercised in `test_cartridge_dense.py`.
"""

from __future__ import annotations

import math
import pathlib
import zlib
from collections.abc import Mapping, Sequence

import torch
from platform_core.power_distributions import McNemarTest
from platform_core.run_record import Observation

from model_trainer.cli import _measurement_hooks, _test_hooks
from model_trainer.core.contracts.model import QuantizationConfig, StoredBf16Precision
from model_trainer.core.services.model.backends.hf_lm import _test_hooks as hf_hooks
from model_trainer.core.services.model.backends.hf_lm._hook_protocols import HFTokenizerProto
from model_trainer.core.services.model.cartridge_dense import EmbedderProto
from model_trainer.core.services.model.cartridge_qa_plans import QaPlan
from model_trainer.core.services.model.known_answer_probe import probe_model_and_input
from model_trainer.core.services.model.probe_shapes import PROBE_SHAPES
from model_trainer.core.types import LMModelProto

#: A plan small enough to run in a test and shaped like the real one.
#:
#: Sized against the tiny rung's 64 positions, which is the binding
#: constraint: a 48-token budget plus an 8-slot prefix is 56, leaving room for
#: an item and some evidence without reaching the position embedding's end.
TINY_PLAN: QaPlan = {
    "model_id": "tiny-under-test",
    "window": 8,
    "held_out_stride": 2,
    "num_slots": 8,
    "max_seq_len": 48,
    "seeds": (7, 8, 9),
    "epochs": 1,
    "learning_rate": 0.05,
    "distractor_count": 2,
    "max_items": 6,
    # DECLARED HONESTLY RATHER THAN SET TO WHATEVER LETS THE FIXTURE THROUGH.
    # Six items resolve nothing smaller than 5/6, so a fixture claiming to
    # hunt a real 0.05 effect would be refused by
    # `require_resolvable_question_set` -- correctly, and the temptation would
    # then be to exempt tests from the gate. That exemption is exactly how a
    # gate stops being one. This plan says what six items can actually do:
    # only a near-total difference. The gate therefore runs in the end-to-end
    # path rather than being skipped in it.
    "smallest_effect_of_interest": 0.9,
    "alpha": 0.05,
    "mcnemar_test": McNemarTest.MID_P,
    "bm25_k1": 1.5,
    "bm25_b": 0.75,
    "retrieved_chunks": 5,
    "expansion_feedback_chunks": 2,
    "expansion_terms": 3,
}

#: Six documents, each naming its own subject in several sentences.
#:
#: FOUR RATHER THAN TWO because a distractor may not be a term from the item's
#: own document: with two documents an item could draw only one distractor,
#: and the builder correctly refuses. Each subject recurs so that it lands in
#: both a held-out window and a training one, which is what makes its item
#: answerable from the corpus rather than a guess.
#:
#: AND SIX RATHER THAN FOUR because four items cannot resolve anything at all.
#: McNemar needs at least five disagreements to reject at alpha 0.05 under
#: mid-p, so a four-item question set has a floor of 5/4 = 1.25 -- above the
#: 1.0 that bounds any accuracy difference. This fixture had been exercising
#: the whole benchmark end to end on a question set incapable of producing a
#: significant result however the arms fell, which is a small instance of the
#: defect `cartridge_qa_power` exists to refuse. Six items put the floor at
#: 5/6, which `TINY_PLAN` declares and can therefore clear honestly.
#: NO SENTENCE BEGINS WITH ITS SUBJECT, and that is a constraint of the fake
#: tokenizer rather than of the corpus. It is word-level, so each name is one
#: token; a name at position zero is the sequence's first token, which no
#: causal model can score because nothing precedes it, and `answer_nll`
#: correctly refuses the item. Real byte-pair encoding splits these names into
#: several tokens and the question does not arise.
DOCUMENTS: tuple[str, ...] = tuple(
    (
        f"The engine called {name} rebuilt the measurement path inside one core. "
        f"A later pass moved {name} onto a faster route for speed and memory. "
        f"The team measured {name} against the usual baseline over many weeks. "
        f"Written notes about {name} explain the design in considerable detail."
    )
    for name in ("ClearGBM", "TankpitBot", "NavProbe", "CoverGate", "LedgerVane", "QuartzMill")
)


class Tokenizer:
    """A reversible word-level tokenizer inside the tiny rung's vocabulary.

    WORD-LEVEL RATHER THAN CHARACTER-LEVEL, and the reason is a real
    constraint rather than convenience. The tiny rung has 64 positions, the
    item builder requires sentences of at least
    :data:`~corpus_cloze.MIN_SENTENCE_CHARS` characters, and a character
    tokenizer turns such a sentence into sixty-odd tokens -- more than the
    whole window, before any evidence. `with_evidence` then correctly refuses
    every item, and the test measures nothing.

    REVERSIBLE because the pipeline decodes windows back to text to build
    items from them. A hashing tokenizer would encode fine and decode to
    nothing, so the vocabulary is kept both ways and grown on demand.
    """

    _to_id: dict[str, int]
    _to_word: dict[int, str]

    def __init__(self) -> None:
        self._to_id = {}
        self._to_word = {}

    @property
    def eos_token_id(self) -> int | None:
        return 0

    @property
    def pad_token_id(self) -> int | None:
        return 1

    def __len__(self) -> int:
        return PROBE_SHAPES["tiny"]["vocab_size"]

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for word in text.split():
            known = self._to_id.get(word)
            if known is None:
                # Ids start at 2 so neither collides with eos or pad, and stay
                # inside the rung's vocabulary or the embedding lookup fails.
                known = len(self._to_id) + 2
                assert known < len(self), "the fake corpus outgrew the tiny vocabulary"
                self._to_id[word] = known
                self._to_word[known] = word
            ids.append(known)
        return ids

    def decode(self, ids: list[int]) -> str:
        return " ".join(self._to_word[value] for value in ids)

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.encode(token)[0] if token.split() else 0


def _fake_tokenizer(model_id_or_path: str) -> HFTokenizerProto:
    """Stand in for the hub tokenizer loader."""
    assert model_id_or_path == TINY_PLAN["model_id"]
    return Tokenizer()


def _fake_model(
    model_id_or_path: str, quantization: QuantizationConfig | StoredBf16Precision | None
) -> LMModelProto:
    """Stand in for the hub model loader, returning a real tiny GPT-2.

    The model is real; only its provenance is faked, so the arms run real
    attention without needing a cache.
    """
    assert model_id_or_path == TINY_PLAN["model_id"]
    assert quantization is None
    model, _ids = probe_model_and_input("cpu", PROBE_SHAPES["tiny"])
    return model


def _fake_plans() -> Mapping[str, QaPlan]:
    """Stand in for the production plan table, with one runnable plan."""
    return {"tiny": TINY_PLAN}


def _fake_corpus_reader(corpus_dir: pathlib.Path, /) -> tuple[str, ...]:
    """Stand in for the corpus reader."""
    return DOCUMENTS


def _fake_embed(texts: Sequence[str], /) -> torch.Tensor:
    """Stand in for the gte embedder with a deterministic hashed vector.

    FAKED HERE FOR TWO REASONS, and speed is the lesser one. The real
    embedder loads 400 MB of weights, which every `measure_qa_plan` test
    would pay -- measured, it took the module from 18 to 161 seconds. More
    importantly it would make these tests depend on what one particular
    model thinks two sentences mean, so a model upgrade would move
    assertions about the CLI's plumbing.

    Deterministic rather than random: the same text must embed the same way
    within a run, or the ranking is not a function of the corpus.

    Args:
        texts: Texts to embed.

    Returns:
        One L2-normalised row per input.
    """
    # Normalised in plain Python rather than through torch: `Tensor.tolist`
    # is typed `Any`, and this repo refuses an Any even in a test fake.
    rows: list[list[float]] = []
    for text in texts:
        seed = zlib.crc32(text.encode("utf-8"))
        raw: list[float] = [float((seed >> shift) & 0xFF) + 1.0 for shift in (0, 8, 16, 24)]
        length = math.sqrt(sum(value * value for value in raw))
        rows.append([value / length for value in raw])
    return torch.tensor(rows, dtype=torch.float32)


def fake_embedder_factory(device: str, /) -> EmbedderProto:
    """Stand in for the gte embedder factory.

    Takes the device and ignores it: the fake hashes text rather than
    running a model, so there is nothing to place. The PARAMETER is kept
    because the protocol has it, and a fake whose signature drifts from the
    real one stops testing the call it stands in for.

    Args:
        device: Where a real encoder would live. Unused here.

    Returns:
        The hashing embedder.
    """
    return _fake_embed


def install_fakes() -> None:
    """Point every seam the benchmark uses at a fake."""
    _measurement_hooks.qa_plans = _fake_plans
    _test_hooks.read_corpus_documents = _fake_corpus_reader
    _test_hooks.make_embedder = fake_embedder_factory
    hf_hooks.Hooks.load_hf_tokenizer = _fake_tokenizer
    hf_hooks.Hooks.load_hf_model = _fake_model


def restore_fakes() -> None:
    """Put the production hooks back."""
    _measurement_hooks.qa_plans = _measurement_hooks._default_qa_plans
    _test_hooks.read_corpus_documents = _test_hooks._default_read_corpus_documents
    _test_hooks.make_embedder = _test_hooks._default_embedder_factory
    hf_hooks.Hooks.reset()


def values(observations: Sequence[Observation]) -> dict[str, float]:
    """Read a record's observations into a name-to-value mapping.

    Args:
        observations: The observations to read.

    Returns:
        Each observation's value, keyed by its name.
    """
    return {observation["name"]: observation["value"] for observation in observations}


__all__ = [
    "DOCUMENTS",
    "TINY_PLAN",
    "Tokenizer",
    "fake_embedder_factory",
    "install_fakes",
    "restore_fakes",
    "values",
]
