"""The language-model arm on the real tiny GPT-2, with a real edit.

The edit silences one attention head of the first block by zeroing the rows
of its output projection that carry that head (GPT-2's ``Conv1D`` stores
``c_proj.weight`` as input by output, so a head's 64 output dimensions are 64
ROWS). That is a write whose size is known on paper -- 64 rows of 128 -- and
whose effect on the model is real, so the arm's before-and-after plumbing is
exercised on an edit that actually moves things.
"""

from __future__ import annotations

import pytest
from platform_core.power_distributions import exact_mcnemar_p

from model_trainer.core.contracts.cloze import BLANK_MARKER, ClozeItem
from model_trainer.core.contracts.weight_program import WeightOp, WeightProgram
from model_trainer.core.services.model.backends.hf_lm.encoding import HFTokenizerEncoder
from model_trainer.core.services.model.editing.verify import parameter_digests, weights_digest
from model_trainer.core.services.model.known_answer_probe import probe_model_and_input
from model_trainer.core.services.model.probe_shapes import PROBE_SHAPES
from model_trainer.core.services.model.weight_space.lm_edit_arm import (
    lm_edit_observations,
    paired_counts,
    run_lm_edit_arm,
)
from model_trainer.core.services.model.weight_space.repetition import (
    RepetitionSettings,
    measure_repetition,
)
from model_trainer.core.types import TracedLMModelProto
from tests._qa_benchmark_support import Tokenizer, values

_ITEMS: tuple[ClozeItem, ...] = (
    ClozeItem(
        item_id="i00",
        template=f"The measured run put NavProbe beside {BLANK_MARKER} in one harness.",
        answer="ClearGBM",
        distractors=["CoverGate", "TankpitBot"],
    ),
    ClozeItem(
        item_id="i01",
        template=f"A later pass moved {BLANK_MARKER} onto a faster route for speed.",
        answer="NavProbe",
        distractors=["ClearGBM", "CoverGate"],
    ),
)

_SETTINGS = RepetitionSettings(new_tokens=10, window=6, max_period=3, ngram=2)


def _silence_head_zero(model: TracedLMModelProto) -> WeightProgram:
    """Zero block 0's attention-output rows for head 0, written against ``model``.

    Args:
        model: The model the program is for.

    Returns:
        The program.
    """
    return WeightProgram(
        program_id="silence-0.0",
        written_against=weights_digest(parameter_digests(model)),
        ops=(
            WeightOp(
                op="scale",
                parameter="transformer.h.0.attn.c_proj.weight",
                axis=0,
                indices=tuple(range(64)),
                values=(0.0,),
            ),
        ),
    )


def test_paired_counts_read_both_discordant_cells() -> None:
    """Control T T F F T, edited T F T F F: lost at 1 and 4, gained at 2."""
    counts = paired_counts([True, True, False, False, True], [True, False, True, False, False])
    assert counts == {"lost": 2, "gained": 1, "p_value": exact_mcnemar_p(1, 3)}
    assert counts["p_value"] == 1.0


def test_paired_counts_refuse_unequal_lengths() -> None:
    with pytest.raises(ValueError, match="cannot pair 2 outcomes with 1"):
        paired_counts([True, False], [True])


def test_the_arm_measures_both_models_and_restores_the_original() -> None:
    model, _ids = probe_model_and_input("cpu", PROBE_SHAPES["tiny"])
    encoder = HFTokenizerEncoder(Tokenizer())
    original = parameter_digests(model)
    prompts = ((5, 17, 3), (40, 41, 42, 43))
    outcome = run_lm_edit_arm(
        model=model,
        encoder=encoder,
        items=_ITEMS,
        prompts=prompts,
        program=_silence_head_zero(model),
        settings=_SETTINGS,
        device="cpu",
        max_seq_len=48,
    )
    assert parameter_digests(model) == original
    effect = outcome["effect"]
    assert effect["elements_changed"] == 64 * 128
    assert effect["elements_total"] == sum(
        int(parameter.detach().numel()) for _, parameter in model.named_parameters()
    )
    assert effect["weights_before"] == weights_digest(original)
    assert outcome["control_cloze"]["total"] == len(_ITEMS)
    assert len(outcome["edited_repetition"]["continuations"]) == len(prompts)


def test_the_control_measurement_is_the_untouched_model() -> None:
    """The arm's control continuations equal a fresh measurement of an unedited copy."""
    model, _ids = probe_model_and_input("cpu", PROBE_SHAPES["tiny"])
    untouched, _ = probe_model_and_input("cpu", PROBE_SHAPES["tiny"])
    encoder = HFTokenizerEncoder(Tokenizer())
    prompts = ((5, 17, 3),)
    outcome = run_lm_edit_arm(
        model=model,
        encoder=encoder,
        items=_ITEMS,
        prompts=prompts,
        program=_silence_head_zero(model),
        settings=_SETTINGS,
        device="cpu",
        max_seq_len=48,
    )
    fresh = measure_repetition(untouched, prompts, _SETTINGS, device="cpu")
    assert outcome["control_repetition"] == fresh
    assert outcome["control_repetition"] != outcome["edited_repetition"]


def test_observations_name_both_models_and_the_paired_counts() -> None:
    model, _ids = probe_model_and_input("cpu", PROBE_SHAPES["tiny"])
    outcome = run_lm_edit_arm(
        model=model,
        encoder=HFTokenizerEncoder(Tokenizer()),
        items=_ITEMS,
        prompts=((5, 17, 3), (40, 41, 42, 43)),
        program=_silence_head_zero(model),
        settings=_SETTINGS,
        device="cpu",
        max_seq_len=48,
    )
    named = values(lm_edit_observations(outcome))
    control = outcome["control_cloze"]
    edited = outcome["edited_cloze"]
    lost = sum(
        1
        for first, second in zip(control["outcomes"], edited["outcomes"], strict=True)
        if first["correct"] and not second["correct"]
    )
    assert named["cloze_items"] == 2.0
    assert named["control_cloze_correct"] == float(control["correct"])
    assert named["edited_cloze_accuracy"] == edited["accuracy"]
    assert named["cloze_right_to_wrong"] == float(lost)
    assert named["prompts"] == 2.0
    assert named["control_looped"] == float(sum(outcome["control_repetition"]["looped"]))
    assert named["edited_seq_rep"] == pytest.approx(
        sum(outcome["edited_repetition"]["seq_rep"]) / 2.0
    )
    assert named["fraction_changed"] == pytest.approx(float(64 * 128) / named["elements_total"])
