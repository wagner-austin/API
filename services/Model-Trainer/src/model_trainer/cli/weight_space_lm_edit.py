"""Measure one written edit on a hub language model, against it untouched.

Board task 7964f9f6, criterion A2. The reasoning lives in
:mod:`~model_trainer.core.services.model.weight_space.lm_edit_arm`; this is
the entry: pin determinism, load the model and the staged cloze items, take
the repetition prompts from those items, run the arm, and record it.

WHAT IT LEAVES BEHIND: ``--out``, a
:class:`~platform_core.run_record.RunRecord` whose payload digest is the
edited weights' digest; and ``--outcomes``, every cloze item's verdict under
both models and every prompt's two continuations as text, which is where a
reader goes to see what the edit actually did to the model's prose.

Usage:
    python -m model_trainer.cli.weight_space_lm_edit --model gpt2
        --items ITEMS.jsonl --program EDIT.json --prompts 256
        --min-prompt-tokens 8 --max-prompt-tokens 48 --new-tokens 96
        --window 32 --max-period 16 --ngram 4 --max-seq-len 512
        --device cuda --controls both --out REC.json --outcomes OUT.json
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Sequence

from platform_core import cli_args
from platform_core.json_utils import JSONValue, dump_json_str
from platform_core.logging import get_logger
from platform_core.run_record import RunRecord, encode_run_record, run_record

from model_trainer.cli._weight_space_flags import require_int_flag
from model_trainer.cli.known_answer_probe import probe_determinism
from model_trainer.cli.weight_space_policy_edit import read_program
from model_trainer.core.encoding import Encoder
from model_trainer.core.run_fingerprint import capture_run_fingerprint
from model_trainer.core.services.model.backends.hf_lm import _test_hooks as hf_hooks
from model_trainer.core.services.model.backends.hf_lm.encoding import HFTokenizerEncoder
from model_trainer.core.services.model.control_arms import CONTROLS_FLAG, require_control_arm
from model_trainer.core.services.model.editing.sites import require_traceable_model
from model_trainer.core.services.model.weight_space.lm_edit_arm import (
    LMEditOutcome,
    lm_edit_observations,
    run_lm_edit_arm,
)
from model_trainer.core.services.model.weight_space.repetition import (
    RepetitionSettings,
    cloze_prompts,
)
from model_trainer.core.types import TracedLMModelProto
from model_trainer.worker.cloze_job import parse_items

_log = get_logger(__name__)

#: Name under which every run of the language-model arm is comparable.
LM_EDIT_EXPERIMENT = "weight-space-lm-edit"

MODEL_FLAG = "--model"
ITEMS_FLAG = "--items"
PROGRAM_FLAG = "--program"
PROMPTS_FLAG = "--prompts"
MIN_PROMPT_TOKENS_FLAG = "--min-prompt-tokens"
MAX_PROMPT_TOKENS_FLAG = "--max-prompt-tokens"
NEW_TOKENS_FLAG = "--new-tokens"
WINDOW_FLAG = "--window"
MAX_PERIOD_FLAG = "--max-period"
NGRAM_FLAG = "--ngram"
MAX_SEQ_LEN_FLAG = "--max-seq-len"
DEVICE_FLAG = "--device"
OUT_FLAG = "--out"
OUTCOMES_FLAG = "--outcomes"

_FLAGS = (
    MODEL_FLAG,
    ITEMS_FLAG,
    PROGRAM_FLAG,
    PROMPTS_FLAG,
    MIN_PROMPT_TOKENS_FLAG,
    MAX_PROMPT_TOKENS_FLAG,
    NEW_TOKENS_FLAG,
    WINDOW_FLAG,
    MAX_PERIOD_FLAG,
    NGRAM_FLAG,
    MAX_SEQ_LEN_FLAG,
    DEVICE_FLAG,
    CONTROLS_FLAG,
    OUT_FLAG,
    OUTCOMES_FLAG,
)


def load_editable_lm(model_id: str) -> tuple[TracedLMModelProto, Encoder]:
    """Load a hub model whose parameters a program can write, and its tokenizer.

    Args:
        model_id: The hub id, resolved from the local cache.

    Returns:
        The model, narrowed to one with a module graph, and its encoder.

    Raises:
        AppError: With ``EDIT_MODULE_NOT_FOUND`` if the loaded model has no
            module graph to write into.
    """
    tokenizer = hf_hooks.Hooks.load_hf_tokenizer(model_id)
    model = require_traceable_model(hf_hooks.Hooks.load_hf_model(model_id, None))
    return model, HFTokenizerEncoder(tokenizer)


def encode_outcomes(outcome: LMEditOutcome, encoder: Encoder) -> str:
    """Render both models' per-item verdicts and continuations for a reader.

    Args:
        outcome: What the arm measured.
        encoder: The tokenizer, to turn continuations back into text.

    Returns:
        Canonical JSON: ``cloze`` as ``[item_id, control_correct,
        edited_correct]`` triples, and ``continuations`` as one object per
        prompt with both texts and both loop verdicts.
    """
    cloze: list[JSONValue] = [
        [first["item_id"], first["correct"], second["correct"]]
        for first, second in zip(
            outcome["control_cloze"]["outcomes"], outcome["edited_cloze"]["outcomes"], strict=True
        )
    ]
    control = outcome["control_repetition"]
    edited = outcome["edited_repetition"]
    continuations: list[JSONValue] = [
        {
            "control": encoder.decode(list(control["continuations"][index])),
            "edited": encoder.decode(list(edited["continuations"][index])),
            "control_looped": control["looped"][index],
            "edited_looped": edited["looped"][index],
        }
        for index in range(len(control["continuations"]))
    ]
    return dump_json_str({"cloze": cloze, "continuations": continuations})


def lm_edit_run_record(parsed: dict[str, str]) -> tuple[RunRecord, str]:
    """Pin determinism, run the arm, and record it.

    Determinism is pinned before the model is loaded, for the reason
    :mod:`~model_trainer.cli.score_baseline` gives: the cuBLAS setting is read
    when the handle is created, and loading weights creates it.

    Args:
        parsed: The parsed flags.

    Returns:
        The record and the rendered outcomes.

    Raises:
        ValueError: When a flag is missing or malformed.
        AppError: Propagated from the loader, the prompt builder or the arm.
    """
    device = cli_args.require_flag(parsed, DEVICE_FLAG)
    remove_split_k, math_attention = require_control_arm(
        cli_args.require_flag(parsed, CONTROLS_FLAG)
    )
    fingerprint = capture_run_fingerprint(
        device,
        probe_determinism(device, remove_split_k=remove_split_k, math_attention=math_attention),
    )
    program = read_program(pathlib.Path(cli_args.require_flag(parsed, PROGRAM_FLAG)))
    items_path = pathlib.Path(cli_args.require_flag(parsed, ITEMS_FLAG))
    items = parse_items(items_path.read_text(encoding="utf-8"))
    model, encoder = load_editable_lm(cli_args.require_flag(parsed, MODEL_FLAG))
    prompts = cloze_prompts(
        items,
        count=require_int_flag(parsed, PROMPTS_FLAG, 1),
        min_tokens=require_int_flag(parsed, MIN_PROMPT_TOKENS_FLAG, 1),
        max_tokens=require_int_flag(parsed, MAX_PROMPT_TOKENS_FLAG, 1),
        encoder=encoder,
    )
    settings = RepetitionSettings(
        new_tokens=require_int_flag(parsed, NEW_TOKENS_FLAG, 1),
        window=require_int_flag(parsed, WINDOW_FLAG, 2),
        max_period=require_int_flag(parsed, MAX_PERIOD_FLAG, 1),
        ngram=require_int_flag(parsed, NGRAM_FLAG, 1),
    )
    outcome = run_lm_edit_arm(
        model=model,
        encoder=encoder,
        items=items,
        prompts=prompts,
        program=program,
        settings=settings,
        device=device,
        max_seq_len=require_int_flag(parsed, MAX_SEQ_LEN_FLAG, 2),
    )
    record = run_record(
        experiment=LM_EDIT_EXPERIMENT,
        label=f"{cli_args.require_flag(parsed, MODEL_FLAG)}-{program['program_id']}",
        fingerprint=fingerprint,
        observations=lm_edit_observations(outcome),
        payload_digest=outcome["effect"]["weights_after"],
    )
    return record, encode_outcomes(outcome, encoder)


def main(argv: Sequence[str] | None = None) -> int:
    """Measure one edit and write its record and outcomes.

    Args:
        argv: Command-line arguments excluding the program name.

    Returns:
        0 once both files are written.

    Raises:
        ValueError: When a flag is unknown, repeated, missing or malformed.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    parsed = cli_args.parse_single_flags(tokens, _FLAGS)
    record, outcomes = lm_edit_run_record(parsed)
    out = pathlib.Path(cli_args.require_flag(parsed, OUT_FLAG))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dump_json_str(encode_run_record(record)), encoding="utf-8")
    outcomes_path = pathlib.Path(cli_args.require_flag(parsed, OUTCOMES_FLAG))
    outcomes_path.parent.mkdir(parents=True, exist_ok=True)
    outcomes_path.write_text(outcomes, encoding="utf-8")
    _log.info("measured %s -> %s, outcomes %s", record["label"], out, outcomes_path)
    return 0


def entrypoint() -> None:
    """Console-script entry point.

    Raises:
        SystemExit: Always, carrying :func:`main`'s exit code.
    """
    raise SystemExit(main())


__all__ = [
    "LM_EDIT_EXPERIMENT",
    "encode_outcomes",
    "entrypoint",
    "lm_edit_run_record",
    "load_editable_lm",
    "main",
]


# Without this, `python -m model_trainer.cli.weight_space_lm_edit` imports the
# module, runs nothing and exits 0.
if __name__ == "__main__":
    entrypoint()
