"""The language-model commands, over the real tiny GPT-2 behind the hub loader.

WHAT IS FAKED AND WHY. Only the hub loader's PROVENANCE: it returns the
production probe builder's real tiny GPT-2 and a word-level tokenizer instead
of reading a 124-million-parameter checkpoint from a cache, exactly as the
question-set benchmark's tests do (``tests/_qa_benchmark_support``). The
model, the edit, both scorers and the records are real.
"""

from __future__ import annotations

import pathlib
from collections.abc import Generator

import pytest
from platform_core.errors import AppError, ModelTrainerErrorCode
from platform_core.json_utils import (
    JSONObject,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    narrow_json_to_list,
    require_list,
    require_str,
)
from platform_core.run_record import decode_run_record

from model_trainer.cli import weight_space_lm_digest as digest_cli
from model_trainer.cli import weight_space_lm_edit as edit_cli
from model_trainer.core.contracts.cloze import BLANK_MARKER
from model_trainer.core.services.model.backends.hf_lm import _test_hooks as hf_hooks
from model_trainer.core.services.model.backends.hf_lm._hook_protocols import HFTokenizerProto
from model_trainer.core.services.model.editing.verify import parameter_digests, weights_digest
from model_trainer.core.services.model.known_answer_probe import probe_model_and_input
from model_trainer.core.services.model.probe_shapes import PROBE_SHAPES
from tests._qa_benchmark_support import Tokenizer, install_fakes, restore_fakes, values


class _ContinuationTokenizer(Tokenizer):
    """The word tokenizer, able to write back ids it never handed out.

    A greedy continuation of a randomly initialised model emits ids anywhere
    in the 512-word vocabulary, most of which the word tokenizer never
    assigned to a word, and the outcomes file decodes every continuation.
    Each unassigned id is written as ``<id>``, one word per token, so the
    decoded text keeps the token count the test reads.
    """

    def decode(self, ids: list[int]) -> str:
        """Render known ids as their words and the rest as ``<id>``."""
        return " ".join(self._to_word.get(value, f"<{value}>") for value in ids)


def _continuation_tokenizer(model_id_or_path: str) -> HFTokenizerProto:
    """Stand in for the hub tokenizer loader with the decoding tokenizer.

    Args:
        model_id_or_path: The hub id, which must be the faked one.

    Returns:
        The tokenizer.
    """
    assert model_id_or_path == "gpt2"
    return _ContinuationTokenizer()


@pytest.fixture(autouse=True)
def _fakes() -> Generator[None, None, None]:
    """Point the hub loader at the tiny model for one test."""
    install_fakes()
    hf_hooks.Hooks.load_hf_tokenizer = _continuation_tokenizer
    yield
    restore_fakes()


def _tiny_digest() -> str:
    """The weights digest of the model the faked loader returns.

    Returns:
        The digest.
    """
    model, _ids = probe_model_and_input("cpu", PROBE_SHAPES["tiny"])
    return weights_digest(parameter_digests(model))


def _items(path: pathlib.Path) -> pathlib.Path:
    """Write a three-item cloze file in the staged format.

    Args:
        path: Directory to write into.

    Returns:
        The file.
    """
    rows: list[JSONObject] = [
        {
            "item_id": f"i{index}",
            "template": f"The engine called {name} rebuilt the path {BLANK_MARKER} in one core.",
            "answer": name,
            "distractors": ["CoverGate"],
        }
        for index, name in enumerate(("ClearGBM", "NavProbe", "TankpitBot"))
    ]
    out = path / "items.jsonl"
    out.write_text("\n".join(dump_json_str(row) for row in rows) + "\n", "utf-8")
    return out


def test_the_digest_command_names_the_weights_and_their_inventory(tmp_path: pathlib.Path) -> None:
    out = tmp_path / "digest.json"
    assert digest_cli.main(["--model", "gpt2", "--out", str(out)]) == 0
    described = narrow_json_to_dict(load_json_str(out.read_text("utf-8")))
    assert require_str(described, "weights_digest") == _tiny_digest()
    inventory = require_list(described, "parameters")
    assert len(inventory) == 28
    first = narrow_json_to_list(inventory[0])
    assert first[0] == "transformer.wte.weight"
    assert first[1] == [512, 128]


def _program(path: pathlib.Path, written_against: str) -> pathlib.Path:
    """Write a program that zeroes block 0's head-0 output rows.

    Args:
        path: Directory to write into.
        written_against: The digest it claims to have been read from.

    Returns:
        The file.
    """
    program: JSONObject = {
        "program_id": "silence-0.0",
        "written_against": written_against,
        "ops": [
            {
                "op": "scale",
                "parameter": "transformer.h.0.attn.c_proj.weight",
                "axis": 0,
                "indices": list(range(64)),
                "values": [0.0],
            }
        ],
    }
    out = path / "program.json"
    out.write_text(dump_json_str(program), "utf-8")
    return out


def _argv(tmp_path: pathlib.Path, program: pathlib.Path) -> list[str]:
    """The measuring command's arguments for the tiny run.

    Args:
        tmp_path: Where outputs go.
        program: The program file.

    Returns:
        The arguments.
    """
    return [
        "--model",
        "gpt2",
        "--items",
        str(_items(tmp_path)),
        "--program",
        str(program),
        "--prompts",
        "2",
        "--min-prompt-tokens",
        "3",
        "--max-prompt-tokens",
        "6",
        "--new-tokens",
        "8",
        "--window",
        "6",
        "--max-period",
        "3",
        "--ngram",
        "2",
        "--max-seq-len",
        "48",
        "--device",
        "cpu",
        "--controls",
        "none",
        "--out",
        str(tmp_path / "out" / "record.json"),
        "--outcomes",
        str(tmp_path / "out" / "outcomes.json"),
    ]


def test_the_measuring_command_records_both_models(tmp_path: pathlib.Path) -> None:
    assert edit_cli.main(_argv(tmp_path, _program(tmp_path, _tiny_digest()))) == 0
    record = decode_run_record(
        narrow_json_to_dict(load_json_str((tmp_path / "out" / "record.json").read_text("utf-8")))
    )
    named = values(record["observations"])
    assert record["experiment"] == "weight-space-lm-edit"
    assert record["label"] == "gpt2-silence-0.0"
    assert named["cloze_items"] == 3.0
    assert named["prompts"] == 2.0
    assert named["elements_changed"] == 64.0 * 128.0
    outcomes = narrow_json_to_dict(
        load_json_str((tmp_path / "out" / "outcomes.json").read_text("utf-8"))
    )
    cloze = require_list(outcomes, "cloze")
    assert narrow_json_to_list(cloze[0])[0] == "i0"
    continuation = narrow_json_to_dict(require_list(outcomes, "continuations")[0])
    assert len(require_str(continuation, "control").split()) == 8
    assert isinstance(continuation["edited_looped"], bool)


def test_a_program_read_from_other_weights_is_refused(tmp_path: pathlib.Path) -> None:
    with pytest.raises(AppError) as caught:
        edit_cli.main(_argv(tmp_path, _program(tmp_path, "sha256:" + "0" * 64)))
    assert caught.value.code is ModelTrainerErrorCode.EDIT_VERIFICATION_FAILED
    assert not (tmp_path / "out" / "record.json").exists()
