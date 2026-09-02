"""The attribution probe and the fully-owned scorer, on real CUDA work.

The probe's record is checked by recomputation: same operands, same stages,
same digests -- what a second card of the same OS would have to produce.
The scorer runs the REAL scoring loop over the real synthetic tiny model
with only the hub loader faked, because a test that faked the scorer would
never drive the swapped attention through a forward.
"""

from __future__ import annotations

import pathlib
import runpy
import sys
from collections.abc import Generator

import pytest
import torch
from model_trainer.cli import _test_hooks as cli_hooks
from model_trainer.core.contracts.model import PreparedLMModel
from model_trainer.core.encoding import ListEncoded
from model_trainer.core.services.model.known_answer_probe import probe_model_and_input
from model_trainer.core.services.model.probe_shapes import require_probe_shape
from model_trainer.core.services.model.tensor_digest import describe_tensor
from platform_core.json_utils import load_json_str, narrow_json_to_dict
from platform_core.run_record import decode_run_record

from ordered_kernels.cli import attn_probe as attn_cli
from ordered_kernels.cli import score as score_cli
from ordered_kernels.cli.train_step import require_swapped
from ordered_kernels.modules import use_ordered_attention

_ITEMS_JSONL = (
    '{"item_id": "a::0", "template": "The tower is <<BLANK>> metres.", '
    '"answer": "324", "distractors": ["12", "99", "700"]}\n'
    '{"item_id": "a::1", "template": "It opened in <<BLANK>>.", '
    '"answer": "1889", "distractors": ["1789", "1989", "1689"]}\n'
)


class _ByteEncoder:
    """A real, deterministic encoder: one token per byte.

    The synthetic tiny model's vocabulary is 512, so byte values are always
    in range; nothing about scoring depends on WHICH ids a text becomes,
    only that the mapping is deterministic.
    """

    def encode(self, text: str) -> ListEncoded:
        return ListEncoded(list(text.encode("utf-8")))

    def decode(self, ids: list[int]) -> str:
        return bytes(ids).decode("utf-8", errors="replace")

    def token_to_id(self, token: str) -> int | None:
        ids = list(token.encode("utf-8"))
        return ids[0] if len(ids) == 1 else None

    def get_vocab_size(self) -> int:
        return 256


def _fake_hub() -> Generator[None, None, None]:
    """Serve the synthetic tiny model in place of the hub, restoring after.

    Yields:
        Nothing; the loader is faked for the test body.
    """

    def load(hub_model_id: str, /) -> PreparedLMModel:
        model, _ = probe_model_and_input("cuda", require_probe_shape("tiny"))
        return PreparedLMModel(
            model=model,
            tokenizer_id=None,
            eos_id=0,
            pad_id=0,
            max_seq_len=64,
            tok_for_dataset=_ByteEncoder(),
        )

    cli_hooks.load_hub_model = load
    try:
        yield
    finally:
        cli_hooks.load_hub_model = cli_hooks._default_load_hub_model


fake_hub = pytest.fixture(_fake_hub)


class TestTheLengthParser:
    def test_the_lengths_come_back_in_order(self) -> None:
        assert attn_cli.require_lengths("15,16,64") == (15, 16, 64)

    def test_an_empty_list_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            attn_cli.require_lengths(" , ")

    def test_junk_is_refused(self) -> None:
        with pytest.raises(ValueError, match="positive integers"):
            attn_cli.require_lengths("15,-2")

    def test_zero_is_refused(self) -> None:
        with pytest.raises(ValueError, match="positive integers"):
            attn_cli.require_lengths("0")

    def test_duplicates_are_refused(self) -> None:
        with pytest.raises(ValueError, match="duplicates"):
            attn_cli.require_lengths("15,15")


class TestTheOperands:
    def test_the_strided_layout_carries_split_heads_strides(self) -> None:
        q, k, v = attn_cli.attn_operands(9, "strided", "cuda")

        assert q.shape == (1, attn_cli.HEADS, 9, attn_cli.HEAD_DIM)
        assert not q.is_contiguous()
        assert not k.is_contiguous()
        assert not v.is_contiguous()

    def test_the_contiguous_layout_is_contiguous(self) -> None:
        q, _, _ = attn_cli.attn_operands(9, "contig", "cuda")

        assert q.is_contiguous()

    def test_an_unknown_layout_is_refused(self) -> None:
        with pytest.raises(ValueError, match="unknown layout"):
            attn_cli.attn_operands(9, "diagonal", "cuda")

    def test_the_seed_is_per_length(self) -> None:
        q_nine, _, _ = attn_cli.attn_operands(9, "contig", "cuda")
        q_nine_again, _, _ = attn_cli.attn_operands(9, "contig", "cuda")
        q_ten, _, _ = attn_cli.attn_operands(10, "contig", "cuda")

        assert torch.equal(q_nine, q_nine_again)
        assert not torch.equal(q_nine, q_ten[:, :, :9, :])


class TestTheProbeCli:
    def test_its_record_reproduces_by_recomputation(self, tmp_path: pathlib.Path) -> None:
        out = tmp_path / "attn.json"

        assert attn_cli.main(["--device", "cuda", "--lengths", "5,7", "--out", str(out)]) == 0

        decoded = decode_run_record(load_json_str(out.read_text(encoding="utf-8")))
        assert decoded["label"] == f"{attn_cli.ATTN_EXPERIMENT}-both-stages"
        by_name = {o["name"]: o["value"] for o in decoded["observations"]}
        probe_stages = attn_cli.stage_tensors(*attn_cli.attn_operands(5, "contig", "cuda"))
        assert len(by_name) == 2 * len(probe_stages) * len(attn_cli.LAYOUTS) * 2
        for length in (5, 7):
            for layout in attn_cli.LAYOUTS:
                q, k, v = attn_cli.attn_operands(length, layout, "cuda")
                for name, tensor in attn_cli.stage_tensors(q, k, v):
                    digest, total = describe_tensor(tensor.cpu())
                    base = f"attn-L{length}-{layout}-{name}"
                    assert by_name[f"{base}|digest48"] == digest, base
                    assert by_name[f"{base}|sum"] == total, base

    def test_an_absent_lengths_flag_is_refused(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ValueError, match="--lengths"):
            attn_cli.main(["--device", "cuda", "--out", str(tmp_path / "a.json")])

    def test_the_entry_point_carries_the_exit_code(self, tmp_path: pathlib.Path) -> None:
        out = tmp_path / "attn-entry.json"
        saved = sys.argv
        sys.argv = ["ordered-attn-probe", "--device", "cuda", "--lengths", "5", "--out", str(out)]
        try:
            with pytest.raises(SystemExit) as excinfo:
                attn_cli.entrypoint()
        finally:
            sys.argv = saved

        assert excinfo.value.code == 0

    def test_running_the_module_as_main_actually_probes(self, tmp_path: pathlib.Path) -> None:
        module_name = "ordered_kernels.cli.attn_probe"
        out = tmp_path / "attn-main.json"
        saved_argv = sys.argv
        saved_module = sys.modules.pop(module_name, None)
        sys.argv = ["ordered-attn-probe", "--device", "cuda", "--lengths", "5", "--out", str(out)]
        try:
            with pytest.raises(SystemExit) as excinfo:
                runpy.run_module(module_name, run_name="__main__", alter_sys=False)
        finally:
            sys.argv = saved_argv
            if saved_module is not None:
                sys.modules[module_name] = saved_module

        assert excinfo.value.code == 0
        assert out.is_file()


class TestTheScoreCli:
    def test_it_scores_for_real_and_reproduces_itself_byte_for_byte(
        self, tmp_path: pathlib.Path, fake_hub: None
    ) -> None:
        items = tmp_path / "items.jsonl"
        items.write_text(_ITEMS_JSONL, encoding="utf-8")
        args = [
            "--model",
            "gpt2",
            "--items",
            str(items),
            "--device",
            "cuda",
            "--max-seq-len",
            "64",
            "--experiment",
            "attn-closure-test",
            "--label",
            "fully-owned-tiny",
        ]

        assert (
            score_cli.main(
                [*args, "--out", str(tmp_path / "r1.json"), "--outcomes", str(tmp_path / "o1.json")]
            )
            == 0
        )
        assert (
            score_cli.main(
                [*args, "--out", str(tmp_path / "r2.json"), "--outcomes", str(tmp_path / "o2.json")]
            )
            == 0
        )

        first = (tmp_path / "o1.json").read_bytes()
        second = (tmp_path / "o2.json").read_bytes()
        assert first == second
        raw_record = (tmp_path / "r1.json").read_text(encoding="utf-8")
        record = decode_run_record(load_json_str(raw_record))
        assert record["label"] == "fully-owned-tiny"
        by_name = {o["name"]: o["value"] for o in record["observations"]}
        assert by_name["cloze_total"] == 2.0
        assert by_name["cloze_chance"] == 0.25
        outcomes = load_json_str(first.decode("utf-8"))
        if not isinstance(outcomes, list) or len(outcomes) != 2:
            raise AssertionError(f"expected two outcomes, got {outcomes!r}")
        scores = narrow_json_to_dict(outcomes[0])["scores"]
        if not isinstance(scores, list) or len(scores) != 4:
            raise AssertionError(f"expected four per-option scores, got {scores!r}")

    def test_a_model_without_attention_to_own_is_refused(self) -> None:
        # The exact composition score_fully_owned runs, on a module graph
        # with no attention in it: the count must refuse, not pass zero.
        bare = torch.nn.Linear(4, 4, bias=False)

        with pytest.raises(RuntimeError, match="replaced nothing"):
            require_swapped(use_ordered_attention(bare))

    def test_a_zero_max_seq_len_is_refused(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            score_cli.main(
                [
                    "--model",
                    "gpt2",
                    "--items",
                    str(tmp_path / "i.jsonl"),
                    "--device",
                    "cuda",
                    "--max-seq-len",
                    "0",
                    "--experiment",
                    "e",
                    "--label",
                    "l",
                    "--out",
                    str(tmp_path / "r.json"),
                    "--outcomes",
                    str(tmp_path / "o.json"),
                ]
            )

    def test_the_entry_point_carries_the_exit_code(
        self, tmp_path: pathlib.Path, fake_hub: None
    ) -> None:
        items = tmp_path / "items.jsonl"
        items.write_text(_ITEMS_JSONL, encoding="utf-8")
        saved = sys.argv
        sys.argv = [
            "ordered-score",
            "--model",
            "gpt2",
            "--items",
            str(items),
            "--device",
            "cuda",
            "--max-seq-len",
            "64",
            "--experiment",
            "attn-closure-test",
            "--label",
            "entry",
            "--out",
            str(tmp_path / "r.json"),
            "--outcomes",
            str(tmp_path / "o.json"),
        ]
        try:
            with pytest.raises(SystemExit) as excinfo:
                score_cli.entrypoint()
        finally:
            sys.argv = saved

        assert excinfo.value.code == 0

    def test_running_the_module_as_main_actually_scores(
        self, tmp_path: pathlib.Path, fake_hub: None
    ) -> None:
        module_name = "ordered_kernels.cli.score"
        items = tmp_path / "items.jsonl"
        items.write_text(_ITEMS_JSONL, encoding="utf-8")
        out = tmp_path / "score-main.json"
        saved_argv = sys.argv
        saved_module = sys.modules.pop(module_name, None)
        sys.argv = [
            "ordered-score",
            "--model",
            "gpt2",
            "--items",
            str(items),
            "--device",
            "cuda",
            "--max-seq-len",
            "64",
            "--experiment",
            "attn-closure-test",
            "--label",
            "main",
            "--out",
            str(out),
            "--outcomes",
            str(tmp_path / "score-main-outcomes.json"),
        ]
        try:
            with pytest.raises(SystemExit) as excinfo:
                runpy.run_module(module_name, run_name="__main__", alter_sys=False)
        finally:
            sys.argv = saved_argv
            if saved_module is not None:
                sys.modules[module_name] = saved_module

        assert excinfo.value.code == 0
        assert out.is_file()
