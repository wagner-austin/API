from __future__ import annotations

from pathlib import Path

from model_trainer.core.services.model.backends.gpt2.io import _TokWrapper, token_ids
from model_trainer.core.services.registries import TokenizerRegistry
from model_trainer.core.services.tokenizer.bpe_backend import BPEBackend
from model_trainer.core.services.tokenizer.spm_backend import SentencePieceBackend


def test_tokenizer_registry_get_known() -> None:
    reg = TokenizerRegistry(backends={"bpe": BPEBackend()})
    out = reg.get("bpe")
    assert hasattr(out, "inspect") and hasattr(out, "load")


def test_bpe_backend_inspect_reads_manifest(tmp_path: Path) -> None:
    # Create folder with manifest
    base = tmp_path / "tok"
    base.mkdir()
    manifest = {
        "stats": {
            "coverage": 1.0,
            "oov_rate": 0.0,
            "token_count": 1,
            "char_coverage": 1.0,
        }
    }
    from platform_core.json_utils import dump_json_str

    (base / "manifest.json").write_text(dump_json_str(manifest), encoding="utf-8")
    stats = BPEBackend().inspect(str(base))
    assert stats.token_count == 1


def test_spm_backend_adapter_vocab_read_error(tmp_path: Path) -> None:
    # Create .model path with missing .vocab to trigger warning path;
    # adapter should still initialize
    model_path = str(tmp_path / "tok.model")
    Path(model_path).write_text("", encoding="utf-8")
    h = SentencePieceBackend().load(model_path)
    assert h.get_vocab_size() == 0


def test_tokwrapper_token_info_methods() -> None:
    class _Handle:
        def token_to_id(self: _Handle, token: str) -> int | None:
            return 7 if token == "[EOS]" else 0

        def get_vocab_size(self: _Handle) -> int:
            return 42

        def encode(self: _Handle, text: str) -> list[int]:
            return [1, 2]

    tw = _TokWrapper(_Handle())
    assert tw.token_to_id("[EOS]") == 7
    assert tw.get_vocab_size() == 42
    e, p, v = token_ids(_Handle())
    assert isinstance(e, int) and isinstance(p, int) and v == 42
