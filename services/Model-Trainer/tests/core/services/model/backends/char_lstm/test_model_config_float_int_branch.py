from __future__ import annotations

from pathlib import Path

from platform_core.json_utils import dump_json_str, load_json_str

from model_trainer.core.services.model.backends.char_lstm.model import CharLSTM


def test_from_pretrained_accepts_int_dropout(tmp_path: Path) -> None:
    # Save a model, then rewrite config with integer dropout to cover int->float branch.
    mdir = tmp_path / "m"
    mdir.mkdir()
    model = CharLSTM(
        vocab_size=8,
        embed_dim=4,
        hidden_dim=6,
        num_layers=1,
        dropout=0.0,
        max_seq_len=8,
    )
    model.save_pretrained(str(mdir))
    cfg_text = (mdir / "config.json").read_text(encoding="utf-8")
    obj = load_json_str(cfg_text)
    assert isinstance(obj, dict) and "dropout" in obj
    obj["dropout"] = 0  # integer
    (mdir / "config.json").write_text(dump_json_str(obj), encoding="utf-8")
    _ = CharLSTM.from_pretrained(str(mdir))
