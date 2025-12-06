from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol

from model_trainer.core.config.settings import Settings
from model_trainer.worker.job_utils import load_tokenizer_for_training as load_tok


class _SettingsFactory(Protocol):
    def __call__(
        self,
        *,
        artifacts_root: str | None = ...,
        runs_root: str | None = ...,
        logs_root: str | None = ...,
        data_root: str | None = ...,
        data_bank_api_url: str | None = ...,
        data_bank_api_key: str | None = ...,
        threads: int | None = ...,
        redis_url: str | None = ...,
        app_env: Literal["dev", "prod"] | None = ...,
        security_api_key: str | None = ...,
    ) -> Settings: ...


def test_load_tokenizer_for_training_char(
    tmp_path: Path, settings_factory: _SettingsFactory
) -> None:
    # Prepare a minimal char tokenizer artifact
    artifacts = tmp_path / "artifacts"
    tok_id = "tok-char"
    tok_dir = artifacts / "tokenizers" / tok_id
    tok_dir.mkdir(parents=True)
    (tok_dir / "tokenizer.json").write_text(
        '{"kind":"char","specials":["[PAD]","[UNK]","[BOS]","[EOS]"],"vocab":{"[PAD]":0,"[UNK]":1,"[BOS]":2,"[EOS]":3,"a":4}}',
        encoding="utf-8",
    )
    (tok_dir / "manifest.json").write_text(
        '{"created_at":0,"config":{"special_tokens":["[PAD]","[UNK]","[BOS]","[EOS]"]},"stats":{"coverage":1.0,"oov_rate":0.0,"token_count":5,"char_coverage":1.0}}',
        encoding="utf-8",
    )

    settings = settings_factory(artifacts_root=str(artifacts))
    handle = load_tok(settings, tok_id)
    ids = handle.encode("a")
    assert isinstance(ids, list) and ids == [4]
