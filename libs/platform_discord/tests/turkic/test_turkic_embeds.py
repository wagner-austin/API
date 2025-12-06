from __future__ import annotations

import pytest

from platform_discord.turkic.embeds import build_turkic_embed
from platform_discord.turkic.types import JobConfig, JobProgress, JobResult


@pytest.fixture(autouse=True)
def _patch_discord_embed_module(monkeypatch: pytest.MonkeyPatch) -> None:
    import platform_discord.embed_helpers as eh

    class _FakeColor:
        def __init__(self, value: int) -> None:
            self.value = int(value)

    class _Field:
        def __init__(self, name: str, value: str, inline: bool) -> None:
            self.name = name
            self.value = value
            self.inline = inline

    class _Footer:
        def __init__(self) -> None:
            self.text: str | None = None

    class _FakeEmbed:
        def __init__(
            self,
            *,
            title: str | None = None,
            description: str | None = None,
            color: _FakeColor | None = None,
        ) -> None:
            self.title = title
            self.description = description
            self.color = color
            self.footer = _Footer()
            self.fields: list[_Field] = []

        def add_field(self, *, name: str, value: str, inline: bool = True) -> _FakeEmbed:
            self.fields.append(_Field(name, value, inline))
            return self

        def set_footer(self, *, text: str) -> _FakeEmbed:
            self.footer.text = text
            return self

    class _FakeDiscordModule:
        Embed = _FakeEmbed
        Color = _FakeColor

    def _fake_loader() -> type[_FakeDiscordModule]:
        return _FakeDiscordModule

    monkeypatch.setattr(eh, "_load_discord_module", _fake_loader, raising=True)


def test_turkic_build_starting_progress_completed_failed() -> None:
    cfg: JobConfig = {"queue": "turkic"}
    e1 = build_turkic_embed(job_id="j", config=cfg, status="starting")
    d1 = e1.to_dict()
    assert "footer" in d1 and "fields" in d1

    prog: JobProgress = {"progress": 50, "message": "halfway"}
    e2 = build_turkic_embed(job_id="j", config=cfg, status="processing", progress=prog)
    d2 = e2.to_dict()
    names2 = [f["name"] for f in d2.get("fields", [])]
    assert "Status" in names2

    res: JobResult = {"result_id": "fid", "result_bytes": 2048}
    e3 = build_turkic_embed(job_id="j", config=cfg, status="completed", result=res)
    d3 = e3.to_dict()
    names3 = [f["name"] for f in d3.get("fields", [])]
    assert "Result" in names3

    e4 = build_turkic_embed(
        job_id="j", config=cfg, status="failed", error_kind="system", error_message="boom"
    )
    d4 = e4.to_dict()
    names4 = [f["name"] for f in d4.get("fields", [])]
    assert "System Error" in names4


def test_turkic_progress_without_message_and_completed_without_result() -> None:
    cfg: JobConfig = {"queue": "turkic"}
    prog: JobProgress = {"progress": 5}
    e = build_turkic_embed(job_id="j2", config=cfg, status="processing", progress=prog)
    d = e.to_dict()
    # Status field present even without message
    names = [f["name"] for f in d.get("fields", [])]
    assert "Status" in names

    # Completed with no result should not add Results section
    e2 = build_turkic_embed(job_id="j3", config=cfg, status="completed", result=None)
    d2 = e2.to_dict()
    names2 = [f["name"] for f in d2.get("fields", [])]
    assert "Result" not in names2


def test_turkic_user_failure_branch() -> None:
    cfg: JobConfig = {"queue": "turkic"}
    e = build_turkic_embed(
        job_id="j4", config=cfg, status="failed", error_kind="user", error_message="bad"
    )
    d = e.to_dict()
    names = [f["name"] for f in d.get("fields", [])]
    assert "Configuration Issue" in names and "Next Steps" in names
