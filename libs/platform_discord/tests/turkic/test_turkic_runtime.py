from __future__ import annotations

import pytest

from platform_discord.turkic.runtime import (
    RequestAction,
    TurkicRuntime,
    new_runtime,
    on_completed,
    on_failed,
    on_progress,
    on_started,
)


def _rt() -> TurkicRuntime:
    return new_runtime()


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


def test_turkic_runtime_flow_with_user() -> None:
    rt = _rt()
    a1: RequestAction = on_started(rt, user_id=1, job_id="j", queue="turkic")
    if a1["embed"] is None:
        pytest.fail("expected embed in a1")
    assert a1["user_id"] == 1
    a2 = on_progress(rt, user_id=1, job_id="j", progress=10, message="ok")
    if a2["embed"] is None:
        pytest.fail("expected embed in a2")
    a3 = on_completed(rt, user_id=1, job_id="j", result_id="fid", result_bytes=1024)
    if a3["embed"] is None:
        pytest.fail("expected embed in a3")


def test_turkic_runtime_skips_when_no_user() -> None:
    rt = _rt()
    a1 = on_started(rt, user_id=None, job_id="x", queue="turkic")
    assert a1["embed"] is None and a1["user_id"] == 0
    a2 = on_failed(
        rt, user_id=None, job_id="x", error_kind="system", message="boom", status="failed"
    )
    assert a2["embed"] is None


def test_turkic_completed_without_user_skips() -> None:
    rt = _rt()
    a = on_completed(rt, user_id=None, job_id="j5", result_id="fid", result_bytes=10)
    assert a["embed"] is None


def test_turkic_progress_without_message_with_user_and_failed_with_user() -> None:
    rt = _rt()
    on_started(rt, user_id=7, job_id="j6", queue="turkic")
    a1 = on_progress(rt, user_id=7, job_id="j6", progress=25, message=None)
    assert a1["embed"] is not None and a1["user_id"] == 7
    a2 = on_failed(rt, user_id=7, job_id="j6", error_kind="system", message="x", status="failed")
    assert a2["embed"] is not None and a2["user_id"] == 7


def test_turkic_progress_without_user_skips() -> None:
    rt = _rt()
    a = on_progress(rt, user_id=None, job_id="j7", progress=1, message=None)
    assert a["embed"] is None
