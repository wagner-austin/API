"""Tests for scripts/demo_video.py."""

from __future__ import annotations

import sys
from collections.abc import Generator
from pathlib import Path
from types import TracebackType

import pytest
from scripts import _test_hooks, demo_video
from scripts._test_hooks import HttpClientProtocol, RenderRequest, ResponseProtocol


class _FakeResponse:
    """Response that records whether its status was checked."""

    def __init__(self) -> None:
        """Initialize with an empty body and unchecked status."""
        self.raised = False

    @property
    def content(self) -> bytes:
        """Raw body.

        Returns:
            An empty body; the demo never parses it.
        """
        return b""

    def raise_for_status(self) -> ResponseProtocol:
        """Record that the status was checked.

        Returns:
            The response itself.
        """
        self.raised = True
        return self


class _FakeClient:
    """Client that records every request instead of issuing one."""

    def __init__(self) -> None:
        """Initialize with empty call logs."""
        self.gets: list[str] = []
        self.posts: list[tuple[str, RenderRequest]] = []
        self.entered = False
        self.exited = False

    def get(self, url: str) -> ResponseProtocol:
        """Record a GET.

        Args:
            url: Requested path.

        Returns:
            A fake response.
        """
        self.gets.append(url)
        return _FakeResponse()

    def post(self, url: str, *, json: RenderRequest) -> ResponseProtocol:
        """Record a POST.

        Args:
            url: Requested path.
            json: Request body.

        Returns:
            A fake response.
        """
        self.posts.append((url, json))
        return _FakeResponse()

    def __enter__(self) -> HttpClientProtocol:
        """Record context entry.

        Returns:
            The client itself.
        """
        self.entered = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Record context exit.

        Args:
            exc_type: Exception type, if propagating.
            exc_value: Exception instance, if propagating.
            traceback: Traceback, if propagating.
        """
        self.exited = True


@pytest.fixture(autouse=True)
def _restore_script_hooks() -> Generator[None, None, None]:
    """Restore script hooks after each test."""
    original = _test_hooks.http_client_factory
    yield
    _test_hooks.http_client_factory = original


def _install_fake_client() -> _FakeClient:
    """Install a fake HTTP client and return it.

    Returns:
        The fake client the script will use.
    """
    client = _FakeClient()

    def _factory(*, base_url: str, timeout_seconds: float) -> HttpClientProtocol:
        del base_url, timeout_seconds
        return client

    _test_hooks.http_client_factory = _factory
    return client


def test_build_demo_scene_uses_requested_dimensions() -> None:
    """Test the scene carries the requested resolution, timing and layers."""
    scene = demo_video._build_demo_scene(
        scene_id="demo", width=320, height=240, fps=24, duration=1.5
    )

    assert scene["id"] == "demo"
    assert scene["resolution"] == {"width": 320, "height": 240}
    assert scene["timing"]["fps"] == 24
    assert scene["timing"]["duration_seconds"] == 1.5
    assert [layer["module"] for layer in scene["layers"]] == ["black_background", "neon_orbs"]


def test_run_demo_calls_health_frames_and_video(tmp_path: Path) -> None:
    """Test the demo checks health, renders frames, then encodes video."""
    client = _install_fake_client()

    result = demo_video._run_demo("http://svc", tmp_path, width=64, height=64, fps=10, duration=0.5)

    assert client.gets == ["/healthz"]
    assert [url for url, _ in client.posts] == ["/render/frames", "/render/video"]
    assert client.entered is True
    assert client.exited is True
    assert result == (tmp_path.resolve() / "demo" / "demo.mp4").resolve()


def test_run_demo_posts_absolute_output_dir(tmp_path: Path) -> None:
    """Test both render calls receive the resolved output directory."""
    client = _install_fake_client()

    demo_video._run_demo("http://svc", tmp_path, width=64, height=64, fps=10, duration=0.5)

    expected = str(tmp_path.resolve())
    for _, body in client.posts:
        assert body["output_dir"] == expected
        assert body["scene"]["id"] == "demo"


def test_main_logs_video_path_and_returns_zero(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Test main() parses arguments, runs the demo and reports the video path."""
    _install_fake_client()
    original_argv = sys.argv
    sys.argv = [
        "demo_video.py",
        "--base-url",
        "http://svc",
        "--out",
        str(tmp_path),
        "--width",
        "64",
        "--height",
        "48",
        "--fps",
        "12",
        "--duration",
        "0.25",
    ]
    try:
        with caplog.at_level("INFO"):
            rc = demo_video.main()
    finally:
        sys.argv = original_argv

    assert rc == 0
    expected = str((tmp_path.resolve() / "demo" / "demo.mp4").resolve())
    assert any(expected in record.getMessage() for record in caplog.records)


def test_main_uses_defaults_without_arguments(tmp_path: Path) -> None:
    """Test main() falls back to its default arguments."""
    client = _install_fake_client()
    original_argv = sys.argv
    sys.argv = ["demo_video.py"]
    try:
        rc = demo_video.main()
    finally:
        sys.argv = original_argv

    assert rc == 0
    _, frames_body = client.posts[0]
    assert frames_body["scene"]["resolution"] == {"width": 256, "height": 256}
    assert frames_body["scene"]["timing"]["fps"] == 30
    assert frames_body["output_dir"] == str(Path("demo_output").resolve())
    del tmp_path


def test_real_http_client_builds_a_client_for_the_base_url() -> None:
    """Test the production factory returns a usable client (no request issued)."""
    client = _test_hooks._real_http_client(base_url="http://svc", timeout_seconds=1.5)

    with client as entered:
        assert entered is client


def test_reset_hooks_restores_the_real_factory() -> None:
    """Test reset_hooks puts the production implementations back."""
    _install_fake_client()
    assert _test_hooks.http_client_factory is not _test_hooks._real_http_client

    _test_hooks.reset_hooks()

    assert _test_hooks.http_client_factory is _test_hooks._real_http_client
    assert _test_hooks.event_source is _test_hooks._real_event_source


def test_entrypoint_runs_main(tmp_path: Path) -> None:
    """Test the `if __name__ == '__main__'` guard executes main()."""
    _install_fake_client()
    script_path = Path(demo_video.__file__)
    code = script_path.read_text(encoding="utf-8")
    globals_dict: dict[str, str] = {"__name__": "__main__", "__file__": str(script_path)}

    original_argv = sys.argv
    sys.argv = ["demo_video.py", "--out", str(tmp_path)]
    try:
        with pytest.raises(SystemExit) as excinfo:
            exec(compile(code, str(script_path), "exec"), globals_dict, globals_dict)
    finally:
        sys.argv = original_argv

    assert excinfo.value.code == 0
