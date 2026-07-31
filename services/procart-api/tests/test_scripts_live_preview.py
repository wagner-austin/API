"""Tests for scripts/live_preview.py.

The preview is a real pygame application. These tests replace only the two
things a headless run cannot do — opening a window and waiting for input — so
the surfarray conversion, blits and font rendering all execute for real against
an offscreen surface.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pygame
import pytest
from procart.types import Resolution
from scripts import _test_hooks, live_preview
from scripts.live_preview import RenderParams


class _OffscreenDisplay:
    """Display that hands back a plain surface instead of opening a window.

    Everything downstream — the surfarray conversion, the blits, the font
    rendering — runs against this exactly as it would against a window.
    """

    def __init__(self) -> None:
        """Initialize with no created surfaces and no presented frames."""
        self.created_sizes: list[tuple[int, int]] = []
        self.caption = ""
        self.presented = 0
        self.shutdowns = 0

    def create(self, size: tuple[int, int], caption: str) -> pygame.Surface:
        """Create an offscreen surface.

        Args:
            size: Surface size in pixels.
            caption: Window title, recorded but unused.

        Returns:
            A plain surface of the requested size.
        """
        pygame.font.init()
        self.caption = caption
        self.created_sizes.append(size)
        return pygame.Surface(size)

    def present(self) -> None:
        """Count a presented frame."""
        self.presented += 1

    def shutdown(self) -> None:
        """Count a shutdown."""
        self.shutdowns += 1


@pytest.fixture(autouse=True)
def _restore_script_hooks() -> Generator[None, None, None]:
    """Restore script hooks after each test."""
    original_events = _test_hooks.event_source
    original_display = _test_hooks.display
    yield
    _test_hooks.event_source = original_events
    _test_hooks.display = original_display


def _defaults() -> RenderParams:
    """Build the preview's default parameters.

    Returns:
        A fresh copy of the default parameter set.
    """
    return {
        "orb_count": 3,
        "core_radius": 0.03,
        "halo_radius": 0.15,
        "core_intensity": 2.0,
        "halo_intensity": 0.4,
        "speed": 0.5,
    }


def _install_events(frames: list[list[pygame.event.Event]]) -> None:
    """Install an event source that replays the given frames in order.

    Args:
        frames: One list of events per loop iteration. The source returns an
            empty list once the frames run out.
    """
    remaining = list(frames)

    def _source() -> list[pygame.event.Event]:
        if not remaining:
            return []
        return remaining.pop(0)

    _test_hooks.event_source = _source


def test_render_frame_produces_rgb_image() -> None:
    """Test render_frame returns an (h, w, 3) image for the requested size."""
    resolution: Resolution = {"width": 32, "height": 16}

    frame = live_preview.render_frame(0.25, resolution, 2, 0.03, 0.15, 2.0, 0.4)

    assert frame.shape == (16, 32, 3)


def test_apply_delta_clamps_integer_parameter() -> None:
    """Test the orb count stays inside its bounds and stays an int."""
    params = _defaults()

    live_preview._apply_delta_to_param(params, "orb_count", -10.0, 1.0, 20.0)
    assert params["orb_count"] == 1

    live_preview._apply_delta_to_param(params, "orb_count", 100.0, 1.0, 20.0)
    assert params["orb_count"] == 20


def test_apply_delta_clamps_float_parameter() -> None:
    """Test a float parameter is clamped at both ends."""
    params = _defaults()

    live_preview._apply_delta_to_param(params, "core_radius", -1.0, 0.01, 0.2)
    assert params["core_radius"] == 0.01

    live_preview._apply_delta_to_param(params, "core_radius", 1.0, 0.01, 0.2)
    assert params["core_radius"] == 0.2


def test_build_key_map_covers_every_documented_key() -> None:
    """Test the key map wires the twelve adjustment keys."""
    key_map = live_preview._build_key_map()

    assert len(key_map) == 12
    assert key_map[pygame.K_1]["param"] == "orb_count"
    assert key_map[pygame.K_1]["delta"] == -1.0
    assert key_map[pygame.K_EQUALS]["param"] == "speed"


def test_handle_key_event_resets_on_r() -> None:
    """Test R restores the defaults."""
    defaults = _defaults()
    params = _defaults()
    params["orb_count"] = 17

    result = live_preview._handle_key_event(
        pygame.K_r, params, defaults, live_preview._build_key_map()
    )

    assert result == defaults


def test_handle_key_event_applies_mapped_key() -> None:
    """Test a mapped key adjusts its parameter."""
    defaults = _defaults()
    params = _defaults()

    result = live_preview._handle_key_event(
        pygame.K_2, params, defaults, live_preview._build_key_map()
    )

    assert result["orb_count"] == 4


def test_handle_key_event_ignores_unmapped_key() -> None:
    """Test an unmapped key leaves the parameters untouched."""
    defaults = _defaults()
    params = _defaults()

    result = live_preview._handle_key_event(
        pygame.K_z, params, defaults, live_preview._build_key_map()
    )

    assert result == defaults


def test_process_events_stops_on_quit() -> None:
    """Test a QUIT event ends the loop."""
    _install_events([[pygame.event.Event(pygame.QUIT)]])

    params, running = live_preview._process_events(
        _defaults(), _defaults(), live_preview._build_key_map()
    )

    assert running is False
    assert params == _defaults()


def test_process_events_stops_on_escape() -> None:
    """Test ESC ends the loop."""
    _install_events([[pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE)]])

    _, running = live_preview._process_events(
        _defaults(), _defaults(), live_preview._build_key_map()
    )

    assert running is False


def test_process_events_applies_keydown_and_continues() -> None:
    """Test a parameter key is applied and the loop continues."""
    _install_events([[pygame.event.Event(pygame.KEYDOWN, key=pygame.K_2)]])

    params, running = live_preview._process_events(
        _defaults(), _defaults(), live_preview._build_key_map()
    )

    assert running is True
    assert params["orb_count"] == 4


def test_process_events_applies_every_key_in_one_frame() -> None:
    """Test a frame carrying several keys applies all of them."""
    _install_events(
        [
            [
                pygame.event.Event(pygame.KEYDOWN, key=pygame.K_2),
                pygame.event.Event(pygame.KEYDOWN, key=pygame.K_4),
            ]
        ]
    )

    params, running = live_preview._process_events(
        _defaults(), _defaults(), live_preview._build_key_map()
    )

    assert running is True
    assert params["orb_count"] == 4
    assert params["core_radius"] == pytest.approx(0.035)


def test_process_events_ignores_events_that_are_not_keydown_or_quit() -> None:
    """Test a non-key event is skipped without touching the parameters."""
    _install_events(
        [
            [
                pygame.event.Event(pygame.KEYUP, key=pygame.K_2),
                pygame.event.Event(pygame.KEYDOWN, key=pygame.K_2),
            ]
        ]
    )

    params, running = live_preview._process_events(
        _defaults(), _defaults(), live_preview._build_key_map()
    )

    assert running is True
    assert params["orb_count"] == 4


def test_real_event_source_drains_the_pygame_queue() -> None:
    """Test the production event source reads pygame's own queue."""
    pygame.init()
    try:
        pygame.display.set_mode((32, 32))
        pygame.event.clear()
        pygame.event.post(pygame.event.Event(pygame.QUIT))

        events = _test_hooks._real_event_source()

        assert any(event.type == pygame.QUIT for event in events)
    finally:
        pygame.quit()


def test_process_events_with_empty_queue_continues() -> None:
    """Test an empty queue leaves state alone and keeps running."""
    _install_events([[]])

    params, running = live_preview._process_events(
        _defaults(), _defaults(), live_preview._build_key_map()
    )

    assert running is True
    assert params == _defaults()


def test_draw_overlay_writes_to_the_surface() -> None:
    """Test the overlay renders its lines onto the target surface."""
    pygame.font.init()
    screen = pygame.Surface((200, 200))
    font = pygame.font.Font(None, 24)
    clock = pygame.time.Clock()
    before = screen.get_at((10, 10))

    live_preview._draw_overlay(screen, font, clock, _defaults())

    assert screen.get_at((12, 14)) != before


def test_main_runs_frames_then_quits() -> None:
    """Test main() renders a frame per event batch until told to stop."""
    display = _OffscreenDisplay()
    _test_hooks.display = display
    _install_events(
        [
            [],
            [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_2)],
            [pygame.event.Event(pygame.QUIT)],
        ]
    )

    # Three event batches means three rendered frames: the batch carrying QUIT
    # is still drawn before the loop condition is re-checked.
    assert live_preview.main() == 0
    assert display.presented == 3
    assert display.shutdowns == 1
    assert display.created_sizes == [(512, 512)]


def test_entrypoint_runs_main() -> None:
    """Test the `if __name__ == '__main__'` guard executes main()."""
    _test_hooks.display = _OffscreenDisplay()
    _install_events([[pygame.event.Event(pygame.QUIT)]])
    script_path = Path(live_preview.__file__)
    code = script_path.read_text(encoding="utf-8")
    globals_dict: dict[str, str] = {"__name__": "__main__", "__file__": str(script_path)}

    with pytest.raises(SystemExit) as excinfo:
        exec(compile(code, str(script_path), "exec"), globals_dict, globals_dict)

    assert excinfo.value.code == 0
