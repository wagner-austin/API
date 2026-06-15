"""Download field minimap GIFs from tankpit.com.

Fetches ``fieldXX.gif`` for all known field numbers and saves them as
``fieldXX_r.gif`` in the project root, matching the naming convention
that ``_find_field_gif`` in ``sniffer.world_state`` expects.
"""

from __future__ import annotations

import sys
from pathlib import Path

from scripts import _test_hooks

_BASE_URL = "https://tankpit.com/play"
_FIELD_RANGE = range(1, 51)


def _project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parents[1]


def download_field_gifs(
    *,
    base_url: str = _BASE_URL,
    output_dir: Path | None = None,
) -> list[Path]:
    """Download all field GIF minimaps.

    Args:
        base_url: Base URL for tankpit.com.
        output_dir: Directory to save GIF files.

    Returns:
        List of paths to successfully downloaded files.
    """
    resolved_dir = output_dir if output_dir is not None else _project_root()
    downloaded: list[Path] = []

    for field_num in _FIELD_RANGE:
        field_name = f"field{field_num:02d}"
        url = f"{base_url}/{field_name}.gif"
        out_path = resolved_dir / f"{field_name}_r.gif"

        if _test_hooks.path_exists(out_path):
            sys.stdout.write(f"  skip {field_name} (already exists)\n")
            downloaded.append(out_path)
            continue

        response = _test_hooks.http_get(url)
        if response.status_code != 200:
            sys.stdout.write(f"  skip {field_name} (HTTP {response.status_code})\n")
            continue

        if not response.content.startswith(b"GIF"):
            sys.stdout.write(f"  skip {field_name} (not a GIF image)\n")
            continue

        out_path.write_bytes(response.content)
        sys.stdout.write(f"  saved {out_path.name} ({len(response.content)} bytes)\n")
        downloaded.append(out_path)

    return downloaded


def main() -> int:
    """Entry point for field GIF downloader.

    Returns:
        Exit code (0 for success).
    """
    _test_hooks.setup_rich_logging("INFO")
    sys.stdout.write("Downloading field GIFs from tankpit.com...\n")
    paths = download_field_gifs()
    sys.stdout.write(f"Done: {len(paths)} field GIFs available\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
