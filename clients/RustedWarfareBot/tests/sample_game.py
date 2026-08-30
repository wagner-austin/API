"""A real game directory, small enough to build in a test.

Not a fake. Every file here is written to a real filesystem and read back by
the real code, because what the fingerprint records is bytes on disk and a
stand-in that skipped the disk would be measuring the stand-in.

It carries all three of the things the packages axis names -- the engine's
jar, the bundled runtime with its own release file, and an asset tree -- so a
test can move exactly one of them and watch the axis notice.
"""

from __future__ import annotations

from pathlib import Path

from rw_bot.harness.jvm import JVM_RELEASE_FILE, JVM_VERSION_KEY, jvm_dir

#: The platform the cluster runs, and the one these trees are shaped for. The
#: Windows spelling is reachable from the same helper by passing it.
LINUX = "linux"

#: What the Linux depot's runtime states about itself, read off the real
#: ``jvm-linux/release`` on 2026-08-29.
SAMPLE_JAVA_VERSION = "1.8.0_131"

#: Where the sample asset tree puts a map. A path with a directory in it, so
#: the listing exercises a nested entry rather than only a flat one.
SAMPLE_ASSET_PATH = "maps/skirmish.tmx"

#: An empty marker file beside it. The real tree carries these -- the game
#: enables its builtin mods with one -- and the staging contract refuses a
#: zero-byte file, so a tree identity that borrowed that rule would refuse a
#: real game directory.
SAMPLE_MARKER_PATH = "builtin_mods_enabled"


def write_sample_game(
    root: Path,
    *,
    platform: str = LINUX,
    jar: bytes = b"pretend this is a game",
    java_version: str = SAMPLE_JAVA_VERSION,
    runtime: bytes = b"pretend this is a java launcher",
    assets: bytes = b"pretend this is a map",
) -> Path:
    """Write a game directory carrying an engine, a runtime and assets.

    Args:
        root: Directory to build. Created along with its parents.
        platform: A ``sys.platform`` value deciding the runtime's directory
            name, so a Windows workstation can build the tree a Linux node
            would have.
        jar: Bytes of the engine's jar.
        java_version: What the runtime states as its ``JAVA_VERSION``.
        runtime: Bytes of a file inside the runtime's tree.
        assets: Bytes of a file inside the asset tree.

    Returns:
        The game directory.
    """
    runtime_root = root / jvm_dir(platform)
    (runtime_root / "bin").mkdir(parents=True, exist_ok=True)
    (root / "assets" / "maps").mkdir(parents=True, exist_ok=True)

    (root / "game-lib.jar").write_bytes(jar)
    (runtime_root / JVM_RELEASE_FILE).write_text(
        f'{JVM_VERSION_KEY}="{java_version}"\nOS_ARCH="amd64"\n', encoding="utf-8"
    )
    (runtime_root / "bin" / "java").write_bytes(runtime)
    (root / "assets" / SAMPLE_MARKER_PATH).write_bytes(b"")
    (root / "assets" / SAMPLE_ASSET_PATH).write_bytes(assets)
    return root


__all__ = [
    "LINUX",
    "SAMPLE_ASSET_PATH",
    "SAMPLE_JAVA_VERSION",
    "SAMPLE_MARKER_PATH",
    "write_sample_game",
]
