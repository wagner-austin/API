"""The production hook bindings.

The implementations are platform_core's and are tested against the real
filesystem and stdout in its suite; what is this package's own is that
production binds exactly those, and that a reset restores them.
"""

from __future__ import annotations

from platform_core.journal_cursor import file_is_present, read_file_bytes, write_file_text
from platform_core.mcp_client import urllib_mcp_post
from platform_core.report_line import emit_line

from fleet_health_wake import _test_hooks


class TestDefaults:
    def test_the_production_bindings_are_platform_cores(self) -> None:
        def _discard(line: str) -> None:
            del line

        _test_hooks.emit = _discard
        _test_hooks.reset_hooks()
        assert _test_hooks.http_post is urllib_mcp_post
        assert _test_hooks.read_bytes is read_file_bytes
        assert _test_hooks.write_text is write_file_text
        assert _test_hooks.file_exists is file_is_present
        assert _test_hooks.emit is emit_line
