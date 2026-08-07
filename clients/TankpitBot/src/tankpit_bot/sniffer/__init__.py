"""WebSocket traffic sniffer module.

Import from specific submodules:
- ``sniffer.core``: WebSocketSniffer class and entry points
- ``sniffer.decoders``: Message decoding functions
- ``sniffer.trackers``: Tracker instances and initialization
- ``sniffer.world_state``: World state from radar/movement messages
- ``sniffer.formatters``: Message formatting

The cipher lives one layer down in ``capture.xor``: a session's table
is built by ``build_session_xor_table`` and threaded through as a
value, never held module-level ([[session-state-deglobalisation]]).
"""
