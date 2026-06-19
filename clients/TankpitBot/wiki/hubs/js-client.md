# JS Client Source

Complete reverse-engineering of `tpclient.js` (329 lines, ~82k tokens of minified Closure Compiler output). Every class, message handler, command encoder, constant table, and rendering pipeline, deobfuscated and annotated. All content extracted directly from the JS source — no guessing.

[JS Source Map](../pages/js-source-map.md) -- line-by-line annotated structure of tpclient.js: classes, V table, sprite system, rendering, audio, UI
[Client Commands](../pages/client-commands.md) -- every command the client can send to the server: opcodes, byte layouts, when each fires
[V Table Complete](../pages/v-table-complete.md) -- every server→client message handler: JS function name, parse logic, field-by-field layout
[Client Constants](../pages/client-constants.md) -- every hardcoded constant: rank thresholds, equipment names, error strings, timing values, sprite dimensions
[Client State Machine](../pages/client-state-machine.md) -- game states (s field), transitions, action queue, tick loop timing
[XOR Cipher](../pages/xor-cipher.md) -- the qb[] table, magic key derivation, za() encode/decode, which messages use it
[Rendering Pipeline](../pages/rendering-pipeline.md) -- canvas layers, sprite sheets, tile engine, animation system, dirty-rect tracking
[Chat Messages](../pages/chat-messages.md) -- all 65 predefined chat messages: IDs, text, team filters, position flags, voice keywords
[Connection Protocol](../pages/connection-protocol.md) -- full handshake: WebSocket framing, AUTH, game list, select, join, start, disconnect
[Terrain System](../pages/terrain-system.md) -- terrain byte encoding, adjacency lookup, pseudo-random variants, edge tiles, rock types
[Obstacle & Bridge Mechanics](../pages/obstacle-bridge-mechanics.md) -- pickup/drop/build rules, carry state, V.B handler, ferry interaction
[MAP_DATA Algorithm](../pages/map-data-algorithm.md) -- exact skip-RLE fuel dot parser and tank entry format from V.L handler
[Game Modes](../pages/game-modes.md) -- practice/normal/tournament encoding, mode-specific behavior differences
[Sound System](../pages/sound-system.md) -- all 18 audio buffers, playback channels, trigger events
[Decoration Encoding](../pages/decoration-encoding.md) -- 4-byte → 9-slot × 2-bit decoration packing, yg() decode, award names and thresholds
[Fingerprint Algorithm](../pages/fingerprint-algorithm.md) -- browser fingerprint data collection and MurmurHash3 implementation
[Toolbar Layout](../pages/toolbar-layout.md) -- 18 clickable regions with pixel-exact hitboxes, scope direction mapping
[Viewport Update Algorithm](../pages/viewport-update-algorithm.md) -- V.Z position step encoding, 24-bit entity packing, scroll optimization
[Playback System](../pages/playback-system.md) -- recording format, playback controller, timeline UI, speed control, seeking
[Input Handling](../pages/input-handling.md) -- mouse, keyboard, touch event processing, gesture recognition, coordinate conversion
[REST API](../pages/rest-api.md) -- official public API endpoints: active games, tank profiles, leaderboards, tournaments, bulletin board
