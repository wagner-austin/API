---
title: Sim Renderer (the TypeScript client's drawing, any sprite pack)
tags: [sim, architecture, multiplayer, rendering, typescript]
related:
  - "[[rendering-pipeline]]"
  - "[[toolbar-layout]]"
  - "[[sim-network-server]]"
  - "[[js-source-map]]"
source_paths:
  - "web/src/renderer.ts"
  - "web/src/manifest.ts"
  - "web/src/default_pack.ts"
  - "web/src/toolbar.ts"
  - "web/src/tile_grid.ts"
  - "web/src/layers.ts"
  - "web/src/scaled_context.ts"
  - "web/src/dirty_rect.ts"
  - "web/src/preview.ts"
  - "web/tests/renderer.spec.ts"
  - "web/tests/toolbar.spec.ts"
  - "web/tests/manifest.spec.ts"
source_git_blobs:
  "web/src/renderer.ts": "53b08e96c55b9fdab8cf5a8a86abec5225f19e68"
  "web/src/manifest.ts": "0922adddb90b68aec29ffe17939c1545b86ac4b8"
  "web/src/default_pack.ts": "10e2ba6ef56f24688859a5aca60258a51057fc7e"
  "web/src/toolbar.ts": "5e93c123566fa183f04a689ed545ba2360653387"
  "web/src/tile_grid.ts": "80908f529344c9ca0e3d88c62d9c2419d0671248"
  "web/src/layers.ts": "b72ade9b18f5cb6abfb936db6b79b1ff1589f020"
  "web/src/scaled_context.ts": "8a342b7f1c522653a677b858f102be2622d99d99"
  "web/src/dirty_rect.ts": "df686169e49aaee37c54460d79722d1716282cad"
  "web/src/preview.ts": "ec907ddd6fc5ced554dd48314a3a0d7889ceb6ca"
  "web/tests/renderer.spec.ts": "c815bd80a73bddc5566cc68f642a8947163ce615"
  "web/tests/toolbar.spec.ts": "9ba71c7b931f26dc26598e6f46953e69fd124184"
  "web/tests/manifest.spec.ts": "875df0dca23aee5800c3a3ed279b087be8cc8622"
provenance:
  - "Board task b008ab91 (the multiplayer track), Phase 5, 2026-10-06: the built preview page served from web/ and loaded in Playwright's Chromium at device scale 2; six canvases at 768x512 (the menu 768x96), 45,536 opaque pixels on the tanks layer, no page error, the Radar button pressed by a click 10 px into the strip"
fact_checked: "2026-10-06"
confidence: high
hubs: [architecture]
---

# Sim renderer: the TypeScript client's drawing, any sprite pack

*Phase 5 of the multiplayer track (board task `b008ab91`), 2026-10-06.*

`clients/TankpitBot/web` is the TypeScript package that draws the sim.
It follows the game client's own rendering design as
[[rendering-pipeline]] and [[toolbar-layout]] record it. It does not
take the game's art: every picture comes from a **sprite pack**. The
game's own client loads custom packs from its Graphics tab, so a pack
is the faithful design, and it keeps the art separate from the code.

## A pack is a manifest and its sheets

A `SpriteManifest` lists sheets by id and URL. Each frame the renderer
draws names a sheet and a rectangle of it:

- the ground, water and obstacle bases;
- four corner overlays each for water and obstacles;
- the fuel and equipment dots, three rocks and four team mines;
- sixteen facings for each of the four teams, and two corpses.

`decodeManifest` reads a manifest's JSON field by field. A wrong count,
a frame with no size, or a frame naming a sheet the manifest does not
list fails with a `MANIFEST_*` code that names the field.
`encodeManifest` is its inverse, and a test round-trips the default
pack through JSON text.[^1]

The default pack is this project's own block art: flat terrain, corner
blocks, dots, and a team-coloured hull with a barrel along its facing.
It is painted onto one 112x436 canvas at start-up and named by that
canvas's data URL. Facing 0 points north and each step turns 22.5
degrees clockwise, the order of the client's projectile offset table
([[js-source-map]]). The colours are a palette of our own. Nothing in
the package comes from tankpit.com.

## What it draws, and the game client's design it keeps

- **Six layers.** Background, tanks, action, map, overlay and the menu
  strip, stacked by z-index. The game area is 384x256 and the strip is
  384x48 below it.
- **A scaled context.** Each canvas is sized in device pixels and drawn
  on in game pixels. A scale that is a whole number of quarters draws
  exactly. Any other scale floors the corner and draws 3% oversize,
  rounded up, as the client's ScaledContext does. Smoothing is off, so
  sprite edges stay sharp: the first live run showed seams where
  smoothing sampled neighbouring sheet cells.
- **An 18x18 tile grid.** It holds the 16x16 viewport and a one-tile
  border. Setting a tile marks it dirty. A frame draws only the dirty
  viewport tiles, in the client's order: base, open corners, cache dot,
  rock, then mine.
- **Dirty rects for tanks.** A tank that moves or leaves clears exactly
  what it last covered. Every still tank whose pixels that cleared is
  erased and redrawn too, through a chain of overlaps.[^2]
- **The toolbar.** It has the client's eighteen regions, stored as the
  client stores them, so its click test is the client's `xc` with the y
  coordinate shifted by three. `scopeDirection` is the client's `qe`
  remap and `mapScroll` its `le` step. A click presses its button and
  releases the last one.[^3]

## Running it

```
cd clients/TankpitBot/web
npm ci
make check                     # lint, typecheck, vitest at 100% under check-budget
npm run build                  # tsc to dist/
python -m http.server 8093     # then open /index.html, or /index.html?pack=<manifest URL>
```

`index.html` is the pack preview. It lays every frame of a pack onto
the field (`layPreview`) so a pack author can see the pack before
anyone plays on it. The browser seams are `_test_hooks`: the canvas
context, image loading and the canvas's data URL. Tests bind a
`RecordingSurface` that records every draw, clear and fill in order.
The package is its own fleet project, `clients/TankpitBot/web`, so the
Python package's check does not also carry the Node suite.

Checked live on 2026-10-06: the built preview ran in Playwright's
Chromium at device scale 2. It drew six canvases with no page error, and
a click 10 px into the strip pressed Radar.

## Fed from the socket

[[sim-web-client]] is the browser client that feeds this renderer. It
AUTHs and runs the lobby, un-XORs the 0x2E batches, and turns them into
`setTile` and `setTank` through its world view. The play page,
`play.html`, is served by the sim server itself.

[^1]: `web/src/manifest.ts`, `decodeManifest` and `encodeManifest`; `web/tests/manifest.spec.ts`, `reads back exactly what encodeManifest wrote, through JSON text`.
[^2]: `web/src/renderer.ts`, `Renderer.renderFrame` and `Renderer.removeTank`; `web/tests/renderer.spec.ts`, `redraws a still tank whose pixels a moving tank's erase cleared, and only that one` and `follows an erase through a chain of overlapping tanks`.
[^3]: `web/src/toolbar.ts`, `TOOLBAR_REGIONS`, `regionAt` and `scopeDirection`; `web/tests/toolbar.spec.ts`, `are the client's four arrays, region by region`.
