/**
 * The pack preview: every frame a manifest names, laid out on the field.
 *
 * A pack author needs to see a pack before anyone plays on it, and the
 * renderer needs a page that exercises all of it. The preview fills the
 * viewport with ground, sets one row of each terrain kind with every
 * corner combination, a row of caches, rocks and mines, and every team's
 * sixteen facings plus both corpses. bootPreview builds the default pack
 * (or loads another manifest from a URL) and renders the preview into an
 * element; the page index.html calls it.
 */

import { hooks } from "./_test_hooks.js";
import { defaultManifest, paintDefaultSheet, SHEET_HEIGHT, SHEET_WIDTH } from "./default_pack.js";
import { GAME_HEIGHT } from "./layers.js";
import { decodeManifest, FACINGS, TEAMS, type SpriteManifest } from "./manifest.js";
import { Renderer } from "./renderer.js";
import { RenderError } from "./scaled_context.js";
import { decodeTerrain } from "./terrain.js";
import { NO_MINE, RockKind, type TileState } from "./tile_grid.js";

const VIEW = 16;
const WATER = 0x20;
const OBSTACLE = 0x40;

function plain(terrainByte: number): TileState {
  return { terrain: decodeTerrain(terrainByte), cache: 0, mine: NO_MINE, rock: RockKind.None };
}

/**
 * Lay every frame of the renderer's pack onto its grid and tanks.
 *
 * Rows 0 and 1 are water and obstacles with corner masks 0 to 15; row 2
 * holds fuel, equipment, the three rocks and the four mines on ground;
 * rows 4 to 11 hold the tanks, two rows of eight facings per team, and
 * row 12 the two corpses.
 *
 * @param renderer - The renderer to fill.
 * @returns How many tanks were placed.
 */
export function layPreview(renderer: Renderer): number {
  for (let row = 0; row < VIEW; row++) {
    for (let column = 0; column < VIEW; column++) {
      renderer.setTile(column + 1, row + 1, plain(0x0f));
    }
  }
  for (let corners = 0; corners < VIEW; corners++) {
    renderer.setTile(corners + 1, 1, plain(WATER | corners));
    renderer.setTile(corners + 1, 2, plain(OBSTACLE | corners));
  }
  const ground = plain(0x0f);
  const items: TileState[] = [
    { ...ground, cache: 1 },
    { ...ground, cache: -1 },
    { ...ground, rock: RockKind.A },
    { ...ground, rock: RockKind.B },
    { ...ground, rock: RockKind.Ferry },
    ...Array.from({ length: TEAMS }, (_, team) => ({ ...ground, mine: team })),
  ];
  items.forEach((state, index) => renderer.setTile(index + 1, 3, state));
  let id = 0;
  for (let team = 0; team < TEAMS; team++) {
    for (let facing = 0; facing < FACINGS; facing++) {
      const half = FACINGS / 2;
      renderer.setTank(id++, { team, facing, alive: true, deaths: 0, column: (facing % half) * 2, row: 4 + team * 2 + Math.floor(facing / half) });
    }
  }
  renderer.setTank(id++, { team: 0, facing: 0, alive: false, deaths: 0, column: 0, row: 13 });
  renderer.setTank(id++, { team: 0, facing: 0, alive: false, deaths: 1, column: 2, row: 13 });
  return id;
}

/**
 * Build the default pack: paint its sheet on a fresh canvas and name it by URL.
 *
 * @param document - The page's document.
 * @returns The manifest, its one sheet a URL of the painted canvas.
 * @throws RenderError RENDER_NO_CANVAS where the platform gives the canvas no 2D context.
 */
export function buildDefaultPack(document: Document): SpriteManifest {
  const canvas = document.createElement("canvas");
  canvas.width = SHEET_WIDTH;
  canvas.height = SHEET_HEIGHT;
  const surface = hooks().canvasContext(canvas);
  if (surface === null) {
    throw new RenderError("RENDER_NO_CANVAS: the default pack's sheet canvas has no 2D context");
  }
  paintDefaultSheet(surface, defaultManifest(""));
  return defaultManifest(hooks().canvasUrl(canvas));
}

/** Fetch a manifest's JSON from a URL. */
export interface FetchJson {
  (url: string): Promise<unknown>;
}

/**
 * Render the preview of a pack into an element.
 *
 * @param container - The element to draw in.
 * @param scale - Device pixels per game pixel, usually the window's devicePixelRatio.
 * @param packUrl - A manifest's URL, or null for the default pack.
 * @param fetchJson - How the manifest is fetched.
 * @returns The renderer, its first frame drawn.
 * @throws ManifestError For a manifest that does not decode; RenderError and RENDER_IMAGE from the renderer.
 */
export async function bootPreview(container: HTMLElement, scale: number, packUrl: string | null, fetchJson: FetchJson): Promise<Renderer> {
  const manifest = packUrl === null ? buildDefaultPack(container.ownerDocument) : decodeManifest(await fetchJson(packUrl));
  const renderer = await Renderer.create(container, scale, manifest);
  layPreview(renderer);
  renderer.renderFrame();
  container.addEventListener("click", (event) => {
    const bounds = container.getBoundingClientRect();
    const y = event.clientY - bounds.top - GAME_HEIGHT;
    if (y >= 0) {
      renderer.clickToolbar(event.clientX - bounds.left, y);
    }
  });
  return renderer;
}

/** The part of a window the preview page reads; the real window is one. */
export interface PreviewPage {
  readonly document: Document;
  readonly location: { readonly search: string };
  readonly devicePixelRatio: number;
  fetch(url: string): Promise<Response>;
}

/**
 * Start the preview page: the #game element, ?pack= for another pack, the page's fetch.
 *
 * @param page - The page's window.
 * @returns The renderer.
 * @throws RenderError RENDER_NO_GAME when the page has no #game element, RENDER_PACK when the pack URL does not answer 2xx.
 */
export async function startPreview(page: PreviewPage): Promise<Renderer> {
  const game = page.document.getElementById("game");
  if (game === null) {
    throw new RenderError("RENDER_NO_GAME: the page has no #game element");
  }
  const pack = new URLSearchParams(page.location.search).get("pack");
  const fetchJson = async (url: string): Promise<unknown> => {
    const response = await page.fetch(url);
    if (!response.ok) {
      throw new RenderError(`RENDER_PACK: ${url} answered ${response.status}`);
    }
    return response.json();
  };
  return bootPreview(game, page.devicePixelRatio, pack, fetchJson);
}
