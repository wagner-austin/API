/**
 * This project's own sprite pack, painted at start-up.
 *
 * The renderer draws whatever a manifest names, so it needs a pack to
 * draw anything. This one is plain block art painted onto one canvas
 * when the page loads: flat terrain, corner blocks, dots for caches and
 * mines, and a tank as a team-coloured hull with a barrel along its
 * facing. It is deliberately nothing like the game's own art, which this
 * project does not ship; a better pack replaces it by publishing a
 * manifest and its sheets, with no change to the renderer.
 *
 * The sheet, 112x436, in rows of 24x16 tile cells and then 28x20 tank
 * cells:
 *
 *   y   0  ground, water, obstacle
 *   y  16  water corners NE, SE, SW, NW (a block in that corner, the rest clear)
 *   y  32  obstacle corners, likewise
 *   y  48  fuel dot, equipment dot, rock A, rock B
 *   y  64  ferry rock
 *   y  80  mines of team 0 to 3
 *   y  96  tanks: a column per team, a row per facing, 0 north clockwise
 *   y 416  corpses, two variants
 *
 * Facing 0 points north and each step turns 22.5 degrees clockwise, the
 * game's own order (its projectile offset table, wiki js-source-map).
 */

import { CORNERS, FACINGS, TEAMS, type SpriteFrame, type SpriteManifest } from "./manifest.js";
import { RenderError, type Surface } from "./scaled_context.js";

export const DEFAULT_SHEET = "default";
export const TILE_W = 24;
export const TILE_H = 16;
export const TANK_W = 28;
export const TANK_H = 20;

/** The team colours, troop 0 to 3: this project's own palette. */
export const TEAM_COLOURS: readonly string[] = ["#d94c3d", "#3d7bd9", "#e0b83a", "#4caf6a"];

/** The terrain and dot colours. */
export const PALETTE = {
  ground: "#8a9a5b",
  water: "#3a6ea5",
  obstacle: "#5b4a3a",
  waterEdge: "#b7d3ea",
  obstacleEdge: "#2f261d",
  fuel: "#f2e94e",
  equipment: "#e07be0",
  rock: "#9e9e9e",
  ferry: "#c8a165",
  hull: "#1d1d1d",
  corpse: "#4a4a4a",
};

const ROW_TERRAIN = 0;
const ROW_WATER_CORNERS = 16;
const ROW_OBSTACLE_CORNERS = 32;
const ROW_DOTS = 48;
const ROW_FERRY = 64;
const ROW_MINES = 80;
const TANK_TOP = 96;
const CORPSE_TOP = TANK_TOP + FACINGS * TANK_H;

export const SHEET_WIDTH = TEAMS * TANK_W;
export const SHEET_HEIGHT = CORPSE_TOP + TANK_H;

const CORNER_W = 8;
const CORNER_H = 5;
const DOT = 4;

function range(count: number): number[] {
  return Array.from({ length: count }, (_, index) => index);
}

function cell(column: number, top: number): SpriteFrame {
  return { sheet: DEFAULT_SHEET, x: column * TILE_W, y: top, w: TILE_W, h: TILE_H };
}

function tankCell(column: number, top: number): SpriteFrame {
  return { sheet: DEFAULT_SHEET, x: column * TANK_W, y: top, w: TANK_W, h: TANK_H };
}

/**
 * The default pack's manifest.
 *
 * @param sheetUrl - Where the painted sheet is served from, usually a data URL.
 * @returns The manifest.
 */
export function defaultManifest(sheetUrl: string): SpriteManifest {
  return {
    name: "default",
    sheets: new Map([[DEFAULT_SHEET, sheetUrl]]),
    tileWidth: TILE_W,
    tileHeight: TILE_H,
    terrain: {
      ground: cell(0, ROW_TERRAIN),
      water: cell(1, ROW_TERRAIN),
      obstacle: cell(2, ROW_TERRAIN),
      waterCorners: range(CORNERS).map((corner) => cell(corner, ROW_WATER_CORNERS)),
      obstacleCorners: range(CORNERS).map((corner) => cell(corner, ROW_OBSTACLE_CORNERS)),
    },
    fuel: cell(0, ROW_DOTS),
    equipment: cell(1, ROW_DOTS),
    rocks: [cell(2, ROW_DOTS), cell(3, ROW_DOTS), cell(0, ROW_FERRY)],
    mines: range(TEAMS).map((team) => cell(team, ROW_MINES)),
    tanks: range(TEAMS).map((team) => range(FACINGS).map((facing) => tankCell(team, TANK_TOP + facing * TANK_H))),
    corpses: [tankCell(0, CORPSE_TOP), tankCell(1, CORPSE_TOP)],
  };
}

/** Where corner c's block sits in a tile: NE, SE, SW, NW. */
export function cornerOffset(corner: number): { readonly x: number; readonly y: number } {
  const east = corner === 0 || corner === 1;
  const south = corner === 1 || corner === 2;
  return { x: east ? TILE_W - CORNER_W : 0, y: south ? TILE_H - CORNER_H : 0 };
}

/** The unit step a facing points along: 0 north, clockwise in sixteenths. */
export function facingVector(facing: number): { readonly x: number; readonly y: number } {
  const angle = (facing * Math.PI * 2) / FACINGS;
  return { x: Math.sin(angle), y: -Math.cos(angle) };
}

/**
 * A team's colour.
 *
 * @param team - The troop, 0 to 3.
 * @returns Its colour.
 * @throws RenderError RENDER_TEAM for any other team.
 */
export function teamColour(team: number): string {
  const colour = TEAM_COLOURS[team];
  if (colour === undefined) {
    throw new RenderError(`RENDER_TEAM: team ${team} has no colour`);
  }
  return colour;
}

function paintBlock(surface: Surface, colour: string, x: number, y: number, w: number, h: number): void {
  surface.fillStyle = colour;
  surface.fillRect(x, y, w, h);
}

function paintDot(surface: Surface, colour: string, frame: SpriteFrame): void {
  paintBlock(surface, colour, frame.x + (frame.w - DOT) / 2, frame.y + (frame.h - DOT) / 2, DOT, DOT);
}

function paintTank(surface: Surface, colour: string, frame: SpriteFrame, facing: number): void {
  const centreX = frame.x + frame.w / 2;
  const centreY = frame.y + frame.h / 2;
  paintBlock(surface, PALETTE.hull, centreX - 7, centreY - 6, 14, 12);
  paintBlock(surface, colour, centreX - 5, centreY - 4, 10, 8);
  const step = facingVector(facing);
  for (let along = 2; along <= 9; along++) {
    paintBlock(surface, PALETTE.hull, Math.round(centreX + step.x * along) - 1, Math.round(centreY + step.y * along) - 1, 2, 2);
  }
}

/**
 * Paint the default sheet onto a surface sized SHEET_WIDTH x SHEET_HEIGHT.
 *
 * @param surface - The sheet canvas's 2D context.
 * @param manifest - The default manifest, whose frames are the rectangles painted.
 */
export function paintDefaultSheet(surface: Surface, manifest: SpriteManifest): void {
  surface.clearRect(0, 0, SHEET_WIDTH, SHEET_HEIGHT);
  const terrain = manifest.terrain;
  for (const [frame, colour] of [
    [terrain.ground, PALETTE.ground],
    [terrain.water, PALETTE.water],
    [terrain.obstacle, PALETTE.obstacle],
  ] as const) {
    paintBlock(surface, colour, frame.x, frame.y, frame.w, frame.h);
  }
  for (const [frames, colour] of [
    [terrain.waterCorners, PALETTE.waterEdge],
    [terrain.obstacleCorners, PALETTE.obstacleEdge],
  ] as const) {
    frames.forEach((frame, corner) => {
      const offset = cornerOffset(corner);
      paintBlock(surface, colour, frame.x + offset.x, frame.y + offset.y, CORNER_W, CORNER_H);
    });
  }
  paintDot(surface, PALETTE.fuel, manifest.fuel);
  paintDot(surface, PALETTE.equipment, manifest.equipment);
  manifest.rocks.forEach((frame, rock) => paintDot(surface, rock === 2 ? PALETTE.ferry : PALETTE.rock, frame));
  manifest.mines.forEach((frame, team) => paintDot(surface, teamColour(team), frame));
  manifest.tanks.forEach((facings, team) => facings.forEach((frame, facing) => paintTank(surface, teamColour(team), frame, facing)));
  for (const frame of manifest.corpses) {
    paintBlock(surface, PALETTE.corpse, frame.x + 7, frame.y + 4, 14, 12);
  }
}
