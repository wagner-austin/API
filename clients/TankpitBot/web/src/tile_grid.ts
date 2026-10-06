/**
 * The viewport's tiles, redrawn only where they changed.
 *
 * The game's client keeps an 18x18 grid of tiles: the 16x16 a viewport
 * shows and a one-tile border, so a scroll has its next edge ready (wiki
 * rendering-pipeline, "Tile Engine"). Each tile carries its terrain, a
 * cache (fuel above zero, equipment below), a team's mine and a rock, and
 * a dirty flag. Setting a tile marks it dirty; drawing draws the dirty
 * tiles in the game's order -- base, corners, cache dot, rock, mine --
 * and clears the flags.
 */

import type { SpriteManifest } from "./manifest.js";
import { RenderError, type ScaledContext } from "./scaled_context.js";
import { drawFrame, pickFrame, type SheetImages } from "./sprites.js";
import { openCorners, TerrainKind, type Terrain } from "./terrain.js";

export const GRID_SIZE = 18;
export const NO_MINE = 255;

/** The rocks a tile can carry. */
export enum RockKind {
  None = -1,
  A = 0,
  B = 1,
  Ferry = 2,
}

/** What a tile shows. */
export interface TileState {
  readonly terrain: Terrain;
  readonly cache: number;
  readonly mine: number;
  readonly rock: RockKind;
}

interface Cell {
  state: TileState;
  dirty: boolean;
}

/** The grid of tiles behind the viewport. */
export class TileGrid {
  private readonly cells: Cell[];

  /**
   * Start every tile as plain ground, dirty, so the first draw fills the layer.
   */
  public constructor() {
    const ground: TileState = {
      terrain: { kind: TerrainKind.Ground, sharedCorners: 0x0f },
      cache: 0,
      mine: NO_MINE,
      rock: RockKind.None,
    };
    this.cells = Array.from({ length: GRID_SIZE * GRID_SIZE }, () => ({ state: ground, dirty: true }));
  }

  private cell(column: number, row: number): Cell {
    const inside = Number.isInteger(column) && Number.isInteger(row) && column >= 0 && row >= 0 && column < GRID_SIZE && row < GRID_SIZE;
    const found = inside ? this.cells[row * GRID_SIZE + column] : undefined;
    if (found === undefined) {
      throw new RenderError(`RENDER_TILE: (${column}, ${row}) is outside the ${GRID_SIZE}x${GRID_SIZE} grid`);
    }
    return found;
  }

  /**
   * Set a tile, marking it dirty.
   *
   * @param column - Grid column, 0 to 17; column 1 is the viewport's first.
   * @param row - Grid row, 0 to 17.
   * @param state - What it shows.
   * @throws RenderError RENDER_TILE outside the grid, RENDER_MINE for a mine that is neither a team nor none.
   */
  public set(column: number, row: number, state: TileState): void {
    if (state.mine !== NO_MINE && !(Number.isInteger(state.mine) && state.mine >= 0 && state.mine <= 3)) {
      throw new RenderError(`RENDER_MINE: mine ${state.mine} is not a team (0 to 3) or none (${NO_MINE})`);
    }
    const cell = this.cell(column, row);
    cell.state = state;
    cell.dirty = true;
  }

  /** What a tile shows. */
  public get(column: number, row: number): TileState {
    return this.cell(column, row).state;
  }

  /** How many tiles wait to be drawn. */
  public dirtyCount(): number {
    return this.cells.filter((cell) => cell.dirty).length;
  }

  /**
   * Draw every dirty tile of the viewport and clear its flag.
   *
   * Border tiles stay dirty: they are not on screen until a scroll brings them in.
   *
   * @param context - The background layer.
   * @param manifest - The pack's manifest.
   * @param sheets - The pack's sheets.
   * @returns How many tiles were drawn.
   */
  public drawDirty(context: ScaledContext, manifest: SpriteManifest, sheets: SheetImages): number {
    let drawn = 0;
    for (let row = 1; row < GRID_SIZE - 1; row++) {
      for (let column = 1; column < GRID_SIZE - 1; column++) {
        const cell = this.cell(column, row);
        if (!cell.dirty) {
          continue;
        }
        drawTile(context, manifest, sheets, cell.state, (column - 1) * manifest.tileWidth, (row - 1) * manifest.tileHeight);
        cell.dirty = false;
        drawn++;
      }
    }
    return drawn;
  }
}

/**
 * Draw one tile at a point, in the game's order.
 *
 * @param context - The background layer.
 * @param manifest - The pack's manifest.
 * @param sheets - The pack's sheets.
 * @param state - What the tile shows.
 * @param x - Its left, in game pixels.
 * @param y - Its top, in game pixels.
 */
export function drawTile(context: ScaledContext, manifest: SpriteManifest, sheets: SheetImages, state: TileState, x: number, y: number): void {
  const terrain = manifest.terrain;
  const kind = state.terrain.kind;
  const base = kind === TerrainKind.Water ? terrain.water : kind === TerrainKind.Obstacle ? terrain.obstacle : terrain.ground;
  drawFrame(context, sheets, base, x, y);
  if (kind !== TerrainKind.Ground) {
    const corners = kind === TerrainKind.Water ? terrain.waterCorners : terrain.obstacleCorners;
    for (const corner of openCorners(state.terrain)) {
      drawFrame(context, sheets, pickFrame(corners, corner, `${kind} corner`), x, y);
    }
  }
  if (state.cache !== 0) {
    drawFrame(context, sheets, state.cache > 0 ? manifest.fuel : manifest.equipment, x, y);
  }
  if (state.rock !== RockKind.None) {
    drawFrame(context, sheets, pickFrame(manifest.rocks, state.rock, "rock"), x, y);
  }
  if (state.mine !== NO_MINE) {
    drawFrame(context, sheets, pickFrame(manifest.mines, state.mine, "mine"), x, y);
  }
}
