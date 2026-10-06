/**
 * What the client knows of the field, from the messages of play, painted onto the renderer.
 *
 * The world view is the client's EnvelopeHandler: each message updates
 * what it holds, and paint() hands the renderer what changed. It holds
 * the viewport's window (set only by 0x5A, wiki viewport-shift-protocol),
 * the field's dynamic tile layers (rocks and ferries, caches, mines) on
 * top of the static terrain, and every tank it has heard of: team, tile,
 * facing, and whether it is a corpse.
 *
 * A tank is drawn while it is placed (a 0x3D or 0x47 put it on a tile and
 * no 0x58 or 0x29 has taken it off), its team is known, and its tile is
 * inside the window. A 0x41 makes it a corpse; the next 0x3D or 0x47 is
 * the tank back in play. The renderer's grid column 1 is the window's
 * left edge, so world x maps to column x - left + 1; a tank's column is
 * x - left, counted from the window's first column.
 */

import type { EnvelopeHandler, TankIdentity, TankPlace } from "./envelope.js";
import type { FieldTerrain } from "./field.js";
import type { FrameReport, Renderer } from "./renderer.js";
import { RenderError } from "./scaled_context.js";
import type { TankView } from "./tanks.js";
import { decodeTerrain } from "./terrain.js";
import { GRID_SIZE, NO_MINE, RockKind, type TileState } from "./tile_grid.js";
import { PATCH_WIDTH, type RadarScan, type TileWrite, type ViewportUpdate } from "./tile_messages.js";

/** Tiles along each side of the window. */
export const WINDOW = 16;

const SPAN = 256;
const PATCH_BORDER = (PATCH_WIDTH - WINDOW) / 2;

/** The wire's dynamic terrain values (0x5A, 0x4A, 0x42), as the rocks the renderer draws. */
const ROCKS = new Map<number, RockKind>([
  [0, RockKind.None],
  [1, RockKind.A],
  [2, RockKind.B],
  [3, RockKind.B],
  [5, RockKind.Ferry],
  [7, RockKind.Ferry],
]);

interface TankState {
  team: number | null;
  x: number;
  y: number;
  facing: number;
  alive: boolean;
  deaths: number;
  placed: boolean;
}

/**
 * A field tile's key in the view's layers.
 *
 * @param x - The tile's x.
 * @param y - The tile's y.
 * @returns Its key.
 * @throws RenderError RENDER_TILE for a tile off the field.
 */
export function tileKey(x: number, y: number): number {
  if (!Number.isInteger(x) || !Number.isInteger(y) || x < 0 || y < 0 || x >= SPAN || y >= SPAN) {
    throw new RenderError(`RENDER_TILE: (${x}, ${y}) is not on the field`);
  }
  return y * SPAN + x;
}

/** The client's knowledge of the field, and what of it the renderer has yet to draw. */
export class WorldView implements EnvelopeHandler {
  private left = 0;
  private top = 0;
  private windowMoved = true;
  private readonly touched = new Set<number>();
  private readonly changedTanks = new Set<number>();
  private readonly tanks = new Map<number, TankState>();
  private readonly rockLayer = new Map<number, RockKind>();
  private readonly cacheLayer = new Map<number, number>();
  private readonly mineLayer = new Map<number, number>();
  private readonly ignored = new Map<number, number>();

  /**
   * Bind the view to its field and the renderer it paints.
   *
   * @param field - The room's static terrain.
   * @param renderer - What the view paints on.
   */
  public constructor(
    private readonly field: FieldTerrain,
    private readonly renderer: Renderer,
  ) {}

  /** The window's left and top, in field tiles. */
  public get window(): readonly [number, number] {
    return [this.left, this.top];
  }

  /** How many messages of each subtype were not drawn from. */
  public get unrenderedCounts(): ReadonlyMap<number, number> {
    return this.ignored;
  }

  private tank(tankId: number): TankState {
    const known = this.tanks.get(tankId);
    if (known !== undefined) {
      return known;
    }
    const fresh: TankState = { team: null, x: 0, y: 0, facing: 0, alive: true, deaths: 0, placed: false };
    this.tanks.set(tankId, fresh);
    return fresh;
  }

  public identity(tank: TankIdentity): void {
    this.tank(tank.tankId).team = tank.team;
    this.changedTanks.add(tank.tankId);
  }

  public entry(tankId: number, team: number): void {
    this.tank(tankId).team = team;
    this.changedTanks.add(tankId);
  }

  public place(place: TankPlace, team: number): void {
    this.tank(place.tankId).team = team;
    this.walk(place);
  }

  public walk(place: TankPlace): void {
    const tank = this.tank(place.tankId);
    Object.assign(tank, { x: place.x, y: place.y, facing: place.facing, alive: true, placed: true });
    this.changedTanks.add(place.tankId);
  }

  public remove(tankId: number): void {
    this.tank(tankId).placed = false;
    this.changedTanks.add(tankId);
  }

  public exit(tankId: number): void {
    this.remove(tankId);
  }

  public destroyed(tankId: number): void {
    const tank = this.tank(tankId);
    if (tank.alive) {
      tank.alive = false;
      tank.deaths += 1;
    }
    this.changedTanks.add(tankId);
  }

  public viewport(update: ViewportUpdate): void {
    if (update.left !== this.left || update.top !== this.top) {
      this.left = update.left;
      this.top = update.top;
      this.windowMoved = true;
    }
    for (const tile of update.tiles) {
      const x = update.left + tile.column - PATCH_BORDER;
      const y = update.top + tile.row - PATCH_BORDER;
      this.writeRock(x, y, tile.terrain);
      this.write(this.cacheLayer, { x, y, value: tile.cache });
      this.write(this.mineLayer, { x, y, value: tile.mine });
    }
  }

  public caches(writes: readonly TileWrite[]): void {
    writes.forEach((write) => this.write(this.cacheLayer, write));
  }

  public radar(scan: RadarScan): void {
    this.caches(scan.caches);
    scan.mines.forEach((write) => this.write(this.mineLayer, write));
  }

  public rocks(writes: readonly TileWrite[]): void {
    writes.forEach((write) => this.writeRock(write.x, write.y, write.value));
  }

  public unrendered(subtype: number): void {
    this.ignored.set(subtype, (this.ignored.get(subtype) ?? 0) + 1);
  }

  private write(layer: Map<number, number>, write: TileWrite): void {
    layer.set(tileKey(write.x, write.y), write.value);
    this.touched.add(tileKey(write.x, write.y));
  }

  private writeRock(x: number, y: number, value: number): void {
    const rock = ROCKS.get(value);
    if (rock === undefined) {
      throw new RenderError(`RENDER_ROCK: tile (${x}, ${y}) carries dynamic terrain ${value}, which the game does not define`);
    }
    this.rockLayer.set(tileKey(x, y), rock);
    this.touched.add(tileKey(x, y));
  }

  /**
   * What one field tile shows: its static terrain and its dynamic layers.
   *
   * @param x - The tile's x, on the field.
   * @param y - The tile's y, on the field.
   * @returns The renderer's tile state.
   */
  public tileState(x: number, y: number): TileState {
    const key = tileKey(x, y);
    return {
      terrain: decodeTerrain(this.field.terrainByte(x, y)),
      cache: this.cacheLayer.get(key) ?? 0,
      mine: this.mineLayer.get(key) ?? NO_MINE,
      rock: this.rockLayer.get(key) ?? RockKind.None,
    };
  }

  private inWindow(x: number, y: number): boolean {
    return x >= this.left && x < this.left + WINDOW && y >= this.top && y < this.top + WINDOW;
  }

  private paintTiles(): void {
    for (let column = 1; column < GRID_SIZE - 1; column++) {
      for (let row = 1; row < GRID_SIZE - 1; row++) {
        const x = this.left + column - 1;
        const y = this.top + row - 1;
        if (this.windowMoved || this.touched.has(tileKey(x, y))) {
          this.renderer.setTile(column, row, this.tileState(x, y));
        }
      }
    }
  }

  private tankView(tank: TankState): TankView | null {
    if (tank.team === null || !tank.placed || !this.inWindow(tank.x, tank.y)) {
      return null;
    }
    return { team: tank.team, facing: tank.facing, alive: tank.alive, deaths: tank.deaths, column: tank.x - this.left, row: tank.y - this.top };
  }

  /**
   * How the renderer is told to draw a tank.
   *
   * @param tankId - The tank.
   * @returns Its view, or null for a tank not drawn or never heard of.
   */
  public drawnAs(tankId: number): TankView | null {
    const tank = this.tanks.get(tankId);
    return tank === undefined ? null : this.tankView(tank);
  }

  /**
   * Hand the renderer what changed since the last paint, and draw the frame.
   *
   * @returns What the frame drew.
   */
  public paint(): FrameReport {
    this.paintTiles();
    const tankIds = this.windowMoved ? [...this.tanks.keys()] : [...this.changedTanks];
    for (const tankId of tankIds) {
      const view = this.tankView(this.tank(tankId));
      if (view === null) {
        this.renderer.removeTank(tankId);
      } else {
        this.renderer.setTank(tankId, view);
      }
    }
    this.windowMoved = false;
    this.touched.clear();
    this.changedTanks.clear();
    return this.renderer.renderFrame();
  }
}
