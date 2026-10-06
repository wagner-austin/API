/**
 * The renderer: one sprite pack, six layers, a tile grid and the tanks on it.
 *
 * It is built from a manifest and nothing else about the art. Tiles are
 * set on the grid and drawn when a frame is rendered, only where they
 * changed. Tanks are drawn on their own layer, each with a dirty rect:
 * a tank that moves or leaves erases what it last covered, and every
 * other tank that overlapped the erased region is erased and drawn again
 * too, the game's own erase-then-redraw sequence. The toolbar strip is
 * drawn once and its pressed button again when that changes; a click on
 * it is resolved by toolbar.regionAt.
 */

import { DirtyRect, overlaps, type Rect } from "./dirty_rect.js";
import { createLayerStack, GAME_WIDTH, LayerName, MENU_HEIGHT, type LayerStack } from "./layers.js";
import type { SpriteManifest } from "./manifest.js";
import type { ScaledContext } from "./scaled_context.js";
import { loadSheets, type SheetImages } from "./sprites.js";
import { drawTank, tankFrame, type TankView } from "./tanks.js";
import { TileGrid, type TileState } from "./tile_grid.js";
import { regionAt, TOOLBAR_REGIONS, type ToolbarAction, type ToolbarRegion } from "./toolbar.js";

export const TOOLBAR_FACE = "#30343a";
export const TOOLBAR_BUTTON = "#565c66";
export const TOOLBAR_PRESSED = "#a8b0bc";

interface TankSlot {
  view: TankView;
  pending: boolean;
  readonly dirty: DirtyRect;
}

/** What one rendered frame changed. */
export interface FrameReport {
  readonly tilesDrawn: number;
  readonly tanksDrawn: number;
}

/** A pack and the layers it draws on. */
export class Renderer {
  public readonly grid = new TileGrid();
  private readonly tanks = new Map<number, TankSlot>();
  private pressed: ToolbarRegion | null = null;

  private constructor(
    public readonly stack: LayerStack,
    public readonly manifest: SpriteManifest,
    private readonly sheets: SheetImages,
  ) {}

  /**
   * Load a pack's sheets and lay the six layers into a container.
   *
   * @param container - The element the game is drawn in.
   * @param scale - Device pixels per game pixel.
   * @param manifest - The pack.
   * @returns The renderer, its toolbar drawn.
   * @throws Error RENDER_IMAGE when a sheet does not load; RenderError from the layer stack.
   */
  public static async create(container: HTMLElement, scale: number, manifest: SpriteManifest): Promise<Renderer> {
    const sheets = await loadSheets(manifest);
    const renderer = new Renderer(createLayerStack(container, scale), manifest, sheets);
    renderer.drawToolbar();
    return renderer;
  }

  private context(name: LayerName): ScaledContext {
    return this.stack.layers[name].context;
  }

  /** Set one grid tile; it is drawn on the next frame. */
  public setTile(column: number, row: number, state: TileState): void {
    this.grid.set(column, row, state);
  }

  /**
   * Place or move a tank; it is drawn on the next frame.
   *
   * @param id - The tank's id.
   * @param view - Where it is and how it faces.
   * @throws RenderError RENDER_FRAME at once for a team or facing the pack cannot draw.
   */
  public setTank(id: number, view: TankView): void {
    tankFrame(this.manifest, view);
    const slot = this.tanks.get(id);
    if (slot === undefined) {
      this.tanks.set(id, { view, pending: true, dirty: new DirtyRect() });
      return;
    }
    slot.view = view;
    slot.pending = true;
  }

  /**
   * Take a tank off the field, erasing what it covered now.
   *
   * @param id - The tank's id; an id not on the field changes nothing.
   * @returns Whether a tank was removed.
   */
  public removeTank(id: number): boolean {
    const slot = this.tanks.get(id);
    if (slot === undefined) {
      return false;
    }
    this.tanks.delete(id);
    const erased = slot.dirty.erase(this.context(LayerName.Tanks));
    if (erased !== null) {
      this.markOverlapping(erased);
    }
    return true;
  }

  private markOverlapping(erased: Rect): void {
    for (const slot of this.tanks.values()) {
      const drawn = slot.dirty.current;
      if (drawn !== null && overlaps(erased, drawn)) {
        slot.pending = true;
      }
    }
  }

  /**
   * Draw what changed since the last frame: dirty tiles, then every pending tank.
   *
   * @returns How many tiles and tanks were drawn.
   */
  public renderFrame(): FrameReport {
    const tilesDrawn = this.grid.drawDirty(this.context(LayerName.Background), this.manifest, this.sheets);
    const layer = this.context(LayerName.Tanks);
    let erasing = true;
    while (erasing) {
      erasing = false;
      for (const slot of this.tanks.values()) {
        const erased = slot.pending ? slot.dirty.erase(layer) : null;
        if (erased !== null) {
          this.markOverlapping(erased);
          erasing = true;
        }
      }
    }
    let tanksDrawn = 0;
    for (const slot of this.tanks.values()) {
      if (slot.pending) {
        slot.dirty.include(drawTank(layer, this.manifest, this.sheets, slot.view));
        slot.pending = false;
        tanksDrawn++;
      }
    }
    return { tilesDrawn, tanksDrawn };
  }

  private drawToolbar(): void {
    this.context(LayerName.Menu).fill(TOOLBAR_FACE, 0, 0, GAME_WIDTH, MENU_HEIGHT);
    for (const region of TOOLBAR_REGIONS) {
      this.drawButton(region, false);
    }
  }

  private drawButton(region: ToolbarRegion, pressed: boolean): void {
    this.context(LayerName.Menu).fill(pressed ? TOOLBAR_PRESSED : TOOLBAR_BUTTON, region.x + 1, region.y + 1, region.w - 2, region.h - 2);
  }

  /**
   * Resolve a click on the toolbar strip and show its button pressed.
   *
   * @param x - Click x, in strip pixels.
   * @param y - Click y, in strip pixels.
   * @returns The action, or -1 for a click between buttons, which releases the pressed one.
   */
  public clickToolbar(x: number, y: number): ToolbarAction | -1 {
    const hit = regionAt(x, y);
    if (this.pressed !== null) {
      this.drawButton(this.pressed, false);
    }
    this.pressed = hit;
    if (hit === null) {
      return -1;
    }
    this.drawButton(hit, true);
    return hit.action;
  }

  /** The button shown pressed, or -1. */
  public get pressedButton(): ToolbarAction | -1 {
    return this.pressed === null ? -1 : this.pressed.action;
  }
}
