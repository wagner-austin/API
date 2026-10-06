/**
 * The toolbar strip under the game: its eighteen buttons and what they do.
 *
 * The regions are the game's client's own hitbox arrays, and a click is
 * tested exactly as its xc function tests one: the y coordinate is moved
 * down three pixels first, the regions are tried in order, and a miss is
 * -1 (wiki toolbar-layout). Scope buttons are numbered clockwise from N
 * on the strip, but the scroll they ask for is numbered from S, which
 * scopeDirection remaps; mapScroll turns a scroll direction into the
 * minimap's 64-pixel step.
 */

/** What each region does, in region order. */
export enum ToolbarAction {
  OpenMap = 0,
  Radar = 1,
  PlaceMine = 2,
  ScopeNorthWest = 3,
  ScopeNorth = 4,
  ScopeNorthEast = 5,
  ScopeWest = 6,
  ScopeCenter = 7,
  ScopeEast = 8,
  ScopeSouthWest = 9,
  ScopeSouth = 10,
  ScopeSouthEast = 11,
  ArmorShield = 12,
  DualShot = 13,
  MissileShot = 14,
  HomingShot = 15,
  ExtraRadar = 16,
  Promotion = 17,
}

/** One clickable rectangle of the strip. */
export interface ToolbarRegion {
  readonly action: ToolbarAction;
  readonly x: number;
  readonly y: number;
  readonly w: number;
  readonly h: number;
}

function region(action: ToolbarAction, x: number, y: number, w: number, h: number): ToolbarRegion {
  return { action, x, y, w, h };
}

/** The eighteen regions, in the order a click tries them: the client's pc, qc, rc and sc arrays, column by column. */
export const TOOLBAR_REGIONS: readonly ToolbarRegion[] = [
  region(ToolbarAction.OpenMap, 10, 2, 43, 44),
  region(ToolbarAction.Radar, 53, 2, 44, 44),
  region(ToolbarAction.PlaceMine, 97, 2, 43, 44),
  region(ToolbarAction.ScopeNorthWest, 151, 2, 24, 14),
  region(ToolbarAction.ScopeNorth, 175, 2, 22, 14),
  region(ToolbarAction.ScopeNorthEast, 197, 2, 24, 14),
  region(ToolbarAction.ScopeWest, 151, 16, 15, 15),
  region(ToolbarAction.ScopeCenter, 166, 16, 40, 15),
  region(ToolbarAction.ScopeEast, 206, 16, 15, 15),
  region(ToolbarAction.ScopeSouthWest, 151, 31, 24, 15),
  region(ToolbarAction.ScopeSouth, 175, 31, 22, 15),
  region(ToolbarAction.ScopeSouthEast, 197, 31, 24, 15),
  region(ToolbarAction.ArmorShield, 233, 8, 30, 26),
  region(ToolbarAction.DualShot, 263, 8, 19, 26),
  region(ToolbarAction.MissileShot, 282, 8, 22, 26),
  region(ToolbarAction.HomingShot, 304, 8, 24, 26),
  region(ToolbarAction.ExtraRadar, 328, 8, 31, 26),
  region(ToolbarAction.Promotion, 362, 8, 20, 30),
];

const CLICK_Y_OFFSET = 3;

/**
 * The region a click on the strip lands in.
 *
 * @param x - Click x, in strip pixels.
 * @param y - Click y, in strip pixels (0 at the strip's top).
 * @returns The region's action, or -1 for a click between regions.
 */
export function hitTest(x: number, y: number): ToolbarAction | -1 {
  const hit = regionAt(x, y);
  return hit === null ? -1 : hit.action;
}

/**
 * The region a click on the strip lands in, tested as hitTest tests it.
 *
 * @param x - Click x, in strip pixels.
 * @param y - Click y, in strip pixels.
 * @returns The region, or null between regions.
 */
export function regionAt(x: number, y: number): ToolbarRegion | null {
  const shifted = y + CLICK_Y_OFFSET;
  return TOOLBAR_REGIONS.find((r) => x >= r.x && x < r.x + r.w && shifted >= r.y && shifted < r.y + r.h) ?? null;
}

/** The equipment slot, 0 to 4, an equipment action toggles. */
export function equipmentSlot(action: ToolbarAction): number | null {
  return action >= ToolbarAction.ArmorShield && action <= ToolbarAction.ExtraRadar ? action - ToolbarAction.ArmorShield : null;
}

/**
 * The game's client's qe remap: a scope button number to a scroll direction.
 *
 * @param button - 0 N, 1 NE, 2 E, 3 SE, 4 S, 5 SW, 6 W, 7 NW; anything else passes through.
 * @returns The direction le reads (0 N ... 7 NW, 8 centre).
 */
export function scopeDirection(button: number): number {
  return button >= 0 && button <= 7 ? (button + 4) % 8 : button;
}

/** A minimap scroll step, in map pixels. */
export interface ScrollStep {
  readonly dx: number;
  readonly dy: number;
}

const STEP = 64;
const STEPS: readonly ScrollStep[] = [
  { dx: 0, dy: -STEP },
  { dx: STEP, dy: -STEP },
  { dx: STEP, dy: 0 },
  { dx: STEP, dy: STEP },
  { dx: 0, dy: STEP },
  { dx: -STEP, dy: STEP },
  { dx: -STEP, dy: 0 },
  { dx: -STEP, dy: -STEP },
];

/**
 * The game's client's le step for a scroll direction.
 *
 * @param direction - 0 N clockwise to 7 NW.
 * @returns The step; null for 8, the centre, which resets rather than steps.
 * @throws RangeError For a direction outside 0 to 8.
 */
export function mapScroll(direction: number): ScrollStep | null {
  if (direction === 8) {
    return null;
  }
  const step = STEPS[direction];
  if (step === undefined) {
    throw new RangeError(`RENDER_SCROLL: direction ${direction} is not 0 to 8`);
  }
  return step;
}
