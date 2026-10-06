/**
 * The sprite manifest: where every picture the renderer draws comes from.
 *
 * The renderer is sprite-atlas-agnostic. It never knows what a tank or a
 * tile looks like, only which rectangle of which sheet to copy for it,
 * and the manifest says so. A sprite pack is a manifest plus the sheets
 * it names, so the art is a decision separate from the code: the default
 * pack (default_pack.ts) is this project's own, painted at start-up, and
 * any other pack loads the same way. Nothing here carries tankpit.com's
 * art.
 *
 * The geometry follows the game's own client (wiki rendering-pipeline):
 * a viewport tile is 24x16 pixels, a tank frame 28x20, sixteen facings
 * per team, four teams.
 */

export const TEAMS = 4;
export const FACINGS = 16;
export const CORNERS = 4;

/** One frame: a rectangle of one sheet. */
export interface SpriteFrame {
  readonly sheet: string;
  readonly x: number;
  readonly y: number;
  readonly w: number;
  readonly h: number;
}

/** The terrain frames: a base per kind, and the corner overlays water and obstacles draw where a neighbour differs. */
export interface TerrainFrames {
  readonly ground: SpriteFrame;
  readonly water: SpriteFrame;
  readonly obstacle: SpriteFrame;
  readonly waterCorners: readonly SpriteFrame[];
  readonly obstacleCorners: readonly SpriteFrame[];
}

/** A whole sprite pack's geometry. */
export interface SpriteManifest {
  readonly name: string;
  readonly sheets: ReadonlyMap<string, string>;
  readonly tileWidth: number;
  readonly tileHeight: number;
  readonly terrain: TerrainFrames;
  readonly fuel: SpriteFrame;
  readonly equipment: SpriteFrame;
  readonly mines: readonly SpriteFrame[];
  readonly rocks: readonly SpriteFrame[];
  readonly tanks: readonly (readonly SpriteFrame[])[];
  readonly corpses: readonly SpriteFrame[];
}

/** A manifest that cannot be used, with a MANIFEST_* code in its message. */
export class ManifestError extends Error {
  public constructor(message: string) {
    super(message);
    this.name = "ManifestError";
  }
}

/** A JSON object, read field by field. */
interface JsonObject {
  readonly [key: string]: unknown;
}

function isObject(value: unknown): value is JsonObject {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function requireObject(value: unknown, where: string): JsonObject {
  if (!isObject(value)) {
    throw new ManifestError(`MANIFEST_SHAPE: ${where} is not an object`);
  }
  return value;
}

function requireString(object: JsonObject, key: string, where: string): string {
  const value = object[key];
  if (typeof value !== "string" || value === "") {
    throw new ManifestError(`MANIFEST_SHAPE: ${where}.${key} is not a non-empty string`);
  }
  return value;
}

function requireSize(object: JsonObject, key: string, where: string, least: number): number {
  const value = object[key];
  if (typeof value !== "number" || !Number.isInteger(value) || value < least) {
    throw new ManifestError(`MANIFEST_SHAPE: ${where}.${key} is not an integer of ${least} or more`);
  }
  return value;
}

function requireArray(object: JsonObject, key: string, where: string, length: number): unknown[] {
  const value = object[key];
  if (!Array.isArray(value) || value.length !== length) {
    throw new ManifestError(`MANIFEST_SHAPE: ${where}.${key} is not a list of ${length}`);
  }
  return value;
}

/**
 * Read one frame and check it names a sheet the manifest lists.
 *
 * @param value - The frame's JSON.
 * @param where - Its path, for messages.
 * @param sheets - The manifest's sheets.
 * @returns The frame.
 * @throws ManifestError MANIFEST_SHAPE for a malformed frame, MANIFEST_SHEET for an unknown sheet.
 */
export function decodeFrame(value: unknown, where: string, sheets: ReadonlyMap<string, string>): SpriteFrame {
  const object = requireObject(value, where);
  const sheet = requireString(object, "sheet", where);
  if (!sheets.has(sheet)) {
    throw new ManifestError(`MANIFEST_SHEET: ${where} names sheet ${JSON.stringify(sheet)}, which the manifest does not list`);
  }
  return {
    sheet,
    x: requireSize(object, "x", where, 0),
    y: requireSize(object, "y", where, 0),
    w: requireSize(object, "w", where, 1),
    h: requireSize(object, "h", where, 1),
  };
}

function decodeFrames(object: JsonObject, key: string, where: string, length: number, sheets: ReadonlyMap<string, string>): SpriteFrame[] {
  return requireArray(object, key, where, length).map((item, index) => decodeFrame(item, `${where}.${key}[${index}]`, sheets));
}

function decodeSheets(object: JsonObject): Map<string, string> {
  const raw = requireObject(object["sheets"], "manifest.sheets");
  const sheets = new Map<string, string>();
  for (const id of Object.keys(raw)) {
    sheets.set(id, requireString(raw, id, "manifest.sheets"));
  }
  if (sheets.size === 0) {
    throw new ManifestError("MANIFEST_SHAPE: manifest.sheets lists no sheet");
  }
  return sheets;
}

/**
 * Read a sprite manifest from its JSON, checking every frame.
 *
 * @param raw - The parsed JSON.
 * @returns The manifest.
 * @throws ManifestError For any field missing, mis-typed, of the wrong count, or naming an unknown sheet.
 */
export function decodeManifest(raw: unknown): SpriteManifest {
  const object = requireObject(raw, "manifest");
  const sheets = decodeSheets(object);
  const terrain = requireObject(object["terrain"], "manifest.terrain");
  const tanks = requireArray(object, "tanks", "manifest", TEAMS).map((team, index) =>
    decodeFrames({ facings: team }, "facings", `manifest.tanks[${index}]`, FACINGS, sheets),
  );
  return {
    name: requireString(object, "name", "manifest"),
    sheets,
    tileWidth: requireSize(object, "tileWidth", "manifest", 1),
    tileHeight: requireSize(object, "tileHeight", "manifest", 1),
    terrain: {
      ground: decodeFrame(terrain["ground"], "manifest.terrain.ground", sheets),
      water: decodeFrame(terrain["water"], "manifest.terrain.water", sheets),
      obstacle: decodeFrame(terrain["obstacle"], "manifest.terrain.obstacle", sheets),
      waterCorners: decodeFrames(terrain, "waterCorners", "manifest.terrain", CORNERS, sheets),
      obstacleCorners: decodeFrames(terrain, "obstacleCorners", "manifest.terrain", CORNERS, sheets),
    },
    fuel: decodeFrame(object["fuel"], "manifest.fuel", sheets),
    equipment: decodeFrame(object["equipment"], "manifest.equipment", sheets),
    mines: decodeFrames(object, "mines", "manifest", TEAMS, sheets),
    rocks: decodeFrames(object, "rocks", "manifest", 3, sheets),
    tanks,
    corpses: decodeFrames(object, "corpses", "manifest", 2, sheets),
  };
}

/**
 * Write a manifest back to its JSON shape, the inverse of {@link decodeManifest}.
 *
 * @param manifest - The manifest.
 * @returns Its JSON object.
 */
export function encodeManifest(manifest: SpriteManifest): Record<string, unknown> {
  return {
    name: manifest.name,
    sheets: Object.fromEntries(manifest.sheets),
    tileWidth: manifest.tileWidth,
    tileHeight: manifest.tileHeight,
    terrain: {
      ground: manifest.terrain.ground,
      water: manifest.terrain.water,
      obstacle: manifest.terrain.obstacle,
      waterCorners: manifest.terrain.waterCorners,
      obstacleCorners: manifest.terrain.obstacleCorners,
    },
    fuel: manifest.fuel,
    equipment: manifest.equipment,
    mines: manifest.mines,
    rocks: manifest.rocks,
    tanks: manifest.tanks,
    corpses: manifest.corpses,
  };
}
