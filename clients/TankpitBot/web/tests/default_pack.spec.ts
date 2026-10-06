import { describe, expect, it } from "vitest";

import {
  cornerOffset,
  defaultManifest,
  facingVector,
  PALETTE,
  paintDefaultSheet,
  SHEET_HEIGHT,
  SHEET_WIDTH,
  TEAM_COLOURS,
  teamColour,
} from "../src/default_pack.js";
import { decodeManifest, encodeManifest, type SpriteFrame } from "../src/manifest.js";
import { RecordingSurface } from "./fakes.js";

const manifest = defaultManifest("data:sheet");

function everyFrame(): SpriteFrame[] {
  const t = manifest.terrain;
  return [
    t.ground,
    t.water,
    t.obstacle,
    ...t.waterCorners,
    ...t.obstacleCorners,
    manifest.fuel,
    manifest.equipment,
    ...manifest.rocks,
    ...manifest.mines,
    ...manifest.tanks.flat(),
    ...manifest.corpses,
  ];
}

describe("defaultManifest", () => {
  it("is a manifest the decoder accepts, naming its one sheet by the URL given", () => {
    expect(decodeManifest(JSON.parse(JSON.stringify(encodeManifest(manifest))))).toEqual(manifest);
    expect([...manifest.sheets]).toEqual([["default", "data:sheet"]]);
  });

  it("keeps every frame on the sheet and no two frames on the same pixels", () => {
    const frames = everyFrame();
    expect(frames).toHaveLength(3 + 8 + 2 + 3 + 4 + 64 + 2);
    for (const frame of frames) {
      expect(frame.x + frame.w <= SHEET_WIDTH && frame.y + frame.h <= SHEET_HEIGHT).toBe(true);
    }
    const keys = new Set(frames.map((frame) => `${frame.x},${frame.y}`));
    expect(keys.size).toBe(frames.length);
  });
});

describe("paintDefaultSheet", () => {
  it("clears the sheet, then paints each frame with its colour inside its own cell", () => {
    const surface = new RecordingSurface();
    paintDefaultSheet(surface, manifest);
    expect(surface.calls[0]).toEqual({ op: "clear", image: null, fill: "", source: [], dest: [0, 0, SHEET_WIDTH, SHEET_HEIGHT] });
    const fills = surface.only("fill");
    expect(fills.slice(0, 3).map((call) => [call.fill, ...call.dest])).toEqual([
      [PALETTE.ground, 0, 0, 24, 16],
      [PALETTE.water, 24, 0, 24, 16],
      [PALETTE.obstacle, 48, 0, 24, 16],
    ]);
    expect(fills.slice(3, 7).map((call) => call.dest)).toEqual([
      [16, 16, 8, 5],
      [40, 27, 8, 5],
      [48, 27, 8, 5],
      [72, 16, 8, 5],
    ]);
    const mineFills = fills.filter((call) => call.dest[1] === 86);
    expect(mineFills.map((call) => call.fill)).toEqual(TEAM_COLOURS);
    expect(fills.filter((call) => call.fill === PALETTE.ferry).map((call) => call.dest)).toEqual([[10, 70, 4, 4]]);
    expect(fills.filter((call) => call.fill === PALETTE.corpse)).toHaveLength(2);
  });

  it("paints each tank's hull in its team colour and its barrel along its facing", () => {
    const surface = new RecordingSurface();
    paintDefaultSheet(surface, manifest);
    const fills = surface.only("fill");
    const team1North = fills.filter((call) => call.dest[0] !== undefined && call.dest[0] >= 28 && call.dest[0] < 56 && call.dest[1] !== undefined && call.dest[1] >= 96 && call.dest[1] < 116);
    expect(team1North.map((call) => call.fill)).toEqual([PALETTE.hull, TEAM_COLOURS[1], ...Array.from({ length: 8 }, () => PALETTE.hull)]);
    expect(team1North.slice(2).map((call) => call.dest[0])).toEqual(Array.from({ length: 8 }, () => 41));
    expect(team1North.slice(2).map((call) => call.dest[1])).toEqual([103, 102, 101, 100, 99, 98, 97, 96]);
  });
});

describe("teamColour", () => {
  it("gives the four troops their colours and refuses a fifth", () => {
    expect([0, 1, 2, 3].map(teamColour)).toEqual(TEAM_COLOURS);
    expect(() => teamColour(4)).toThrow("RENDER_TEAM: team 4 has no colour");
  });
});

describe("cornerOffset and facingVector", () => {
  it("put corners NE, SE, SW, NW in that order", () => {
    expect([0, 1, 2, 3].map(cornerOffset)).toEqual([
      { x: 16, y: 0 },
      { x: 16, y: 11 },
      { x: 0, y: 11 },
      { x: 0, y: 0 },
    ]);
  });

  it("point facing 0 north and turn clockwise", () => {
    const round = (v: { x: number; y: number }): number[] => [Math.round(v.x * 1000) / 1000 + 0, Math.round(v.y * 1000) / 1000 + 0];
    expect([0, 4, 8, 12].map((facing) => round(facingVector(facing)))).toEqual([
      [0, -1],
      [1, 0],
      [0, 1],
      [-1, 0],
    ]);
  });
});
