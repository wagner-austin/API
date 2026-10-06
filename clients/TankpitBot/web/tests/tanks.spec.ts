import { describe, expect, it } from "vitest";

import { defaultManifest, TANK_H, TANK_W } from "../src/default_pack.js";
import { ScaledContext } from "../src/scaled_context.js";
import { drawTank, tankFrame, tankRect, type TankView } from "../src/tanks.js";
import { RecordingSurface } from "./fakes.js";

const manifest = defaultManifest("sheet.png");
const alive: TankView = { team: 2, facing: 5, alive: true, deaths: 0, column: 3, row: 4 };

describe("tankFrame", () => {
  it("picks the team's column and the facing's row", () => {
    expect(tankFrame(manifest, alive)).toEqual({ sheet: "default", x: 2 * TANK_W, y: 96 + 5 * TANK_H, w: TANK_W, h: TANK_H });
  });

  it("picks a corpse by the parity of the death count", () => {
    expect(tankFrame(manifest, { ...alive, alive: false, deaths: 4 })).toBe(manifest.corpses[0]);
    expect(tankFrame(manifest, { ...alive, alive: false, deaths: 7 })).toBe(manifest.corpses[1]);
  });

  it("refuses a team or a facing the pack has no frame for", () => {
    expect(() => tankFrame(manifest, { ...alive, team: 4 })).toThrow("RENDER_FRAME: the pack has no tanks for team 4");
    expect(() => tankFrame(manifest, { ...alive, facing: 16 })).toThrow("RENDER_FRAME: the pack has no team 2 facing frame 16 (it has 16)");
  });
});

describe("tankRect and drawTank", () => {
  it("centre the frame on the tank's tile", () => {
    const rect = tankRect(manifest, alive);
    expect(rect).toEqual({ x: 3 * 24 - 2, y: 4 * 16 - 2, w: 28, h: 20 });
    const surface = new RecordingSurface();
    expect(drawTank(new ScaledContext(surface, 1), manifest, { images: new Map([["default", document.createElement("canvas")]]) }, alive)).toEqual(rect);
    expect(surface.only("draw").map((call) => call.dest)).toEqual([[70, 62, 28, 20]]);
  });
});
