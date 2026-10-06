import { describe, expect, it } from "vitest";

import { readEnvelope, walkEnd, type EnvelopeHandler, type TankIdentity, type TankPlace } from "../src/envelope.js";
import type { RadarScan, TileWrite, ViewportUpdate } from "../src/tile_messages.js";
import { envelope, GOLDEN, hex, TABLE } from "./golden.js";

/** One handler call: its method and what it was given. */
interface Call {
  readonly method: string;
  readonly args: readonly unknown[];
}

/** A handler that records every call, in order. */
class RecordingHandler implements EnvelopeHandler {
  public readonly calls: Call[] = [];

  private record(method: string, ...args: unknown[]): void {
    this.calls.push({ method, args });
  }

  public identity(tank: TankIdentity): void {
    this.record("identity", tank);
  }
  public entry(tankId: number, team: number): void {
    this.record("entry", tankId, team);
  }
  public place(place: TankPlace, team: number): void {
    this.record("place", place, team);
  }
  public walk(place: TankPlace): void {
    this.record("walk", place);
  }
  public remove(tankId: number): void {
    this.record("remove", tankId);
  }
  public exit(tankId: number): void {
    this.record("exit", tankId);
  }
  public destroyed(tankId: number): void {
    this.record("destroyed", tankId);
  }
  public viewport(update: ViewportUpdate): void {
    this.record("viewport", update.left, update.top, update.tiles.length);
  }
  public caches(writes: readonly TileWrite[]): void {
    this.record("caches", writes);
  }
  public radar(scan: RadarScan): void {
    this.record("radar", scan.caches.length, scan.mines.length);
  }
  public rocks(writes: readonly TileWrite[]): void {
    this.record("rocks", writes);
  }
  public unrendered(subtype: number): void {
    this.record("unrendered", subtype);
  }
}

function read(body: Uint8Array): Call[] {
  const handler = new RecordingHandler();
  readEnvelope(envelope(body), TABLE, handler);
  return handler.calls;
}

describe("readEnvelope", () => {
  it("hands each tank message the server writes to its method", () => {
    expect(read(GOLDEN.info)).toEqual([{ method: "identity", args: [{ tankId: 2000, team: 2, name: "austin" }] }]);
    expect(read(GOLDEN.status)).toEqual([{ method: "identity", args: [{ tankId: 513, team: 1, name: "red-513" }] }]);
    expect(read(GOLDEN.entry)).toEqual([{ method: "entry", args: [527, 3] }]);
    expect(read(GOLDEN.exit)).toEqual([{ method: "exit", args: [500] }]);
    expect(read(GOLDEN.position)).toEqual([{ method: "place", args: [{ tankId: 2000, x: 246, y: 1, facing: 5 }, 2] }]);
    expect(read(GOLDEN.movement)).toEqual([{ method: "walk", args: [{ tankId: 2000, x: 248, y: 2, facing: 4 }] }]);
    expect(read(GOLDEN.remove)).toEqual([{ method: "remove", args: [509] }]);
    expect(read(GOLDEN.deactivation)).toEqual([{ method: "destroyed", args: [518] }]);
  });

  it("hands each tile message to its method", () => {
    expect(read(GOLDEN.viewport)).toEqual([{ method: "viewport", args: [238, 0, 3] }]);
    expect(read(GOLDEN.pickup)).toEqual([{ method: "caches", args: [[{ x: 40, y: 41, value: 257 }]] }]);
    expect(read(GOLDEN.radar)).toEqual([{ method: "radar", args: [2, 2] }]);
    expect(read(GOLDEN.terrain)).toEqual([
      {
        method: "rocks",
        args: [
          [
            { x: 10, y: 20, value: 2 },
            { x: 11, y: 20, value: 0 },
          ],
        ],
      },
    ]);
  });

  it("leaves to unrendered a subtype it does not draw from, and one whose body has another message's shape", () => {
    const shapes = ["3f01", "2102d007", "3e3d01", "2803", "2900f40100", "3d02d0", "47d007", "58fd", "410106", "5aee", "432829", "4f01", "530000"];
    for (const body of shapes) {
      expect(read(hex(body))).toEqual([{ method: "unrendered", args: [parseInt(body.slice(0, 2), 16)] }]);
    }
  });

  it("refuses a frame that is not an envelope with a body", () => {
    const handler = new RecordingHandler();
    expect(() => readEnvelope(hex("2e"), TABLE, handler)).toThrow("WIRE_ENVELOPE: a 1-byte frame is not an envelope with a body");
    expect(() => readEnvelope(hex("2b3131"), TABLE, handler)).toThrow("WIRE_ENVELOPE: a 3-byte frame");
  });
});

describe("walkEnd", () => {
  it("follows each step letter, faces along the last, and skips other bytes", () => {
    expect(walkEnd(7, 10, 10, 0, "nnww")).toEqual({ tankId: 7, x: 8, y: 8, facing: 12 });
    expect(walkEnd(7, 10, 10, 0, "e?")).toEqual({ tankId: 7, x: 11, y: 10, facing: 4 });
  });

  it("leaves a tank with no steps where it was, facing the direction's low nibble", () => {
    expect(walkEnd(7, 10, 10, 0x3c, "")).toEqual({ tankId: 7, x: 10, y: 10, facing: 12 });
  });
});
