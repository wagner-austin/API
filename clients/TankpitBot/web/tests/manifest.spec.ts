import { describe, expect, it } from "vitest";

import { defaultManifest } from "../src/default_pack.js";
import { decodeFrame, decodeManifest, encodeManifest, ManifestError } from "../src/manifest.js";

function json(): Record<string, unknown> {
  const encoded: unknown = JSON.parse(JSON.stringify(encodeManifest(defaultManifest("sheet.png"))));
  if (typeof encoded !== "object" || encoded === null || Array.isArray(encoded)) {
    throw new Error("encodeManifest did not give an object");
  }
  return { ...encoded };
}

describe("decodeManifest", () => {
  it("reads back exactly what encodeManifest wrote, through JSON text", () => {
    const manifest = defaultManifest("sheet.png");
    expect(decodeManifest(json())).toEqual(manifest);
  });

  it("refuses a manifest that is not an object", () => {
    expect(() => decodeManifest([])).toThrow("MANIFEST_SHAPE: manifest is not an object");
    expect(() => decodeManifest(null)).toThrow(ManifestError);
  });

  it("refuses a manifest with no sheets", () => {
    expect(() => decodeManifest({ ...json(), sheets: {} })).toThrow("MANIFEST_SHAPE: manifest.sheets lists no sheet");
  });

  it("refuses a sheet whose URL is not a non-empty string", () => {
    expect(() => decodeManifest({ ...json(), sheets: { default: "" } })).toThrow("MANIFEST_SHAPE: manifest.sheets.default is not a non-empty string");
  });

  it("refuses a frame naming a sheet the manifest does not list", () => {
    const raw = json();
    expect(() => decodeManifest({ ...raw, fuel: { sheet: "other", x: 0, y: 0, w: 1, h: 1 } })).toThrow(
      'MANIFEST_SHEET: manifest.fuel names sheet "other", which the manifest does not list',
    );
  });

  it("refuses a list of the wrong length, naming it", () => {
    expect(() => decodeManifest({ ...json(), mines: [] })).toThrow("MANIFEST_SHAPE: manifest.mines is not a list of 4");
    const raw = json();
    const tanks = raw["tanks"];
    if (!Array.isArray(tanks)) {
      throw new Error("tanks is not a list");
    }
    expect(() => decodeManifest({ ...raw, tanks: [tanks[0], tanks[1], tanks[2], []] })).toThrow(
      "MANIFEST_SHAPE: manifest.tanks[3].facings is not a list of 16",
    );
  });

  it("refuses a missing terrain block and a missing name", () => {
    expect(() => decodeManifest({ ...json(), terrain: 3 })).toThrow("MANIFEST_SHAPE: manifest.terrain is not an object");
    expect(() => decodeManifest({ ...json(), name: 7 })).toThrow("MANIFEST_SHAPE: manifest.name is not a non-empty string");
  });

  it("refuses a tile size below one", () => {
    expect(() => decodeManifest({ ...json(), tileWidth: 0 })).toThrow("MANIFEST_SHAPE: manifest.tileWidth is not an integer of 1 or more");
  });
});

describe("decodeFrame", () => {
  const sheets = new Map([["s", "s.png"]]);

  it("reads a frame on a listed sheet", () => {
    expect(decodeFrame({ sheet: "s", x: 0, y: 4, w: 2, h: 3 }, "f", sheets)).toEqual({ sheet: "s", x: 0, y: 4, w: 2, h: 3 });
  });

  it("refuses negative, fractional, zero-sized and missing coordinates", () => {
    expect(() => decodeFrame({ sheet: "s", x: -1, y: 0, w: 1, h: 1 }, "f", sheets)).toThrow("MANIFEST_SHAPE: f.x is not an integer of 0 or more");
    expect(() => decodeFrame({ sheet: "s", x: 0, y: 0.5, w: 1, h: 1 }, "f", sheets)).toThrow("MANIFEST_SHAPE: f.y is not an integer of 0 or more");
    expect(() => decodeFrame({ sheet: "s", x: 0, y: 0, w: 0, h: 1 }, "f", sheets)).toThrow("MANIFEST_SHAPE: f.w is not an integer of 1 or more");
    expect(() => decodeFrame({ sheet: "s", x: 0, y: 0, w: 1 }, "f", sheets)).toThrow("MANIFEST_SHAPE: f.h is not an integer of 1 or more");
  });

  it("refuses a frame that is not an object or has no sheet", () => {
    expect(() => decodeFrame("frame", "f", sheets)).toThrow("MANIFEST_SHAPE: f is not an object");
    expect(() => decodeFrame({ x: 0 }, "f", sheets)).toThrow("MANIFEST_SHAPE: f.sheet is not a non-empty string");
  });
});
