import { describe, it, expect } from "vitest";
import {
  encodeAppConfig,
  decodeAppConfig,
  requireAppConfig,
  encodeTranslateResponse,
  decodeTranslateResponse,
  requireTranslateResponse,
  encodeErrorResponse,
  decodeErrorResponse,
  createRecorderState,
  createAppState,
} from "../../src/types.js";

describe("types", () => {
  describe("AppConfig", () => {
    describe("encodeAppConfig", () => {
      it("encodes AppConfig to object", () => {
        const config = { API_BASE_URL: "https://api.example.com" };
        const encoded = encodeAppConfig(config);

        expect(encoded).toEqual({ API_BASE_URL: "https://api.example.com" });
      });
    });

    describe("decodeAppConfig", () => {
      it("decodes valid object to AppConfig", () => {
        const input = { API_BASE_URL: "https://api.example.com" };
        const result = decodeAppConfig(input);

        expect(result).toEqual({ API_BASE_URL: "https://api.example.com" });
      });

      it("returns null for non-object", () => {
        expect(decodeAppConfig(null)).toBeNull();
        expect(decodeAppConfig(undefined)).toBeNull();
        expect(decodeAppConfig("string")).toBeNull();
        expect(decodeAppConfig(123)).toBeNull();
      });

      it("returns null for missing API_BASE_URL", () => {
        expect(decodeAppConfig({})).toBeNull();
        expect(decodeAppConfig({ other: "value" })).toBeNull();
      });

      it("returns null for non-string API_BASE_URL", () => {
        expect(decodeAppConfig({ API_BASE_URL: 123 })).toBeNull();
        expect(decodeAppConfig({ API_BASE_URL: null })).toBeNull();
      });

      it("returns null for empty API_BASE_URL", () => {
        expect(decodeAppConfig({ API_BASE_URL: "" })).toBeNull();
      });
    });

    describe("requireAppConfig", () => {
      it("returns AppConfig for valid input", () => {
        const input = { API_BASE_URL: "https://api.example.com" };
        const result = requireAppConfig(input);

        expect(result).toEqual({ API_BASE_URL: "https://api.example.com" });
      });

      it("throws for invalid input", () => {
        expect(() => requireAppConfig({})).toThrow(
          "Invalid config.json: missing or invalid API_BASE_URL"
        );
      });
    });
  });

  describe("TranslateResponse", () => {
    describe("encodeTranslateResponse", () => {
      it("encodes TranslateResponse to object", () => {
        const response = { text: "Hello grandmother" };
        const encoded = encodeTranslateResponse(response);

        expect(encoded).toEqual({ text: "Hello grandmother" });
      });
    });

    describe("decodeTranslateResponse", () => {
      it("decodes valid object to TranslateResponse", () => {
        const input = { text: "Hello grandmother" };
        const result = decodeTranslateResponse(input);

        expect(result).toEqual({ text: "Hello grandmother" });
      });

      it("decodes empty text", () => {
        const result = decodeTranslateResponse({ text: "" });
        expect(result).toEqual({ text: "" });
      });

      it("returns null for non-object", () => {
        expect(decodeTranslateResponse(null)).toBeNull();
        expect(decodeTranslateResponse(undefined)).toBeNull();
        expect(decodeTranslateResponse("string")).toBeNull();
      });

      it("returns null for missing text", () => {
        expect(decodeTranslateResponse({})).toBeNull();
        expect(decodeTranslateResponse({ other: "value" })).toBeNull();
      });

      it("returns null for non-string text", () => {
        expect(decodeTranslateResponse({ text: 123 })).toBeNull();
        expect(decodeTranslateResponse({ text: null })).toBeNull();
      });
    });

    describe("requireTranslateResponse", () => {
      it("returns TranslateResponse for valid input", () => {
        const input = { text: "Hello grandmother" };
        const result = requireTranslateResponse(input);

        expect(result).toEqual({ text: "Hello grandmother" });
      });

      it("throws for invalid input", () => {
        expect(() => requireTranslateResponse({})).toThrow("Invalid response from server");
      });
    });
  });

  describe("ErrorResponse", () => {
    describe("encodeErrorResponse", () => {
      it("encodes ErrorResponse to object", () => {
        const response = { detail: "Invalid token" };
        const encoded = encodeErrorResponse(response);

        expect(encoded).toEqual({ detail: "Invalid token" });
      });
    });

    describe("decodeErrorResponse", () => {
      it("decodes valid object to ErrorResponse", () => {
        const input = { detail: "Invalid token" };
        const result = decodeErrorResponse(input);

        expect(result).toEqual({ detail: "Invalid token" });
      });

      it("returns null for non-object", () => {
        expect(decodeErrorResponse(null)).toBeNull();
        expect(decodeErrorResponse(undefined)).toBeNull();
        expect(decodeErrorResponse("string")).toBeNull();
      });

      it("returns null for missing detail", () => {
        expect(decodeErrorResponse({})).toBeNull();
      });

      it("returns null for non-string detail", () => {
        expect(decodeErrorResponse({ detail: 123 })).toBeNull();
      });
    });
  });

  describe("RecorderState", () => {
    describe("createRecorderState", () => {
      it("creates initial recorder state", () => {
        const state = createRecorderState();

        expect(state.mediaRecorder).toBeNull();
        expect(state.audioChunks).toEqual([]);
        expect(state.isRecording).toBe(false);
      });
    });
  });

  describe("AppState", () => {
    describe("createAppState", () => {
      it("creates initial app state", () => {
        const state = createAppState();

        expect(state.config).toBeNull();
        expect(state.token).toBeNull();
        expect(state.recorderState.isRecording).toBe(false);
        expect(state.transcripts).toEqual([]);
        expect(state.pendingOperation).toBeNull();
      });
    });
  });
});
