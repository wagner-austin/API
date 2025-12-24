import { describe, it, expect, afterEach } from "vitest";
import { setHooks, resetHooks } from "../../src/_test_hooks.js";
import { createFakeHooks } from "../../src/testing.js";
import {
  getElementById,
  showElement,
  hideElement,
  setText,
  addClass,
  removeClass,
} from "../../src/dom.js";

describe("dom utilities", () => {
  afterEach(() => {
    resetHooks();
  });

  describe("getElementById", () => {
    it("returns element when found", () => {
      const div = document.createElement("div");
      const elements = new Map<string, HTMLElement>([["test-element", div]]);

      const { hooks } = createFakeHooks({ elements });
      setHooks(hooks);

      expect(getElementById("test-element")).toBe(div);
    });

    it("throws when element not found", () => {
      const { hooks } = createFakeHooks({ elements: new Map() });
      setHooks(hooks);

      expect(() => getElementById("nonexistent")).toThrow("Element not found: #nonexistent");
    });
  });

  describe("showElement", () => {
    it("removes hidden class", () => {
      const el = document.createElement("div");
      el.classList.add("hidden");

      showElement(el);

      expect(el.classList.contains("hidden")).toBe(false);
    });

    it("does nothing if not hidden", () => {
      const el = document.createElement("div");

      showElement(el);

      expect(el.classList.contains("hidden")).toBe(false);
    });
  });

  describe("hideElement", () => {
    it("adds hidden class", () => {
      const el = document.createElement("div");

      hideElement(el);

      expect(el.classList.contains("hidden")).toBe(true);
    });

    it("keeps hidden class if already present", () => {
      const el = document.createElement("div");
      el.classList.add("hidden");

      hideElement(el);

      expect(el.classList.contains("hidden")).toBe(true);
    });
  });

  describe("setText", () => {
    it("sets text content", () => {
      const el = document.createElement("div");

      setText(el, "Hello World");

      expect(el.textContent).toBe("Hello World");
    });

    it("replaces existing text", () => {
      const el = document.createElement("div");
      el.textContent = "Old";

      setText(el, "New");

      expect(el.textContent).toBe("New");
    });
  });

  describe("addClass", () => {
    it("adds class to element", () => {
      const el = document.createElement("div");

      addClass(el, "active");

      expect(el.classList.contains("active")).toBe(true);
    });
  });

  describe("removeClass", () => {
    it("removes class from element", () => {
      const el = document.createElement("div");
      el.classList.add("active");

      removeClass(el, "active");

      expect(el.classList.contains("active")).toBe(false);
    });
  });
});
