import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { createLogger } from "../../src/logger.js";

describe("logger", () => {
  let consoleLogSpy: ReturnType<typeof vi.spyOn>;
  let consoleWarnSpy: ReturnType<typeof vi.spyOn>;
  let consoleErrorSpy: ReturnType<typeof vi.spyOn>;
  let consoleDebugSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    consoleLogSpy = vi.spyOn(console, "log").mockImplementation(() => {});
    consoleWarnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    consoleErrorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    consoleDebugSpy = vi.spyOn(console, "debug").mockImplementation(() => {});
  });

  afterEach(() => {
    consoleLogSpy.mockRestore();
    consoleWarnSpy.mockRestore();
    consoleErrorSpy.mockRestore();
    consoleDebugSpy.mockRestore();
  });

  describe("createLogger", () => {
    it("creates logger with debug, info, warn, and error methods", () => {
      const log = createLogger("test");
      expect(typeof log.debug).toBe("function");
      expect(typeof log.info).toBe("function");
      expect(typeof log.warn).toBe("function");
      expect(typeof log.error).toBe("function");
    });

    it("debug logs with timestamp and namespace prefix", () => {
      const log = createLogger("mymodule");
      log.debug("debug message", 123);
      expect(consoleDebugSpy).toHaveBeenCalledTimes(1);
      const args = consoleDebugSpy.mock.calls[0];
      expect(args).toBeDefined();
      expect(args[0]).toMatch(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/);
      expect(args[1]).toBe("[mymodule]");
      expect(args[2]).toBe("debug message");
      expect(args[3]).toBe(123);
    });

    it("info logs with timestamp and namespace prefix", () => {
      const log = createLogger("mymodule");
      log.info("test message", 123);
      expect(consoleLogSpy).toHaveBeenCalledTimes(1);
      const args = consoleLogSpy.mock.calls[0];
      expect(args).toBeDefined();
      expect(args[0]).toMatch(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/);
      expect(args[1]).toBe("[mymodule]");
      expect(args[2]).toBe("test message");
      expect(args[3]).toBe(123);
    });

    it("warn logs with timestamp and namespace prefix", () => {
      const log = createLogger("mymodule");
      log.warn("warning message", { key: "value" });
      expect(consoleWarnSpy).toHaveBeenCalledTimes(1);
      const args = consoleWarnSpy.mock.calls[0];
      expect(args).toBeDefined();
      expect(args[0]).toMatch(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/);
      expect(args[1]).toBe("[mymodule]");
      expect(args[2]).toBe("warning message");
      expect(args[3]).toEqual({ key: "value" });
    });

    it("error logs with timestamp and namespace prefix", () => {
      const log = createLogger("mymodule");
      const err = new Error("test error");
      log.error("error occurred", err);
      expect(consoleErrorSpy).toHaveBeenCalledTimes(1);
      const args = consoleErrorSpy.mock.calls[0];
      expect(args).toBeDefined();
      expect(args[0]).toMatch(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/);
      expect(args[1]).toBe("[mymodule]");
      expect(args[2]).toBe("error occurred");
      expect(args[3]).toBe(err);
    });

    it("supports multiple arguments", () => {
      const log = createLogger("multi");
      log.info("a", "b", "c", 1, 2, 3);
      expect(consoleLogSpy).toHaveBeenCalledTimes(1);
      const args = consoleLogSpy.mock.calls[0];
      expect(args).toBeDefined();
      expect(args[0]).toMatch(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/);
      expect(args[1]).toBe("[multi]");
      expect(args[2]).toBe("a");
      expect(args[3]).toBe("b");
      expect(args[4]).toBe("c");
      expect(args[5]).toBe(1);
      expect(args[6]).toBe(2);
      expect(args[7]).toBe(3);
    });

    it("different namespaces create independent loggers", () => {
      const logA = createLogger("moduleA");
      const logB = createLogger("moduleB");

      logA.info("from A");
      logB.info("from B");

      expect(consoleLogSpy).toHaveBeenCalledTimes(2);
      const argsA = consoleLogSpy.mock.calls[0];
      const argsB = consoleLogSpy.mock.calls[1];
      expect(argsA).toBeDefined();
      expect(argsB).toBeDefined();
      expect(argsA[1]).toBe("[moduleA]");
      expect(argsA[2]).toBe("from A");
      expect(argsB[1]).toBe("[moduleB]");
      expect(argsB[2]).toBe("from B");
    });
  });
});
