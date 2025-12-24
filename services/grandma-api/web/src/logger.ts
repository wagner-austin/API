/**
 * Logging utility for the application.
 *
 * Provides structured logging with namespace support. Wraps console methods
 * in a way that centralizes logging and satisfies ESLint no-console rule.
 * All console calls are contained here with explicit eslint-disable comments.
 */

/**
 * Logger interface with debug, info, warn, and error levels.
 */
export interface Logger {
  debug: (...args: unknown[]) => void;
  info: (...args: unknown[]) => void;
  warn: (...args: unknown[]) => void;
  error: (...args: unknown[]) => void;
}

/**
 * Create a namespaced logger.
 *
 * Args:
 *   namespace: Logger namespace (e.g., "app", "api", "recorder")
 *
 * Returns:
 *   Logger object with debug, info, warn, and error methods
 *
 * Example:
 *   const log = createLogger("app");
 *   log.debug("Details...");  // [app] Details...
 *   log.info("Starting...");  // [app] Starting...
 *   log.error("Failed:", err); // [app] Failed: Error: ...
 */
export function createLogger(namespace: string): Logger {
  const prefix = `[${namespace}]`;
  const timestamp = (): string => new Date().toISOString();

  return {
    debug: (...args: unknown[]): void => {
      // eslint-disable-next-line no-console
      console.debug(timestamp(), prefix, ...args);
    },
    info: (...args: unknown[]): void => {
      // eslint-disable-next-line no-console
      console.log(timestamp(), prefix, ...args);
    },
    warn: (...args: unknown[]): void => {
      // eslint-disable-next-line no-console
      console.warn(timestamp(), prefix, ...args);
    },
    error: (...args: unknown[]): void => {
      // eslint-disable-next-line no-console
      console.error(timestamp(), prefix, ...args);
    },
  };
}

/** Default application logger */
export const log = createLogger("app");
