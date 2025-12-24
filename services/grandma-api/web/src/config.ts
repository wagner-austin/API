/**
 * Configuration loading using hooks for testability.
 *
 * Loads application configuration from config.json. Uses the _test_hooks
 * system for fetch dependency injection.
 */

import { getFetch } from "./_test_hooks.js";
import { AppConfig, requireAppConfig } from "./types.js";

let cached: AppConfig | null = null;

/**
 * Load application configuration from config.json.
 *
 * Caches the result after first successful load. Uses the fetch hook
 * from _test_hooks for testability.
 *
 * Returns:
 *   Promise resolving to AppConfig
 *
 * Raises:
 *   Error if config.json fetch fails
 *   Error if config.json is invalid or missing required fields
 */
export async function loadConfig(): Promise<AppConfig> {
  if (cached !== null) {
    return cached;
  }

  const fetch = getFetch();
  const resp = await fetch("config.json", {
    headers: { Accept: "application/json" },
  });

  if (!resp.ok) {
    throw new Error(`Failed to load config: ${resp.status}`);
  }

  const data: unknown = await resp.json();
  const config = requireAppConfig(data);

  cached = config;
  return config;
}

/**
 * Clear cached config.
 *
 * Resets the config cache, forcing the next loadConfig() call to fetch
 * fresh configuration from the server.
 */
export function clearConfigCache(): void {
  cached = null;
}
