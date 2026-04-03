/**
 * Configuration loading using hooks for testability.
 *
 * Loads application configuration from config.json. Uses the _test_hooks
 * system for fetch dependency injection. Auto-detects local vs production
 * environment based on hostname.
 */

import { getFetch, getLocation } from "./_test_hooks.js";
import { AppConfig, requireAppConfig } from "./types.js";

/** Local API port for development. */
const LOCAL_API_PORT = "8090";

let cached: AppConfig | null = null;

/**
 * Detect API base URL based on current hostname.
 *
 * Returns local API URL for localhost/127.0.0.1 or private IPs,
 * otherwise returns the configured production URL.
 *
 * Args:
 *   configuredUrl: The URL from config.json (production default).
 *
 * Returns:
 *   API base URL appropriate for current environment.
 */
export function detectApiBaseUrl(configuredUrl: string): string {
  const location = getLocation();
  const hostname = location.hostname;

  // Local development: localhost, 127.0.0.1, or private IPs (10.x, 192.168.x, 100.x Tailscale)
  const isLocal =
    hostname === "localhost" ||
    hostname === "127.0.0.1" ||
    hostname.startsWith("192.168.") ||
    hostname.startsWith("10.") ||
    hostname.startsWith("100.");

  if (isLocal) {
    return `https://${hostname}:${LOCAL_API_PORT}`;
  }

  // Production - use configured URL
  return configuredUrl;
}

/**
 * Load application configuration from config.json.
 *
 * Caches the result after first successful load. Uses the fetch hook
 * from _test_hooks for testability. Auto-detects local vs production
 * API URL based on hostname.
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

  // Auto-detect API URL based on environment
  const resolvedConfig: AppConfig = {
    API_BASE_URL: detectApiBaseUrl(config.API_BASE_URL),
  };

  cached = resolvedConfig;
  return resolvedConfig;
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
