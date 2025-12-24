/**
 * Token storage using hooks for testability.
 *
 * Provides functions to persist the authentication token to localStorage.
 * Uses the _test_hooks system for storage dependency injection.
 */

import { getStorage } from "./_test_hooks.js";

const TOKEN_KEY = "grandma_token";

/**
 * Save token to storage.
 *
 * Persists the authentication token using the storage hook.
 *
 * Args:
 *   token: Authentication token string
 */
export function saveToken(token: string): void {
  const storage = getStorage();
  storage.setItem(TOKEN_KEY, token);
}

/**
 * Load token from storage.
 *
 * Retrieves the stored authentication token if present.
 *
 * Returns:
 *   Token string if found, null otherwise
 */
export function loadToken(): string | null {
  const storage = getStorage();
  return storage.getItem(TOKEN_KEY);
}

/**
 * Clear token from storage.
 *
 * Removes the stored authentication token.
 */
export function clearToken(): void {
  const storage = getStorage();
  storage.removeItem(TOKEN_KEY);
}
