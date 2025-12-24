/**
 * DOM utility functions using hooks for testability.
 *
 * Provides helper functions for common DOM operations.
 * Uses the _test_hooks system for document dependency injection.
 */

import { getDocument } from "./_test_hooks.js";

/**
 * Get element by ID with type safety.
 *
 * Args:
 *   id: Element ID
 *
 * Returns:
 *   Element
 *
 * Raises:
 *   Error if element not found
 */
export function getElementById<T extends HTMLElement>(id: string): T {
  const doc = getDocument();
  const el = doc.getElementById(id);
  if (el === null) {
    throw new Error(`Element not found: #${id}`);
  }
  return el as T;
}

/**
 * Show an element by removing the 'hidden' class.
 *
 * Args:
 *   el: Element to show
 */
export function showElement(el: HTMLElement): void {
  el.classList.remove("hidden");
}

/**
 * Hide an element by adding the 'hidden' class.
 *
 * Args:
 *   el: Element to hide
 */
export function hideElement(el: HTMLElement): void {
  el.classList.add("hidden");
}

/**
 * Set element text content.
 *
 * Args:
 *   el: Element to update
 *   text: Text content to set
 */
export function setText(el: HTMLElement, text: string): void {
  el.textContent = text;
}

/**
 * Add class to element.
 *
 * Args:
 *   el: Element to update
 *   className: Class name to add
 */
export function addClass(el: HTMLElement, className: string): void {
  el.classList.add(className);
}

/**
 * Remove class from element.
 *
 * Args:
 *   el: Element to update
 *   className: Class name to remove
 */
export function removeClass(el: HTMLElement, className: string): void {
  el.classList.remove(className);
}
