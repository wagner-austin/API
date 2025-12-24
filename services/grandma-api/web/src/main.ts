/**
 * Application entry point for browser.
 *
 * This module imports the app and triggers auto-initialization.
 * It should be the script loaded by index.html.
 */

import { autoInit } from "./app.js";
import { initTsEbml } from "./_test_hooks.js";
import { log } from "./logger.js";

log.info("main.ts: Starting application...");

// Initialize ts-ebml for WebM processing, then start app
initTsEbml()
  .then(() => {
    log.info("main.ts: ts-ebml initialized successfully");
    autoInit();
  })
  .catch((err: unknown) => {
    log.error("main.ts: Failed to initialize ts-ebml:", err);
    // Try to show error to user
    const loginError = document.getElementById("login-error");
    if (loginError !== null) {
      loginError.textContent = "Failed to load audio processing library";
      loginError.classList.remove("hidden");
    }
  });
