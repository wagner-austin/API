import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    environment: "jsdom",
    include: ["tests/unit/**/*.spec.ts"],
    setupFiles: ["tests/setup.ts"],
    isolate: true, // Isolate test environments
    fileParallelism: false, // Run test files sequentially to avoid interference
    coverage: {
      provider: "istanbul",
      include: ["src/**/*.ts"],
      exclude: ["assets/**", "src/main.ts"],
      thresholds: {
        statements: 100,
        branches: 100,
        functions: 100,
        lines: 100,
      },
    },
  },
});
