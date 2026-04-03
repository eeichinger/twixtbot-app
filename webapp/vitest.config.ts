import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    environment: 'node',
    include: ['src/**/*.test.ts'],
  },
  resolve: {
    // Allow imports ending in .js to resolve to .ts files
    extensions: ['.ts', '.js'],
  },
});
