import { defineConfig } from 'vitest/config';
import { fileURLToPath } from 'node:url';

export default defineConfig({
  resolve: { alias: { '@': fileURLToPath(new URL('.', import.meta.url)) } },
  css: { postcss: { plugins: [] } },
  test: {
    restoreMocks: true,
    maxWorkers: 2,
    projects: [
      { extends: true, test: { name: 'contracts', environment: 'node', include: ['lib/**/*.test.ts', 'app/api/**/*.test.ts'] } },
      { extends: true, test: { name: 'ui', environment: 'jsdom', setupFiles: ['./vitest.setup.ts'], include: ['components/**/*.test.tsx', 'app/lab/**/*.test.tsx'] } },
    ],
  },
  oxc: { jsx: { runtime: 'automatic' } },
});
