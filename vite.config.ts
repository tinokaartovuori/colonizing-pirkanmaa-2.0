import { defineConfig } from 'vite';

// Static SPA build. base: './' keeps asset URLs relative so the build works
// from any sub-path (e.g. GitHub Pages project pages).
export default defineConfig({
  base: './',
  build: {
    target: 'es2020',
    assetsInlineLimit: 0,
  },
  server: {
    host: '127.0.0.1',
    port: 5173,
    // Always use 5173 — fail loudly (so a stale instance can be killed) instead of
    // silently hopping to 5174/5175, which makes the running build unpredictable.
    strictPort: true,
  },
  preview: {
    host: '127.0.0.1',
    port: 5173,
    strictPort: true,
  },
});
