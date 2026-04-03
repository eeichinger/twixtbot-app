import { defineConfig } from 'vite';
import { VitePWA } from 'vite-plugin-pwa';

export default defineConfig({
  plugins: [
    VitePWA({
      registerType: 'autoUpdate',
      includeAssets: ['icons/*.png', 'model.onnx'],
      workbox: {
        globPatterns: ['**/*.{js,css,html,wasm,mjs}'],
        // Don't inline model.onnx in precache manifest (too large for revision hash);
        // use runtime CacheFirst instead.
        additionalManifestEntries: [],
        runtimeCaching: [
          {
            urlPattern: /model\.onnx$/,
            handler: 'CacheFirst',
            options: {
              cacheName: 'twixt-model-cache',
              expiration: { maxAgeSeconds: 30 * 24 * 60 * 60 },
            },
          },
          {
            urlPattern: /\.wasm$/,
            handler: 'CacheFirst',
            options: { cacheName: 'twixt-wasm-cache' },
          },
        ],
      },
      manifest: {
        name: 'TwixT vs AI',
        short_name: 'TwixT',
        description: 'Play TwixT against a neural net AI — fully offline',
        display: 'standalone',
        orientation: 'portrait',
        background_color: '#1a1a2e',
        theme_color: '#16213e',
        icons: [
          { src: 'icons/192.png', sizes: '192x192', type: 'image/png' },
          { src: 'icons/512.png', sizes: '512x512', type: 'image/png', purpose: 'any maskable' },
        ],
      },
    }),
  ],
  // Allow onnxruntime-web to load its own .wasm files from the same origin.
  optimizeDeps: {
    exclude: ['onnxruntime-web'],
  },
  server: {
    headers: {
      // Required for SharedArrayBuffer (used by ONNX WASM threading).
      'Cross-Origin-Opener-Policy':   'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  preview: {
    headers: {
      'Cross-Origin-Opener-Policy':   'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
});
