import { defineConfig } from 'vite';
import { VitePWA } from 'vite-plugin-pwa';

export default defineConfig({
  base: '/twixtbot-app/',
  plugins: [
    VitePWA({
      strategies: 'injectManifest',
      srcDir: 'src',
      filename: 'sw.ts',
      registerType: 'autoUpdate',
      includeAssets: ['icons/*.png'],
      injectManifest: {
        globPatterns: ['**/*.{js,css,html,mjs}'],
      },
      manifest: {
        name: 'TwixT vs AI',
        short_name: 'TwixT',
        description: 'Play TwixT against a neural net AI — fully offline',
        display: 'standalone',
        orientation: 'portrait',
        background_color: '#f0f4f8',
        theme_color: '#0072b2',
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
