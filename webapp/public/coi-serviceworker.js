/**
 * Cross-Origin Isolation Service Worker
 *
 * Injects Cross-Origin-Opener-Policy and Cross-Origin-Embedder-Policy headers
 * into every same-origin response, enabling SharedArrayBuffer (required by
 * onnxruntime-web's WASM threading) on hosts like GitHub Pages that cannot
 * set HTTP headers directly.
 *
 * Registration: add a <script> in index.html that calls
 *   navigator.serviceWorker.register('/coi-serviceworker.js')
 * before the main app module loads.
 */

self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', (event) => event.waitUntil(self.clients.claim()));

self.addEventListener('fetch', (event) => {
  const { request } = event;

  // Only intercept GET requests to the same origin.
  if (request.method !== 'GET') return;
  if (new URL(request.url).origin !== self.location.origin) return;

  event.respondWith(
    fetch(request).then((response) => {
      // Pass through error/opaque responses unchanged.
      if (response.status === 0 || response.type === 'opaque') return response;

      const headers = new Headers(response.headers);
      headers.set('Cross-Origin-Opener-Policy', 'same-origin');
      headers.set('Cross-Origin-Embedder-Policy', 'credentialless');

      return new Response(response.body, {
        status: response.status,
        statusText: response.statusText,
        headers,
      });
    }),
  );
});
