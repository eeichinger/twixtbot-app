/**
 * Custom service worker — replaces both the Workbox-generated sw.js and the
 * separate coi-serviceworker.js.  Does two jobs in one:
 *
 * 1. Injects Cross-Origin-Opener-Policy / Cross-Origin-Embedder-Policy into
 *    every same-origin response so that SharedArrayBuffer (required by
 *    onnxruntime-web's WASM backend) is available on GitHub Pages.
 *
 * 2. Provides offline support:
 *    - Precaches the app shell (JS, CSS, HTML, icons) via __WB_MANIFEST
 *    - CacheFirst for model.onnx  (large, changes only on model retraining)
 *    - CacheFirst for .wasm files (large, stable)
 */

declare const self: ServiceWorkerGlobalScope;

const PRECACHE   = 'twixtbot-shell-v1';
const MODEL_CACHE = 'twixt-model';
const WASM_CACHE  = 'twixt-wasm';

// ── Install: populate precache ───────────────────────────────────────────────

self.addEventListener('install', (event) => {
  // self.__WB_MANIFEST is replaced at build time by vite-plugin-pwa
  const urls: string[] = ((self as unknown as Record<string, unknown>).__WB_MANIFEST as Array<{ url: string }> ?? [])
    .map((e) => e.url);

  // skipWaiting() lets the new SW take control immediately instead of waiting
  // for all tabs to close.  The client (main.ts) listens for 'controllerchange'
  // and only reloads the page when no game is in progress, so mid-move
  // computation is never interrupted.
  self.skipWaiting();
  event.waitUntil(
    caches.open(PRECACHE).then((cache) => cache.addAll(urls)),
  );
});

// ── Activate: take control immediately ──────────────────────────────────────

self.addEventListener('activate', (event) => {
  event.waitUntil(self.clients.claim());
});

// ── Helpers ──────────────────────────────────────────────────────────────────

/** Wrap a response with COOP/COEP headers (enables SharedArrayBuffer). */
function withCOI(response: Response): Response {
  if (response.type === 'opaque') return response;
  const headers = new Headers(response.headers);
  headers.set('Cross-Origin-Opener-Policy', 'same-origin');
  headers.set('Cross-Origin-Embedder-Policy', 'require-corp');
  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers,
  });
}

/** CacheFirst helper: return cached copy or fetch + store + return. */
async function cacheFirst(request: Request, cacheName: string): Promise<Response> {
  const cache  = await caches.open(cacheName);
  const cached = await cache.match(request);
  if (cached) return withCOI(cached);
  try {
    const fresh = await fetch(request);
    if (fresh.ok) cache.put(request, fresh.clone());
    return withCOI(fresh);
  } catch {
    return new Response('Resource not cached and network unavailable', { status: 503 });
  }
}

// ── Fetch: single handler, always adds COI headers ───────────────────────────

self.addEventListener('fetch', (event) => {
  const { request } = event;
  if (request.method !== 'GET') return;
  if (new URL(request.url).origin !== self.location.origin) return;

  event.respondWith(handleFetch(request));
});

async function handleFetch(request: Request): Promise<Response> {
  // 1. App shell / precached assets
  const precached = await caches.match(request, { cacheName: PRECACHE });
  if (precached) return withCOI(precached);

  const url = request.url;

  // 2. ONNX model — large, stable, cache indefinitely
  if (/\/model\.onnx(\?|$)/.test(url)) return cacheFirst(request, MODEL_CACHE);

  // 3. WASM binaries — large, stable, cache indefinitely
  if (/\.wasm(\?|$)/.test(url)) return cacheFirst(request, WASM_CACHE);

  // 4. Navigation requests (e.g. start_url variants) — fall back to cached index.html
  if (request.mode === 'navigate') {
    const indexFallback = await caches.match('/twixtbot-app/index.html');
    if (indexFallback) return withCOI(indexFallback);
  }

  // 5. Anything else — network with COI headers; return offline error if network unavailable
  try {
    return withCOI(await fetch(request));
  } catch {
    return new Response('Offline', { status: 503, statusText: 'Service Unavailable' });
  }
}
