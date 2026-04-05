# CLAUDE.md — Project Memory

This file records architectural decisions, lessons learned, and recommendations
for future development of the twixtbot-app project.

---

## Project Overview

A web app that lets a human play TwixT against a neural network AI, deployed
as a PWA on GitHub Pages (`https://eeichinger.github.io/twixtbot-app/`).

- **Human is BLACK**, **AI is WHITE**
- Board size: 24×24, 528 legal move positions (24 × 22 inner grid)
- AI uses MCTS + neural network policy/value evaluation
- Fully offline-capable (PWA with Service Worker)

---

## Original Python-Based Approach

### Architecture

The original codebase (`src/`) is Python 2 + TensorFlow 1.12, designed for
server-side use or desktop play.

**Key files:**
- `src/nns.py` — Neural Net Server: loads TF model, listens on a Unix socket + shared memory, batches inference queries from multiple clients
- `src/nnclient.py` + `src/smmpp.py` — client-side IPC to the NNS
- `src/nnmcts.py` / `src/nnmplayer.py` — synchronous MCTS player (used with `one.py` or `battle.py`)
- `src/asn_player.py` — Asynchronous Net Player, preferred for self-play
- `src/one.py` — CLI to get a single best move given a position history
- `src/battle.py` + `src/pmany.py` — parallel self-play framework (runs ~80 workers against a single NNS)
- `src/train.py` — training loop (self-play output → model update)
- `src/twixt.py` — canonical `Game` class (ported to TypeScript for the webapp)
- `src/naf.py` — position → numpy arrays for model input (ported to TypeScript)
- `src/swapmodel.py` — instead of MCTSing the swap rule, fitted a near-linear model on bot self-play results to decide swap/keep

**Model format:** TensorFlow 1 checkpoint (`models/six-917000.*`)

### PyTorch Rewrite (TRAINING.md)

A modern PyTorch version of the model has been implemented (`src/model.py`),
targeting AMD Ryzen 7800X3D + RTX 5070 Ti. Key differences from original:

- Activation: GELU (default) instead of `abs()`
- Policy head: 528 outputs (fixes off-by-one in original 529)
- Value head: always 3-class logits (Loss/Draw/Win)
- `value_padding='valid'` (default) or `'same'` to match TF1 original

**Swap rule model** is maintained as a fitted linear model; see
`src/swapmodel.py`. This avoids expensive MCTS search at move 2.

---

## AI Opponent: Improvement Ideas & Recommendations

### Current Setup (webapp)
- MCTS with a 5–60 second time budget (user-configurable)
- Neural network policy + value from an ONNX-exported model
- `MAX_TRIALS = 100_000` (effectively unlimited; real constraint is time)
- Hard deadline pattern: `setTimeout` fires at `timeLimitMs` and sends the
  best move found so far; MCTS `await` may still be running (WASM blocking)
  when the result is sent

### Improvement Ideas

**1. Swap rule implementation (RECOMMENDED)**
Currently the AI plays a random/policy move for swap. The original codebase
uses a fitted logistic model (`swapmodel.py`) to decide whether to swap.
Porting this to TypeScript would significantly improve early-game play.
The swap model coefficients are simple enough to hardcode.

**2. Progressive widening / better MCTS tuning**
The current MCTS uses uniform Dirichlet noise (α=0.3 default). Tuning the
exploration constant (currently 1.0) and noise weight (0.0 in competitive
mode) could improve strength.

**3. Larger/stronger model (RECOMMENDED for quality)**
The deployed model (`models/six-917000`) is the TF1-era model, which the
README acknowledges "can probably be improved with more training and/or a
bigger net." The PyTorch rewrite enables training a stronger model. A
`filters=128, blocks=12` config would be a significant upgrade, but would
increase ONNX model file size and memory usage — measure iOS impact first.

**4. Quantized model (RECOMMENDED for iOS memory)**
Converting the ONNX model to int8 or fp16 would reduce model weight memory
by 2–4×. This directly addresses the iOS deferred-kill problem (see iOS
section). Tools: `onnxruntime.quantization` or `onnxmltools`.

**5. MCTS opening book**
For the first 2–3 moves, replace MCTS with a hardcoded or precomputed
opening book. Avoids burning 5+ seconds on trivially-decided early moves.

**6. Worker restart on new game vs. keep-alive**
Currently the worker is terminated after each AI move (to free WASM heap
memory during the human turn — see iOS section). Restarting takes ~0.5–1s
from HTTP cache. For fast think times (≤5s) this overhead is noticeable.
If iOS memory issues are resolved (e.g., via model quantization), switching
back to keep-alive would improve responsiveness.

**7. `computing-done` message for diagnostics**
The worker sends `{ type: 'computing-done', elapsed }` from its `finally`
block. This reveals how long WASM was still running after the hard deadline
fired. Useful for tuning `timeLimitMs` vs. actual MCTS depth.

---

## PWA Implementation

### Stack
- **Vite** + **vite-plugin-pwa** (`injectManifest` strategy)
- **Service Worker** (`src/sw.ts`): sets COOP/COEP headers on all responses,
  enabling `SharedArrayBuffer` (required by the threaded WASM binary)
- **Web Worker** (`src/worker.ts`): runs MCTS + ONNX off the main thread
- **onnxruntime-web** 1.20.x for in-browser ONNX inference

### WASM Binary Selection (CRITICAL)

ORT ships multiple WASM binaries. The npm package default entry point
(`import * as ort from 'onnxruntime-web'`) resolves to `ort.bundle.min.mjs`
which **always** loads the 24MB JSEP binary (`ort-wasm-simd-threaded.jsep.wasm`),
even when only the WASM backend is used. This wastes ~12MB of compiled WASM
memory vs. the standard binary.

**Correct import:**
```typescript
import * as ort from 'onnxruntime-web/wasm';
```

This uses `ort.wasm.bundle.min.mjs` which loads `ort-wasm-simd-threaded.wasm`
(12MB instead of 24MB).

**Required public files** (both must be served at `wasmPaths` directory):
- `ort-wasm-simd-threaded.wasm` (12MB) — the WASM binary
- `ort-wasm-simd-threaded.mjs` (24KB) — the Emscripten JS glue module

If either file is missing, ORT throws `TypeError: Importing a module script failed`.

**prebuild script** (in `webapp/package.json`):
```json
"prebuild": "cp node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.mjs public/"
```

### ORT Session Options

```typescript
ort.env.wasm.numThreads = 1;   // Prevents pthread worker creation (see iOS section)
ort.env.wasm.wasmPaths = import.meta.env.BASE_URL;  // Load from root, not assets/

this.session = await ort.InferenceSession.create(modelUrl, {
  executionProviders: ['wasm'],
  enableCpuMemArena: false,   // Prevents large WASM heap pre-allocation
});
```

`enableCpuMemArena: false` prevents ORT from holding large heap slabs between
inference calls, reducing peak memory on iOS.

### Why numThreads=1 is Required

Setting `numThreads > 1` causes ORT to spawn pthread Worker threads using a
URL derived from `import.meta.url`. In Vite's module worker context this
resolves to a bundled file path that doesn't exist as a separate file,
causing `Importing a module script failed`. With `numThreads=1`, ORT skips
thread-pool creation entirely.

The threaded WASM binary is still used (it's the only binary available in
ORT 1.18+), and SharedArrayBuffer is still needed — but no actual threads
are spawned.

### Model File

The ONNX model (`model.onnx`) is served from the GitHub Pages root but is
**not** precached by the Service Worker (only JS/CSS/HTML/MJS are precached).
It is loaded fresh by the worker each time and cached by the browser's
standard HTTP cache. This is intentional — the model file is large and its
SW cache entry would bloat the SW cache.

### Service Worker Headers

The SW intercepts all fetch responses and injects:
```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

These are required for `crossOriginIsolated=true`, which enables
`SharedArrayBuffer`. Without these headers, `new SharedArrayBuffer()` throws.

### Hard Deadline Pattern in Worker

MCTS runs in a `try/finally` block with a `setTimeout` hard deadline:

```typescript
const hardDeadlineId = setTimeout(() => {
  self.postMessage({ type: 'result', move: bestMoveFoundSoFar });
}, timeLimitMs);

try {
  result = await mcts.mcts(game, MAX_TRIALS, timeLimitMs);
} finally {
  clearTimeout(hardDeadlineId);
  self.postMessage({ type: 'computing-done', elapsed: Date.now() - moveStart });
}
```

The hard deadline fires and sends the result early; the `mcts.mcts()` `await`
may still be blocking on WASM after the result is sent. The `computing-done`
message tells the main thread how long WASM ran total.

### Diagnostic Logging System

A localStorage-backed ring buffer (`DIAG_KEY`, 150 entries) persists log
entries across page reloads. On `pagehide`, the current session log is
copied to `DIAG_PREV_KEY` for inspection after crashes. The diag panel is
toggled by triple-tapping the version label on either the intro or game
screen. A Share button invokes `navigator.share()` on iOS (native share
sheet) or `navigator.clipboard.writeText()` on other platforms.

---

## iOS-Specific Findings and Learnings

### The Core Problem: iOS Deferred Page Kill

iOS WebKit kills pages that exceed a memory threshold. The kill is:
1. **Deferred**: iOS observes peak memory during AI computation (5s MCTS),
   schedules the kill, but executes it 10–30 seconds later during the human
   turn — giving the appearance that crashes happen "at random during the
   human turn"
2. **Hard**: no `beforeunload` or `pagehide` events fire before the kill
3. **Reproducible**: first 1–2 sessions after a cold start crash quickly;
   later sessions survive longer (WASM module caching reduces peak allocation)

### Memory Pressure Evidence

From diagnostic logs when iOS was already under memory pressure:
```
worker-error: Error: Failed to load ONNX model from /twixtbot-app/model.onnx:
  Error: no available backend found. ERR: [wasm] RangeError: Out of memory
```

This OOM error directly confirms the WASM heap could not be allocated.

### What Does NOT Cause the Crash

- **Auto-lock / screen turning off** — user confirmed phone stays awake
- **JavaScript errors** — no `js-error` or `unhandled-rejection` entries
  appear before crashes
- **Foreground/background cycle** — crashes happen with phone in hand,
  screen on, app in foreground
- **Service Worker updates** — `sw-controllerchange` never fires before crashes
- **Long human turns** — crashes happen within 2–24 seconds of turn start,
  independent of whether the user has done anything

### What DOES Cause the Crash

Peak WASM heap during the 5-second MCTS computation. The heap holds:
- ORT compiled WASM binary code (~50–100MB for the 12MB binary)
- ONNX model weights (size of `model.onnx` when expanded in memory)
- Activation tensors during forward passes
- ORT memory arena allocations (disabled with `enableCpuMemArena: false`)

### Mitigation: Terminate Worker After Each AI Move

The worker (holding the WASM heap) is terminated immediately after the AI
move result is processed. This frees the heap before the human turn begins.
The worker is restarted in `requestAiMove()` when the human makes their move.

**Important**: this does NOT prevent the deferred kill if iOS has already
scheduled it based on peak memory during computation. But it prevents
compounding kills across multiple moves.

Model reload from HTTP cache takes ~0.5–1s (consistently observed in logs).
The thinking overlay appears during this reload period.

### Cross-Origin Isolation on iOS

`crossOriginIsolated=true` is required for `SharedArrayBuffer` (needed by the
threaded WASM binary). This is achieved via Service Worker header injection.
iOS Safari 18.7 handles COI correctly in our setup.

### WebGPU / JSEP Binary Failure on iOS

Using `executionProviders: ['webgpu']` or the default `ort.bundle.min.mjs`
(JSEP binary) inside a module Worker on iOS Safari fails with:
```
TypeError: Importing a module script failed
```
This error "poisons" ORT's internal WASM init state permanently, making the
WASM fallback also fail in the same process. Fix: use `onnxruntime-web/wasm`
entry point and `executionProviders: ['wasm']` only.

### SharedArrayBuffer Even with numThreads=1

ORT 1.18+ ships only threaded WASM binaries. Even with `numThreads=1`,
the threaded binary requires SharedArrayBuffer to be available (it's used
for the WASM heap's memory model). Therefore COOP/COEP headers remain
required even in single-thread mode.

### iOS Does Not Free Worker Memory Immediately

When `worker.terminate()` is called, iOS does not immediately run GC to
reclaim the WASM heap. The memory may remain allocated for several seconds.
This means the deferred kill can fire even after the worker has been
terminated, if iOS already observed the peak memory and scheduled the kill
before termination.

### Rapid Reload Loop

Under heavy memory pressure, iOS enters a reload loop:
- Page loads → worker starts → OOM on WASM heap → crash → repeat
- Sessions survive 0.038–4 seconds before the next kill
- Once iOS clears memory (usually after 3–5 kills), a session survives

This pattern is consistently observed in logs as multiple rapid
`app-start` → `worker-error: RangeError: Out of memory` sequences.

### Recommended Next Steps for iOS Memory

1. **Quantize the ONNX model** (int8 or fp16) — 2–4× reduction in model
   weight memory, directly reduces peak WASM heap. This is the single most
   impactful remaining improvement.
2. **Reduce think time** on iOS — detect iOS and default to 5s instead of
   10s to reduce the duration of peak memory usage (may not help if iOS
   already observed the peak).
3. **Profile actual model memory** — add `performance.memory` logging if
   running on Chrome (not iOS) to baseline model vs. runtime overhead split.

### Version History (crash investigation)

| Version | Key change | Outcome |
|---|---|---|
| `2026-04-04-b` | Baseline with diagnostic logging | Crashes after 1 AI move (15–18s into human turn) |
| `2026-04-04-c` | Terminate worker after each move (first attempt) | Worse — double model load due to initWorker called twice |
| `2026-04-04-d` | Wake Lock API | Failed — iOS denies with `NotAllowedError` |
| `2026-04-04-e` | Roll back to -b + extended diagnostics | Confirmed deferred-kill pattern; `worker-computing-done` added |
| `2026-04-04-f` | Intro screen + return to intro after game end | UX improvement |
| `2026-04-04-g` | Terminate worker after each AI move (fixed) | Confirms deferred-kill — crash still happens after termination |
| `2026-04-04-h` | Switch to `onnxruntime-web/wasm` (12MB binary); `enableCpuMemArena: false` | **FIXED** — game runs stably on iOS. The combination of halving the WASM binary (24MB JSEP → 12MB standard) and disabling the memory arena eliminated the peak memory that was triggering the deferred OS kill. |

---

## Service Worker Cache Busting

### Problem

The SW used `injectManifest` strategy with precached hashed assets but did **not** call
`skipWaiting()`. New SWs waited until all tabs were closed before activating. On iOS Safari
this meant users ran old cached CSS/JS for days even after a fresh deploy and manual
tab-closing. Chrome was unaffected (faster SW lifecycle management).

### Solution: skipWaiting + conditional reload

`sw.ts` now calls `self.skipWaiting()` in the install handler. The new SW activates
immediately. The client (`main.ts`) listens for the resulting `controllerchange` event and
calls `window.location.reload()` **only when `userClickedStart === false`** (i.e. the user
is on the intro screen, not mid-game). This gives instant cache busting with zero risk of
disrupting an active AI computation.

```typescript
// sw.ts — install handler
self.skipWaiting();

// main.ts — controllerchange listener
navigator.serviceWorker.addEventListener('controllerchange', () => {
  if (!userClickedStart) window.location.reload();
});
```

**Note:** The very first deploy of the `skipWaiting()` change still requires a manual cache
clear, because the old SW (without `skipWaiting`) is what runs that one time. All subsequent
deploys are automatic.

---

## Safari "Request Desktop Website" Viewport Quirk

### Problem

iOS Safari has a per-site (and global) **"Request Desktop Website"** toggle. When enabled,
Safari inflates the CSS viewport to ~980px while the physical screen remains 393px (iPhone 15).

This breaks CSS `@media (max-width: 430px)` and `@media (min-width: 431px)` queries —
they see the fake 980px viewport, not the real screen width, so phone-targeted font sizes
never apply. Chrome on the same device works correctly because it uses the real viewport.

Symptoms: font sizes look correct in Chrome but remain at tablet/desktop compact sizes in
Safari, even after clearing all caches.

### Solution: `window.screen.width` instead of CSS viewport queries

`screen.width` always returns the **physical device CSS pixel width** regardless of Safari's
viewport scaling. An inline script in `<head>` (runs before CSS paint, no FOUC) adds a
class to `<html>` based on the real screen size:

```html
<!-- index.html <head> -->
<script>if(window.screen.width<=430)document.documentElement.classList.add('phone-screen');</script>
```

CSS then scopes phone-size rules to `.phone-screen` instead of a media query:

```css
.phone-screen .mode-btn        { font-size: 14px; letter-spacing: 1.5px; }
.phone-screen .bp-tagline      { font-size: 14px; }
.phone-screen .intro-attribution { font-size: 12px; }
/* etc. */
```

**Rule:** Never use CSS viewport-width media queries for phone vs. tablet detection in this
app. Always use `screen.width` via the `phone-screen` class mechanism.

### Touch drag offset

The board's touch drag callout offset (`-this.cellSize * 2`) was too small on iPhone
(~31px, less than a fingertip width of ~44px) while fine on the Fire HD 10 tablet (~60px).

Fix: `Math.max(this.cellSize * 2, 65)` — a 65px floor ensures the preview peg always
clears the fingertip on small-cell boards without meaningfully changing the tablet experience.
