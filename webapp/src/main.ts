/**
 * main.ts — App shell and game loop.
 *
 * Manages:
 *   - Worker lifecycle (init, move requests)
 *   - Game state (human is BLACK, AI is WHITE)
 *   - UI state (loading, thinking, game-over)
 */

import { Game, pt, WHITE, BLACK } from './twixt.js';
import type { MoveRecord } from './twixt.js';
import { BoardUI } from './ui.js';

// -------------------------------------------------------------------------
// Version — update this string with every deploy to confirm new code loaded
// -------------------------------------------------------------------------

const APP_VERSION = '2026-04-04-g';

// -------------------------------------------------------------------------
// Constants
// -------------------------------------------------------------------------

const MODEL_URL   = import.meta.env.BASE_URL + 'model.onnx';
const HUMAN_COLOR = BLACK;
const AI_COLOR    = WHITE;

const THINK_TIME_OPTIONS = [5, 10, 15, 25, 30, 45, 60];  // seconds
const THINK_TIME_KEY     = 'twixt-think-time-sec';
const DEFAULT_THINK_TIME = 10;

function getThinkTimeSec(): number {
  const stored = parseInt(localStorage.getItem(THINK_TIME_KEY) ?? '', 10);
  return THINK_TIME_OPTIONS.includes(stored) ? stored : DEFAULT_THINK_TIME;
}

// -------------------------------------------------------------------------
// Diagnostic logging — survives page reloads via localStorage
// -------------------------------------------------------------------------

const DIAG_KEY = 'twixt-diag-log';
const DIAG_PREV_KEY = 'twixt-diag-log-prev';
const t0 = Date.now();

function diagLog(event: string): void {
  const elapsed = ((Date.now() - t0) / 1000).toFixed(3);
  const ts = new Date().toISOString().substring(11, 23); // HH:MM:SS.mmm
  const entry = `[${ts} +${elapsed}s] ${event}`;
  console.log('[DIAG]', entry);

  // Persist to localStorage (ring buffer, 150 entries max)
  try {
    const prev: string[] = JSON.parse(localStorage.getItem(DIAG_KEY) || '[]');
    prev.push(entry);
    if (prev.length > 150) prev.splice(0, prev.length - 150);
    localStorage.setItem(DIAG_KEY, JSON.stringify(prev));
  } catch { /* storage full — ignore */ }

  // Update on-screen current-session panel
  const cur = document.getElementById('diag-log-cur');
  if (cur) {
    const div = document.createElement('div');
    div.textContent = entry;
    cur.appendChild(div);
    // Keep last 30 entries visible
    while (cur.children.length > 30) cur.removeChild(cur.firstChild!);
    cur.scrollTop = cur.scrollHeight;
  }
}

function diagInit(): void {
  // Show version on both screens
  for (const id of ['app-version', 'intro-version']) {
    const el = document.getElementById(id);
    if (el) el.textContent = `v${APP_VERSION}`;
  }

  // Load PREVIOUS session's log into the "prev" panel
  try {
    const prev: string[] = JSON.parse(localStorage.getItem(DIAG_PREV_KEY) || '[]');
    const prevPanel = document.getElementById('diag-log-prev');
    if (prevPanel && prev.length > 0) {
      prevPanel.innerHTML = '<em>— previous session —</em><br>';
      // Show last 40 entries
      for (const entry of prev.slice(-40)) {
        const div = document.createElement('div');
        div.textContent = entry;
        prevPanel.appendChild(div);
      }
    }
  } catch { /* ignore */ }

  // Save current session's log as "prev" on unload
  window.addEventListener('pagehide', () => {
    try {
      const cur = localStorage.getItem(DIAG_KEY);
      if (cur) localStorage.setItem(DIAG_PREV_KEY, cur);
    } catch { /* ignore */ }
  });

  // Helper: attach triple-tap diag-panel toggle to an element
  function setupVersionTap(id: string): void {
    const el = document.getElementById(id);
    if (!el) return;
    let tapCount = 0;
    let tapTimer: ReturnType<typeof setTimeout> | null = null;
    el.addEventListener('click', () => {
      tapCount++;
      if (tapTimer) clearTimeout(tapTimer);
      tapTimer = setTimeout(() => { tapCount = 0; }, 600);
      if (tapCount >= 3) {
        tapCount = 0;
        const panel = document.getElementById('diag-panel');
        if (panel) panel.style.display = panel.style.display === 'none' ? 'block' : 'none';
      }
    });
  }

  setupVersionTap('app-version');
  setupVersionTap('intro-version');

  // Share button — opens native share sheet on iOS; falls back to clipboard elsewhere
  document.getElementById('diag-share-btn')?.addEventListener('click', async () => {
    try {
      const prev: string[] = JSON.parse(localStorage.getItem(DIAG_PREV_KEY) || '[]');
      const cur:  string[] = JSON.parse(localStorage.getItem(DIAG_KEY)      || '[]');
      const lines: string[] = [];
      if (prev.length) { lines.push('=== PREVIOUS SESSION ==='); lines.push(...prev); }
      lines.push('=== CURRENT SESSION ===');
      lines.push(...cur);
      const text = lines.join('\n');
      const btn = document.getElementById('diag-share-btn') as HTMLButtonElement;
      if (navigator.share) {
        await navigator.share({ title: 'TwixtBot Diagnostics', text });
        btn.textContent = 'Shared!';
      } else {
        await navigator.clipboard.writeText(text);
        btn.textContent = 'Copied!';
      }
      setTimeout(() => { btn.textContent = 'Share'; }, 2000);
    } catch { /* user dismissed share sheet or clipboard unavailable */ }
  });

  // Clear button
  document.getElementById('diag-clear-btn')?.addEventListener('click', () => {
    try { localStorage.removeItem(DIAG_KEY); localStorage.removeItem(DIAG_PREV_KEY); } catch { /* ignore */ }
    const cur = document.getElementById('diag-log-cur');
    const prev = document.getElementById('diag-log-prev');
    if (cur) cur.innerHTML = '';
    if (prev) prev.innerHTML = '';
  });
}

// -------------------------------------------------------------------------
// JS heartbeat — a setTimeout that reschedules itself every second during
// the human turn. If JS execution is suspended by iOS (backgrounded, or
// some other reason), the next tick fires late.  The gap is logged so we
// can see in the following session's grey section how long JS was frozen.
// -------------------------------------------------------------------------

let heartbeatTimer: ReturnType<typeof setTimeout> | null = null;
let heartbeatLast = 0;

function startHeartbeat(): void {
  heartbeatLast = Date.now();
  function tick() {
    const now = Date.now();
    const gap = now - heartbeatLast;
    heartbeatLast = now;
    // Only log if there was a significant gap (> 1.5s means JS was suspended)
    if (gap > 1500) diagLog(`js-gap ${gap}ms — JS was suspended`);
    heartbeatTimer = setTimeout(tick, 1000);
  }
  heartbeatTimer = setTimeout(tick, 1000);
}

function stopHeartbeat(): void {
  if (heartbeatTimer !== null) {
    clearTimeout(heartbeatTimer);
    heartbeatTimer = null;
  }
}

// -------------------------------------------------------------------------
// DOM references
// -------------------------------------------------------------------------

const $introScreen    = document.getElementById('intro-screen')!;
const $loadingScreen  = document.getElementById('loading-screen')!;
const $gameScreen     = document.getElementById('game-screen')!;
const $statusText     = document.getElementById('status-text')!;
const $thinkingOverlay = document.getElementById('thinking-overlay')!;
const $loadingMsg     = document.getElementById('loading-msg')!;
const $undoBtn          = document.getElementById('undo-btn')!;
const $newGameBtn       = document.getElementById('new-game-btn')!;
const $thinkTimeSelect  = document.getElementById('think-time-select') as HTMLSelectElement;
const boardCanvas     = document.getElementById('board-canvas') as HTMLCanvasElement;

// -------------------------------------------------------------------------
// State
// -------------------------------------------------------------------------

let game  = new Game();
let board: BoardUI;
let worker: Worker;
/** True once the worker has sent 'ready'; false after terminate(). */
let workerAlive = false;
/** Set when the user taps Start; gates whether onWorkerReady() begins the game. */
let userClickedStart = false;
let gameOver = false;
let aiThinking = false;
let aiMoveTimer: ReturnType<typeof setTimeout> | null = null;

// -------------------------------------------------------------------------
// Worker bridge
// -------------------------------------------------------------------------

type MoveMsg = { x: number; y: number } | 'swap';

function initWorker(onReady: () => void = onWorkerReady): void {
  diagLog('worker-init-start');
  workerAlive = false;
  worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });

  worker.onmessage = (e) => {
    const msg = e.data;
    if (msg.type === 'ready') {
      workerAlive = true;
      diagLog('worker-ready');
      onReady();
    } else if (msg.type === 'result') {
      diagLog(`worker-result x=${(msg.move as {x:number,y:number})?.x} y=${(msg.move as {x:number,y:number})?.y}`);
      clearAiMoveTimer();
      onAiMove(msg.move as MoveMsg);
    } else if (msg.type === 'computing-done') {
      // Worker's MCTS fully finished (may be AFTER result was already sent by hard deadline).
      // Tells us the worker was still running computation during the human turn.
      diagLog(`worker-computing-done elapsed=${msg.elapsed}ms`);
    } else if (msg.type === 'error') {
      diagLog(`worker-error: ${msg.message}`);
      clearAiMoveTimer();
      console.error('[Worker]', msg.message);
      $loadingMsg.textContent = `Error: ${msg.message}`;
    } else if (msg.type === 'ping') {
      diagLog(`worker-ping elapsed=${msg.elapsed}ms iters=${msg.iterations}`);
    }
  };

  worker.onerror = (e) => {
    diagLog(`worker-onerror: ${e.message}`);
    console.error('[Worker] crashed:', e.message);
    workerAlive = false;
    if (aiThinking && !gameOver) onAiMoveTimeout();
  };

  $loadingMsg.textContent = 'Loading AI model…';
  worker.postMessage({ type: 'init', modelUrl: MODEL_URL, timeLimitMs: getThinkTimeSec() * 1000 });
}

function clearAiMoveTimer(): void {
  if (aiMoveTimer !== null) {
    diagLog('watchdog-cleared');
    clearTimeout(aiMoveTimer);
    aiMoveTimer = null;
  }
}

function onAiMoveTimeout(): void {
  diagLog('watchdog-fired — worker did not respond; playing fallback move');
  aiMoveTimer = null;
  if (!aiThinking || gameOver) return;
  const legal = game.legalPlays();
  const fallbackMove = legal.at(Math.floor(Math.random() * legal.length))!;
  worker.terminate();
  workerAlive = false;
  diagLog('worker-terminated (by watchdog)');
  initWorker(() => { diagLog('worker-restarted after watchdog'); });
  onAiMove({ x: fallbackMove.x, y: fallbackMove.y });
}

function requestAiMove(): void {
  if (gameOver || aiThinking) return;
  aiThinking = true;
  setThinking(true);
  stopHeartbeat();  // not needed while AI is thinking

  const timeSec = getThinkTimeSec();
  const watchdogMs = (timeSec + 15) * 1000;

  clearAiMoveTimer();
  aiMoveTimer = setTimeout(onAiMoveTimeout, watchdogMs);
  diagLog(`request-ai-move timeSec=${timeSec} watchdog=${watchdogMs}ms`);

  const history: MoveMsg[] = game.history.map(m =>
    m === 'swap' ? 'swap' : { x: (m as {x:number,y:number}).x, y: (m as {x:number,y:number}).y }
  );

  if (workerAlive) {
    worker.postMessage({ type: 'move', history, timeLimitMs: timeSec * 1000 });
  } else {
    // Worker was terminated (after watchdog or mid-move "New Game").
    diagLog('worker-restarting-for-move');
    initWorker(() => {
      diagLog('worker-ready-sending-move');
      worker.postMessage({ type: 'move', history, timeLimitMs: timeSec * 1000 });
    });
  }
}

// -------------------------------------------------------------------------
// Event handlers
// -------------------------------------------------------------------------

function onWorkerReady(): void {
  if (userClickedStart) {
    // User already tapped Start while model was loading — transition now.
    $loadingScreen.classList.add('hidden');
    $gameScreen.classList.remove('hidden');
    startNewGame();
  }
  // else: model is silently ready; the Start button handler will begin the game.
}

function onHumanMove(p: { x: number; y: number }): void {
  if (gameOver || aiThinking || game.turn !== HUMAN_COLOR) return;
  if (!game.legalPlays().contains(p)) return;

  diagLog(`human-move x=${p.x} y=${p.y}`);
  stopHeartbeat();
  game.play(p);
  board.setGame(game, false);

  if (game.justWon()) {
    endGame('You win!');
    return;
  }
  if (game.legalPlays().length === 0) {
    endGame('Draw');
    return;
  }

  requestAiMove();
}

function onAiMove(moveMsg: MoveMsg): void {
  aiThinking = false;
  setThinking(false);
  if (gameOver) return;

  const move: MoveRecord = moveMsg === 'swap' ? 'swap' : pt((moveMsg as {x:number;y:number}).x, (moveMsg as {x:number;y:number}).y);
  game.play(move);
  board.setGame(game, true);

  if (game.justWon()) {
    endGame('AI wins');
    return;
  }
  if (game.legalPlays().length === 0) {
    endGame('Draw');
    return;
  }

  diagLog('human-turn-start');
  // Free WASM heap during human turn — iOS kills the page if ~300MB stays allocated.
  // Worker will be restarted in requestAiMove() when the human makes their move.
  worker.terminate();
  workerAlive = false;
  diagLog('worker-terminated (freeing memory for human turn)');
  $statusText.textContent = 'Your turn (Black)';
  startHeartbeat();  // monitor JS suspension during human turn
}

function onUndoClick(): void {
  if (gameOver || aiThinking) return;
  if (game.history.length >= 2) {
    game.undo();
    game.undo();
    board.setGame(game, true);
    $statusText.textContent = 'Your turn (Black)';
  } else if (game.history.length === 1) {
    game.undo();
    board.setGame(game, true);
    $statusText.textContent = 'Your turn (Black)';
  }
}

function showIntro(result?: string): void {
  diagLog(`show-intro${result ? ': ' + result : ''}`);
  stopHeartbeat();
  clearAiMoveTimer();
  if (aiThinking) {
    try { worker.terminate(); } catch { /* ignore */ }
    workerAlive = false;
    aiThinking = false;
    diagLog('worker-terminated (returning to intro)');
  }
  gameOver = true;
  userClickedStart = false;
  setThinking(false);
  const subtitleEl = document.getElementById('intro-subtitle');
  if (subtitleEl) subtitleEl.textContent = result ?? 'vs AI';
  const startBtn = document.getElementById('start-btn') as HTMLButtonElement | null;
  if (startBtn) startBtn.textContent = result ? 'Play Again' : 'Start Game';
  $gameScreen.classList.add('hidden');
  $loadingScreen.classList.add('hidden');
  $introScreen.classList.remove('hidden');
}

function startNewGame(): void {
  diagLog('new-game-start');
  stopHeartbeat();
  clearAiMoveTimer();
  // Only terminate if AI was mid-move; otherwise reuse the loaded worker.
  if (aiThinking) {
    worker.terminate();
    workerAlive = false;
    diagLog('worker-terminated (new-game cancelled in-flight move)');
  }
  gameOver   = false;
  aiThinking = false;
  game = new Game();
  $thinkingOverlay.classList.add('hidden');
  board.setGame(game, false);
  requestAiMove();
}

function endGame(msg: string): void {
  diagLog(`game-over: ${msg}`);
  showIntro(msg);
}

function setThinking(thinking: boolean): void {
  if (thinking) {
    $statusText.textContent = 'AI is thinking…';
    $thinkingOverlay.classList.remove('hidden');
    board.setEnabled(false);
  } else {
    $thinkingOverlay.classList.add('hidden');
    if (!gameOver) board.setEnabled(true);
  }
}

// -------------------------------------------------------------------------
// Bootstrap
// -------------------------------------------------------------------------

function init(): void {
  diagInit();
  diagLog(`app-start v=${APP_VERSION}`);
  diagLog(`sw-controller: ${navigator.serviceWorker?.controller?.scriptURL ?? 'none'}`);
  diagLog(`ua: ${navigator.userAgent.substring(0, 80)}`);

  // Environment facts that may explain iOS-specific behaviour
  diagLog(`crossOriginIsolated=${self.crossOriginIsolated}`);
  diagLog(`hardwareConcurrency=${navigator.hardwareConcurrency}`);
  // deviceMemory is not available on iOS Safari (will log undefined)
  diagLog(`deviceMemory=${(navigator as unknown as {deviceMemory?: number}).deviceMemory ?? 'n/a'}`);

  // Page lifecycle — hard OS kills won't fire these at all
  window.addEventListener('beforeunload', () => diagLog('page-beforeunload'));
  window.addEventListener('pagehide',     (e) => diagLog(`page-pagehide persisted=${e.persisted}`));
  window.addEventListener('pageshow',     (e) => diagLog(`page-pageshow persisted=${e.persisted}`));

  // Visibility — fires when user backgrounds the app or screen turns off
  document.addEventListener('visibilitychange', () => {
    diagLog(`visibility=${document.visibilityState}`);
  });

  // Unhandled JS errors — might reveal a crash we're not catching elsewhere
  window.addEventListener('error', (e) => {
    diagLog(`js-error: ${e.message} @ ${e.filename}:${e.lineno}`);
  });
  window.addEventListener('unhandledrejection', (e) => {
    diagLog(`unhandled-rejection: ${e.reason}`);
  });

  // SW controller changes — rules out SW as reload cause
  if (navigator.serviceWorker) {
    navigator.serviceWorker.addEventListener('controllerchange', () => {
      diagLog('sw-controllerchange');
    });
  }

  board = new BoardUI(boardCanvas, { onMove: onHumanMove });

  const savedSec = getThinkTimeSec();
  for (const sec of THINK_TIME_OPTIONS) {
    const opt = document.createElement('option');
    opt.value = String(sec);
    opt.textContent = `${sec}s`;
    if (sec === savedSec) opt.selected = true;
    $thinkTimeSelect.appendChild(opt);
  }
  $thinkTimeSelect.addEventListener('change', () => {
    localStorage.setItem(THINK_TIME_KEY, $thinkTimeSelect.value);
  });

  $undoBtn.addEventListener('click',    onUndoClick);
  $newGameBtn.addEventListener('click', () => showIntro());

  // Start button: hide intro, begin game (or show loading if model not ready yet).
  document.getElementById('start-btn')?.addEventListener('click', () => {
    userClickedStart = true;
    $introScreen.classList.add('hidden');
    if (workerAlive) {
      $gameScreen.classList.remove('hidden');
      startNewGame();
    } else {
      // Model still loading in background — show loading screen until onWorkerReady fires.
      $loadingScreen.classList.remove('hidden');
    }
  });

  initWorker();  // begin loading model silently while intro screen is shown (userClickedStart=false)
}

init();
