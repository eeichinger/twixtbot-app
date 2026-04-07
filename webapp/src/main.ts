/**
 * main.ts — App shell and game loop.
 *
 * Manages:
 *   - Worker lifecycle (init, move requests) — PvC mode only
 *   - Game state (human is BLACK, AI is WHITE in PvC; both human in PvP)
 *   - UI state (loading, thinking, game-over)
 *   - Game mode selection (PvC vs PvP)
 */

import { Game, pt, WHITE, BLACK } from './twixt.js';
import type { MoveRecord } from './twixt.js';
import { BoardUI } from './ui.js';
import {
  loadGameMode, saveGameMode,
  isHumanTurn, turnStatusText, resultMessage,
  type GameMode,
} from './game-mode.js';
import { fetchGame, fetchPlayerGamesByPlid, searchPlayers, type GameSummary, type PlayerResult } from './lg-api.js';
import { parseTSGF, serializeTSGF, formatResult, type ParsedGame } from './lg-sgf.js';

// -------------------------------------------------------------------------
// Version — update this string with every deploy to confirm new code loaded
// -------------------------------------------------------------------------

const APP_VERSION = '2026-04-07-c';

// -------------------------------------------------------------------------
// Constants
// -------------------------------------------------------------------------

const MODEL_URL   = import.meta.env.BASE_URL + 'model.onnx';

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

const $introScreen     = document.getElementById('intro-screen')!;
const $loadingScreen   = document.getElementById('loading-screen')!;
const $gameScreen      = document.getElementById('game-screen')!;
const $statusText      = document.getElementById('status-text')!;
const $thinkingOverlay = document.getElementById('thinking-overlay')!;
const $loadingMsg      = document.getElementById('loading-msg')!;
const $hintBtn         = document.getElementById('hint-btn')!;
const $swapBtn         = document.getElementById('swap-btn')!;
const $undoBtn         = document.getElementById('undo-btn')!;
const $newGameBtn      = document.getElementById('new-game-btn')!;
const $exportBtn       = document.getElementById('export-btn')!;
const $thinkTimeSelect = document.getElementById('think-time-select') as HTMLSelectElement;
const boardCanvas      = document.getElementById('board-canvas') as HTMLCanvasElement;

// LG Explore screen
const $lgScreen        = document.getElementById('lg-screen')!;
const $lgSearchInput   = document.getElementById('lg-search-input') as HTMLInputElement;
const $lgSearchBtn     = document.getElementById('lg-search-btn')!;
const $lgEmptyMsg      = document.getElementById('lg-empty-msg')!;
const $lgLoadingMsg    = document.getElementById('lg-loading-msg')!;
const $lgErrorMsg      = document.getElementById('lg-error-msg')!;
const $lgResults       = document.getElementById('lg-results')!;

// Replay viewer screen
const $replayScreen    = document.getElementById('replay-screen')!;
const $replayTitle     = document.getElementById('replay-title')!;
const $replayCounter   = document.getElementById('replay-counter')!;
const $replayFirstBtn  = document.getElementById('replay-first-btn') as HTMLButtonElement;
const $replayPrevBtn   = document.getElementById('replay-prev-btn')  as HTMLButtonElement;
const $replayNextBtn   = document.getElementById('replay-next-btn')  as HTMLButtonElement;
const $replayLastBtn   = document.getElementById('replay-last-btn')  as HTMLButtonElement;
const replayCanvas     = document.getElementById('replay-canvas') as HTMLCanvasElement;
const $lgPasteInput    = document.getElementById('lg-paste-input') as HTMLTextAreaElement;
const $lgFileInput     = document.getElementById('lg-file-input') as HTMLInputElement;

// -------------------------------------------------------------------------
// State
// -------------------------------------------------------------------------

let game  = new Game();
let board: BoardUI;
let worker: Worker;
/** True once the worker has sent 'ready'; false after terminate(). */
let workerAlive = false;
/** True while initWorker() has been called but 'ready' not yet received. */
let workerLoading = false;
/** Set when the user taps Start; gates whether onWorkerReady() begins the game. */
let userClickedStart = false;
let gameOver = false;
let aiThinking = false;
/** TSGF result for the current game: 'B+' | 'W+' | '0' | '?'. WHITE is first mover = TSGF Black. */
let tsgfResult = '?';
let aiMoveTimer: ReturnType<typeof setTimeout> | null = null;
let gameMode: GameMode = loadGameMode();

// -------------------------------------------------------------------------
// UI helpers
// -------------------------------------------------------------------------

/** Show/hide the think-time selector: always in PvC; in PvP only when the AI-move button is live. */
function syncThinkTimeVisibility(): void {
  const show = gameMode === 'pvc' ||
    (!gameOver && !aiThinking && isHumanTurn(game.turn, gameMode));
  $thinkTimeSelect.classList.toggle('hidden', !show);
}

/** Show the "AI move" button only when a human can meaningfully delegate their turn. */
function syncHintButton(): void {
  const show = !gameOver && !aiThinking && isHumanTurn(game.turn, gameMode);
  $hintBtn.classList.toggle('hidden', !show);
}

// -------------------------------------------------------------------------
// Worker bridge (PvC only)
// -------------------------------------------------------------------------

type MoveMsg = { x: number; y: number } | 'swap';

function initWorker(onReady: () => void = onWorkerReady): void {
  diagLog('worker-init-start');
  workerAlive = false;
  workerLoading = true;
  worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });

  worker.onmessage = (e) => {
    const msg = e.data;
    if (msg.type === 'ready') {
      workerAlive = true;
      workerLoading = false;
      diagLog('worker-ready');
      onReady();
    } else if (msg.type === 'result') {
      const moveStr = msg.move === 'swap' ? 'swap' : `x=${(msg.move as {x:number,y:number}).x} y=${(msg.move as {x:number,y:number}).y}`;
      diagLog(`worker-result ${moveStr}`);
      clearAiMoveTimer();
      onAiMove(msg.move as MoveMsg);
    } else if (msg.type === 'computing-done') {
      diagLog(`worker-computing-done elapsed=${msg.elapsed}ms`);
    } else if (msg.type === 'error') {
      diagLog(`worker-error: ${msg.message}`);
      clearAiMoveTimer();
      console.error('[Worker]', msg.message);
      if (!workerAlive) {
        // Model failed to load (e.g. offline and not cached). Return to intro after a delay.
        workerLoading = false;
        const isOffline = !navigator.onLine;
        $loadingMsg.textContent = isOffline
          ? 'Offline — AI model not cached yet. Connect once to enable offline play.'
          : `Failed to load AI: ${msg.message}`;
        setTimeout(() => showIntro(), 3000);
      } else {
        $loadingMsg.textContent = `Error: ${msg.message}`;
      }
    } else if (msg.type === 'ping') {
      diagLog(`worker-ping elapsed=${msg.elapsed}ms iters=${msg.iterations}`);
    }
  };

  worker.onerror = (e) => {
    diagLog(`worker-onerror: ${e.message}`);
    console.error('[Worker] crashed:', e.message);
    workerAlive = false;
    workerLoading = false;
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

/** Show the Swap button only when the human can legally swap (move 2, human's turn). */
function updateSwapBtn(): void {
  const canSwap = !gameOver && !aiThinking &&
    isHumanTurn(game.turn, gameMode) && game.history.length === 1;
  $swapBtn.classList.toggle('hidden', !canSwap);
}

function onHumanMove(p: { x: number; y: number }): void {
  if (gameOver || aiThinking) return;
  if (!isHumanTurn(game.turn, gameMode)) return;
  if (!game.legalPlays().contains(p)) return;

  diagLog(`human-move x=${p.x} y=${p.y}`);
  stopHeartbeat();
  game.play(p);
  updateSwapBtn();
  board.setGame(game, false);

  if (game.justWon()) {
    // The player who just moved won: turn has already switched to the next player,
    // so the winner is the opposite of game.turn.
    const winner = game.turn === WHITE ? BLACK : WHITE;
    tsgfResult = winner === WHITE ? 'B+' : 'W+';  // WHITE = first mover = TSGF Black
    endGame(resultMessage(winner, gameMode));
    return;
  }
  if (game.legalPlays().length === 0) {
    tsgfResult = '0';
    endGame(resultMessage(null, gameMode));
    return;
  }

  if (gameMode === 'pvp') {
    // Other human player takes over.
    board.setEnabled(true);
    $statusText.textContent = turnStatusText(game.turn, gameMode);
    syncHintButton();
    syncThinkTimeVisibility();
    startHeartbeat();
  } else {
    syncHintButton();
    syncThinkTimeVisibility();
    requestAiMove();
  }
}

function onHumanSwap(): void {
  if (gameOver || aiThinking || !isHumanTurn(game.turn, gameMode) || game.history.length !== 1) return;

  diagLog('human-swap');
  stopHeartbeat();
  game.play('swap');
  updateSwapBtn();
  board.setGame(game, false);

  if (gameMode === 'pvp') {
    board.setEnabled(true);
    $statusText.textContent = turnStatusText(game.turn, gameMode);
    syncHintButton();
    syncThinkTimeVisibility();
    startHeartbeat();
  } else {
    syncHintButton();
    syncThinkTimeVisibility();
    requestAiMove();
  }
}

function onAiMove(moveMsg: MoveMsg): void {
  aiThinking = false;
  setThinking(false);
  if (gameOver) return;

  const move: MoveRecord = moveMsg === 'swap' ? 'swap' : pt((moveMsg as {x:number;y:number}).x, (moveMsg as {x:number;y:number}).y);
  // Remember who just moved before game.play() advances game.turn.
  const movedColor = game.turn;
  game.play(move);

  if (game.justWon()) {
    board.setGame(game, false);
    tsgfResult = movedColor === WHITE ? 'B+' : 'W+';  // WHITE = first mover = TSGF Black
    endGame(resultMessage(movedColor, gameMode));
    return;
  }
  if (game.legalPlays().length === 0) {
    board.setGame(game, false);
    tsgfResult = '0';
    endGame(resultMessage(null, gameMode));
    return;
  }

  if (isHumanTurn(game.turn, gameMode)) {
    // Human's turn next — hand control back.
    board.setGame(game, true);
    diagLog('human-turn-start');
    // Free WASM heap during human turn — iOS kills the page if ~300MB stays allocated.
    // Worker will be restarted in requestAiMove() when the human makes their next move.
    worker.terminate();
    workerAlive = false;
    diagLog('worker-terminated (freeing memory for human turn)');
    $statusText.textContent = turnStatusText(game.turn, gameMode);
    updateSwapBtn();
    syncHintButton();
    syncThinkTimeVisibility();
    startHeartbeat();  // monitor JS suspension during human turn
  } else {
    // AI's turn again (e.g. human used "AI move" and it's still AI's turn in PvC).
    board.setGame(game, false);
    requestAiMove();
  }
}

function onHintClick(): void {
  if (gameOver || aiThinking || !isHumanTurn(game.turn, gameMode)) return;
  diagLog(`hint-requested turn=${game.turn}`);
  requestAiMove();
}

function onUndoClick(): void {
  if (gameOver || aiThinking) return;

  if (gameMode === 'pvp') {
    // PvP: undo exactly 1 move (the last player's move).
    if (game.history.length >= 1) {
      game.undo();
      board.setGame(game, true);
      $statusText.textContent = turnStatusText(game.turn, gameMode);
      updateSwapBtn();
      syncHintButton();
      syncThinkTimeVisibility();
    }
  } else {
    // PvC: undo the human move + the preceding AI move together, so it's
    // always the human's turn after undo.
    if (game.history.length >= 2) {
      game.undo();
      game.undo();
      board.setGame(game, true);
      $statusText.textContent = turnStatusText(BLACK, gameMode);
      updateSwapBtn();
    } else if (game.history.length === 1) {
      game.undo();
      board.setGame(game, true);
      $statusText.textContent = turnStatusText(BLACK, gameMode);
      updateSwapBtn();
    }
    syncHintButton();
    syncThinkTimeVisibility();
  }
}

function onExportClick(): void {
  if (game.history.length === 0) return;  // nothing to export yet

  // In the webapp WHITE is the first mover; TSGF labels the first mover "B" (Black).
  // So webapp WHITE → TSGF PB, webapp BLACK → TSGF PW.
  let pbName: string, pwName: string;
  if (gameMode === 'pvc') {
    pbName = 'TwixtBot';  // WHITE = AI = first mover
    pwName = 'Human';     // BLACK = human = second mover
  } else {
    pbName = 'Orange';    // WHITE = first mover
    pwName = 'Blue';      // BLACK = second mover
  }

  const text = serializeTSGF(game.history, {
    blackPlayer: pbName,
    whitePlayer: pwName,
    result: tsgfResult,
  });

  const now = new Date();
  const pad = (n: number) => String(n).padStart(2, '0');
  const ts = `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}` +
             `${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`;
  const filename = `twixt${ts}.tsgf`;

  const blob = new Blob([text], { type: 'application/octet-stream' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href     = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);

  diagLog(`game-exported ${filename} moves=${game.history.length}`);
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
  const startBtn = document.getElementById('start-btn') as HTMLButtonElement | null;
  if (startBtn) startBtn.textContent = result ? 'Play Again' : 'Start Game';
  hideAllScreens();
  $introScreen.classList.remove('hidden');
}

function startNewGame(): void {
  diagLog(`new-game-start mode=${gameMode}`);
  stopHeartbeat();
  clearAiMoveTimer();
  // Only terminate if AI was mid-move; otherwise reuse the loaded worker.
  if (aiThinking) {
    worker.terminate();
    workerAlive = false;
    diagLog('worker-terminated (new-game cancelled in-flight move)');
  }
  gameOver    = false;
  aiThinking  = false;
  tsgfResult  = '?';
  game = new Game();
  $thinkingOverlay.classList.add('hidden');
  $swapBtn.classList.add('hidden');
  $undoBtn.classList.remove('hidden');
  board.setGame(game, false);
  syncThinkTimeVisibility();

  syncHintButton();
  syncThinkTimeVisibility();
  if (gameMode === 'pvc') {
    requestAiMove();
  } else {
    // PvP: WHITE (first mover) takes the first turn.
    board.setEnabled(true);
    $statusText.textContent = turnStatusText(game.turn, gameMode);
    syncHintButton();
    syncThinkTimeVisibility();
    startHeartbeat();
  }
}

function endGame(msg: string): void {
  diagLog(`game-over: ${msg}`);
  gameOver = true;
  userClickedStart = false;
  stopHeartbeat();
  clearAiMoveTimer();
  if (aiThinking) {
    try { worker.terminate(); } catch { /* ignore */ }
    workerAlive = false;
    aiThinking = false;
    diagLog('worker-terminated (game over)');
  }
  setThinking(false);
  $statusText.textContent = msg;
  board.setEnabled(false);
  $hintBtn.classList.add('hidden');
  $swapBtn.classList.add('hidden');
  $undoBtn.classList.add('hidden');
  $thinkTimeSelect.classList.add('hidden');
}

function setThinking(thinking: boolean): void {
  if (thinking) {
    $statusText.textContent = 'AI is thinking…';
    $thinkingOverlay.classList.remove('hidden');
    board.setEnabled(false);
    $hintBtn.classList.add('hidden');
    $thinkTimeSelect.classList.remove('hidden'); // keep visible so user can adjust mid-think
  } else {
    $thinkingOverlay.classList.add('hidden');
    if (!gameOver) board.setEnabled(true);
    // syncHintButton() / syncThinkTimeVisibility() called by whoever transitions out of thinking.
  }
}

// -------------------------------------------------------------------------
// Little Golem — Explore & Replay
// -------------------------------------------------------------------------

let replayBoardUI: BoardUI | null = null;
let replayParsedGame: ParsedGame | null = null;
let replayMoveIndex = 0;

/** Cached player search results for instant back-navigation without re-fetch. */
let lastPlayerResults: PlayerResult[] = [];

/** All screens that must be hidden when switching between them. */
function hideAllScreens(): void {
  $introScreen.classList.add('hidden');
  $loadingScreen.classList.add('hidden');
  $gameScreen.classList.add('hidden');
  $lgScreen.classList.add('hidden');
  $replayScreen.classList.add('hidden');
}

function showLgScreen(): void {
  diagLog('lg-screen-open');
  hideAllScreens();
  $lgScreen.classList.remove('hidden');
}

/** Set the LG explore UI into one of four status states. */
function lgSetState(state: 'empty' | 'loading' | 'error' | 'results', errorMsg?: string): void {
  $lgEmptyMsg.classList.toggle('hidden',   state !== 'empty');
  $lgLoadingMsg.classList.toggle('hidden', state !== 'loading');
  $lgErrorMsg.classList.toggle('hidden',   state !== 'error');
  $lgResults.classList.toggle('hidden',    state !== 'results');
  if (state === 'error' && errorMsg) $lgErrorMsg.textContent = errorMsg;
}

// ---- Shared card builder ------------------------------------------------

function makeLgCard(line1: string, line2: string, onClick: () => void): HTMLDivElement {
  const card = document.createElement('div');
  card.className = 'lg-game-card';
  const el1 = document.createElement('div');
  el1.className = 'lg-card-players';
  el1.textContent = line1;
  const el2 = document.createElement('div');
  el2.className = 'lg-card-meta';
  el2.textContent = line2;
  card.appendChild(el1);
  card.appendChild(el2);
  card.addEventListener('click', onClick);
  return card;
}

function lgHandleError(diagKey: string, err: unknown): void {
  const msg = err instanceof Error ? err.message : String(err);
  diagLog(`${diagKey}: ${msg}`);
  lgSetState('error', `Error: ${msg}`);
}

// ---- Player result cards ------------------------------------------------

function renderPlayerResults(players: PlayerResult[]): void {
  $lgResults.innerHTML = '';
  if (players.length === 0) {
    lgSetState('error', 'No TwixT players found with that name.');
    return;
  }
  lgSetState('results');
  for (const p of players) {
    const meta = p.rating ? `Rating: ${p.rating}  ·  id ${p.plid}` : `id ${p.plid}`;
    $lgResults.appendChild(makeLgCard(p.name, meta, () => openPlayerGames(p)));
  }
}

// ---- Game result cards --------------------------------------------------

function renderGameResults(games: GameSummary[]): void {
  $lgResults.innerHTML = '';
  if (games.length === 0) {
    lgSetState('error', 'No finished TwixT PP games found for this player.');
    return;
  }
  lgSetState('results');

  // "← Players" back link restores cached player search results instantly
  const backEl = document.createElement('div');
  backEl.className = 'lg-card-back';
  backEl.textContent = '← Players';
  backEl.addEventListener('click', () => renderPlayerResults(lastPlayerResults));
  $lgResults.appendChild(backEl);

  for (const g of games) {
    const line1 = g.opponent ? `vs ${g.opponent}` : `${g.blackPlayer} (B) vs ${g.whitePlayer} (W)`;
    const movePart = g.moveCount > 0 ? `${g.moveCount} moves  ·  ` : '';
    const line2 = `#${g.id}  ·  ${movePart}${formatResult(g.result)}`;
    $lgResults.appendChild(makeLgCard(line1, line2, () => openReplayById(g.id)));
  }
}

// ---- Search & navigation ------------------------------------------------

async function performLgSearch(): Promise<void> {
  const query = $lgSearchInput.value.trim();
  if (!query) return;

  lgSetState('loading');
  $lgResults.innerHTML = '';

  try {
    if (/^\d+$/.test(query)) {
      diagLog(`lg-fetch-game id=${query}`);
      const parsed = await fetchGame(query);
      openReplayForParsedGame(parsed);
    } else {
      diagLog(`lg-search-players name=${query}`);
      const players = await searchPlayers(query);
      lastPlayerResults = players;
      renderPlayerResults(players);
    }
  } catch (err) {
    lgHandleError('lg-search-error', err);
  }
}

async function openPlayerGames(player: PlayerResult): Promise<void> {
  lgSetState('loading');
  $lgResults.innerHTML = '';
  try {
    diagLog(`lg-fetch-player-games plid=${player.plid}`);
    const games = await fetchPlayerGamesByPlid(player.plid);
    renderGameResults(games);
  } catch (err) {
    lgHandleError('lg-player-games-error', err);
  }
}

async function openReplayById(id: string): Promise<void> {
  lgSetState('loading');
  try {
    diagLog(`lg-fetch-game id=${id}`);
    const parsed = await fetchGame(id);
    openReplayForParsedGame(parsed);
  } catch (err) {
    lgHandleError('lg-open-replay-error', err);
    // Restore the results panel so the user can try another game
    if ($lgResults.children.length > 0) lgSetState('results');
  }
}

function openReplayFromText(text: string): void {
  const trimmed = text.trim();
  if (!trimmed) return;
  try {
    const parsed = parseTSGF(trimmed, 'custom');
    if (parsed.moves.length === 0) {
      lgSetState('error', 'No moves found — is this a valid .tsgf file?');
      return;
    }
    openReplayForParsedGame(parsed);
  } catch (err) {
    lgHandleError('lg-paste-error', err);
  }
}

function openReplayForParsedGame(parsed: ParsedGame): void {
  replayParsedGame = parsed;
  replayMoveIndex = parsed.moves.length; // start at final position

  // Lazy-init the replay BoardUI
  if (!replayBoardUI) {
    replayBoardUI = new BoardUI(replayCanvas, { onMove: () => {} });
  }

  const title = `${parsed.blackPlayer} vs ${parsed.whitePlayer}  #${parsed.id}`;
  $replayTitle.textContent = title;
  diagLog(`lg-replay-open id=${parsed.id} moves=${parsed.moves.length}`);

  hideAllScreens();
  $replayScreen.classList.remove('hidden');

  replayShowAtIndex(replayMoveIndex);
}

function replayShowAtIndex(index: number): void {
  if (!replayParsedGame || !replayBoardUI) return;
  const total = replayParsedGame.moves.length;
  replayMoveIndex = Math.max(0, Math.min(index, total));

  // Replay moves onto a fresh game
  const g = new Game();
  for (let i = 0; i < replayMoveIndex; i++) {
    g.play(replayParsedGame.moves[i]);
  }
  replayBoardUI.setGame(g, false); // read-only

  $replayCounter.textContent = `Move ${replayMoveIndex} / ${total}`;
  $replayFirstBtn.disabled = replayMoveIndex === 0;
  $replayPrevBtn.disabled  = replayMoveIndex === 0;
  $replayNextBtn.disabled  = replayMoveIndex === total;
  $replayLastBtn.disabled  = replayMoveIndex === total;
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

  // SW controller changes: new SW has activated (skipWaiting fired).
  // Reload to pick up the new assets — but only when no game is in progress
  // (userClickedStart is true only while a game is active).
  if (navigator.serviceWorker) {
    navigator.serviceWorker.addEventListener('controllerchange', () => {
      diagLog('sw-controllerchange');
      if (!userClickedStart) window.location.reload();
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

  $hintBtn.addEventListener('click',    onHintClick);
  $undoBtn.addEventListener('click',    onUndoClick);
  $swapBtn.addEventListener('click',    onHumanSwap);
  $newGameBtn.addEventListener('click', () => showIntro());
  $exportBtn.addEventListener('click',  onExportClick);

  // LG Explore button on intro screen
  document.getElementById('lg-explore-btn')?.addEventListener('click', () => {
    showLgScreen();
    lgSetState('empty');
    $lgResults.innerHTML = '';
    $lgSearchInput.value = '';
  });

  // LG screen — back, search
  document.getElementById('lg-back-btn')?.addEventListener('click', () => showIntro());
  $lgSearchBtn.addEventListener('click', () => performLgSearch());
  $lgSearchInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') performLgSearch();
  });

  // LG screen — paste / upload
  document.getElementById('lg-paste-replay-btn')?.addEventListener('click', () => {
    openReplayFromText($lgPasteInput.value);
  });
  $lgFileInput.addEventListener('change', () => {
    const file = $lgFileInput.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      $lgPasteInput.value = reader.result as string;
      openReplayFromText(reader.result as string);
    };
    reader.readAsText(file);
    $lgFileInput.value = ''; // reset so the same file can be re-selected
  });

  // Replay screen — back and step controls
  document.getElementById('replay-back-btn')?.addEventListener('click', () => showLgScreen());
  $replayFirstBtn.addEventListener('click', () => replayShowAtIndex(0));
  $replayPrevBtn.addEventListener('click',  () => replayShowAtIndex(replayMoveIndex - 1));
  $replayNextBtn.addEventListener('click',  () => replayShowAtIndex(replayMoveIndex + 1));
  $replayLastBtn.addEventListener('click',  () => replayShowAtIndex(replayParsedGame?.moves.length ?? 0));

  // Keyboard navigation in replay (arrow keys)
  document.addEventListener('keydown', (e) => {
    if ($replayScreen.classList.contains('hidden')) return;
    if (e.key === 'ArrowLeft'  || e.key === 'ArrowUp')   replayShowAtIndex(replayMoveIndex - 1);
    if (e.key === 'ArrowRight' || e.key === 'ArrowDown')  replayShowAtIndex(replayMoveIndex + 1);
    if (e.key === 'Home') replayShowAtIndex(0);
    if (e.key === 'End')  replayShowAtIndex(replayParsedGame?.moves.length ?? 0);
  });

  // Mode selector buttons on intro screen
  document.querySelectorAll<HTMLButtonElement>('.mode-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      gameMode = btn.dataset.mode as GameMode;
      saveGameMode(gameMode);
      document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      syncThinkTimeVisibility();
      // Pre-load AI model when user switches to PvC (if not already loading/loaded).
      if (gameMode === 'pvc' && !workerAlive) initWorker();
    });
  });

  // Restore persisted mode selection visually
  document.querySelector<HTMLButtonElement>(`.mode-btn[data-mode="${gameMode}"]`)
    ?.classList.add('active');

  // Start button: hide intro, begin game.
  document.getElementById('start-btn')?.addEventListener('click', () => {
    userClickedStart = true;
    $introScreen.classList.add('hidden');

    if (gameMode === 'pvp') {
      // PvP: no worker needed — go directly to game screen.
      $gameScreen.classList.remove('hidden');
      startNewGame();
      return;
    }

    // PvC: worker must be ready (or loading).
    if (workerAlive) {
      $gameScreen.classList.remove('hidden');
      startNewGame();
    } else {
      // Show loading screen; start loading the worker if it isn't already.
      $loadingScreen.classList.remove('hidden');
      if (!workerLoading) initWorker();
    }
  });

  diagLog(`game-mode=${gameMode}`);
  if (gameMode === 'pvc') {
    // Pre-load AI model in PvC mode while intro screen is shown.
    initWorker();
  } else {
    // In PvP mode the worker isn't started, but we still warm the SW model cache
    // in the background so it's available if the user switches to PvC while offline.
    fetch(MODEL_URL).catch(() => {});
  }
}

init();
