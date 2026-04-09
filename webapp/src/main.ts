/**
 * main.ts — App shell and game loop.
 *
 * Manages:
 *   - Worker lifecycle (init, move requests) — PvC mode only
 *   - Game state (human is BLACK, AI is WHITE in PvC; both human in PvP)
 *   - UI state (loading, thinking, game-over)
 *   - Game mode selection (PvC vs PvP)
 */

import { Game, pt, WHITE, BLACK, ptToString } from './twixt.js';
import type { MoveRecord, Point } from './twixt.js';
import { BoardUI } from './ui.js';
import {
  loadGameMode, saveGameMode,
  isHumanTurn, turnStatusText, resultMessage, resignTsgfResult, winProbBarStyle, formatWinProb,
  type GameMode,
} from './game-mode.js';
import type { Top3Move } from './naf.js';
import { fetchGame, fetchGameRaw, fetchPlayerGamesByPlid, searchPlayers, filterGameSummaries, type GameSummary, type PlayerResult, type ResultFilter } from './lg-api.js';
import { parseTSGF, serializeTSGF, formatResult, type ParsedGame } from './lg-sgf.js';

// -------------------------------------------------------------------------
// Version — update this string with every deploy to confirm new code loaded
// -------------------------------------------------------------------------

const APP_VERSION = '2026-04-09-f';

// -------------------------------------------------------------------------
// Constants
// -------------------------------------------------------------------------

const MODEL_URL   = import.meta.env.BASE_URL + 'model.onnx';

const THINK_TIME_OPTIONS = [2, 5, 10, 15, 25, 30, 45, 60];  // seconds
const THINK_TIME_KEY     = 'twixt-think-time-sec';
const DEFAULT_THINK_TIME = 5;

function getThinkTimeSec(): number {
  const stored = parseInt(localStorage.getItem(THINK_TIME_KEY) ?? '', 10);
  return THINK_TIME_OPTIONS.includes(stored) ? stored : DEFAULT_THINK_TIME;
}

interface BotStrengthOption {
  id: string;
  label: string;
  maxTrials: number;
  temperature: number;
}

const BOT_STRENGTH_OPTIONS: BotStrengthOption[] = [
  { id: 'beginner', label: 'Beginner', maxTrials:     50, temperature: 2.0 },
  { id: 'club',     label: 'Club',     maxTrials:    500, temperature: 0.5 },
  { id: 'master',   label: 'Master',   maxTrials: 100_000, temperature: 0   },
];
const BOT_STRENGTH_KEY     = 'twixt-bot-strength';
const DEFAULT_BOT_STRENGTH = 'beginner';

function getBotStrength(): BotStrengthOption {
  const stored = localStorage.getItem(BOT_STRENGTH_KEY);
  return BOT_STRENGTH_OPTIONS.find(o => o.id === stored)
      ?? BOT_STRENGTH_OPTIONS.find(o => o.id === DEFAULT_BOT_STRENGTH)!;
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
  // Show version on all screens
  for (const id of ['app-version', 'intro-version', 'lg-version', 'replay-version']) {
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
  setupVersionTap('lg-version');
  setupVersionTap('replay-version');

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

  // Close button — always reachable even when the overlay covers the version label
  document.getElementById('diag-close-btn')?.addEventListener('click', () => {
    const panel = document.getElementById('diag-panel');
    if (panel) panel.style.display = 'none';
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
const $thinkingOverlay  = document.getElementById('thinking-overlay')!;
const $thinkingProgress = document.getElementById('thinking-progress')!;
const $loadingMsg      = document.getElementById('loading-msg')!;
const $hintBtn         = document.getElementById('hint-btn')!;
const $swapBtn         = document.getElementById('swap-btn')!;
const $undoBtn         = document.getElementById('undo-btn')!;
const $resignBtn       = document.getElementById('resign-btn')!;
const $heatmapBtn      = document.getElementById('heatmap-btn') as HTMLButtonElement;
const $winProbBar      = document.getElementById('win-prob-bar')!;
const $analysisPanel   = document.getElementById('analysis-panel')!;
const $analysisToggle  = document.getElementById('analysis-toggle')!;
const $winProbText     = document.getElementById('win-prob-text')!;
const $top3Bars        = document.getElementById('top3-bars')!;
const $evalSparkline   = document.getElementById('eval-sparkline') as HTMLCanvasElement;
const $newGameBtn      = document.getElementById('new-game-btn')!;
const $exportBtn       = document.getElementById('export-btn')!;
const $redoBtn         = document.getElementById('redo-btn') as HTMLButtonElement;
const $settingsBtn     = document.getElementById('settings-btn')!;
const $settingsPanel   = document.getElementById('settings-panel')!;
const $settingsBackdrop = document.getElementById('settings-backdrop')!;
const $thinkTimeSelect = document.getElementById('think-time-select') as HTMLSelectElement;
const $strengthSelect  = document.getElementById('strength-select')  as HTMLSelectElement;
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

// Move list panels
const $moveListPanel   = document.getElementById('move-list-panel')!;
const $moveListLabel   = document.getElementById('move-list-label')!;
const $moveListBody    = document.getElementById('move-list-body')!;
const $replayMoveListPanel       = document.getElementById('replay-move-list-panel')!;
const $replayMoveListLabel       = document.getElementById('replay-move-list-label')!;
const $replayMoveListBody        = document.getElementById('replay-move-list-body')!;
const $replayAnalyseGameBtn      = document.getElementById('replay-analyse-game-btn') as HTMLButtonElement;
const $replayGameSparklineWrap   = document.getElementById('replay-game-sparkline-wrap')!;
const $replayGameSparkline       = document.getElementById('replay-game-sparkline') as HTMLCanvasElement;

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

/** Per-move AI win-probability history (topQ after each AI move) for the sparkline. */
let evalHistory: number[] = [];
/** Top-3 move data from the most recent AI result, applied in onAiMove(). */
let pendingAnalysis: { topQ: number; top3: Top3Move[] } | null = null;

/** V5: Heatmap state — true while the heatmap overlay is shown. */
let heatmapActive  = false;
/** True while a heatmap eval-position request is in-flight. */
let pendingHeatmap = false;
/** Routes the eval-position-result to the correct board ('game' or 'replay'). */
let heatmapContext: 'game' | 'replay' = 'game';

/** Replay-screen heatmap state (independent of game-screen heatmap). */
let replayHeatmapActive  = false;
let replayHeatmapPending = false;

// -------------------------------------------------------------------------
// UI helpers
// -------------------------------------------------------------------------

/** Show/hide the settings gear button. Visible during active play in both modes. */
function syncThinkTimeVisibility(): void {
  $settingsBtn.classList.toggle('hidden', gameOver);
}

/** Enable/disable the Redo button based on game state. */
function syncRedoButton(): void {
  $redoBtn.disabled = gameOver || aiThinking || !game.canRedo;
}

/** Show the "AI move" button only when a human can meaningfully delegate their turn. */
function syncHintButton(): void {
  const show = !gameOver && !aiThinking && isHumanTurn(game.turn, gameMode);
  $hintBtn.classList.toggle('hidden', !show);
}

/** Show the resign button once at least one move has been played and the game is active. */
function syncResignButton(): void {
  const show = !gameOver && !aiThinking && game.history.length > 0;
  $resignBtn.classList.toggle('hidden', !show);
}

/** Show the heatmap button whenever the game is active and the AI is not mid-move. */
function syncHeatmapButton(): void {
  const show = !gameOver && !aiThinking;
  $heatmapBtn.classList.toggle('hidden', !show);
  $heatmapBtn.disabled = pendingHeatmap;
}

/** Clear the heatmap overlay and reset active state (called on move / undo / redo). */
function clearHeatmap(): void {
  if (heatmapActive) {
    heatmapActive = false;
    $heatmapBtn.classList.remove('active');
    board.setHeatmap(null, 0);
  }
}

function clearReplayHeatmap(): void {
  replayHeatmapActive = false;
  replayHeatmapPending = false;
  $replayHeatmapBtn.classList.remove('active');
  $replayHeatmapBtn.disabled = false;
  replayBoardUI?.setHeatmap(null, 0);
}

// -------------------------------------------------------------------------
// Move list (U1 / L4) — shared rendering
// -------------------------------------------------------------------------

function moveToStr(m: MoveRecord): string {
  return m === 'swap' ? 'swap' : ptToString(m as Point);
}

/**
 * Render a numbered move list into `container`.
 * Moves are paired into rounds: WHITE (even indices) left, BLACK (odd) right.
 * `currentIndex` is the 0-based half-move index to highlight; -1 = none.
 * If `onHalfMoveClick` is provided, moves become clickable links.
 * The highlighted row is scrolled into view after rendering.
 */
function renderMoveList(
  container: HTMLElement,
  moves: MoveRecord[],
  currentIndex: number,
  onHalfMoveClick?: (index: number) => void,
): void {
  container.innerHTML = '';
  const rounds = Math.ceil(moves.length / 2);
  let highlightedRow: HTMLElement | null = null;
  for (let r = 0; r < rounds; r++) {
    const row = document.createElement('div');
    row.className = 'ml-row';

    const num = document.createElement('span');
    num.className = 'ml-num';
    num.textContent = `${r + 1}.`;
    row.appendChild(num);

    for (let side = 0; side < 2; side++) {
      const halfIdx = 2 * r + side;
      if (halfIdx >= moves.length) break;
      const span = document.createElement('span');
      span.className = 'ml-move' +
        (halfIdx === currentIndex ? ' ml-current' : '') +
        (onHalfMoveClick ? ' ml-clickable' : '');
      span.textContent = moveToStr(moves[halfIdx]);
      if (onHalfMoveClick) {
        span.addEventListener('click', () => onHalfMoveClick(halfIdx));
      }
      row.appendChild(span);
      if (halfIdx === currentIndex) highlightedRow = row;
    }

    container.appendChild(row);
  }
  if (highlightedRow) highlightedRow.scrollIntoView({ block: 'nearest' });
}

function syncMoveList(): void {
  const moves = game.history;
  if (moves.length === 0) {
    $moveListPanel.classList.add('hidden');
    return;
  }
  $moveListPanel.classList.remove('hidden');
  $moveListLabel.textContent = `Moves (${moves.length})`;
  if ($moveListPanel.classList.contains('expanded')) {
    renderMoveList($moveListBody, moves, moves.length - 1);
  }
}

// -------------------------------------------------------------------------
// Analysis panel (V2 — top-3 moves + win prob + eval sparkline)
// -------------------------------------------------------------------------

function updateAnalysisPanel(topQ: number, top3: Top3Move[]): void {
  $winProbText.textContent = formatWinProb(topQ);
  $analysisPanel.classList.remove('hidden');

  // Top-3 bars
  $top3Bars.innerHTML = '';
  for (const m of top3) {
    const row = document.createElement('div');
    row.className = 'bar-row';
    const coord = document.createElement('span');
    coord.className = 'bar-coord';
    coord.textContent = `(${m.x},${m.y})`;
    const track = document.createElement('div');
    track.className = 'bar-track';
    const fill = document.createElement('div');
    fill.className = 'bar-fill' + (m.q >= 0 ? ' ai-winning' : '');
    fill.style.width = `${m.pct.toFixed(1)}%`;
    track.appendChild(fill);
    const pct = document.createElement('span');
    pct.className = 'bar-pct';
    pct.textContent = `${m.pct.toFixed(0)}%`;
    row.appendChild(coord);
    row.appendChild(track);
    row.appendChild(pct);
    $top3Bars.appendChild(row);
  }

  drawSparkline();
}

function drawSparkline(): void {
  const canvas = $evalSparkline;
  // Size canvas to its CSS display size
  const rect = canvas.getBoundingClientRect();
  const w = Math.max(rect.width || 200, 200);
  const h = 28;
  canvas.width  = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  ctx.clearRect(0, 0, w, h);
  const mid = h / 2;

  // Centre line
  ctx.strokeStyle = '#1a3050';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, mid);
  ctx.lineTo(w, mid);
  ctx.stroke();

  if (evalHistory.length < 1) return;

  // Fill area under the curve
  const points = evalHistory.map((q, i) => ({
    x: evalHistory.length === 1 ? w / 2 : (i / (evalHistory.length - 1)) * w,
    y: mid - q * (mid - 2),   // q in [-1,1] → y in [h-2, 2]
  }));

  ctx.beginPath();
  ctx.moveTo(points[0].x, mid);
  for (const pt of points) ctx.lineTo(pt.x, pt.y);
  ctx.lineTo(points[points.length - 1].x, mid);
  ctx.closePath();
  // Use AI colour (red) when AI winning, human colour (blue) when human winning
  const lastQ = evalHistory[evalHistory.length - 1];
  ctx.fillStyle = lastQ >= 0 ? 'rgba(231,76,60,0.2)' : 'rgba(93,173,226,0.2)';
  ctx.fill();

  // Line on top
  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  for (let i = 1; i < points.length; i++) ctx.lineTo(points[i].x, points[i].y);
  ctx.strokeStyle = lastQ >= 0 ? '#e74c3c' : '#5dade2';
  ctx.lineWidth = 1.5;
  ctx.stroke();
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
      if (replayAnalysisMode) {
        replayAnalysisMode = false;
        $replayAnalyseBtn.disabled = false;
        $replayAnalyseBtn.textContent = 'Analyse';
        if (msg.top3) {
          updateReplayAnalysisPanel(msg.topQ as number, msg.top3 as Top3Move[]);
        }
        worker.terminate();
        workerAlive = false;
        return;
      }
      const moveStr = msg.move === 'swap' ? 'swap' : `x=${(msg.move as {x:number,y:number}).x} y=${(msg.move as {x:number,y:number}).y}`;
      if (msg.trials != null) {
        const qSign = (msg.topQ as number) >= 0 ? '+' : '';
        diagLog(`worker-result ${moveStr} | trials=${msg.trials} topPct=${(msg.topPct as number).toFixed(1)}% topQ=${qSign}${(msg.topQ as number).toFixed(3)} elapsed=${msg.elapsed}ms [budget=${msg.timeLimitMs}ms cap=${msg.maxTrials} temp=${msg.temperature}]`);
        const barStyle = winProbBarStyle(msg.topQ as number);
        $winProbBar.style.backgroundColor = barStyle.color;
        $winProbBar.style.opacity = String(barStyle.opacity);
        pendingAnalysis = { topQ: msg.topQ as number, top3: (msg.top3 as Top3Move[]) ?? [] };
      } else {
        diagLog(`worker-result ${moveStr}`);
        pendingAnalysis = null;
      }
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
      const sec = Math.round((msg.elapsed as number) / 1000);
      const budget = msg.timeLimitMs != null ? ` / ${Math.round((msg.timeLimitMs as number) / 1000)}s` : '';
      $thinkingProgress.textContent = `${sec}s${budget}`;
    } else if (msg.type === 'eval-game-progress') {
      // L2+L3: accumulate per-move eval as it streams in.
      const idx = msg.moveIndex as number;
      replayGameEvals[idx] = { topQ: msg.topQ as number, rank: msg.rank as number };
      const total = replayParsedGame?.moves.length ?? 0;
      $replayAnalyseGameBtn.textContent = `Analysing… (${idx + 1}/${total})`;
      // Update sparkline incrementally.
      $replayGameSparklineWrap.classList.remove('hidden');
      drawReplayGameSparkline();
      // Apply quality colours to move list if it is currently expanded.
      if ($replayMoveListPanel.classList.contains('expanded')) {
        applyMoveQuality($replayMoveListBody);
      }
    } else if (msg.type === 'eval-game-done') {
      replayGameAnalysisRunning = false;
      $replayAnalyseGameBtn.disabled = false;
      $replayAnalyseGameBtn.textContent = 'Re-analyse';
      // Final redraw with position marker.
      drawReplayGameSparkline();
      if ($replayMoveListPanel.classList.contains('expanded')) {
        applyMoveQuality($replayMoveListBody);
      }
      // Free the worker heap now that batch eval is complete.
      worker.terminate();
      workerAlive = false;
    } else if (msg.type === 'eval-position-result') {
      // V5: heatmap result — route to game board or replay board by context.
      if (heatmapContext === 'replay') {
        replayHeatmapPending = false;
        $replayHeatmapBtn.disabled = false;
        if (replayHeatmapActive) {
          replayBoardUI?.setHeatmap(msg.policy as Float32Array, msg.turn as number);
        }
      } else {
        pendingHeatmap = false;
        syncHeatmapButton();
        if (heatmapActive) {
          board.setHeatmap(msg.policy as Float32Array, msg.turn as number);
        }
      }
      worker.terminate();
      workerAlive = false;
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
  const initStrength = getBotStrength();
  worker.postMessage({ type: 'init', modelUrl: MODEL_URL, timeLimitMs: getThinkTimeSec() * 1000, maxTrials: initStrength.maxTrials, temperature: initStrength.temperature });
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

  const timeSec  = getThinkTimeSec();
  const strength = getBotStrength();
  const watchdogMs = (timeSec + 15) * 1000;

  clearAiMoveTimer();
  aiMoveTimer = setTimeout(onAiMoveTimeout, watchdogMs);
  diagLog(`request-ai-move timeSec=${timeSec} strength=${strength.id} watchdog=${watchdogMs}ms`);

  const history: MoveMsg[] = game.history.map(m =>
    m === 'swap' ? 'swap' : { x: (m as {x:number,y:number}).x, y: (m as {x:number,y:number}).y }
  );
  const moveMsg = { type: 'move', history, timeLimitMs: timeSec * 1000, maxTrials: strength.maxTrials, temperature: strength.temperature };

  if (workerAlive) {
    worker.postMessage(moveMsg);
  } else {
    // Worker was terminated (after watchdog or mid-move "New Game").
    diagLog('worker-restarting-for-move');
    initWorker(() => {
      diagLog('worker-ready-sending-move');
      worker.postMessage(moveMsg);
    });
  }
}

// -------------------------------------------------------------------------
// V5: Policy heatmap
// -------------------------------------------------------------------------

function requestHeatmap(): void {
  if (pendingHeatmap || gameOver || aiThinking) return;
  pendingHeatmap = true;
  heatmapContext = 'game';
  $heatmapBtn.disabled = true;
  diagLog('heatmap-request');

  const history: MoveMsg[] = game.history.map(m =>
    m === 'swap' ? 'swap' : { x: (m as {x:number,y:number}).x, y: (m as {x:number,y:number}).y }
  );
  const req = { type: 'eval-position', history };

  if (workerAlive) {
    worker.postMessage(req);
  } else {
    initWorker(() => worker.postMessage(req));
  }
}

function requestReplayHeatmap(): void {
  if (!replayParsedGame || replayHeatmapPending) return;
  replayHeatmapPending = true;
  heatmapContext = 'replay';
  $replayHeatmapBtn.disabled = true;
  diagLog('replay-heatmap-request');

  const history = replayParsedGame.moves
    .slice(0, replayMoveIndex)
    .map(m => m === 'swap' ? 'swap' : { x: (m as {x:number,y:number}).x, y: (m as {x:number,y:number}).y });
  const req = { type: 'eval-position', history };

  if (workerAlive) {
    worker.postMessage(req);
  } else {
    initWorker(() => worker.postMessage(req));
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

  clearHeatmap();
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

  syncMoveList();
  if (gameMode === 'pvp') {
    // Other human player takes over.
    board.setEnabled(true);
    $statusText.textContent = turnStatusText(game.turn, gameMode);
    syncHintButton();
    syncResignButton();
    syncHeatmapButton();
    syncRedoButton();
    syncThinkTimeVisibility();
    startHeartbeat();
  } else {
    syncHintButton();
    syncResignButton();
    syncHeatmapButton();
    syncRedoButton();
    syncThinkTimeVisibility();
    requestAiMove();
  }
}

function onHumanSwap(): void {
  if (gameOver || aiThinking || !isHumanTurn(game.turn, gameMode) || game.history.length !== 1) return;

  clearHeatmap();
  diagLog('human-swap');
  stopHeartbeat();
  game.play('swap');
  updateSwapBtn();
  board.setGame(game, false);

  if (gameMode === 'pvp') {
    board.setEnabled(true);
    $statusText.textContent = turnStatusText(game.turn, gameMode);
    syncHintButton();
    syncResignButton();
    syncHeatmapButton();
    syncRedoButton();
    syncThinkTimeVisibility();
    startHeartbeat();
  } else {
    syncHintButton();
    syncResignButton();
    syncHeatmapButton();
    syncRedoButton();
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

  // Apply analysis data from the pending result (if MCTS ran)
  if (pendingAnalysis) {
    evalHistory.push(pendingAnalysis.topQ);
    updateAnalysisPanel(pendingAnalysis.topQ, pendingAnalysis.top3);
    pendingAnalysis = null;
  }

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
    syncMoveList();
    syncHintButton();
    syncResignButton();
    syncHeatmapButton();
    syncRedoButton();
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
  clearHeatmap();
  diagLog(`hint-requested turn=${game.turn}`);
  requestAiMove();
}

function onRedoClick(): void {
  if (gameOver || aiThinking || !game.canRedo) return;

  clearHeatmap();
  if (gameMode === 'pvp') {
    game.redo();
    board.setGame(game, true);
    $statusText.textContent = turnStatusText(game.turn, gameMode);
    updateSwapBtn();
  } else {
    // PvC: redo up to 2 moves (AI + human), stopping at first human move
    // that would trigger a new AI calculation. Re-apply stored moves only.
    game.redo();  // first undone move (AI's)
    if (game.canRedo) game.redo();  // second undone move (human's)
    board.setGame(game, true);
    $statusText.textContent = turnStatusText(BLACK, gameMode);
    updateSwapBtn();
  }
  syncMoveList();
  syncHintButton();
  syncResignButton();
  syncHeatmapButton();
  syncRedoButton();
  syncThinkTimeVisibility();
}

function onUndoClick(): void {
  if (aiThinking) return;
  if (game.history.length === 0) return;

  // Undoing after game over reactivates the game.
  if (gameOver) {
    gameOver = false;
    userClickedStart = true;
    board.setEnabled(true);
  }

  clearHeatmap();
  if (gameMode === 'pvp') {
    // PvP: undo exactly 1 move (the last player's move).
    if (game.history.length >= 1) {
      game.undo();
      board.setGame(game, true);
      $statusText.textContent = turnStatusText(game.turn, gameMode);
      updateSwapBtn();
      syncMoveList();
      syncHintButton();
      syncResignButton();
      syncHeatmapButton();
      syncRedoButton();
      syncThinkTimeVisibility();
    }
  } else {
    // PvC: undo the human move + the preceding AI move together, so it's
    // always the human's turn after undo.
    if (game.history.length >= 2) {
      game.undo();
      game.undo();
      evalHistory.pop();  // remove the AI move's eval entry
      board.setGame(game, true);
      $statusText.textContent = turnStatusText(BLACK, gameMode);
      updateSwapBtn();
    } else if (game.history.length === 1) {
      game.undo();
      board.setGame(game, true);
      $statusText.textContent = turnStatusText(BLACK, gameMode);
      updateSwapBtn();
    }
    // Refresh analysis panel (redraw sparkline without the removed entry)
    if (evalHistory.length > 0) {
      drawSparkline();
    } else {
      $analysisPanel.classList.add('hidden');
      $analysisPanel.classList.remove('expanded');
    }
    syncMoveList();
    syncHintButton();
    syncResignButton();
    syncHeatmapButton();
    syncRedoButton();
    syncThinkTimeVisibility();
  }
}

function onResignClick(): void {
  if (gameOver || aiThinking || game.history.length === 0) return;
  diagLog(`resign turn=${game.turn}`);
  tsgfResult = resignTsgfResult(game.turn);
  const opponent = game.turn === BLACK ? WHITE : BLACK;
  endGame(resultMessage(opponent, gameMode));
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
  gameOver       = false;
  aiThinking     = false;
  tsgfResult     = '?';
  heatmapActive  = false;
  pendingHeatmap = false;
  game = new Game();
  evalHistory = [];
  pendingAnalysis = null;
  $winProbBar.style.opacity = '0';
  $analysisPanel.classList.add('hidden');
  $analysisPanel.classList.remove('expanded');
  $moveListPanel.classList.add('hidden');
  $moveListPanel.classList.remove('expanded');
  $moveListBody.innerHTML = '';
  $redoBtn.disabled = true;
  $thinkingOverlay.classList.add('hidden');
  $swapBtn.classList.add('hidden');
  $undoBtn.classList.remove('hidden');
  $heatmapBtn.classList.add('hidden');
  $heatmapBtn.classList.remove('active');
  board.setGame(game, false);
  syncThinkTimeVisibility();

  syncHintButton();
  syncResignButton();
  syncThinkTimeVisibility();
  if (gameMode === 'pvc') {
    requestAiMove();
  } else {
    // PvP: WHITE (first mover) takes the first turn.
    board.setEnabled(true);
    $statusText.textContent = turnStatusText(game.turn, gameMode);
    syncHintButton();
    syncResignButton();
    syncHeatmapButton();
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
  $resignBtn.classList.add('hidden');
  $redoBtn.disabled = true;
  syncThinkTimeVisibility();
  // Undo stays visible so players can roll back and continue the game.
}

function setThinking(thinking: boolean): void {
  if (thinking) {
    $statusText.textContent = 'AI is thinking…';
    $thinkingOverlay.classList.remove('hidden');
    board.setEnabled(false);
    $hintBtn.classList.add('hidden');
    $heatmapBtn.classList.add('hidden');
    $thinkTimeSelect.classList.remove('hidden'); // keep visible so user can adjust mid-think
  } else {
    $thinkingOverlay.classList.add('hidden');
    $thinkingProgress.textContent = '';
    if (!gameOver) board.setEnabled(true);
    // syncHintButton() / syncThinkTimeVisibility() / syncHeatmapButton() called by whoever transitions out of thinking.
  }
}

// -------------------------------------------------------------------------
// Little Golem — Explore & Replay
// -------------------------------------------------------------------------

let replayBoardUI: BoardUI | null = null;
let replayParsedGame: ParsedGame | null = null;
let replayMoveIndex = 0;
let replayAnalysisMode = false;

/** Per-move eval results from eval-game batch run (L2+L3). Index = half-move index. */
let replayGameEvals: Array<{ topQ: number; rank: number }> = [];
let replayGameAnalysisRunning = false;

const $replayAnalyseBtn    = document.getElementById('replay-analyse-btn') as HTMLButtonElement;
const $replayHeatmapBtn    = document.getElementById('replay-heatmap-btn') as HTMLButtonElement;
const $replayAnalysisPanel = document.getElementById('replay-analysis-panel')!;
const $replayWinProbText   = document.getElementById('replay-win-prob-text')!;
const $replayTop3Bars      = document.getElementById('replay-top3-bars')!;

/** Cached player search results for instant back-navigation without re-fetch. */
let lastPlayerResults: PlayerResult[] = [];

/** Full unfiltered game list for the current player (populated by openPlayerGames). */
let allLoadedGames: GameSummary[] = [];
/** Active result filter (L10). */
let activeResultFilter: ResultFilter = 'all';
/** Active opponent filter (L11); null = any. */
let activeOpponentFilter: string | null = null;

/** All screens that must be hidden when switching between them. */
function hideAllScreens(): void {
  $introScreen.classList.add('hidden');
  $loadingScreen.classList.add('hidden');
  $gameScreen.classList.add('hidden');
  $lgScreen.classList.add('hidden');
  $replayScreen.classList.add('hidden');
  replayAnalysisMode = false;
  replayGameAnalysisRunning = false;
  replayHeatmapActive  = false;
  replayHeatmapPending = false;
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
  const content = document.createElement('div');
  content.className = 'lg-card-content';
  const el1 = document.createElement('div');
  el1.className = 'lg-card-players';
  el1.textContent = line1;
  const el2 = document.createElement('div');
  el2.className = 'lg-card-meta';
  el2.textContent = line2;
  content.appendChild(el1);
  content.appendChild(el2);
  card.appendChild(content);
  card.addEventListener('click', onClick);
  return card;
}

async function saveLgGame(id: string): Promise<void> {
  try {
    const text = await fetchGameRaw(id);
    const filename = `game${id}.tsgf`;
    const blob = new Blob([text], { type: 'application/octet-stream' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href     = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    diagLog(`lg-game-saved id=${id} filename=${filename}`);
  } catch (err) {
    diagLog(`lg-save-error id=${id}: ${err instanceof Error ? err.message : String(err)}`);
  }
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

/** Re-render the game cards below the filter bar using current active filters. */
function applyGameFilters(): void {
  const filtered = filterGameSummaries(allLoadedGames, activeResultFilter, activeOpponentFilter);

  // Remove existing game cards (everything after .lg-filter-bar)
  const filterBar = $lgResults.querySelector('.lg-filter-bar');
  let el = filterBar ? filterBar.nextElementSibling : null;
  while (el) {
    const next = el.nextElementSibling;
    $lgResults.removeChild(el);
    el = next;
  }

  // Update chip active states and counts
  const counts: Record<string, number> = { all: allLoadedGames.length, win: 0, lost: 0, draw: 0 };
  for (const g of allLoadedGames) {
    if (g.result === 'win' || g.result === 'lost' || g.result === 'draw') counts[g.result]++;
  }
  $lgResults.querySelectorAll<HTMLButtonElement>('.lg-chip').forEach(btn => {
    const f = btn.dataset.filter as ResultFilter;
    btn.classList.toggle('active', f === activeResultFilter);
    btn.textContent = f === 'all' ? `All (${counts.all})`
                    : f === 'win' ? `Win (${counts.win})`
                    : f === 'lost' ? `Loss (${counts.lost})`
                    : `Draw (${counts.draw})`;
  });

  // Update opponent select value
  const sel = $lgResults.querySelector<HTMLSelectElement>('.lg-opponent-select');
  if (sel) sel.value = activeOpponentFilter ?? '';

  for (const g of filtered) {
    const line1 = g.opponent ? `vs ${g.opponent}` : `${g.blackPlayer} (B) vs ${g.whitePlayer} (W)`;
    const movePart = g.moveCount > 0 ? `${g.moveCount} moves  ·  ` : '';
    const line2 = `#${g.id}  ·  ${movePart}${formatResult(g.result)}`;
    const card = makeLgCard(line1, line2, () => openReplayById(g.id));
    const saveBtn = document.createElement('button');
    saveBtn.className = 'lg-save-btn';
    saveBtn.title = 'Save game file to device';
    saveBtn.textContent = '\u2B07';  // ⬇ downwards black arrow
    saveBtn.addEventListener('click', (e) => { e.stopPropagation(); saveLgGame(g.id); });
    card.appendChild(saveBtn);
    $lgResults.appendChild(card);
  }
}

function renderGameResults(games: GameSummary[]): void {
  allLoadedGames = games;
  activeResultFilter = 'all';
  activeOpponentFilter = null;

  $lgResults.innerHTML = '';
  if (games.length === 0) {
    lgSetState('error', 'No finished TwixT PP games found for this player.');
    return;
  }
  lgSetState('results');

  // "← Players" back link
  const backEl = document.createElement('div');
  backEl.className = 'lg-card-back';
  backEl.textContent = '← Players';
  backEl.addEventListener('click', () => renderPlayerResults(lastPlayerResults));
  $lgResults.appendChild(backEl);

  // Filter bar (L10 + L11)
  const filterBar = document.createElement('div');
  filterBar.className = 'lg-filter-bar';

  // Result chips (L10)
  const chipRow = document.createElement('div');
  chipRow.className = 'lg-chip-row';
  for (const f of ['all', 'win', 'lost', 'draw'] as ResultFilter[]) {
    const btn = document.createElement('button');
    btn.className = 'lg-chip';
    btn.dataset.filter = f;
    btn.addEventListener('click', () => {
      activeResultFilter = f;
      applyGameFilters();
    });
    chipRow.appendChild(btn);
  }
  filterBar.appendChild(chipRow);

  // Opponent dropdown (L11) — hidden when fewer than 2 distinct opponents
  const opponents = [...new Set(games.map(g => g.opponent).filter(Boolean) as string[])].sort();
  if (opponents.length >= 2) {
    const opponentRow = document.createElement('div');
    opponentRow.className = 'lg-opponent-row';
    const label = document.createElement('span');
    label.textContent = 'vs';
    opponentRow.appendChild(label);
    const sel = document.createElement('select');
    sel.className = 'lg-opponent-select';
    const anyOpt = document.createElement('option');
    anyOpt.value = '';
    anyOpt.textContent = 'any opponent';
    sel.appendChild(anyOpt);
    for (const opp of opponents) {
      const opt = document.createElement('option');
      opt.value = opp;
      opt.textContent = opp;
      sel.appendChild(opt);
    }
    sel.addEventListener('change', () => {
      activeOpponentFilter = sel.value || null;
      applyGameFilters();
    });
    opponentRow.appendChild(sel);
    filterBar.appendChild(opponentRow);
  }

  $lgResults.appendChild(filterBar);
  applyGameFilters();
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

  // Reset move list, game-analysis, and heatmap state for new game.
  replayGameEvals = [];
  replayGameAnalysisRunning = false;
  replayHeatmapActive  = false;
  replayHeatmapPending = false;
  $replayHeatmapBtn.classList.remove('active');
  $replayHeatmapBtn.disabled = false;
  $replayMoveListPanel.classList.remove('expanded');
  $replayMoveListBody.innerHTML = '';
  $replayMoveListLabel.textContent = `Moves (${parsed.moves.length})`;
  $replayGameSparklineWrap.classList.add('hidden');
  $replayAnalyseGameBtn.disabled = false;
  $replayAnalyseGameBtn.textContent = 'Analyse game';

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

  // Hide analysis panel and heatmap when navigating to a different move.
  $replayAnalysisPanel.classList.add('hidden');
  $replayAnalyseBtn.disabled = false;
  $replayAnalyseBtn.textContent = 'Analyse';
  clearReplayHeatmap();

  // Refresh move list label and body (if expanded).
  $replayMoveListLabel.textContent = `Moves (${total})`;
  if ($replayMoveListPanel.classList.contains('expanded')) {
    renderMoveList(
      $replayMoveListBody,
      replayParsedGame.moves,
      replayMoveIndex - 1,
      (i) => replayShowAtIndex(i + 1),
    );
    if (replayGameEvals.length > 0) applyMoveQuality($replayMoveListBody);
  }
  // Redraw sparkline position marker if game analysis has run.
  if (replayGameEvals.length > 0) drawReplayGameSparkline();
}

function updateReplayAnalysisPanel(topQ: number, top3: Top3Move[]): void {
  $replayWinProbText.textContent = formatWinProb(topQ);
  $replayAnalysisPanel.classList.remove('hidden');
  $replayTop3Bars.innerHTML = '';
  for (const m of top3) {
    const row = document.createElement('div');
    row.className = 'bar-row';
    const coord = document.createElement('span');
    coord.className = 'bar-coord';
    coord.textContent = `(${m.x},${m.y})`;
    const track = document.createElement('div');
    track.className = 'bar-track';
    const fill = document.createElement('div');
    fill.className = 'bar-fill' + (m.q >= 0 ? ' ai-winning' : '');
    fill.style.width = `${m.pct.toFixed(1)}%`;
    track.appendChild(fill);
    const pct = document.createElement('span');
    pct.className = 'bar-pct';
    pct.textContent = `${m.pct.toFixed(0)}%`;
    row.appendChild(coord);
    row.appendChild(track);
    row.appendChild(pct);
    $replayTop3Bars.appendChild(row);
  }
}

function requestReplayAnalysis(): void {
  if (!replayParsedGame || replayAnalysisMode) return;
  replayAnalysisMode = true;
  $replayAnalyseBtn.disabled = true;
  $replayAnalyseBtn.textContent = 'Analysing…';
  $replayAnalysisPanel.classList.add('hidden');

  const history = replayParsedGame.moves
    .slice(0, replayMoveIndex)
    .map(m => m === 'swap' ? 'swap' : { x: (m as {x:number,y:number}).x, y: (m as {x:number,y:number}).y });

  const strength = getBotStrength();
  const moveMsg = {
    type: 'move',
    history,
    timeLimitMs: getThinkTimeSec() * 1000,
    maxTrials: strength.maxTrials,
    temperature: 0,
  };

  if (workerAlive) {
    worker.postMessage(moveMsg);
  } else {
    initWorker(() => {
      worker.postMessage(moveMsg);
    });
  }
}

function drawReplayGameSparkline(): void {
  if (replayGameEvals.length === 0) return;
  const canvas = $replayGameSparkline;
  const rect = canvas.getBoundingClientRect();
  const w = Math.max(rect.width || 200, 200);
  const h = 28;
  canvas.width  = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  ctx.clearRect(0, 0, w, h);
  const mid = h / 2;

  ctx.strokeStyle = '#1a3050';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, mid);
  ctx.lineTo(w, mid);
  ctx.stroke();

  const evals = replayGameEvals;
  const points = evals.map((e, i) => ({
    x: evals.length === 1 ? w / 2 : (i / (evals.length - 1)) * w,
    y: mid - e.topQ * (mid - 2),
  }));

  ctx.beginPath();
  ctx.moveTo(points[0].x, mid);
  for (const p of points) ctx.lineTo(p.x, p.y);
  ctx.lineTo(points[points.length - 1].x, mid);
  ctx.closePath();
  const lastQ = evals[evals.length - 1].topQ;
  ctx.fillStyle = lastQ >= 0 ? 'rgba(231,76,60,0.2)' : 'rgba(93,173,226,0.2)';
  ctx.fill();

  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  for (let i = 1; i < points.length; i++) ctx.lineTo(points[i].x, points[i].y);
  ctx.strokeStyle = lastQ >= 0 ? '#e74c3c' : '#5dade2';
  ctx.lineWidth = 1.5;
  ctx.stroke();

  // Draw a vertical marker at the current replay position
  if (replayMoveIndex > 0 && evals.length > 0) {
    const evalIdx = Math.min(replayMoveIndex - 1, evals.length - 1);
    const markerX = evals.length === 1 ? w / 2 : (evalIdx / (evals.length - 1)) * w;
    ctx.strokeStyle = 'rgba(200,220,240,0.5)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(markerX, 0);
    ctx.lineTo(markerX, h);
    ctx.stroke();
  }
}

/** Apply quality CSS classes to move list rows based on replayGameEvals. */
function applyMoveQuality(container: HTMLElement): void {
  const spans = container.querySelectorAll<HTMLElement>('.ml-move');
  spans.forEach((span, halfIdx) => {
    span.classList.remove('ml-q-best', 'ml-q-good', 'ml-q-poor');
    if (halfIdx < replayGameEvals.length) {
      const { rank } = replayGameEvals[halfIdx];
      if (rank < 0) return;  // swap move
      if (rank === 0)      span.classList.add('ml-q-best');
      else if (rank <= 4)  span.classList.add('ml-q-good');
      else                 span.classList.add('ml-q-poor');
    }
  });
}

function requestReplayGameAnalysis(): void {
  if (!replayParsedGame || replayGameAnalysisRunning) return;
  replayGameAnalysisRunning = true;
  replayGameEvals = [];
  $replayAnalyseGameBtn.disabled = true;
  $replayAnalyseGameBtn.textContent = 'Analysing…';

  // Ensure move list is expanded so quality colours are visible.
  if (!$replayMoveListPanel.classList.contains('expanded')) {
    $replayMoveListPanel.classList.add('expanded');
    renderMoveList(
      $replayMoveListBody,
      replayParsedGame.moves,
      replayMoveIndex - 1,
      (i) => replayShowAtIndex(i + 1),
    );
  }

  const history = replayParsedGame.moves
    .map(m => m === 'swap' ? 'swap' : { x: (m as {x:number;y:number}).x, y: (m as {x:number;y:number}).y });

  const evalMsg = { type: 'eval-game', history };

  if (workerAlive) {
    worker.postMessage(evalMsg);
  } else {
    initWorker(() => {
      worker.postMessage(evalMsg);
    });
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

  const savedStrength = getBotStrength();
  for (const s of BOT_STRENGTH_OPTIONS) {
    const opt = document.createElement('option');
    opt.value = s.id;
    opt.textContent = s.label;
    if (s.id === savedStrength.id) opt.selected = true;
    $strengthSelect.appendChild(opt);
  }
  $strengthSelect.addEventListener('change', () => {
    localStorage.setItem(BOT_STRENGTH_KEY, $strengthSelect.value);
  });

  $hintBtn.addEventListener('click',    onHintClick);
  $undoBtn.addEventListener('click',    onUndoClick);
  $redoBtn.addEventListener('click',    onRedoClick);
  $resignBtn.addEventListener('click',  onResignClick);
  $heatmapBtn.addEventListener('click', () => {
    if (pendingHeatmap || gameOver || aiThinking) return;
    heatmapActive = !heatmapActive;
    $heatmapBtn.classList.toggle('active', heatmapActive);
    if (!heatmapActive) {
      board.setHeatmap(null, 0);
    } else {
      requestHeatmap();
    }
  });
  $swapBtn.addEventListener('click',    onHumanSwap);
  $newGameBtn.addEventListener('click', () => showIntro());
  $exportBtn.addEventListener('click',  onExportClick);
  $analysisToggle.addEventListener('click', () => {
    $analysisPanel.classList.toggle('expanded');
    if ($analysisPanel.classList.contains('expanded')) drawSparkline();
  });

  // Move list toggle (game screen U1)
  document.getElementById('move-list-toggle')?.addEventListener('click', () => {
    $moveListPanel.classList.toggle('expanded');
    if ($moveListPanel.classList.contains('expanded')) {
      renderMoveList($moveListBody, game.history, game.history.length - 1);
    }
  });

  // Move list toggle (replay screen L4)
  document.getElementById('replay-move-list-toggle')?.addEventListener('click', () => {
    $replayMoveListPanel.classList.toggle('expanded');
    if ($replayMoveListPanel.classList.contains('expanded') && replayParsedGame) {
      renderMoveList(
        $replayMoveListBody,
        replayParsedGame.moves,
        replayMoveIndex - 1,
        (i) => replayShowAtIndex(i + 1),
      );
    }
  });
  $settingsBtn.addEventListener('click', () => {
    $settingsPanel.classList.remove('hidden');
  });
  $settingsBackdrop.addEventListener('click', () => {
    $settingsPanel.classList.add('hidden');
  });

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

  // Replay screen — back, analyse, and step controls
  document.getElementById('replay-back-btn')?.addEventListener('click', () => {
    showLgScreen();
    if ($lgResults.children.length > 0) lgSetState('results');
  });
  $replayAnalyseBtn.addEventListener('click', () => requestReplayAnalysis());
  $replayHeatmapBtn.addEventListener('click', () => {
    if (replayHeatmapPending) return;
    replayHeatmapActive = !replayHeatmapActive;
    $replayHeatmapBtn.classList.toggle('active', replayHeatmapActive);
    if (!replayHeatmapActive) {
      replayBoardUI?.setHeatmap(null, 0);
    } else {
      requestReplayHeatmap();
    }
  });
  $replayAnalyseGameBtn.addEventListener('click', () => requestReplayGameAnalysis());
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
