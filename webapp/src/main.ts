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
// Constants
// -------------------------------------------------------------------------

const MODEL_URL   = import.meta.env.BASE_URL + 'model.onnx';
const HUMAN_COLOR = BLACK;
const AI_COLOR    = WHITE;

const THINK_TIME_OPTIONS = [5, 10, 15, 25, 30, 45, 60];  // seconds
const THINK_TIME_KEY     = 'twixt-think-time-sec';
const DEFAULT_THINK_TIME = 10;  // 25s caused ~30s total worker CPU → iOS kills page

function getThinkTimeSec(): number {
  const stored = parseInt(localStorage.getItem(THINK_TIME_KEY) ?? '', 10);
  return THINK_TIME_OPTIONS.includes(stored) ? stored : DEFAULT_THINK_TIME;
}

// -------------------------------------------------------------------------
// DOM references
// -------------------------------------------------------------------------

const $loadingScreen  = document.getElementById('loading-screen')!;
const $gameScreen     = document.getElementById('game-screen')!;
const $statusText     = document.getElementById('status-text')!;
const $thinkingOverlay = document.getElementById('thinking-overlay')!;
const $gameoverOverlay = document.getElementById('gameover-overlay')!;
const $gameoverMsg    = document.getElementById('gameover-msg')!;
const $gameoverNewBtn = document.getElementById('gameover-new-btn')!;
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
let gameOver = false;
let aiThinking = false;
let aiMoveTimer: ReturnType<typeof setTimeout> | null = null;

// -------------------------------------------------------------------------
// Worker bridge
// -------------------------------------------------------------------------

type MoveMsg = { x: number; y: number } | 'swap';

function initWorker(onReady: () => void = onWorkerReady): void {
  worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });

  worker.onmessage = (e) => {
    const msg = e.data;
    if (msg.type === 'ready') {
      onReady();
    } else if (msg.type === 'result') {
      clearAiMoveTimer();
      onAiMove(msg.move as MoveMsg);
    } else if (msg.type === 'error') {
      clearAiMoveTimer();
      console.error('[Worker]', msg.message);
      $loadingMsg.textContent = `Error: ${msg.message}`;
    }
  };

  // If iOS kills the worker we won't hear from it again — onerror won't always
  // fire, so we rely on the main-thread watchdog (aiMoveTimer) instead.
  worker.onerror = (e) => {
    console.error('[Worker] crashed:', e.message);
    if (aiThinking && !gameOver) onAiMoveTimeout();
  };

  $loadingMsg.textContent = 'Loading AI model…';
  worker.postMessage({ type: 'init', modelUrl: MODEL_URL, timeLimitMs: getThinkTimeSec() * 1000 });
}

function clearAiMoveTimer(): void {
  if (aiMoveTimer !== null) { clearTimeout(aiMoveTimer); aiMoveTimer = null; }
}

/** Called when the worker doesn't respond within the timeout budget.
 *  Plays a random legal move so the game can continue, then restarts the worker. */
function onAiMoveTimeout(): void {
  aiMoveTimer = null;
  if (!aiThinking || gameOver) return;
  console.warn('[Main] AI worker timed out — playing random move and restarting worker');
  const legal = game.legalPlays();
  const fallbackMove = legal.at(Math.floor(Math.random() * legal.length));
  // Restart the worker silently (no new game) so future AI moves work again
  worker.terminate();
  initWorker(() => { /* worker restarted; game already in progress */ });
  onAiMove({ x: fallbackMove.x, y: fallbackMove.y });
}

function requestAiMove(): void {
  if (gameOver || aiThinking) return;
  aiThinking = true;
  setThinking(true);

  const timeSec = getThinkTimeSec();

  // Watchdog: if the worker is killed by iOS (or hangs) and never responds,
  // this fires after the budget + a generous buffer and plays a fallback move.
  clearAiMoveTimer();
  aiMoveTimer = setTimeout(onAiMoveTimeout, (timeSec + 15) * 1000);

  const history: MoveMsg[] = game.history.map(m =>
    m === 'swap' ? 'swap' : { x: (m as {x:number,y:number}).x, y: (m as {x:number,y:number}).y }
  );
  worker.postMessage({ type: 'move', history, timeLimitMs: timeSec * 1000 });
}

// -------------------------------------------------------------------------
// Event handlers
// -------------------------------------------------------------------------

function onWorkerReady(): void {
  $loadingScreen.classList.add('hidden');
  $gameScreen.classList.remove('hidden');
  startNewGame();
}

function onHumanMove(p: { x: number; y: number }): void {
  if (gameOver || aiThinking || game.turn !== HUMAN_COLOR) return;
  if (!game.legalPlays().contains(p)) return;

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

  $statusText.textContent = 'Your turn (Black)';
}

function onUndoClick(): void {
  if (gameOver || aiThinking) return;
  // Undo both AI move and human move (one full round)
  if (game.history.length >= 2) {
    game.undo(); // undo AI move
    game.undo(); // undo human move
    board.setGame(game, true);
    $statusText.textContent = 'Your turn (Black)';
  } else if (game.history.length === 1) {
    game.undo();
    board.setGame(game, true);
    $statusText.textContent = 'Your turn (Black)';
  }
}

function startNewGame(): void {
  gameOver   = false;
  aiThinking = false;
  game = new Game();
  $gameoverOverlay.classList.add('hidden');
  $thinkingOverlay.classList.add('hidden');
  board.setGame(game, false);  // AI (WHITE) moves first; board disabled until then
  requestAiMove();
}

function endGame(msg: string): void {
  gameOver = true;
  setThinking(false);
  board.setGame(game, false);
  $gameoverMsg.textContent   = msg;
  $gameoverOverlay.classList.remove('hidden');
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
  board = new BoardUI(boardCanvas, { onMove: onHumanMove });

  // Populate think-time selector and restore saved value
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

  $undoBtn.addEventListener('click',     onUndoClick);
  $newGameBtn.addEventListener('click',  startNewGame);
  $gameoverNewBtn.addEventListener('click', startNewGame);

  initWorker();
}

init();
