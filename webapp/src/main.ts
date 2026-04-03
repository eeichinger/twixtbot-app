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

const MODEL_URL     = import.meta.env.BASE_URL + 'model.onnx';
const MCTS_TRIALS   = 100;   // reduce to 50 on slow devices
const HUMAN_COLOR   = BLACK;
const AI_COLOR      = WHITE;

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
const $undoBtn        = document.getElementById('undo-btn')!;
const $newGameBtn     = document.getElementById('new-game-btn')!;
const boardCanvas     = document.getElementById('board-canvas') as HTMLCanvasElement;

// -------------------------------------------------------------------------
// State
// -------------------------------------------------------------------------

let game  = new Game();
let board: BoardUI;
let worker: Worker;
let gameOver = false;
let aiThinking = false;

// -------------------------------------------------------------------------
// Worker bridge
// -------------------------------------------------------------------------

type MoveMsg = { x: number; y: number } | 'swap';

function initWorker(): void {
  worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });

  worker.onmessage = (e) => {
    const msg = e.data;
    if (msg.type === 'ready') {
      onWorkerReady();
    } else if (msg.type === 'result') {
      onAiMove(msg.move as MoveMsg);
    } else if (msg.type === 'error') {
      console.error('[Worker]', msg.message);
      $loadingMsg.textContent = `Error: ${msg.message}`;
    }
  };

  $loadingMsg.textContent = 'Loading AI model…';
  worker.postMessage({ type: 'init', modelUrl: MODEL_URL, trials: MCTS_TRIALS });
}

function requestAiMove(): void {
  if (gameOver || aiThinking) return;
  aiThinking = true;
  setThinking(true);

  const history: MoveMsg[] = game.history.map(m =>
    m === 'swap' ? 'swap' : { x: (m as {x:number,y:number}).x, y: (m as {x:number,y:number}).y }
  );
  worker.postMessage({ type: 'move', history, trials: MCTS_TRIALS });
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
  board.setGame(game, true);
  $statusText.textContent = 'Your turn (Black)';
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

  $undoBtn.addEventListener('click',     onUndoClick);
  $newGameBtn.addEventListener('click',  startNewGame);
  $gameoverNewBtn.addEventListener('click', startNewGame);

  initWorker();
}

// Register PWA service worker
if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('/sw.js').catch(console.warn);
  });
}

init();
