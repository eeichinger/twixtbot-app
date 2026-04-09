/**
 * game-mode.ts — Pure helpers for game mode logic (PvC vs PvP).
 *
 * Kept as pure functions (no DOM, no side effects) so they are
 * straightforwardly unit-testable.
 */

import { BLACK, WHITE } from './twixt.js';

export type GameMode = 'pvc' | 'pvp';

export const GAME_MODE_KEY = 'twixt-game-mode';

// -------------------------------------------------------------------------
// Persistence
// -------------------------------------------------------------------------

export function loadGameMode(): GameMode {
  return localStorage.getItem(GAME_MODE_KEY) === 'pvp' ? 'pvp' : 'pvc';
}

export function saveGameMode(mode: GameMode): void {
  localStorage.setItem(GAME_MODE_KEY, mode);
}

// -------------------------------------------------------------------------
// Logic helpers
// -------------------------------------------------------------------------

/**
 * Returns true when the player whose turn it is should be controlled by a
 * human click (i.e. the app should accept board input).
 *
 * PvC: only BLACK is human (WHITE is the AI).
 * PvP: both colors are human.
 */
export function isHumanTurn(turn: number, mode: GameMode): boolean {
  return mode === 'pvp' || turn === BLACK;
}

/** Status-bar text to display for the current turn. */
export function turnStatusText(turn: number, mode: GameMode): string {
  if (mode === 'pvp') return turn === BLACK ? "Blue's turn" : "Orange's turn";
  return 'Your turn (Blue)';
}

/**
 * Result message shown on the intro screen after a game ends.
 *
 * Pass `null` for a draw, or the Color of the winner.
 */
export function resultMessage(winner: number | null, mode: GameMode): string {
  if (winner === null) return 'Draw';
  if (mode === 'pvp') return winner === BLACK ? 'Blue wins!' : 'Orange wins!';
  return winner === BLACK ? 'You win!' : 'AI wins';
}

/**
 * TSGF result string when the player whose turn it is resigns.
 *
 * In TSGF/SGF, WHITE is the first mover (written as "B"), so
 * game.turn===WHITE means the TSGF Black player resigns → TSGF White wins → 'W+'.
 * game.turn===BLACK means the TSGF White player resigns → TSGF Black wins → 'B+'.
 */
export function resignTsgfResult(turn: number): string {
  return turn === WHITE ? 'W+' : 'B+';
}
