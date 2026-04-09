/**
 * game-mode.test.ts — unit tests for game-mode.ts helpers.
 *
 * All helpers are pure functions (no DOM, no worker), so these tests run in
 * the standard Vitest node environment without any special setup.
 */

import { describe, it, expect } from 'vitest';
import { BLACK, WHITE } from '../twixt.js';
import {
  isHumanTurn,
  turnStatusText,
  resultMessage,
  resignTsgfResult,
} from '../game-mode.js';

// -------------------------------------------------------------------------
// isHumanTurn
// -------------------------------------------------------------------------

describe('isHumanTurn — PvP mode', () => {
  it('returns true when it is WHITE\'s turn', () => {
    expect(isHumanTurn(WHITE, 'pvp')).toBe(true);
  });

  it('returns true when it is BLACK\'s turn', () => {
    expect(isHumanTurn(BLACK, 'pvp')).toBe(true);
  });
});

describe('isHumanTurn — PvC mode', () => {
  it('returns false when it is WHITE\'s turn (WHITE is AI)', () => {
    expect(isHumanTurn(WHITE, 'pvc')).toBe(false);
  });

  it('returns true when it is BLACK\'s turn (BLACK is human)', () => {
    expect(isHumanTurn(BLACK, 'pvc')).toBe(true);
  });
});

// -------------------------------------------------------------------------
// turnStatusText
// -------------------------------------------------------------------------

describe('turnStatusText — PvP mode', () => {
  it('shows "Orange\'s turn" when WHITE (orange) is to move', () => {
    expect(turnStatusText(WHITE, 'pvp')).toBe("Orange's turn");
  });

  it('shows "Blue\'s turn" when BLACK (blue) is to move', () => {
    expect(turnStatusText(BLACK, 'pvp')).toBe("Blue's turn");
  });
});

describe('turnStatusText — PvC mode', () => {
  it('always shows the human-turn message regardless of whose turn it is', () => {
    expect(turnStatusText(BLACK, 'pvc')).toBe('Your turn (Blue)');
    // In PvC this function is only called when it genuinely is the human's
    // turn, but the output should be consistent regardless.
    expect(turnStatusText(WHITE, 'pvc')).toBe('Your turn (Blue)');
  });
});

// -------------------------------------------------------------------------
// resultMessage
// -------------------------------------------------------------------------

describe('resultMessage — draw', () => {
  it('returns "Draw" in PvC mode', () => {
    expect(resultMessage(null, 'pvc')).toBe('Draw');
  });

  it('returns "Draw" in PvP mode', () => {
    expect(resultMessage(null, 'pvp')).toBe('Draw');
  });
});

describe('resultMessage — BLACK wins', () => {
  it('returns "You win!" in PvC mode (human is BLACK)', () => {
    expect(resultMessage(BLACK, 'pvc')).toBe('You win!');
  });

  it('returns "Blue wins!" in PvP mode', () => {
    expect(resultMessage(BLACK, 'pvp')).toBe('Blue wins!');
  });
});

describe('resultMessage — WHITE wins', () => {
  it('returns "AI wins" in PvC mode (AI is WHITE)', () => {
    expect(resultMessage(WHITE, 'pvc')).toBe('AI wins');
  });

  it('returns "Orange wins!" in PvP mode', () => {
    expect(resultMessage(WHITE, 'pvp')).toBe('Orange wins!');
  });
});

// -------------------------------------------------------------------------
// resignTsgfResult
// -------------------------------------------------------------------------

describe('resignTsgfResult', () => {
  // In TSGF, WHITE = first mover = written as "B" (Black in SGF notation).
  // When WHITE resigns, TSGF White (i.e. the second mover) wins → 'W+'.
  it('returns "W+" when WHITE (first mover / TSGF Black) resigns', () => {
    expect(resignTsgfResult(WHITE)).toBe('W+');
  });

  // When BLACK (second mover, TSGF White) resigns, TSGF Black (first mover) wins → 'B+'.
  it('returns "B+" when BLACK (second mover / TSGF White) resigns', () => {
    expect(resignTsgfResult(BLACK)).toBe('B+');
  });
});
