/**
 * lg-sgf.test.ts — unit tests for the LG TwixT SGF parser.
 *
 * Uses a real game downloaded from Little Golem (game 2060663) as the
 * primary test fixture to verify coordinate parsing, player names, move
 * counts, and result handling against known-correct data.
 */

import { describe, it, expect } from 'vitest';
import { parseTSGF, formatResult } from '../lg-sgf.js';

// ---------------------------------------------------------------------------
// Real game fixture — Alan Hensel (B) vs TwixtBot (W), LG game 2060663.
// Uses LG's native notation: b[..] = Black, r[..] = Red (White).
// 31 moves; ends with r[resign] which is excluded from the moves array.
// ---------------------------------------------------------------------------

const REAL_GAME_2060663 =
  '(;FF[4]EV[twixt]PB[Alan Hensel]PW[TwixtBot]SZ[24]' +
  'SO[https://www.littlegolem.net]' +
  ';b[kg];r[ki];b[ii];r[jg];b[gh];r[ke];b[ei];r[fd];b[fc];r[in]' +
  ';b[po];r[km];b[mk];r[lk];b[qh];r[qk];b[tk];r[sl];b[ol];r[pm]' +
  ';b[nn];r[qf];b[og];r[oe];b[tf];r[sf];b[sd];r[rh];b[ui];r[vg]' +
  ';b[sh];r[resign])';

// ---------------------------------------------------------------------------
// parseTSGF — real game 2060663
// ---------------------------------------------------------------------------

describe('parseTSGF — real game 2060663 (Alan Hensel vs TwixtBot)', () => {
  const game = parseTSGF(REAL_GAME_2060663, '2060663');

  it('sets the id', () => {
    expect(game.id).toBe('2060663');
  });

  it('parses PB (Black player)', () => {
    expect(game.blackPlayer).toBe('Alan Hensel');
  });

  it('parses PW (White/Red player)', () => {
    expect(game.whitePlayer).toBe('TwixtBot');
  });

  it('parses board size SZ[24]', () => {
    expect(game.boardSize).toBe(24);
  });

  it('result is unknown — no RE field in this SGF', () => {
    expect(game.result).toBe('?');
  });

  it('parses 31 moves (r[resign] excluded)', () => {
    expect(game.moves).toHaveLength(31);
  });

  it('move 1: b[kg] → {x:10, y:6}  (k=10, g=6)', () => {
    expect(game.moves[0]).toEqual({ x: 10, y: 6 });
  });

  it('move 2: r[ki] → {x:10, y:8}  (k=10, i=8)  — r-notation parses correctly', () => {
    expect(game.moves[1]).toEqual({ x: 10, y: 8 });
  });

  it('move 3: b[ii] → {x:8, y:8}', () => {
    expect(game.moves[2]).toEqual({ x: 8, y: 8 });
  });

  it('move 4: r[jg] → {x:9, y:6}', () => {
    expect(game.moves[3]).toEqual({ x: 9, y: 6 });
  });

  it('move 31 (last): b[sh] → {x:18, y:7}  (s=18, h=7)', () => {
    expect(game.moves[30]).toEqual({ x: 18, y: 7 });
  });

  it('all moves are Points (no swap in this game)', () => {
    for (const move of game.moves) {
      expect(move).not.toBe('swap');
      expect(move).toMatchObject({ x: expect.any(Number), y: expect.any(Number) });
    }
  });

  it('all coordinates are within 0–23', () => {
    for (const move of game.moves) {
      if (move === 'swap') continue;
      expect((move as { x: number }).x).toBeGreaterThanOrEqual(0);
      expect((move as { x: number }).x).toBeLessThanOrEqual(23);
      expect((move as { y: number }).y).toBeGreaterThanOrEqual(0);
      expect((move as { y: number }).y).toBeLessThanOrEqual(23);
    }
  });
});

// ---------------------------------------------------------------------------
// parseTSGF — standard B/W notation with swap and result
// ---------------------------------------------------------------------------

describe('parseTSGF — standard B/W notation, swap move, RE field', () => {
  const sgf =
    '(;GM[21]FF[4]SZ[24]PB[Alice]PW[Bob]RE[B+]' +
    ';B[hd];W[qd];B[swap];W[hd];B[me];W[tt])';
  const game = parseTSGF(sgf, 'test-bw');

  it('parses result RE[B+]', () => {
    expect(game.result).toBe('B+');
  });

  it('parses 5 moves (W[tt] = resign/pass sentinel, excluded)', () => {
    expect(game.moves).toHaveLength(5);
  });

  it('move 3 is swap', () => {
    expect(game.moves[2]).toBe('swap');
  });

  it('move 1: B[hd] → {x:7, y:3}  (h=7, d=3)', () => {
    expect(game.moves[0]).toEqual({ x: 7, y: 3 });
  });
});

// ---------------------------------------------------------------------------
// parseTSGF — edge cases
// ---------------------------------------------------------------------------

describe('parseTSGF — edge cases', () => {
  it('returns defaults for an empty string', () => {
    const g = parseTSGF('');
    expect(g.blackPlayer).toBe('?');
    expect(g.whitePlayer).toBe('?');
    expect(g.boardSize).toBe(24);
    expect(g.result).toBe('?');
    expect(g.moves).toHaveLength(0);
  });

  it('stops parsing after B[resign]', () => {
    const g = parseTSGF('(;B[hd];W[qd];B[resign];W[ab])');
    expect(g.moves).toHaveLength(2); // hd, qd only
  });

  it('stops parsing after r[resign] (LG notation)', () => {
    const g = parseTSGF('(;b[hd];r[qd];b[ii];r[resign])');
    expect(g.moves).toHaveLength(3); // hd, qd, ii
  });
});

// ---------------------------------------------------------------------------
// formatResult
// ---------------------------------------------------------------------------

describe('formatResult', () => {
  it('B+ → Black wins', () => expect(formatResult('B+')).toBe('Black wins'));
  it('W+ → White wins', () => expect(formatResult('W+')).toBe('White wins'));
  it('0  → Draw',       () => expect(formatResult('0')).toBe('Draw'));
  it('draw → Draw',     () => expect(formatResult('draw')).toBe('Draw'));
  it('win  → Win',      () => expect(formatResult('win')).toBe('Win'));
  it('lost → Lost',     () => expect(formatResult('lost')).toBe('Lost'));
  it('?    → ?',        () => expect(formatResult('?')).toBe('?'));
  it('empty → —',       () => expect(formatResult('')).toBe('—'));
});
