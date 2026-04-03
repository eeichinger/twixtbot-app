/**
 * game.test.ts — outside-in behavioral tests for the TwixT game engine.
 *
 * Tests focus on observable outcomes (legal moves, win conditions, undo
 * symmetry, board state after play) — not internal data structures.
 */

import { describe, it, expect } from 'vitest';
import { Game, pt, SIZE, BLACK, WHITE, allLinks, replayHistory } from '../twixt.js';

// -------------------------------------------------------------------------
// Helpers
// -------------------------------------------------------------------------

/** Returns the number of pegs on the board for the given color. */
function pegCount(g: Game, color: number): number {
  return Array.from(g.pegs[color]).filter(v => v !== 0).length;
}

// -------------------------------------------------------------------------
// Initial state
// -------------------------------------------------------------------------

describe('Game — initial state', () => {
  it('starts on WHITE to move', () => {
    expect(new Game().turn).toBe(WHITE);
  });

  it('starts with empty board', () => {
    const g = new Game();
    expect(pegCount(g, WHITE)).toBe(0);
    expect(pegCount(g, BLACK)).toBe(0);
  });

  it('has no history', () => {
    expect(new Game().history).toHaveLength(0);
  });

  it('WHITE has 22×24 = 528 legal plays initially (border columns excluded)', () => {
    const g = new Game();
    // WHITE may not play at x=0 or x=23, so 22 columns × 24 rows = 528
    expect(g.legalPlays().length).toBe(22 * SIZE);
  });

  it('after WHITE plays, BLACK has 24×22 − 1 legal plays', () => {
    const g = new Game();
    g.play(pt(1, 1));
    // BLACK may not play at y=0 or y=23, so 24 columns × 22 rows = 528 minus the 1 already occupied
    expect(g.legalPlays().length).toBe(24 * 22 - 1);
  });

  it('WHITE cannot play at border columns x=0 or x=23', () => {
    const g = new Game();
    const legal = g.legalPlays();
    expect(legal.contains(pt(0, 5))).toBe(false);
    expect(legal.contains(pt(SIZE - 1, 5))).toBe(false);
  });

  it('BLACK cannot play at border rows y=0 or y=23', () => {
    const g = new Game();
    g.play(pt(5, 5)); // WHITE moves first
    const legal = g.legalPlays();
    expect(legal.contains(pt(5, 0))).toBe(false);
    expect(legal.contains(pt(5, SIZE - 1))).toBe(false);
  });
});

// -------------------------------------------------------------------------
// Playing moves
// -------------------------------------------------------------------------

describe('Game — playing moves', () => {
  it('alternates turns after each move', () => {
    const g = new Game();
    expect(g.turn).toBe(WHITE);
    g.play(pt(3, 3));
    expect(g.turn).toBe(BLACK);
    g.play(pt(3, 4));
    expect(g.turn).toBe(WHITE);
  });

  it('played cell is removed from both colors open pegs', () => {
    const g = new Game();
    g.play(pt(5, 5)); // WHITE plays
    // BLACK cannot play at (5,5) either
    expect(g.legalPlays().contains(pt(5, 5))).toBe(false);
  });

  it('records history in order', () => {
    const g = new Game();
    g.play(pt(1, 2));
    g.play(pt(3, 4));
    expect(g.history).toHaveLength(2);
    expect((g.history[0] as {x:number;y:number}).x).toBe(1);
    expect((g.history[1] as {x:number;y:number}).x).toBe(3);
  });

  it('does not place a link when there is no adjacent friendly peg', () => {
    const g = new Game();
    g.play(pt(5, 5)); // WHITE alone — no neighbor
    expect(allLinks(g)).toHaveLength(0);
  });

  it('places a link when two friendly pegs are a knight-move apart', () => {
    const g = new Game();
    g.play(pt(5, 5)); // WHITE move 1
    g.play(pt(2, 2)); // BLACK move (somewhere far)
    g.play(pt(7, 6)); // WHITE move 2 — knight (+2,+1) from (5,5)
    const links = allLinks(g).filter(l => l.color === WHITE);
    expect(links.length).toBeGreaterThanOrEqual(1);
  });
});

// -------------------------------------------------------------------------
// Undo
// -------------------------------------------------------------------------

describe('Game — undo', () => {
  it('restores turn after undo', () => {
    const g = new Game();
    g.play(pt(3, 3));
    expect(g.turn).toBe(BLACK);
    g.undo();
    expect(g.turn).toBe(WHITE);
  });

  it('restores legal plays after undo', () => {
    const g = new Game();
    const legalBefore = g.legalPlays().length;
    g.play(pt(3, 3));
    g.undo();
    expect(g.legalPlays().length).toBe(legalBefore);
    expect(g.legalPlays().contains(pt(3, 3))).toBe(true);
  });

  it('removes peg from board after undo', () => {
    const g = new Game();
    g.play(pt(3, 3));
    expect(pegCount(g, WHITE)).toBe(1);
    g.undo();
    expect(pegCount(g, WHITE)).toBe(0);
  });

  it('undo + redo reaches same board state', () => {
    const g = new Game();
    g.play(pt(3, 3));  // WHITE
    g.play(pt(5, 5));  // BLACK
    // Save state using named variables to avoid BLACK=0 / WHITE=1 index confusion
    const whitePegsBefore = new Int8Array(g.pegs[WHITE]);
    const blackPegsBefore = new Int8Array(g.pegs[BLACK]);
    g.undo(); // undo BLACK's (5,5); now BLACK to move again
    g.play(pt(5, 5)); // replay BLACK's (5,5)
    expect(Array.from(g.pegs[WHITE])).toEqual(Array.from(whitePegsBefore));
    expect(Array.from(g.pegs[BLACK])).toEqual(Array.from(blackPegsBefore));
  });
});

// -------------------------------------------------------------------------
// Win detection
// -------------------------------------------------------------------------

describe('Game — win detection', () => {
  it('reports no winner on fresh game', () => {
    const g = new Game();
    expect(g.isWinning(WHITE)).toBe(false);
    expect(g.isWinning(BLACK)).toBe(false);
    expect(g.justWon()).toBe(false);
  });

  it('WHITE wins by connecting y=0 row to y=23 row via a zigzag chain', () => {
    // WHITE chain using (+1,+2) knight moves from y=0 to y=22, then (+2,+1) to y=23.
    // Each consecutive pair is a valid knight move; no BLACK fillers cross the chain.
    const whiteChain = [
      {x:5,y:0}, {x:6,y:2}, {x:7,y:4}, {x:8,y:6}, {x:9,y:8},
      {x:10,y:10}, {x:11,y:12}, {x:12,y:14}, {x:13,y:16}, {x:14,y:18},
      {x:15,y:20}, {x:16,y:22}, {x:18,y:23},
    ];
    const blackFiller = [
      {x:1,y:2},{x:1,y:4},{x:1,y:6},{x:1,y:8},{x:1,y:10},
      {x:1,y:12},{x:1,y:14},{x:1,y:16},{x:1,y:18},{x:1,y:20},
      {x:1,y:22},{x:2,y:2},
    ];

    const g = new Game();
    for (let i = 0; i < whiteChain.length; i++) {
      g.play(pt(whiteChain[i].x, whiteChain[i].y));
      // Check win immediately after WHITE's last move (before any subsequent BLACK move)
      if (i === whiteChain.length - 1) break;
      if (i < blackFiller.length) g.play(pt(blackFiller[i].x, blackFiller[i].y));
    }

    // After WHITE plays (18,23), it is now BLACK's turn.
    // justWon() = isWinning(1 - BLACK) = isWinning(WHITE) ✓
    expect(g.isWinning(WHITE)).toBe(true);
    expect(g.justWon()).toBe(true);
  });

  it('win flag is cleared after undoing the winning move', () => {
    const whiteChain = [
      {x:5,y:0}, {x:6,y:2}, {x:7,y:4}, {x:8,y:6}, {x:9,y:8},
      {x:10,y:10}, {x:11,y:12}, {x:12,y:14}, {x:13,y:16}, {x:14,y:18},
      {x:15,y:20}, {x:16,y:22}, {x:18,y:23},
    ];
    const blackFiller = [
      {x:1,y:2},{x:1,y:4},{x:1,y:6},{x:1,y:8},{x:1,y:10},
      {x:1,y:12},{x:1,y:14},{x:1,y:16},{x:1,y:18},{x:1,y:20},
      {x:1,y:22},{x:2,y:2},
    ];

    const g = new Game();
    for (let i = 0; i < whiteChain.length; i++) {
      g.play(pt(whiteChain[i].x, whiteChain[i].y));
      if (i === whiteChain.length - 1) break;
      if (i < blackFiller.length) g.play(pt(blackFiller[i].x, blackFiller[i].y));
    }

    expect(g.isWinning(WHITE)).toBe(true);
    g.undo(); // undo WHITE's winning move (18,23)
    expect(g.isWinning(WHITE)).toBe(false);
  });
});

// -------------------------------------------------------------------------
// Clone independence
// -------------------------------------------------------------------------

describe('Game — clone', () => {
  it('clone is independent: mutating the clone does not affect original', () => {
    const g = new Game();
    g.play(pt(5, 5));
    const c = g.clone();
    c.play(pt(3, 3));  // mutate clone (BLACK's turn)
    expect(pegCount(g, BLACK)).toBe(0);  // original unaffected
    expect(pegCount(c, BLACK)).toBe(1);
  });

  it('clone has identical board state', () => {
    const g = new Game();
    g.play(pt(5, 5));
    g.play(pt(7, 7));
    const c = g.clone();
    expect(Array.from(c.pegs[WHITE])).toEqual(Array.from(g.pegs[WHITE]));
    expect(Array.from(c.pegs[BLACK])).toEqual(Array.from(g.pegs[BLACK]));
    expect(c.turn).toBe(g.turn);
    expect(c.history).toEqual(g.history);
  });
});

// -------------------------------------------------------------------------
// replayHistory
// -------------------------------------------------------------------------

describe('replayHistory', () => {
  it('produces identical state to playing moves manually', () => {
    const moves = [pt(5,5), pt(3,3), pt(7,6)];
    const g1 = new Game();
    for (const m of moves) g1.play(m);

    const g2 = replayHistory(moves);
    expect(Array.from(g2.pegs[WHITE])).toEqual(Array.from(g1.pegs[WHITE]));
    expect(Array.from(g2.pegs[BLACK])).toEqual(Array.from(g1.pegs[BLACK]));
    expect(g2.turn).toBe(g1.turn);
  });
});

// -------------------------------------------------------------------------
// Link blocking (crossing links)
// -------------------------------------------------------------------------

describe('Game — crossing link blocking', () => {
  it('does not form a link when it would cross an existing enemy link', () => {
    // WHITE at (5,5) and (7,6) form a link (knight +2,+1).
    // BLACK at (5,6) and (7,5) would form a link that crosses WHITE's.
    const g = new Game();
    g.play(pt(5, 5));  // WHITE
    g.play(pt(5, 6));  // BLACK
    g.play(pt(7, 6));  // WHITE — forms link (5,5)↔(7,6)
    g.play(pt(7, 5));  // BLACK — (5,6)↔(7,5) would cross WHITE's link

    const blackLinks = allLinks(g).filter(l => l.color === BLACK);
    const crossingLink = blackLinks.find(
      l => (l.p1.x === 5 && l.p1.y === 6 && l.p2.x === 7 && l.p2.y === 5) ||
           (l.p1.x === 7 && l.p1.y === 5 && l.p2.x === 5 && l.p2.y === 6)
    );
    expect(crossingLink).toBeUndefined();
  });
});
