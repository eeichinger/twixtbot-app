/**
 * naf.test.ts — outside-in behavioral tests for the NAF encoding module.
 *
 * Tests focus on what callers observe: correct tensor shapes, legal-move
 * masks, policy index round-trips, and score conversion.
 */

import { describe, it, expect } from 'vitest';
import { Game, pt, SIZE, BLACK, WHITE } from '../twixt.js';
import {
  toInputArrays,
  policyIndexToPoint,
  policyPointToIndex,
  legalMovePolicyArray,
  threeToOne,
  top3FromScores,
} from '../naf.js';

const NUM_MOVES = SIZE * (SIZE - 2); // 528

// -------------------------------------------------------------------------
// toInputArrays — tensor shapes
// -------------------------------------------------------------------------

describe('toInputArrays — shapes', () => {
  it('pegs tensor has length 2 * SIZE * SIZE', () => {
    const { pegs } = toInputArrays(new Game());
    expect(pegs.length).toBe(2 * SIZE * SIZE);
  });

  it('links tensor has length 8 * SIZE * SIZE', () => {
    const { links } = toInputArrays(new Game());
    expect(links.length).toBe(8 * SIZE * SIZE);
  });

  it('locs tensor has length 2 * SIZE * SIZE', () => {
    const { locs } = toInputArrays(new Game());
    expect(locs.length).toBe(2 * SIZE * SIZE);
  });

  it('all tensor values are finite float32', () => {
    const { pegs, links, locs } = toInputArrays(new Game());
    for (const arr of [pegs, links, locs]) {
      for (let i = 0; i < arr.length; i++) {
        expect(isFinite(arr[i])).toBe(true);
      }
    }
  });
});

// -------------------------------------------------------------------------
// toInputArrays — peg encoding
// -------------------------------------------------------------------------

describe('toInputArrays — peg encoding (WHITE to move)', () => {
  it('empty board → all peg values are 0', () => {
    const { pegs } = toInputArrays(new Game());
    expect(Array.from(pegs).every(v => v === 0)).toBe(true);
  });

  it('after WHITE plays at (5,7), the pegs tensor encodes that peg for WHITE-to-move', () => {
    const g = new Game();
    g.play(pt(5, 7)); // WHITE plays; now BLACK to move
    g.play(pt(3, 3)); // BLACK plays; now WHITE to move
    const { pegs } = toInputArrays(g);
    // _nafFromWhite stores game.pegs[j] at naf plane 8+j, so:
    //   channel 0 of output = game.pegs[0] = BLACK pegs (BLACK=0)
    //   channel 1 of output = game.pegs[1] = WHITE pegs (WHITE=1)
    // WHITE peg at (5,7) → channel 1, flat index 1*SIZE*SIZE + 5*SIZE + 7
    const idx = 1 * SIZE * SIZE + 5 * SIZE + 7;
    expect(pegs[idx]).toBe(1);
    // And channel 0 at the same location is 0 (no BLACK peg there)
    expect(pegs[0 * SIZE * SIZE + 5 * SIZE + 7]).toBe(0);
  });
});

// -------------------------------------------------------------------------
// policyIndexToPoint / policyPointToIndex — round-trip
// -------------------------------------------------------------------------

describe('policy index ↔ point round-trip', () => {
  it('WHITE: all 528 indices round-trip through policyIndexToPoint → policyPointToIndex', () => {
    for (let i = 0; i < NUM_MOVES; i++) {
      const p = policyIndexToPoint(WHITE, i);
      expect(policyPointToIndex(WHITE, p)).toBe(i);
    }
  });

  it('BLACK: all 528 indices round-trip through policyIndexToPoint → policyPointToIndex', () => {
    for (let i = 0; i < NUM_MOVES; i++) {
      const p = policyIndexToPoint(BLACK, i);
      expect(policyPointToIndex(BLACK, p)).toBe(i);
    }
  });

  it('WHITE points are restricted to x in [1, SIZE-2]', () => {
    for (let i = 0; i < NUM_MOVES; i++) {
      const p = policyIndexToPoint(WHITE, i);
      expect(p.x).toBeGreaterThanOrEqual(1);
      expect(p.x).toBeLessThanOrEqual(SIZE - 2);
    }
  });

  it('BLACK points are restricted to y in [1, SIZE-2]', () => {
    for (let i = 0; i < NUM_MOVES; i++) {
      const p = policyIndexToPoint(BLACK, i);
      expect(p.y).toBeGreaterThanOrEqual(1);
      expect(p.y).toBeLessThanOrEqual(SIZE - 2);
    }
  });
});

// -------------------------------------------------------------------------
// legalMovePolicyArray
// -------------------------------------------------------------------------

describe('legalMovePolicyArray', () => {
  it('fresh WHITE game: 528 legal moves → 528 ones in mask', () => {
    const mask = legalMovePolicyArray(new Game());
    expect(mask.length).toBe(NUM_MOVES);
    const ones = Array.from(mask).filter(v => v === 1).length;
    expect(ones).toBe(NUM_MOVES);
  });

  it('after one WHITE move the mask has one fewer legal move', () => {
    const g = new Game();
    g.play(pt(5, 5)); // WHITE plays; now BLACK to move
    const mask = legalMovePolicyArray(g);
    const ones = Array.from(mask).filter(v => v === 1).length;
    // BLACK has 24*22 - 1 legal moves; but the mask is 528 entries keyed by BLACK's policy layout
    expect(ones).toBe(NUM_MOVES - 1);
  });

  it('played cell has mask value 0', () => {
    const g = new Game();
    g.play(pt(5, 5)); // WHITE plays
    g.play(pt(3, 3)); // BLACK plays; now WHITE to move
    const mask = legalMovePolicyArray(g);
    // (5,5) was played by WHITE; it must be 0 in WHITE's mask
    const idx = policyPointToIndex(WHITE, pt(5, 5));
    expect(mask[idx]).toBe(0);
  });

  it('mask only contains 0s and 1s', () => {
    const mask = legalMovePolicyArray(new Game());
    for (const v of mask) {
      expect(v === 0 || v === 1).toBe(true);
    }
  });
});

// -------------------------------------------------------------------------
// threeToOne
// -------------------------------------------------------------------------

// -------------------------------------------------------------------------
// top3FromScores
// -------------------------------------------------------------------------

describe('top3FromScores', () => {
  const NUM_MOVES = SIZE * (SIZE - 2); // 528

  it('returns empty array when all scores are zero', () => {
    const scores = new Float64Array(NUM_MOVES);
    expect(top3FromScores(scores, WHITE, null)).toHaveLength(0);
  });

  it('returns at most 3 entries even when many moves are visited', () => {
    const scores = new Float64Array(NUM_MOVES);
    for (let i = 0; i < NUM_MOVES; i++) scores[i] = i + 1;  // all non-zero
    expect(top3FromScores(scores, WHITE, null)).toHaveLength(3);
  });

  it('sorts entries by visit count descending', () => {
    const scores = new Float64Array(NUM_MOVES);
    scores[0] = 10;
    scores[1] = 30;
    scores[2] = 20;
    const result = top3FromScores(scores, WHITE, null);
    expect(result[0].pct).toBeGreaterThan(result[1].pct);
    expect(result[1].pct).toBeGreaterThan(result[2].pct);
  });

  it('pct values sum to approximately 100 when exactly 3 moves are visited', () => {
    const scores = new Float64Array(NUM_MOVES);
    scores[0] = 50;
    scores[1] = 30;
    scores[2] = 20;
    const result = top3FromScores(scores, WHITE, null);
    const total = result.reduce((s, m) => s + m.pct, 0);
    expect(total).toBeCloseTo(100);
  });

  it('maps index to correct (x,y) for WHITE', () => {
    const scores = new Float64Array(NUM_MOVES);
    scores[0] = 100;  // WHITE: index 0 → (1, 0)
    const result = top3FromScores(scores, WHITE, null);
    expect(result[0].x).toBe(1);
    expect(result[0].y).toBe(0);
  });

  it('maps index to correct (x,y) for BLACK', () => {
    const scores = new Float64Array(NUM_MOVES);
    scores[0] = 100;  // BLACK: index 0 → (0, 1)
    const result = top3FromScores(scores, BLACK, null);
    expect(result[0].x).toBe(0);
    expect(result[0].y).toBe(1);
  });

  it('reads Q values from qValues array when provided', () => {
    const scores = new Float64Array(NUM_MOVES);
    scores[5] = 100;
    const qValues = new Float64Array(NUM_MOVES);
    qValues[5] = 0.75;
    const result = top3FromScores(scores, WHITE, qValues);
    expect(result[0].q).toBeCloseTo(0.75);
  });

  it('returns q=0 when qValues is null', () => {
    const scores = new Float64Array(NUM_MOVES);
    scores[0] = 100;
    const result = top3FromScores(scores, WHITE, null);
    expect(result[0].q).toBe(0);
  });

  it('returns only moves with > 0 visits', () => {
    const scores = new Float64Array(NUM_MOVES);
    scores[10] = 5;
    scores[20] = 3;
    const result = top3FromScores(scores, WHITE, null);
    expect(result).toHaveLength(2);
  });
});

describe('threeToOne', () => {
  it('equal logits → score of 0', () => {
    expect(threeToOne([0, 0, 0])).toBeCloseTo(0);
  });

  it('very high Win logit → score near +1', () => {
    expect(threeToOne([-100, 0, 100])).toBeCloseTo(1, 3);
  });

  it('very high Loss logit → score near -1', () => {
    expect(threeToOne([100, 0, -100])).toBeCloseTo(-1, 3);
  });

  it('score is in range [-1, 1] for arbitrary logits', () => {
    const cases: [number, number, number][] = [
      [1, 2, 3], [-5, 0, 5], [10, 10, 10], [0, -3, 7],
    ];
    for (const c of cases) {
      const s = threeToOne(c);
      expect(s).toBeGreaterThanOrEqual(-1);
      expect(s).toBeLessThanOrEqual(1);
    }
  });
});
