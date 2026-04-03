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
