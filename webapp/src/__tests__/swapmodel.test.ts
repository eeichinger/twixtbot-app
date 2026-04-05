/**
 * swapmodel.test.ts — tests for the TypeScript port of src/swapmodel.py
 *
 * Expected values are derived by running the Python swapmodel directly:
 *   python -c "import twixt, swapmodel; p=twixt.Point('g7'); print(swapmodel._point_score(p), swapmodel.want_swap(p))"
 *
 * Coordinate note: twixt.Point("a1") = (x=0,y=0) in 0-indexed coords.
 * In the Python source SIZE=24, S2=12.  Column letters a–x map to x=0–23.
 */

import { describe, it, expect } from 'vitest';
import { pt } from '../twixt.js';
import { swapScore, wantSwap } from '../swapmodel.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Round to 6 decimal places for float comparison. */
function r6(n: number): number {
  return Math.round(n * 1e6) / 1e6;
}

// ---------------------------------------------------------------------------
// swapScore — numeric accuracy against Python reference values
//
// Python reference: score = dot(betas, [1, xres, yres, xres*yres])
//   betas = [0.494481, -0.00366079, 0.0225597, 0.00114293]
//   fold:  if x>=12, x=23-x;  if y>=12, y=23-y
//   xres = x - 6.0,  yres = y - 5.5
// ---------------------------------------------------------------------------

describe('swapScore — folding and arithmetic', () => {
  it('centre point (12,12): folded to (11,11), score > 0.5', () => {
    // x=12 → 23-12=11, y=12 → 23-12=11
    // xres=5, yres=5.5
    // score = 0.494481 + (-0.00366079)*5 + 0.0225597*5.5 + 0.00114293*5*5.5
    //       = 0.494481 - 0.01830395 + 0.12407835 + 0.031430575 ≈ 0.631686
    const s = swapScore(pt(12, 12));
    expect(r6(s)).toBeCloseTo(0.631686, 4);
    expect(s).toBeGreaterThan(0.5);
  });

  it('corner-like point (1,1): score < 0.5', () => {
    // x=1, y=1 (no fold needed)
    // xres=-5, yres=-4.5
    // score = 0.494481 + 0.0183039 - 0.10151865 + 0.025716075
    //       = 0.437000325
    const s = swapScore(pt(1, 1));
    expect(r6(s)).toBeCloseTo(0.437, 3);
    expect(s).toBeLessThan(0.5);
  });

  it('(6,6): xres=0, yres=0.5 → small positive score just above 0.5', () => {
    // xres=0, yres=0.5
    // score = 0.494481 + 0 + 0.0225597*0.5 + 0 = 0.494481 + 0.01127985 = 0.50576085
    const s = swapScore(pt(6, 6));
    expect(r6(s)).toBeCloseTo(0.505761, 5);
    expect(s).toBeGreaterThan(0.5);
  });

  it('(6,5): xres=0, yres=-0.5 → score just below 0.5', () => {
    // xres=0, yres=-0.5
    // score = 0.494481 - 0.01127985 = 0.48320115
    const s = swapScore(pt(6, 5));
    expect(r6(s)).toBeCloseTo(0.483201, 5);
    expect(s).toBeLessThan(0.5);
  });

  it('symmetry: (5,5) and (18,18) have the same score (board symmetry fold)', () => {
    // 18 = 23-5
    const s1 = swapScore(pt(5, 5));
    const s2 = swapScore(pt(18, 18));
    expect(r6(s1)).toBeCloseTo(r6(s2), 10);
  });

  it('symmetry: (3,7) and (20,7) have the same score (x-fold)', () => {
    // 20 = 23-3
    const s1 = swapScore(pt(3, 7));
    const s2 = swapScore(pt(20, 7));
    expect(r6(s1)).toBeCloseTo(r6(s2), 10);
  });

  it('symmetry: (7,3) and (7,20) have the same score (y-fold)', () => {
    const s1 = swapScore(pt(7, 3));
    const s2 = swapScore(pt(7, 20));
    expect(r6(s1)).toBeCloseTo(r6(s2), 10);
  });

  it('board-edge column (12,0): folded x=11, y=0, score < 0.5 (weak peg near row edge)', () => {
    // x=12→11, y=0 (no fold)
    // xres=5, yres=-5.5
    // score = 0.494481 - 0.0183039 - 0.12407835 - 0.031430575 = 0.320567075
    const s = swapScore(pt(12, 0));
    expect(s).toBeLessThan(0.5);
  });
});

// ---------------------------------------------------------------------------
// wantSwap — boolean decision
// ---------------------------------------------------------------------------

describe('wantSwap — swap/no-swap decisions', () => {
  it('returns true for a strong centre peg (12,12)', () => {
    expect(wantSwap(pt(12, 12))).toBe(true);
  });

  it('returns false for a weak corner-adjacent peg (1,1)', () => {
    expect(wantSwap(pt(1, 1))).toBe(false);
  });

  it('returns true for (6,6) — just above the decision boundary', () => {
    expect(wantSwap(pt(6, 6))).toBe(true);
  });

  it('returns false for (6,5) — just below the decision boundary', () => {
    expect(wantSwap(pt(6, 5))).toBe(false);
  });

  it('returns false for a board-edge column peg (12,0)', () => {
    expect(wantSwap(pt(12, 0))).toBe(false);
  });

  it('returns true for (1,12) — weak x but strong y after fold gives score ≈ 0.606', () => {
    // x=1 (no fold), y=12 → fold to 23-12=11
    // xres = 1-6 = -5, yres = 11-5.5 = 5.5
    // score = 0.494481 + 0.0183039 + 0.12407835 - 0.031430575 ≈ 0.606 > 0.5
    expect(wantSwap(pt(1, 12))).toBe(true);
  });

  it('returns false for far-edge peg (1,0) — both coordinates unfavourable', () => {
    // x=1, y=0; xres=-5, yres=-5.5
    // score = 0.494481 + 0.0183039 - 0.12407835 + 0.031430575 ≈ 0.420 < 0.5
    expect(wantSwap(pt(1, 0))).toBe(false);
  });
});
