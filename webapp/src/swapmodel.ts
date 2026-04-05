/**
 * swapmodel.ts — Port of src/swapmodel.py
 *
 * Fitted linear model to decide whether BLACK should swap WHITE's first peg.
 * Derived from bot self-play: positions with score > 0.5 are strong enough for
 * WHITE that BLACK should swap to take them.
 *
 * Model: score = β₀ + β₁·xres + β₂·yres + β₃·xres·yres
 *   where (xres, yres) = first-peg coords folded to first quadrant, centred on (6, 5.5)
 *
 * Coefficients (from swapmodel.py, fitted on self-play data):
 *   β = [0.494481, −0.00366079, 0.0225597, 0.00114293]
 */

import { SIZE } from './twixt.js';
import type { Point } from './twixt.js';

const S2 = SIZE >> 1; // 12

// Regression coefficients: [intercept, xres, yres, xres*yres]
const BETAS = [0.494481, -0.00366079, 0.0225597, 0.00114293] as const;

/**
 * Compute the swap score for a given first-peg position.
 * Returns a value in roughly [0.3, 0.7]; > 0.5 means BLACK should swap.
 */
export function swapScore(p: Point): number {
  let x = p.x;
  let y = p.y;

  // Fold to first quadrant via board symmetry (same as Python: 2*S2 - coord - 1 = SIZE-1-coord)
  if (x >= S2) x = SIZE - 1 - x;
  if (y >= S2) y = SIZE - 1 - y;

  const xres = x - 6.0;
  const yres = y - 5.5;

  return BETAS[0] + BETAS[1] * xres + BETAS[2] * yres + BETAS[3] * xres * yres;
}

/**
 * Returns true if BLACK should swap WHITE's first-peg position.
 * Used by the AI worker when it is playing as BLACK at move 2.
 */
export function wantSwap(firstPeg: Point): boolean {
  return swapScore(firstPeg) > 0.50;
}
