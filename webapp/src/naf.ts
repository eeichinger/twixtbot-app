/**
 * naf.ts — Neural net encoding helpers
 * Minimal port of src/naf.py: only what the AI player needs at runtime.
 * (No LearningState, no hflip/vflip, no serialisation.)
 */

import { Game, Point, pt, SIZE, BLACK, WHITE, DLINKS } from './twixt.js';

const LINK_LONGY    = 4;
const LINK_DIFFSIGN = 2;

// ---------------------------------------------------------------------------
// location_inputs — returns a flat Float32Array for the locs input [1,2,S,S]
// ---------------------------------------------------------------------------

/** Returns a flat Float32Array of shape [2, SIZE, SIZE] (NCHW) with
 *  channel 0 = x/SIZE ramp, channel 1 = y/SIZE ramp. */
export function locationInputs(): Float32Array {
  const out = new Float32Array(2 * SIZE * SIZE);
  for (let x = 0; x < SIZE; x++) {
    for (let y = 0; y < SIZE; y++) {
      out[0 * SIZE * SIZE + x * SIZE + y] = x / SIZE;  // channel 0: x ramp
      out[1 * SIZE * SIZE + x * SIZE + y] = y / SIZE;  // channel 1: y ramp
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// toInputArrays — encode Game → {pegs, links, locs}  (NCHW Float32, batch=1)
// ---------------------------------------------------------------------------

export interface NetInputArrays {
  pegs:  Float32Array;   // [1, 2, SIZE, SIZE]
  links: Float32Array;   // [1, 8, SIZE, SIZE]
  locs:  Float32Array;   // [1, 2, SIZE, SIZE]
}

/**
 * Encode a Game state into the three float32 input tensors expected by TwixNet.
 * Always encodes from the perspective of the side to move ("white on play"),
 * applying the board transpose for BLACK exactly as init_from_game_black does.
 */
export function toInputArrays(game: Game): NetInputArrays {
  // naf[x, y, plane] uint8 — built from game state
  // planes 0-7: links; planes 8-9: pegs
  const naf = new Uint8Array(SIZE * SIZE * 10);

  if (game.turn === WHITE) {
    _nafFromWhite(game, naf);
  } else {
    _nafFromBlack(game, naf);
  }

  // links input: copy link planes and apply the "shift toward left peg" transform
  // (mirrors the NumPy slice ops in old_naf_to_ninputs / to_input_arrays)
  const linksFlat = new Float32Array(8 * SIZE * SIZE);
  for (let i = 0; i < 8; i++) {
    // Copy plane i from naf
    const plane = new Float32Array(SIZE * SIZE);
    for (let x = 0; x < SIZE; x++) {
      for (let y = 0; y < SIZE; y++) {
        plane[x * SIZE + y] = naf[(x * SIZE + y) * 10 + i];
      }
    }

    const vertical  = (i & LINK_LONGY)    !== 0;
    const diffsign  = (i & LINK_DIFFSIGN) !== 0;

    // Shift down (rows): links[1:,:] = links[:-1,:]; links[0,:] = 0
    if (vertical || diffsign) {
      for (let x = SIZE - 1; x >= 1; x--) {
        for (let y = 0; y < SIZE; y++) {
          plane[x * SIZE + y] = plane[(x - 1) * SIZE + y];
        }
      }
      for (let y = 0; y < SIZE; y++) plane[y] = 0;
    }
    // Shift right (cols): links[:,1:] = links[:,:-1]; links[:,0] = 0
    if (!vertical || diffsign) {
      for (let x = 0; x < SIZE; x++) {
        for (let y = SIZE - 1; y >= 1; y--) {
          plane[x * SIZE + y] = plane[x * SIZE + (y - 1)];
        }
        plane[x * SIZE] = 0;
      }
    }

    // Copy into output NCHW tensor (channel i)
    linksFlat.set(plane, i * SIZE * SIZE);
  }

  // pegs input: planes 8 and 9 from naf
  const pegsFlat = new Float32Array(2 * SIZE * SIZE);
  for (let c = 0; c < 2; c++) {
    for (let x = 0; x < SIZE; x++) {
      for (let y = 0; y < SIZE; y++) {
        pegsFlat[c * SIZE * SIZE + x * SIZE + y] = naf[(x * SIZE + y) * 10 + 8 + c];
      }
    }
  }

  return {
    pegs:  pegsFlat,
    links: linksFlat,
    locs:  locationInputs(),
  };
}

// Fill naf in [x, y, plane] order for WHITE to move (no transposition needed)
function _nafFromWhite(game: Game, naf: Uint8Array): void {
  for (let i = 0; i < 8; i++) {
    for (let x = 0; x < SIZE; x++) {
      for (let y = 0; y < SIZE; y++) {
        naf[(x * SIZE + y) * 10 + i] = game.links[i][x * SIZE + y];
      }
    }
  }
  for (let j = 0; j < 2; j++) {
    for (let x = 0; x < SIZE; x++) {
      for (let y = 0; y < SIZE; y++) {
        naf[(x * SIZE + y) * 10 + 8 + j] = game.pegs[j][x * SIZE + y];
      }
    }
  }
}

// Fill naf for BLACK to move: transpose + remap link planes
// Mirrors init_from_game_black in Python
function _nafFromBlack(game: Game, naf: Uint8Array): void {
  // LONGY changes but DIFFSIGN and color do not.
  // Mapping (acolor = 1-color):
  //   naf[:,:,4+color] = links[0+acolor].T
  //   naf[:,:,6+color] = links[2+acolor].T
  //   naf[:,:,0+color] = links[4+acolor].T
  //   naf[:,:,2+color] = links[6+acolor].T
  //   naf[:,:,8+color] = pegs[acolor].T
  for (let color = 0; color < 2; color++) {
    const acolor = 1 - color;

    const linkMappings: [number, number][] = [
      [4 + color, 0 + acolor],
      [6 + color, 2 + acolor],
      [0 + color, 4 + acolor],
      [2 + color, 6 + acolor],
    ];
    for (const [dst, src] of linkMappings) {
      for (let x = 0; x < SIZE; x++) {
        for (let y = 0; y < SIZE; y++) {
          // Transpose: src[x,y] → dst[y,x]
          naf[(y * SIZE + x) * 10 + dst] = game.links[src][x * SIZE + y];
        }
      }
    }
    // pegs
    for (let x = 0; x < SIZE; x++) {
      for (let y = 0; y < SIZE; y++) {
        naf[(y * SIZE + x) * 10 + 8 + color] = game.pegs[acolor][x * SIZE + y];
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Policy index ↔ Point
// ---------------------------------------------------------------------------

/**
 * Convert a flat policy array index (0..527) to a board Point.
 * Policy layout (always from the perspective of the side to move):
 *   WHITE: index = (x-1)*SIZE + y   for x in [1..SIZE-2]
 *   BLACK: index = (y-1)*SIZE + x   for y in [1..SIZE-2]
 */
export function policyIndexToPoint(color: number, index: number): Point {
  const major = Math.floor(index / SIZE);
  const minor = index % SIZE;
  if (color === WHITE) return pt(major + 1, minor);
  else                 return pt(minor, major + 1);
}

export function policyPointToIndex(color: number, p: Point): number {
  let major: number, minor: number;
  if (color === WHITE) { major = p.x - 1; minor = p.y; }
  else                 { major = p.y - 1; minor = p.x; }
  return major * SIZE + minor;
}

// ---------------------------------------------------------------------------
// Legal move mask
// ---------------------------------------------------------------------------

/** Returns Float32Array[528]: 1 at each legal move index, 0 elsewhere. */
export function legalMovePolicyArray(game: Game): Float32Array {
  const out = new Float32Array(SIZE * (SIZE - 2));
  const color = game.turn;
  const open = game.legalPlays();
  for (const p of open) {
    const idx = policyPointToIndex(color, p);
    if (idx >= 0 && idx < out.length) out[idx] = 1;
  }
  return out;
}

// ---------------------------------------------------------------------------
// Top-3 candidate move extraction
// ---------------------------------------------------------------------------

export type Top3Move = { x: number; y: number; pct: number; q: number };

/**
 * Extract the top-3 most-visited moves from an MCTS visit-count array.
 *
 * @param scores  Float64Array[528] of visit counts (indices match policy space)
 * @param color   The side to move (WHITE or BLACK) — used for index→point mapping
 * @param qValues Float64Array[528] of Q-values from mcts.root.Q, or null if unavailable
 * @returns       Up to 3 entries sorted by visit count descending; empty if no visits
 */
export function top3FromScores(
  scores: Float64Array,
  color: number,
  qValues: Float64Array | null,
): Top3Move[] {
  let total = 0;
  for (let i = 0; i < scores.length; i++) total += scores[i];

  const indices: number[] = [];
  for (let i = 0; i < scores.length; i++) {
    if (scores[i] > 0) indices.push(i);
  }
  indices.sort((a, b) => scores[b] - scores[a]);

  return indices.slice(0, 3).map(idx => {
    const p = policyIndexToPoint(color, idx);
    return {
      x: p.x,
      y: p.y,
      pct: total > 0 ? (scores[idx] / total) * 100 : 0,
      q: qValues ? qValues[idx] : 0,
    };
  });
}

// ---------------------------------------------------------------------------
// Value: 3-class logits → scalar score in [-1, 1]
// ---------------------------------------------------------------------------

/** Convert (Loss, Draw, Win) logits to a scalar score.
 *  Exact port of three_to_one() from naf.py. */
export function threeToOne(logits: Float32Array | [number, number, number]): number {
  const lL = logits[0], lD = logits[1], lW = logits[2];
  const eL = Math.exp(lL - lD);
  const eW = Math.exp(lW - lD);
  const div = 1.0 + eL + eW;
  return eW / div - eL / div;
}
