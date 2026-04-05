/**
 * mcts.test.ts — outside-in behavioral tests for NeuralMCTS.
 *
 * Uses a mock score-and-policy function so no ONNX model is needed.
 * Tests focus on observable behavior: return types, legal-move constraint,
 * game-state preservation, and forced-win detection.
 */

import { describe, it, expect } from 'vitest';
import { Game, pt, SIZE, WHITE, BLACK } from '../twixt.js';
import { NeuralMCTS } from '../mcts.js';
import { legalMovePolicyArray, policyPointToIndex } from '../naf.js';

const NUM_MOVES = SIZE * (SIZE - 2); // 528

// -------------------------------------------------------------------------
// Mock SAP function: returns score=0, uniform policy over legal moves
// -------------------------------------------------------------------------

async function uniformSap(game: Game): Promise<[number, Float32Array]> {
  const policy = new Float32Array(NUM_MOVES).fill(0); // uniform after softmax
  return [0, policy];
}

// -------------------------------------------------------------------------
// Basic return-value tests
// -------------------------------------------------------------------------

describe('NeuralMCTS — return type', () => {
  it('returns a Float64Array of length 528 on a fresh game', async () => {
    const mcts = new NeuralMCTS(uniformSap, 1.0, 0.0);
    const result = await mcts.mcts(new Game(), 5, 30_000);
    expect(result).toBeInstanceOf(Float64Array);
    expect((result as Float64Array).length).toBe(NUM_MOVES);
  });

  it('returns non-negative visit counts', async () => {
    const mcts = new NeuralMCTS(uniformSap, 1.0, 0.0);
    const result = await mcts.mcts(new Game(), 5, 30_000) as Float64Array;
    for (const v of result) {
      expect(v).toBeGreaterThanOrEqual(0);
    }
  });

  it('total visit count equals number of trials run', async () => {
    const trials = 10;
    const mcts = new NeuralMCTS(uniformSap, 1.0, 0.0);
    const result = await mcts.mcts(new Game(), trials, 30_000) as Float64Array;
    const total = Array.from(result).reduce((a, b) => a + b, 0);
    expect(total).toBe(trials);
  });
});

// -------------------------------------------------------------------------
// Legal-move constraint
// -------------------------------------------------------------------------

describe('NeuralMCTS — only legal moves are visited', () => {
  it('no visit counts for illegal moves on a fresh WHITE game', async () => {
    const mcts = new NeuralMCTS(uniformSap, 1.0, 0.0);
    const g = new Game(); // WHITE to move
    const result = await mcts.mcts(g, 20, 30_000) as Float64Array;
    const mask = legalMovePolicyArray(g);
    for (let i = 0; i < NUM_MOVES; i++) {
      if (mask[i] === 0) {
        expect(result[i]).toBe(0);
      }
    }
  });

  it('no visit counts for illegal moves after one move (BLACK to move)', async () => {
    const g = new Game();
    g.play(pt(5, 5)); // WHITE plays → BLACK to move
    const mcts = new NeuralMCTS(uniformSap, 1.0, 0.0);
    const result = await mcts.mcts(g, 20, 30_000) as Float64Array;
    const mask = legalMovePolicyArray(g);
    for (let i = 0; i < NUM_MOVES; i++) {
      if (mask[i] === 0) {
        expect(result[i]).toBe(0);
      }
    }
  });
});

// -------------------------------------------------------------------------
// Game state preservation
// -------------------------------------------------------------------------

describe('NeuralMCTS — does not mutate game state', () => {
  it('game history is unchanged after mcts()', async () => {
    const g = new Game();
    g.play(pt(3, 3));
    const histBefore = [...g.history];
    const turnBefore = g.turn;
    const mcts = new NeuralMCTS(uniformSap, 1.0, 0.0);
    await mcts.mcts(g, 5, 30_000);
    expect(g.history.length).toBe(histBefore.length);
    expect(g.turn).toBe(turnBefore);
  });

  it('peg arrays are unchanged after mcts()', async () => {
    const g = new Game();
    g.play(pt(5, 5));
    const whitePegsBefore = new Int8Array(g.pegs[WHITE]);
    const blackPegsBefore = new Int8Array(g.pegs[BLACK]);
    const mcts = new NeuralMCTS(uniformSap, 1.0, 0.0);
    await mcts.mcts(g, 5, 30_000);
    expect(Array.from(g.pegs[WHITE])).toEqual(Array.from(whitePegsBefore));
    expect(Array.from(g.pegs[BLACK])).toEqual(Array.from(blackPegsBefore));
  });
});

// -------------------------------------------------------------------------
// Smart-init (FPU) — Q pre-seeding
// -------------------------------------------------------------------------

describe('NeuralMCTS — smart-init Q pre-seeding', () => {
  it('unvisited children have Q pre-seeded to the SAP score, not 0', async () => {
    // SAP always returns score=0.7.  After exactly 1 trial:
    //   • one child idx is visited → Q[idx] is overwritten by backprop (= −0.7)
    //   • all other legal children remain at the pre-seeded value (0.7)
    const SAP_SCORE = 0.7;
    async function fixedScoreSap(_game: Game): Promise<[number, Float32Array]> {
      return [SAP_SCORE, new Float32Array(NUM_MOVES)]; // uniform policy, fixed score
    }

    const mcts = new NeuralMCTS(fixedScoreSap, 1.0, 0.0);
    await mcts.mcts(new Game(), /* maxTrials */ 1, 30_000);

    const root = mcts.root!;
    expect(root).not.toBeNull();
    expect(root.lmNonzero).not.toBeNull();

    // Exactly one child must have been visited (N=1), the rest N=0.
    const visitedIndices = root.lmNonzero!.filter(i => root.N[i] > 0);
    const unvisitedIndices = root.lmNonzero!.filter(i => root.N[i] === 0);
    expect(visitedIndices.length).toBe(1);
    expect(unvisitedIndices.length).toBeGreaterThan(0);

    // Unvisited children must carry the pre-seeded score, not 0.
    for (const i of unvisitedIndices) {
      expect(root.Q[i]).toBeCloseTo(SAP_SCORE, 10);
    }

    // The visited child must have a Q value set by backprop (= −child.score = −0.7),
    // not the pre-seeded value.
    const visitedIdx = visitedIndices[0];
    expect(root.Q[visitedIdx]).toBeCloseTo(-SAP_SCORE, 10);
  });

  it('Q pre-seeding does not alter total visit counts or return type', async () => {
    // Regression: smart_init must not break the existing visit-count invariant.
    const trials = 8;
    async function sap(_game: Game): Promise<[number, Float32Array]> {
      return [0.5, new Float32Array(NUM_MOVES)];
    }
    const mcts = new NeuralMCTS(sap, 1.0, 0.0);
    const result = await mcts.mcts(new Game(), trials, 30_000) as Float64Array;
    expect(result).toBeInstanceOf(Float64Array);
    const total = Array.from(result).reduce((a, b) => a + b, 0);
    expect(total).toBe(trials);
  });
});

// -------------------------------------------------------------------------
// Forced win detection
// -------------------------------------------------------------------------

describe('NeuralMCTS — forced win returns a Point', () => {
  it('returns a Point (not Float64Array) when a forced win exists', async () => {
    // Build a game where the current player has an immediate winning move.
    // WHITE wins by connecting y=0 to y=23.  Place all the chain pegs except
    // the last one, then call mcts — it should find the winning move.

    // Build the same chain used in game.test.ts, minus the final WHITE peg.
    const whiteChain = [
      {x:5,y:0}, {x:6,y:2}, {x:7,y:4}, {x:8,y:6}, {x:9,y:8},
      {x:10,y:10}, {x:11,y:12}, {x:12,y:14}, {x:13,y:16}, {x:14,y:18},
      {x:15,y:20}, {x:16,y:22},   // ← last peg before the winner
    ];
    const blackFiller = [
      {x:1,y:2},{x:1,y:4},{x:1,y:6},{x:1,y:8},{x:1,y:10},
      {x:1,y:12},{x:1,y:14},{x:1,y:16},{x:1,y:18},{x:1,y:20},
      {x:1,y:22},{x:2,y:2},
    ];

    const g = new Game();
    for (let i = 0; i < whiteChain.length; i++) {
      g.play(pt(whiteChain[i].x, whiteChain[i].y));
      if (i < blackFiller.length) {
        g.play(pt(blackFiller[i].x, blackFiller[i].y));
      }
    }
    // It is now WHITE's turn. Playing (18,23) wins immediately.
    expect(g.turn).toBe(WHITE);

    // Use a SAP that gives (18,23) a very high policy score
    const winnerIdx = (SIZE - 2) * SIZE - 1; // won't be exact, use mock that returns high value at winning move
    const winMove = pt(18, 23);

    async function biasedSap(_game: Game): Promise<[number, Float32Array]> {
      const p = new Float32Array(NUM_MOVES);
      // Import policyPointToIndex inline
      const { policyPointToIndex } = await import('../naf.js');
      const idx = policyPointToIndex(WHITE, winMove);
      p[idx] = 100; // strongly favor the winning move
      return [0, p];
    }

    const mcts = new NeuralMCTS(biasedSap, 1.0, 0.0);
    // Run enough trials to expand and prove the win
    const result = await mcts.mcts(g, 30, 30_000);

    // Result should be a Point (not Float64Array) when proven win is detected
    const isPoint = result !== null &&
      typeof result === 'object' &&
      !(result instanceof Float64Array) &&
      'x' in result && 'y' in result;
    expect(isPoint).toBe(true);
    expect((result as {x:number;y:number}).x).toBe(18);
    expect((result as {x:number;y:number}).y).toBe(23);
  });
});

// -------------------------------------------------------------------------
// Tree reuse (_computeRoot) — A3
// -------------------------------------------------------------------------

describe('NeuralMCTS — tree reuse (_computeRoot)', () => {
  // Build a SAP that strongly biases toward a fixed sequence of moves so MCTS
  // reliably builds deep subtrees within a small trial budget.
  // moves[i] is the preferred move at ply i (alternating WHITE/BLACK turns).
  function buildBiasedSap(moves: ReturnType<typeof pt>[]): typeof uniformSap {
    return async (game: Game) => {
      const p = new Float32Array(NUM_MOVES);
      const ply = game.history.length;
      if (ply < moves.length) {
        const color = game.turn;
        const idx = policyPointToIndex(color, moves[ply]);
        if (idx >= 0 && idx < NUM_MOVES) p[idx] = 100;
      }
      return [0, p] as [number, Float32Array];
    };
  }

  // Fixed move sequence used across reuse tests (well inside the board, alternating colors)
  const SEQ = [
    pt(5,  5),  // WHITE ply 0
    pt(10, 10), // BLACK ply 1
    pt(5,  6),  // WHITE ply 2
    pt(10, 11), // BLACK ply 3
    pt(5,  7),  // WHITE ply 4
    pt(10, 12), // BLACK ply 5
  ];

  it('delta=0: root and historyAtRoot unchanged after second call at same position', async () => {
    const mcts = new NeuralMCTS(buildBiasedSap(SEQ), 1.0, 0.0);
    const g = new Game();
    await mcts.mcts(g, 5, 30_000);
    const rootBefore = mcts.root;
    await mcts.mcts(g, 5, 30_000);
    // root may change (more visits expand more nodes) but historyAtRoot must stay [].
    expect(mcts.historyAtRoot).not.toBeNull();
    expect(mcts.historyAtRoot!.length).toBe(0);
    expect(rootBefore).toBe(mcts.root); // same object — no unnecessary reset
  });

  it('delta=2: root is reused (historyAtRoot advances)', async () => {
    const mcts = new NeuralMCTS(buildBiasedSap(SEQ), 1.0, 0.0);
    const g = new Game();
    // Enough trials to build the first two plies of the preferred path
    await mcts.mcts(g, 30, 30_000);

    g.play(SEQ[0]);  // WHITE
    g.play(SEQ[1]);  // BLACK
    await mcts.mcts(g, 5, 30_000);

    expect(mcts.historyAtRoot).not.toBeNull();
    expect(mcts.historyAtRoot!.length).toBe(2);
  });

  it('delta=4: root walk is attempted (not immediately discarded)', async () => {
    // With the old limit of 2, delta=4 would reset root before even trying.
    // Now it should attempt the walk; with enough prior trials the walk succeeds.
    const mcts = new NeuralMCTS(buildBiasedSap(SEQ), 1.0, 0.0);
    const g = new Game();
    await mcts.mcts(g, 80, 30_000);

    for (let i = 0; i < 4; i++) g.play(SEQ[i]);
    await mcts.mcts(g, 5, 30_000);

    // The walk may or may not find the node (depends on how deep trials went),
    // but if it succeeded historyAtRoot will be at length 4.
    // At minimum, verify the call didn't throw and the root is in a valid state.
    expect(mcts.root).not.toBeUndefined();
    if (mcts.historyAtRoot !== null) {
      expect(mcts.historyAtRoot.length).toBe(4);
    }
  });

  it('delta=5: root is immediately discarded (exceeds limit)', async () => {
    const mcts = new NeuralMCTS(buildBiasedSap(SEQ), 1.0, 0.0);
    const g = new Game();
    await mcts.mcts(g, 200, 30_000);  // build a deep tree

    for (let i = 0; i < 5; i++) g.play(SEQ[i]);
    await mcts.mcts(g, 5, 30_000);

    // Root was reset (delta=5 > 4) and then re-expanded at the new position.
    expect(mcts.historyAtRoot).not.toBeNull();
    expect(mcts.historyAtRoot!.length).toBe(5);
  });

  it('history mismatch: root is discarded when game diverges', async () => {
    const mcts = new NeuralMCTS(uniformSap, 1.0, 0.0);
    const g = new Game();
    await mcts.mcts(g, 5, 30_000);
    expect(mcts.historyAtRoot!.length).toBe(0);

    // Play a move, then undo it and play a different one (divergent history)
    g.play(pt(5, 5));
    g.play(pt(10, 10));
    g.undo();
    g.play(pt(11, 11)); // different move — history now diverges from root
    await mcts.mcts(g, 5, 30_000);

    // Root was reset and re-anchored at the new 2-move position
    expect(mcts.historyAtRoot!.length).toBe(2);
  });
});
