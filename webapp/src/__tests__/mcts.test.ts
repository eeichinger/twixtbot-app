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
import { legalMovePolicyArray } from '../naf.js';

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
    const result = await mcts.mcts(new Game(), 5);
    expect(result).toBeInstanceOf(Float64Array);
    expect((result as Float64Array).length).toBe(NUM_MOVES);
  });

  it('returns non-negative visit counts', async () => {
    const mcts = new NeuralMCTS(uniformSap, 1.0, 0.0);
    const result = await mcts.mcts(new Game(), 5) as Float64Array;
    for (const v of result) {
      expect(v).toBeGreaterThanOrEqual(0);
    }
  });

  it('total visit count equals number of trials run', async () => {
    const trials = 10;
    const mcts = new NeuralMCTS(uniformSap, 1.0, 0.0);
    const result = await mcts.mcts(new Game(), trials) as Float64Array;
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
    const result = await mcts.mcts(g, 20) as Float64Array;
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
    const result = await mcts.mcts(g, 20) as Float64Array;
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
    await mcts.mcts(g, 5);
    expect(g.history.length).toBe(histBefore.length);
    expect(g.turn).toBe(turnBefore);
  });

  it('peg arrays are unchanged after mcts()', async () => {
    const g = new Game();
    g.play(pt(5, 5));
    const whitePegsBefore = new Int8Array(g.pegs[WHITE]);
    const blackPegsBefore = new Int8Array(g.pegs[BLACK]);
    const mcts = new NeuralMCTS(uniformSap, 1.0, 0.0);
    await mcts.mcts(g, 5);
    expect(Array.from(g.pegs[WHITE])).toEqual(Array.from(whitePegsBefore));
    expect(Array.from(g.pegs[BLACK])).toEqual(Array.from(blackPegsBefore));
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
    const result = await mcts.mcts(g, 30);

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
