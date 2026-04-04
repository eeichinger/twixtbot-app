/**
 * worker.ts — Web Worker that runs MCTS + ONNX inference off the main thread.
 *
 * Message protocol (main → worker):
 *   { type: 'init',  modelUrl: string, trials?: number }
 *   { type: 'move',  history: MoveMsg[], trials?: number }
 *   { type: 'abort' }
 *
 * Message protocol (worker → main):
 *   { type: 'ready' }
 *   { type: 'result', move: MoveMsg }
 *   { type: 'error',  message: string }
 *
 * A MoveMsg is either { x: number, y: number } or the string 'swap'.
 */

import { OnnxPlayer } from './onnx-player.js';
import { NeuralMCTS } from './mcts.js';
import { replayHistory, pt } from './twixt.js';
import type { MoveRecord } from './twixt.js';

type MoveMsg = { x: number; y: number } | 'swap';

function toMoveRecord(m: MoveMsg): MoveRecord {
  if (m === 'swap') return 'swap';
  return pt(m.x, m.y);
}

function fromMoveRecord(m: MoveRecord): MoveMsg {
  if (m === 'swap') return 'swap';
  return { x: m.x, y: m.y };
}

const MAX_TRIALS = 100_000;  // effectively unlimited; time limit is the real constraint

let player: OnnxPlayer | null = null;
let mcts:   NeuralMCTS | null = null;
let defaultTimeLimitMs = 25_000;

self.onmessage = async (e: MessageEvent) => {
  const msg = e.data;

  if (msg.type === 'init') {
    try {
      player = new OnnxPlayer();
      await player.load(msg.modelUrl);
      if (msg.timeLimitMs) defaultTimeLimitMs = msg.timeLimitMs;

      mcts = new NeuralMCTS(
        (game) => player!.eval(game),
        /* cpuct   */ 1.0,
        /* addNoise */ 0.0,  // no exploration noise in single-player mode
      );

      self.postMessage({ type: 'ready' });
    } catch (err) {
      self.postMessage({ type: 'error', message: String(err) });
    }

  } else if (msg.type === 'move') {
    if (!mcts || !player) {
      self.postMessage({ type: 'error', message: 'Worker not initialised' });
      return;
    }
    try {
      const history: MoveRecord[] = (msg.history as MoveMsg[]).map(toMoveRecord);
      const timeLimitMs: number = msg.timeLimitMs ?? defaultTimeLimitMs;

      const game   = replayHistory(history);
      const result = await mcts.mcts(game, MAX_TRIALS, timeLimitMs);

      let moveMsg: MoveMsg;
      if (result instanceof Float64Array) {
        // Pick the move with the highest visit count
        let bestIdx = 0;
        for (let i = 1; i < result.length; i++) {
          if (result[i] > result[bestIdx]) bestIdx = i;
        }
        // policyIndexToPoint needs to be imported
        const { policyIndexToPoint } = await import('./naf.js');
        const move = policyIndexToPoint(game.turn, bestIdx);
        moveMsg = { x: move.x, y: move.y };
      } else {
        // Forced winning point returned directly
        moveMsg = { x: (result as {x:number,y:number}).x, y: (result as {x:number,y:number}).y };
      }

      self.postMessage({ type: 'result', move: moveMsg });
    } catch (err) {
      self.postMessage({ type: 'error', message: String(err) });
    }

  } else if (msg.type === 'abort') {
    // Reset tree to free memory; the in-flight async MCTS will finish its current
    // trial and then return (time limit check will stop the loop on next iteration).
    if (mcts) mcts = new NeuralMCTS((game) => player!.eval(game), 1.0, 0.0);
  }
};
