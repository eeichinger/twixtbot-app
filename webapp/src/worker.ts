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
import { replayHistory, pt, BLACK } from './twixt.js';
import type { MoveRecord, Point } from './twixt.js';
import { wantSwap } from './swapmodel.js';

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
let defaultTimeLimitMs = 10_000;
let isProcessing = false;

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
    if (isProcessing) {
      self.postMessage({ type: 'error', message: 'Worker busy' });
      return;
    }
    isProcessing = true;
    const { policyIndexToPoint } = await import('./naf.js');

    /** Pick the best move from a Float64Array of scores (visit counts or priors). */
    function pickBestMove(scores: Float64Array, turn: number): MoveMsg {
      let bestIdx = 0;
      for (let i = 1; i < scores.length; i++) {
        if (scores[i] > scores[bestIdx]) bestIdx = i;
      }
      const move = policyIndexToPoint(turn, bestIdx);
      return { x: move.x, y: move.y };
    }

    try {
      const history: MoveRecord[] = (msg.history as MoveMsg[]).map(toMoveRecord);
      const timeLimitMs: number = msg.timeLimitMs ?? defaultTimeLimitMs;
      const game = replayHistory(history);
      const turn = game.turn;

      // Swap rule: when the AI is playing as BLACK at move 2, use the fitted
      // swap model to decide whether to take WHITE's first peg.  MCTS cannot
      // reason about this because 'swap' is not in the 528-action policy array.
      if (turn === BLACK && game.history.length === 1) {
        const firstPeg = game.history[0];
        if (firstPeg !== 'swap' && wantSwap(firstPeg as Point)) {
          self.postMessage({ type: 'result', move: 'swap' });
          isProcessing = false;
          return;
        }
      }

      // Heartbeat: every 2s send a ping to main thread so it can confirm
      // the worker is still alive. Pings stop when MCTS finishes or is killed.
      let pingCount = 0;
      const pingStart = Date.now();
      const pingId = setInterval(() => {
        pingCount++;
        self.postMessage({ type: 'ping', elapsed: Date.now() - pingStart, iterations: pingCount });
      }, 2000);

      // Hard-deadline timer: fires between await yields if the internal
      // Date.now() check somehow doesn't stop the loop in time.
      // This is the second line of defence after the mcts.ts deadline.
      let resultSent = false;
      const hardDeadlineId = setTimeout(() => {
        if (resultSent) return;
        resultSent = true;
        const root = mcts!.root;
        if (root) {
          const scores = new Float64Array(root.N);
          // Fall back to priors if no iterations ran yet
          const totalN = scores.reduce((s, v) => s + v, 0);
          if (totalN === 0 && root.lmNonzero) {
            for (const i of root.lmNonzero) scores[i] = root.P[i];
          }
          self.postMessage({ type: 'result', move: pickBestMove(scores, turn) });
        } else {
          // Root not yet expanded — pick any legal move
          const legal = game.legalPlays();
          const p = legal.at(Math.floor(Math.random() * legal.length));
          self.postMessage({ type: 'result', move: { x: p.x, y: p.y } });
        }
      }, timeLimitMs);

      let result: Float64Array | { x: number; y: number } | null = null;
      const moveStart = Date.now();
      try {
        result = await mcts.mcts(game, MAX_TRIALS, timeLimitMs) as Float64Array | { x: number; y: number };
      } finally {
        // Always cancel the timers — whether mcts returned normally,
        // threw, or was interrupted.
        clearTimeout(hardDeadlineId);
        clearInterval(pingId);
        // Notify main thread that MCTS has fully finished. If this fires AFTER
        // the main thread already received 'result' (via the hard deadline), it
        // means the worker was still running WASM compute during the human turn.
        self.postMessage({ type: 'computing-done', elapsed: Date.now() - moveStart });
      }

      if (!resultSent && result !== null) {
        resultSent = true;
        let moveMsg: MoveMsg;
        if (result instanceof Float64Array) {
          moveMsg = pickBestMove(result, turn);
        } else {
          moveMsg = { x: (result as {x:number,y:number}).x, y: (result as {x:number,y:number}).y };
        }
        self.postMessage({ type: 'result', move: moveMsg });
      }
    } catch (err) {
      self.postMessage({ type: 'error', message: String(err) });
    } finally {
      isProcessing = false;
    }

  } else if (msg.type === 'abort') {
    // Reset tree to free memory; the in-flight async MCTS will finish its current
    // trial and then return (time limit check will stop the loop on next iteration).
    if (mcts) mcts = new NeuralMCTS((game) => player!.eval(game), 1.0, 0.0);
  }
};
