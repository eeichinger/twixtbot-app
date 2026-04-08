/**
 * worker.ts — Web Worker that runs MCTS + ONNX inference off the main thread.
 *
 * Message protocol (main → worker):
 *   { type: 'init',  modelUrl: string, timeLimitMs?: number, maxTrials?: number, temperature?: number }
 *   { type: 'move',  history: MoveMsg[], timeLimitMs?: number, maxTrials?: number, temperature?: number }
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

let player: OnnxPlayer | null = null;
let mcts:   NeuralMCTS | null = null;
let defaultTimeLimitMs = 10_000;
let defaultMaxTrials   = 100_000;  // effectively unlimited; time limit is the real constraint
let defaultTemperature = 0;        // 0 = argmax (strongest); >0 = temperature sampling (weaker)
let isProcessing = false;

self.onmessage = async (e: MessageEvent) => {
  const msg = e.data;

  if (msg.type === 'init') {
    try {
      player = new OnnxPlayer();
      await player.load(msg.modelUrl);
      if (msg.timeLimitMs  != null) defaultTimeLimitMs = msg.timeLimitMs;
      if (msg.maxTrials   != null) defaultMaxTrials   = msg.maxTrials;
      if (msg.temperature != null) defaultTemperature = msg.temperature;

      mcts = new NeuralMCTS(
        (game) => player!.eval(game),
        /* cpuct   */ 0.5,
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

    /**
     * Pick a move from a Float64Array of scores (visit counts or priors).
     *
     * temperature=0  → argmax (deterministic best move)
     * temperature>0  → sample proportional to scores^(1/temperature);
     *                  higher values flatten the distribution → weaker play
     */
    function pickMoveWithTemperature(scores: Float64Array, turn: number, temperature: number): MoveMsg {
      if (temperature === 0) {
        let bestIdx = 0;
        for (let i = 1; i < scores.length; i++) {
          if (scores[i] > scores[bestIdx]) bestIdx = i;
        }
        const move = policyIndexToPoint(turn, bestIdx);
        return { x: move.x, y: move.y };
      }
      // Build weights = scores^(1/T), then sample
      const inv = 1 / temperature;
      const weights = new Float64Array(scores.length);
      let sum = 0;
      for (let i = 0; i < scores.length; i++) {
        if (scores[i] > 0) {
          weights[i] = Math.pow(scores[i], inv);
          sum += weights[i];
        }
      }
      if (sum === 0) {
        // No MCTS visits yet — fall back to argmax on raw scores
        let bestIdx = 0;
        for (let i = 1; i < scores.length; i++) {
          if (scores[i] > scores[bestIdx]) bestIdx = i;
        }
        const move = policyIndexToPoint(turn, bestIdx);
        return { x: move.x, y: move.y };
      }
      let r = Math.random() * sum;
      for (let i = 0; i < weights.length; i++) {
        r -= weights[i];
        if (r <= 0) {
          const move = policyIndexToPoint(turn, i);
          return { x: move.x, y: move.y };
        }
      }
      // Floating-point rounding fallback: return last nonzero weight
      for (let i = weights.length - 1; i >= 0; i--) {
        if (weights[i] > 0) {
          const move = policyIndexToPoint(turn, i);
          return { x: move.x, y: move.y };
        }
      }
      const move = policyIndexToPoint(turn, 0);
      return { x: move.x, y: move.y };
    }

    try {
      const history: MoveRecord[] = (msg.history as MoveMsg[]).map(toMoveRecord);
      const timeLimitMs: number  = msg.timeLimitMs  ?? defaultTimeLimitMs;
      const maxTrials:   number  = msg.maxTrials    ?? defaultMaxTrials;
      const temperature: number  = msg.temperature  ?? defaultTemperature;
      const game = replayHistory(history);
      const turn = game.turn;

      // Opening book: WHITE's first move.  MCTS burns 5-10 s on move 1 with
      // the highest branching factor and the least tactical content.  Instead,
      // pick randomly from positions that have swap scores in [0.54, 0.58] —
      // strong enough that BLACK will often swap them, but not so extreme that
      // they're trivially decided.  Each folded position and all 4 board mirrors
      // are included so the opening varies across games.
      //
      // Swap scores (BETAS model, folded coords xres=x-6, yres=y-5.5):
      //   (7,8)↔(7,15)↔(16,8)↔(16,15)  ≈ 0.550
      //   (8,9)↔(8,14)↔(15,9)↔(15,14)  ≈ 0.574
      //   (9,8)↔(9,15)↔(14,8)↔(14,15)  ≈ 0.548
      const OPENING_BOOK = [
        {x:7,y:8},  {x:7,y:15},  {x:16,y:8},  {x:16,y:15},
        {x:8,y:9},  {x:8,y:14},  {x:15,y:9},  {x:15,y:14},
        {x:9,y:8},  {x:9,y:15},  {x:14,y:8},  {x:14,y:15},
      ];
      if (game.history.length === 0) {
        const pick = OPENING_BOOK[Math.floor(Math.random() * OPENING_BOOK.length)];
        self.postMessage({ type: 'result', move: pick });
        isProcessing = false;
        return;
      }

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

      // Immediate win detection: before spending any MCTS budget, scan all
      // legal moves for a single-step win.  This is pure game-logic (no ONNX),
      // so it's essentially free and guarantees we never miss a forced win.
      for (const m of game.legalPlays()) {
        game.play(m);
        const won = game.justWon();
        game.undo();
        if (won) {
          self.postMessage({ type: 'result', move: { x: m.x, y: m.y } });
          isProcessing = false;
          return;
        }
      }

      // Heartbeat: every 2s send a ping to main thread so it can confirm
      // the worker is still alive. Pings stop when MCTS finishes or is killed.
      let pingCount = 0;
      const moveStart = Date.now();
      const pingId = setInterval(() => {
        pingCount++;
        self.postMessage({ type: 'ping', elapsed: Date.now() - moveStart, iterations: pingCount });
      }, 2000);

      /** Summarise visit-count distribution for diagnostics.
       *  trials    = total MCTS visits at root
       *  topPct    = % of visits on the most-visited move (search concentration)
       *  topQ      = Q of that move from the bot's perspective (-1 loss … +1 win)
       */
      function gatherStats(scores: Float64Array): { trials: number; topPct: number; topQ: number } {
        let trials = 0, topN = 0, topIdx = 0;
        for (let i = 0; i < scores.length; i++) {
          trials += scores[i];
          if (scores[i] > topN) { topN = scores[i]; topIdx = i; }
        }
        const topPct = trials > 0 ? (topN / trials) * 100 : 0;
        const topQ   = mcts?.root ? mcts.root.Q[topIdx] : 0;
        return { trials, topPct, topQ };
      }

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
          const stats = gatherStats(scores);
          const elapsed = Date.now() - moveStart;
          self.postMessage({ type: 'result', move: pickMoveWithTemperature(scores, turn, temperature), ...stats, elapsed, timeLimitMs, maxTrials, temperature });
        } else {
          // Root not yet expanded — pick any legal move
          const legal = game.legalPlays();
          const p = legal.at(Math.floor(Math.random() * legal.length));
          self.postMessage({ type: 'result', move: { x: p.x, y: p.y } });
        }
      }, timeLimitMs);

      let result: Float64Array | { x: number; y: number } | null = null;
      try {
        result = await mcts.mcts(game, maxTrials, timeLimitMs) as Float64Array | { x: number; y: number };
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
        const elapsed = Date.now() - moveStart;
        let moveMsg: MoveMsg;
        if (result instanceof Float64Array) {
          const stats = gatherStats(result);
          moveMsg = pickMoveWithTemperature(result, turn, temperature);
          self.postMessage({ type: 'result', move: moveMsg, ...stats, elapsed, timeLimitMs, maxTrials, temperature });
        } else {
          // Proven forced move — no MCTS stats available
          moveMsg = { x: (result as {x:number,y:number}).x, y: (result as {x:number,y:number}).y };
          self.postMessage({ type: 'result', move: moveMsg });
        }
      }
    } catch (err) {
      self.postMessage({ type: 'error', message: String(err) });
    } finally {
      isProcessing = false;
    }

  } else if (msg.type === 'abort') {
    // Reset tree to free memory; the in-flight async MCTS will finish its current
    // trial and then return (time limit check will stop the loop on next iteration).
    if (mcts) mcts = new NeuralMCTS((game) => player!.eval(game), 0.5, 0.0);
  }
};
