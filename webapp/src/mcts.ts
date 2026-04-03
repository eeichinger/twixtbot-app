/**
 * mcts.ts — Neural MCTS
 * Direct port of src/nnmcts.py to TypeScript.
 *
 * The sap (score-and-policy) function is async because onnxruntime-web
 * inference is Promise-based.
 */

import { Game, Point, pt, SIZE, DLINKS } from './twixt.js';
import { legalMovePolicyArray, policyIndexToPoint, policyPointToIndex } from './naf.js';

export const NUM_MOVES = SIZE * (SIZE - 2);  // 528

// ---------------------------------------------------------------------------
// EvalNode
// ---------------------------------------------------------------------------

export class EvalNode {
  N: Float64Array = new Float64Array(NUM_MOVES);
  Q: Float64Array = new Float64Array(NUM_MOVES);
  P: Float64Array = new Float64Array(NUM_MOVES);
  subnodes: (EvalNode | null)[] = new Array(NUM_MOVES).fill(null);

  proven = false;
  score: number | null = null;
  winningMove: Point | null = null;
  drawingMove: Point | null = null;

  /** Legal move mask (1=legal). Set by expandLeaf. */
  lm: Float32Array | null = null;
  /** Indices of legal moves (nonzero positions of lm). Set by expandLeaf. */
  lmNonzero: number[] | null = null;
}

// ---------------------------------------------------------------------------
// NeuralMCTS
// ---------------------------------------------------------------------------

/** Score-and-policy function: game → [score, policyLogits[528]] */
export type SapFn = (game: Game) => Promise<[number, Float32Array]>;

export class NeuralMCTS {
  private cpuct: number;
  private addNoise: number;
  private sap: SapFn;

  root: EvalNode | null = null;
  historyAtRoot: (Point | 'swap')[] | null = null;

  constructor(sap: SapFn, cpuct = 1.0, addNoise = 0.0) {
    this.sap       = sap;
    this.cpuct     = cpuct;
    this.addNoise  = addNoise;
  }

  // -------------------------------------------------------------------------
  // Public API
  // -------------------------------------------------------------------------

  /** Run `trials` MCTS simulations from the current `game` state.
   *  Returns visit counts array[528], or a forced Point if win is proven. */
  async mcts(game: Game, trials: number): Promise<Float64Array | Point> {
    this._computeRoot(game);

    if (!this.root) {
      this.root = await this.expandLeaf(game);
      this.historyAtRoot = [...game.history];
    }

    for (let i = 0; i < trials; i++) {
      await this._visitNode(game, this.root, this.root, trials - i);
      if (this.root.proven) break;
    }

    if (this.root.proven && this.root.winningMove) {
      return this.root.winningMove;
    }

    return new Float64Array(this.root.N);
  }

  // -------------------------------------------------------------------------
  // expandLeaf
  // -------------------------------------------------------------------------

  async expandLeaf(game: Game): Promise<EvalNode> {
    const leaf = new EvalNode();

    if (game.justWon()) {
      leaf.proven = true;
      leaf.score  = -1;
      return leaf;
    }

    const lm = legalMovePolicyArray(game);
    leaf.lm = lm;
    leaf.lmNonzero = [];
    for (let i = 0; i < lm.length; i++) {
      if (lm[i]) leaf.lmNonzero.push(i);
    }

    if (leaf.lmNonzero.length === 0) {
      // Draw (no legal moves — shouldn't happen in normal TwixT, but handle defensively)
      leaf.proven = true;
      leaf.score  = 0;
      return leaf;
    }

    const [score, policyLogits] = await this.sap(game);
    leaf.score = score;

    // softmax over legal moves only
    let maxLogit = -Infinity;
    for (const i of leaf.lmNonzero) {
      if (policyLogits[i] > maxLogit) maxLogit = policyLogits[i];
    }
    let sumExp = 0;
    for (const i of leaf.lmNonzero) {
      sumExp += Math.exp(policyLogits[i] - maxLogit);
    }
    for (const i of leaf.lmNonzero) {
      leaf.P[i] = Math.exp(policyLogits[i] - maxLogit) / sumExp;
    }

    // Optional Dirichlet noise at root
    if (this.addNoise > 0) {
      this._addDirichletNoise(leaf);
    }

    return leaf;
  }

  // -------------------------------------------------------------------------
  // _visitNode  (recursive simulation)
  // -------------------------------------------------------------------------

  private async _visitNode(
    game:       Game,
    node:       EvalNode,
    topNode:    EvalNode,
    trialsLeft: number,
  ): Promise<number> {
    if (node.proven) {
      return node.score!;
    }

    // Select best action via UCB
    const idx = this._selectAction(node, topNode, trialsLeft);

    if (idx < 0) {
      // No legal moves (shouldn't happen, but defensive)
      node.proven = true;
      node.score  = 0;
      return 0;
    }

    const move = policyIndexToPoint(game.turn, idx);
    game.play(move);

    let value: number;

    if (!node.subnodes[idx]) {
      // Leaf expansion
      const child = await this.expandLeaf(game);
      node.subnodes[idx] = child;
      value = -(child.score ?? 0);
    } else {
      value = -(await this._visitNode(game, node.subnodes[idx]!, topNode, trialsLeft));
    }

    game.undo();

    // Backpropagate
    node.N[idx] += 1;
    node.Q[idx] += (value - node.Q[idx]) / node.N[idx];

    // Propagate proven states upward
    this._updateProven(node, game);

    return value;
  }

  // -------------------------------------------------------------------------
  // UCB action selection
  // -------------------------------------------------------------------------

  private _selectAction(node: EvalNode, topNode: EvalNode, trialsLeft: number): number {
    if (!node.lmNonzero || node.lmNonzero.length === 0) return -1;

    const totalN = node.N.reduce((s, v) => s + v, 0);
    const sqrtN  = Math.sqrt(totalN + 1);
    const cpuct  = this.cpuct;

    let best    = -Infinity;
    let bestIdx = -1;

    for (const i of node.lmNonzero) {
      if (node.subnodes[i]?.proven && node.subnodes[i]?.score === -1) {
        // Avoid proven losing moves
        continue;
      }
      const q = node.Q[i];
      const u = cpuct * node.P[i] * sqrtN / (1 + node.N[i]);
      const val = q + u;
      if (val > best) { best = val; bestIdx = i; }
    }

    // Fallback: if all moves are proven losses, pick any legal one
    if (bestIdx < 0) bestIdx = node.lmNonzero[0];
    return bestIdx;
  }

  // -------------------------------------------------------------------------
  // Proven state propagation
  // -------------------------------------------------------------------------

  private _updateProven(node: EvalNode, _game: Game): void {
    if (!node.lmNonzero) return;

    let allLosses = true;
    let hasWin    = false;
    let hasDraw   = false;

    for (const i of node.lmNonzero) {
      const child = node.subnodes[i];
      if (!child?.proven) { allLosses = false; continue; }
      if (child.score === -1) {
        // Child is a loss for the child player = WIN for us
        hasWin = true;
        node.proven = true;
        node.score  = 1;
        node.winningMove = policyIndexToPoint(_game.turn, i);
        return;
      }
      if (child.score === 0) hasDraw = true;
    }

    if (allLosses) {
      node.proven = true;
      if (hasDraw) {
        node.score = 0;
        // pick the drawing move
        for (const i of node.lmNonzero) {
          if (node.subnodes[i]?.score === 0) {
            node.drawingMove = policyIndexToPoint(_game.turn, i);
            break;
          }
        }
      } else {
        node.score = -1;
      }
    }
  }

  // -------------------------------------------------------------------------
  // Tree reuse: find the subtree matching the current game history
  // -------------------------------------------------------------------------

  private _computeRoot(game: Game): void {
    if (!this.root || !this.historyAtRoot) return;

    const hist      = game.history;
    const rootHist  = this.historyAtRoot;

    // Check that rootHist is a prefix of hist
    if (hist.length < rootHist.length) { this.root = null; this.historyAtRoot = null; return; }
    for (let i = 0; i < rootHist.length; i++) {
      const a = rootHist[i], b = hist[i];
      if (a === 'swap' || b === 'swap') { if (a !== b) { this.root = null; this.historyAtRoot = null; return; } continue; }
      if ((a as Point).x !== (b as Point).x || (a as Point).y !== (b as Point).y) {
        this.root = null; this.historyAtRoot = null; return;
      }
    }

    // Walk down the tree by the moves added since root
    let node: EvalNode | null = this.root;
    // Dummy game to track turn
    const tmpGame = game.clone();
    // We need to rewind tmpGame to rootHist length and re-play
    // Easier: just reset root when histories diverge significantly.
    // For simplicity, reset root if more than 2 moves ahead.
    if (hist.length - rootHist.length > 2) {
      this.root = null; this.historyAtRoot = null; return;
    }

    for (let i = rootHist.length; i < hist.length; i++) {
      const m = hist[i];
      // Need the turn at move i — use a local game replay
      const g = new Game();
      for (let j = 0; j < i; j++) g.play(hist[j] as Point | 'swap');
      const idx = m === 'swap' ? -1 : policyPointToIndex(g.turn, m as Point);
      if (idx < 0 || !node || !node.subnodes[idx]) {
        node = null; break;
      }
      node = node.subnodes[idx];
    }

    if (node && node !== this.root) {
      this.root = node;
      this.historyAtRoot = [...hist];
    } else if (!node) {
      this.root = null;
      this.historyAtRoot = null;
    }
  }

  // -------------------------------------------------------------------------
  // Dirichlet noise
  // -------------------------------------------------------------------------

  private _addDirichletNoise(node: EvalNode): void {
    if (!node.lmNonzero || node.lmNonzero.length === 0) return;
    const alpha  = 0.03;
    const eps    = this.addNoise;
    const k      = node.lmNonzero.length;
    const noise  = _dirichlet(k, alpha);
    for (let i = 0; i < k; i++) {
      const idx = node.lmNonzero[i];
      node.P[idx] = (1 - eps) * node.P[idx] + eps * noise[i];
    }
  }
}

// ---------------------------------------------------------------------------
// Dirichlet sample (Gamma-based, no external dependency)
// ---------------------------------------------------------------------------

function _gamma(alpha: number): number {
  // Marsaglia-Tsang method
  if (alpha < 1) return _gamma(1 + alpha) * Math.pow(Math.random(), 1 / alpha);
  const d = alpha - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  while (true) {
    let x: number, v: number;
    do { x = _randn(); v = 1 + c * x; } while (v <= 0);
    v = v * v * v;
    const u = Math.random();
    if (u < 1 - 0.0331 * x * x * x * x) return d * v;
    if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
  }
}

function _randn(): number {
  // Box-Muller
  const u = 1 - Math.random(), v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function _dirichlet(k: number, alpha: number): number[] {
  const samples = Array.from({length: k}, () => _gamma(alpha));
  const sum = samples.reduce((a, b) => a + b, 0);
  return samples.map(s => s / sum);
}

// Re-export Game import for worker.ts convenience
import { Game as _Game } from './twixt.js';
export { _Game as Game };
