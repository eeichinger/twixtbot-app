# TwixT AI Improvement — Technical Reference

Generated: 2026-04-05. Based on analysis of the current webapp MCTS + ONNX stack.

> **Status and priority for all items are tracked in `docs/planned-features.md`
> (sections 1–2). This document contains the technical background and implementation
> details for each improvement.**

---

## Algorithmic Improvements (no retraining needed)

### A1 · Swap Rule — DONE

**Problem:** When the AI plays as BLACK (move 2), it runs full MCTS to decide
whether to swap. MCTS is poorly suited to this binary decision because the
swap option isn't in the 528-action policy array, so the tree never explores
the 'swap' branch at all. In practice the AI never swaps, even when it should.

**Solution:** Port `src/swapmodel.py` to TypeScript. The model is a 4-coefficient
linear regression fitted on bot self-play data:

```
score = β₀ + β₁·xres + β₂·yres + β₃·xres·yres
want_swap = score > 0.5
```

where `(xres, yres)` are the first peg's coordinates folded to the first quadrant
and centred on `(6.0, 5.5)`. Coefficients:

```
β = [0.494481, −0.00366079, 0.0225597, 0.00114293]
```

**Impact:** Every game where the AI is BLACK and faces a strong opening peg,
it now makes the correct swap decision rather than ignoring the option entirely.

**Files changed:** `webapp/src/swapmodel.ts` (new), `webapp/src/worker.ts`

---

### A2 · Smart-init Q Pre-seeding (FPU) — DONE

**Problem:** In the MCTS tree, `Q[i]` starts at 0 for all children of a newly
expanded node. With the 50–300 trials per move typical in the webapp, PUCT
selection over unvisited children degenerates to pure policy-prior ranking
because all have Q=0. The neural network's **value** estimate for the position
(the `score` returned by the SAP function) is never used to bias which branches
are explored first.

**Solution:** After expanding a leaf and obtaining `leaf.score` from the neural
network, pre-seed all legal children's Q values:

```typescript
for (const i of leaf.lmNonzero) {
  leaf.Q[i] = leaf.score;
}
```

This is equivalent to KataGo's *First Play Urgency (FPU)*. Unvisited children
inherit the parent's position value rather than a neutral 0 prior.

**Why it helps at low trial counts:**
- If the position is losing (score ≈ −0.8): unvisited Q[i] ≈ −0.8, so already-visited
  moves that returned better values compete fairly instead of being drowned out by
  the optimistic Q=0 baseline of unvisited moves.
- If the position is winning (score ≈ +0.8): unvisited moves get a head-start that
  reflects the position strength, biasing search toward more promising branches.

After the first backpropagated visit, the running-average update overwrites the
pre-seeded value entirely (`Q = Q + (value − Q) / N` at N=1 gives `Q = value`),
so the seeding only influences which child is selected first.

**Files changed:** `webapp/src/mcts.ts`

---

### A3 · Tree Reuse: Extend to 4 Moves + Fix O(N²) Replay

**Problem:** `_computeRoot()` discards the tree if the game is more than 2 moves
ahead of the root. Extending to 4 moves would retain more search work across
turns. Additionally the existing reuse path replays the game history from scratch
for each step (O(N²) total), which is wasteful.

**Recommendation:** Refactor `_computeRoot()` to:
1. Walk the tree up to 4 steps (or until a null child)
2. Track game state incrementally by replaying only the delta moves

**Expected gain:** At 10-second think time the tree can reach 300–500 nodes deep;
reusing 4 plies instead of 2 preserves more of this work for the opponent's reply.

---

### A4 · PUCT Constant Tuning (cpuct 1.0 → 0.5–0.75)

**Problem:** `cpuct=1.0` is the AlphaGo Zero default, calibrated for Go's
~250 legal moves. TwixT starts with 528 legal moves and a flatter prior
distribution. At low trial counts, `cpuct=1.0` produces too much exploration
and not enough exploitation of the top-ranked policy moves.

**Recommendation:** Reduce to 0.5–0.75 and validate with a self-play arena
(e.g. 100 games old vs. new). Expected gain: faster tactical convergence in
positions where one or two moves are clearly best.

---

## Neural Network Improvements (require re-export ± retraining)

### B1 · INT8 Quantization — iOS Memory Relief

**Problem:** The iOS deferred-kill bug (see CLAUDE.md) is driven by peak WASM
heap size during MCTS. The 12 MB ONNX model expands to ~50–100 MB compiled WASM
+ activations.

**Solution:** One Python call:

```python
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic('model.onnx', 'model-int8.onnx', weight_type=QuantType.QInt8)
```

Expected result: 2–4× reduction in model weight memory, ~25% inference slowdown
(negligible at 10-second think time). No retraining required.

---

### B7 · Deeper Policy Head

**Problem:** The current policy head is a 2-channel 1×1 bottleneck (~80 params).
With 50–300 trials per move, the quality of the policy prior dominates: it
determines which lines get explored at all.

**Recommendation:** Upgrade to `3×3 conv (32ch) → 3×3 conv (16ch) → 1×1 → 528`
(~14K params). This better captures local link patterns that inform move
selection. One training run with the PyTorch rewrite.

---

### B9a · KataGo Global Pooling Bias

**Problem:** TwixT is a long-range connectivity game — threats span 12–15 cells —
but 5×5 convolutions can't efficiently aggregate global board state. The model
has no mechanism to detect "who controls the board overall."

**Solution:** Add a `GlobalAvgPool → Linear(3ch → Nch) → broadcast-add` at each
residual block (KataGo 2020 architecture). Adds negligible inference cost, all
standard ONNX ops. Requires a full retrain from scratch.

**Applicability:** Direct fit for TwixT. Global link density (how many links each
player has built across the board) is precisely the signal local convolutions miss.

---

### B5 · Attention / Non-local Blocks (Future)

ViT-style multi-head attention or non-local blocks would structurally capture
long-range link chains better than convolutions. Main concern: WASM inference
cost (one non-local block ≈ 2× a residual block). GNNs modelling the actual
peg graph are not practically ONNX-compatible with static shapes.

Recommended: validate B9a first, then profile inference time before committing
to attention-based architectures.

---

## Recent Research Most Directly Applicable

| Research | Applicability to TwixT |
|---|---|
| **KataGo global pooling bias** (2020) | Direct fit — global link density is exactly what local convs miss |
| **Gumbel MuZero policy targets** (2022) | Training: better policy labels with fewer self-play trials per move |
| **MuZero Reanalyze** (2021) | Training: relabel old data with updated model; improves value calibration |
| **Opening book / retrieval-augmented MCTS** (2023–24) | ~100 KB JSON for first 2–3 moves; eliminates wasted compute on trivially-decided openings |
| **MuZero full / world model** | Not browser-compatible — dynamics network is larger than the entire current model |

