# Further Training Performance Improvements

Ideas to improve self-play + training throughput beyond the current
`--compile --fp16` + `async_calls=32` configuration, ranked by effort-to-reward.

**Current baseline** (as of 2026-04-24, `--num_clones=24 --async_calls=32`):
~14,000 evals/sec, batch size ~480, ~5,600–7,000 games/hour at `trials=100`.
NNS is GPU-busy ~96% of wall clock; model (1.9M params, `num_filters=64`,
`num_blocks=8`) is memory-bandwidth limited on the 5070 Ti.

---

## Quick wins (low effort, low risk)

### 1. Increase `--num_clones` from 16 to 20–24 — **DONE**

Implemented in `train_loop.py` with `NUM_CLONES=24`, `NNS_CAPACITY=3072`.
Result: batch size ~480, throughput ~14,200 evals/sec, GPU busy 96.5%.

### 2. Channels-last memory format

Blackwell Tensor Cores prefer NHWC (channels-last) over NCHW for convolutions.
Two-line change in `src/nneval.py`:

```python
# In __init__:
self.model = net.eval().to(device).to(memory_format=torch.channels_last)

# In to_tensor():
t = t.permute(0, 3, 1, 2).to(self.device)
return t.to(memory_format=torch.channels_last)
```

**Expected gain:** 10–25% kernel speedup on small conv nets.
**Risk:** zero — if it doesn't help, revert the two lines.

---

## Medium effort (1–2 hours work)

### 3. Apply `--compile --fp16` to Phase B (training)

Training is ~30% of iteration time. `src/train.py` currently trains in fp32
without compile. Adding both should cut training wall clock by 30–50%.

**Changes to `src/train.py`:**
- Wrap `trainer.train_step` with `torch.autocast(device_type='cuda', dtype=torch.float16)`
  on the forward+loss path. The optimizer step should stay fp32.
- Add `torch.compile(self.model, mode='default')` inside `Trainer.__init__()`.
- Add a `--batch_size 512` option — current `256` leaves VRAM headroom.
- Consider `torch.set_float32_matmul_precision('high')` for TF32 on non-autocast ops.

Caveat: use `GradScaler` if you see NaN losses; small models often train fine without it.

### 4. Position caching (transposition table)

MCTS tree expansion calls the NN on many positions that are reachable via
different move orders. A hash table keyed on board state → cached
`(policy, value)` skips the NN entirely for repeats.

**Where:** `src/asn_player.py`, in `_expand_leaf()`. Before calling
`self.client.write_query`, compute a hash of `self.game` board state and
check a local LRU cache. On hit, populate `leaf.score` and `leaf.P`
directly (re-apply Dirichlet noise fresh) and append to `finished_leaves`
without any NNS round-trip. On miss, cache the un-rotated result in the
`set_reply` callback.

**Why worker-level, not NNS-level:** `asn_player` applies random rotation
(`random_rotation=1`) before sending queries. The same board position
produces different byte patterns at different rotations, so an NNS-level
byte-hash cache would miss. The cache must be keyed on the pre-rotation
game state.

**Board hash:** `twixt.Game` has no hash function today. Options:
- **Zobrist hashing** (incremental, O(1) per play/undo): pre-generate a
  random 64-bit table indexed by `(color, row, col)` for pegs. XOR on
  play, XOR again on undo. Fast and standard.
- **Symmetry-aware Zobrist**: compute hashes for both the original and
  the mirrored board, take the lexicographic minimum. Doubles the hit
  rate for symmetric positions at negligible cost.

**Cache entry size:** `score` (4 B) + `P` array (528 × 4 = 2,112 B) +
hash key (8 B) + LRU overhead (~40 B) ≈ **2,164 bytes/entry**.

**Memory budget:** 50 GB system RAM, 24 worker processes, NNS + OS ≈ 8 GB.
Leaves ~42 GB for workers → ~1.7 GB per process → **~800K entries/process**.
Conservative target: **500K entries** (~1.1 GB per process, ~26 GB total).
Can tune down to 200K (~430 MB each, ~10 GB total) if memory is tighter.

**Expected benefit on TwixT:** TwixT has no captures, so any permutation
of independent moves reaches the same position — transpositions are
structurally common. Estimated hit rates:
- `trials=50–100` (shallow search): 10–20% cache hits
- `trials=200+` (deeper search): 25–40% cache hits

Since the GPU is the bottleneck (96.5% busy), every cached eval translates
directly to faster game completion: 15% fewer evals → games finish ~15%
faster → ~18% more games/hour (GPU fills freed capacity automatically).

**Interaction with other items:** composes with everything (channels-last,
TensorRT, trials scaling). Benefit grows at higher trials, so pairing
with #9 (per-iteration trials scaling) is natural — the cache matters
most at the higher-trials later iterations where data quality counts.

### 5. Larger model, once iter 2–3 is stable

`num_filters=128, num_blocks=12` (~8M params) makes the GPU *more* efficient
per parameter because compute grows faster than memory traffic — the model
becomes compute-bound rather than memory-bandwidth-bound, pushing nvidia-smi
SM occupancy from ~35% toward 70%+.

**Per-batch time:** roughly 1.5–2× longer.
**Games per hour:** drops ~30–50%, but per-game learning signal is stronger
and fewer iterations are needed to reach target strength.

**When to switch:** after the small model beats random init decisively
(iter 3–5). Don't switch on iter 1 — validate the whole pipeline with the
cheap model first.

### 5a. Deeper policy head (B7 in `docs/improvements.md`)

Keep the trunk, replace the tiny 2-channel 1×1 policy bottleneck
(~80 params) with `3×3 conv (32ch) → 3×3 conv (16ch) → 1×1 → 528`
(~14K params). With `trials=100`, policy-prior quality dominates MCTS
exploration, so this head capacity jump has outsized effect.

**Per-batch time:** ~5–10% slower.
**Data reusability:** accumulated `spdata/*.bin` stays valid
(architecture-independent).
**Pairs with:** the recently-added soft MCTS policy targets in `asn_player` —
the current 80-param head cannot really exploit that signal; a deeper head can.

See `docs/improvements.md` B7 for design rationale.

### 5b. KataGo global pooling bias (B9a in `docs/improvements.md`)

Add `GlobalAvgPool → Linear(Nch → Nch) → broadcast-add` at each residual
block. Lets every spatial position at every depth see board-global statistics
(link density, edge control, overall structure) — precisely the signal that
local 3×3 convs miss, and precisely the signal TwixT's connectivity-based
scoring depends on.

**Per-batch time:** ~10–15% slower.
**Parameter count:** +10–30%.
**Side-effect on GPU utilisation:** *positive* — improves compute/memory
ratio, helping the current memory-bandwidth-bound regime.
**Data reusability:** accumulated `spdata/*.bin` stays valid; the new
architecture can warm-start training on the existing dataset rather than
regenerating self-play from scratch.
**Pairs with:** item #5 above — deeper nets + global pooling compound well.

This is the biggest structural lever for TwixT specifically. Requires a
full retrain from random init, so plan it as a commit-when-ready experiment
rather than mid-iteration.

See `docs/improvements.md` B9a for design rationale and KataGo references.

---

## Big levers (high effort, high reward)

### 6. TensorRT or ONNX Runtime CUDA EP for inference (includes INT8)

Export the PyTorch model to ONNX and run NNS against TensorRT (or ORT CUDA EP).
TensorRT on Blackwell with a 1.9M-param conv net typically gives **2–3×
inference throughput** over PyTorch + `torch.compile` because of:
- Aggressive cross-layer fusion (conv + BN + activation + residual-add into
  a single kernel)
- Calibrated kernel autotuning (benchmarks multiple implementations per
  shape, picks the fastest)
- Layout optimisations beyond what TorchInductor generates

**INT8 quantization for NNS inference** is best done through TensorRT, not
PyTorch. PyTorch's `torch.ao.quantization` targets CPU; its CUDA INT8
path is immature for conv nets. TensorRT's INT8 calibration (post-training
quantization on a small calibration set) is the proven approach and folds
into the same integration work.

FP16→INT8 halves memory bandwidth per eval. Since the model is
memory-bandwidth limited (35% SM utilisation with FP16), INT8 could push
SM utilisation toward 50–60% and add another 1.3–1.5× on top of the
FP16→TRT baseline gain. Combined **FP16 PyTorch → INT8 TRT: 3–4×
throughput** is realistic.

**What it requires:**
- ONNX export step in the NNS startup (already have `tools/export_onnx.py`)
- TensorRT engine build step (one-time per checkpoint, ~30 seconds)
- INT8 calibration dataset: ~1,000 positions from `spdata/`, run once per
  model checkpoint
- Swap out `nneval.NNEvaluater` for a TRT-backed evaluator in `src/nns.py`
- Install `tensorrt` and `pycuda` (or use `polygraphy`)

Biggest potential single win. ~1–2 days of work.

### 7. Pipeline CPU prep and GPU inference

NNS currently runs sequentially within each loop: `prep → GPU wait → postproc`.
With CUDA streams + pinned memory you can overlap:
while batch N is running on GPU, batch N+1 is being prepared on CPU.

**Savings:** ~0.5 ms/batch at current rates (the `preprocessing + pp_shmem + pp_socket`
sum) — ~2% throughput. Minor on its own but compounds with TensorRT (which
reduces GPU time, making CPU prep a larger relative share).

**Where:** refactor `smmpp.py`'s `run_gpu_side()` into a double-buffered loop
with two work slots. Requires care around shared memory lifetimes.

### 8. Automate the iteration loop — **DONE**

Implemented in `train_loop.py` (repo root). Orchestrates Phase A (NNS +
pmany self-play) and Phase B (train.py) unattended, with weighted-tier
`spdata/` rotation, heartbeat logging, and resume support via
`--start_iter`. All tunables are constants at the top of the file.

Still open: automatic arena-gated promotion (run arena at the end of each
iteration; only accept the new model if it beats the prior by >55% at
n=200). Currently the loop accepts every trained model unconditionally.

### 9. Per-iteration MCTS trials scaling

Use fewer MCTS trials at early iterations (when the model is weak) and
ramp up at later iterations (when policy quality makes deeper search
productive).

**Rationale:** at iter 1–2 the policy head is near-random. Running 100
trials explores ~100 leaves out of ~500 legal moves — the search is too
shallow for the extra trials to find tactics the policy missed. At
`trials=50`, each move's MCTS is ~50% faster. Since GPU is the
bottleneck, this directly halves self-play Phase A wall time for early
iterations.

At iter 5+ the policy is strong enough that deeper search (200+ trials)
finds tactical lines the policy ranks incorrectly, producing better
training targets. The cost per game doubles, but each game's training
signal is worth more.

**Implementation:** add `TRIALS_CADENCE` to `train_loop.py`:
```python
TRIALS_CADENCE = [
    (2,  50),    # iter 1-2: weak model, fast games
    (4,  100),   # iter 3-4: developing model
    (99, 200),   # iter 5+: strong model, high-quality data
]
```
Look up trials per iteration in `main()` alongside games/batches from
`ITERATION_CADENCE`, and pass it into `run_self_play()` which interpolates
it into the `asn_spec` string.

**Expected gain:** ~50% faster Phase A at iter 1–2; improved data quality
at iter 5+.
**Effort:** Low (15-minute change to `train_loop.py`).
**Risk:** None — does not affect model architecture or data format.

---

## Things NOT to do

### Don't set `cudagraph_skip_dynamic_graphs=True`

You already use `mode='default'` which avoids CUDA graphs entirely.
This flag only matters if you reverted to `mode='reduce-overhead'`.

### Don't pad batches to fixed sizes for CUDA graphs

The waste from running conv on padded-zero positions cancels the graph benefit.
Let TorchInductor handle dynamic shapes via `mode='default'`.

### Don't increase `trials` for throughput reasons

Higher `trials` improves training data quality (stronger self-play → better
targets) but does **not** improve throughput — the NNS is already saturated.
Raise `trials` to 200 in iter 3+ once the model knows basic tactics, as a
quality lever, not a performance lever.

### Don't raise `async_calls` above 32 on this hardware

Measured: at `async_calls=48` the NNS idle time jumped from 1.24 ms → 8.14 ms
per cycle, dropping throughput from 12,093 → 11,441 evals/sec. The CPU burst
pattern on 8 cores cannot sustain the post-reply tree traversal fast enough.
Only revisit this on a 9950X3D (16 cores) where the CPU burst has more
headroom.

---

## Recommended sequence

**Done:** #1 (24 clones), #8 (train_loop.py).

**Active TODO — self-play speed:**

1. **Immediately** — #9 (per-iteration trials scaling). 15-minute change,
   halves Phase A at early iterations. No downside.
2. **Immediately** — #2 (channels-last). 2-line change, test on current
   config. 10–25% GPU kernel speedup if it lands, zero-cost revert if not.
3. **Next priority** — #4 (position cache). Medium effort (~half day).
   10–20% fewer GPU evals at trials=100, 25–40% at trials=200+. Biggest
   return when combined with #9 (higher trials at later iters = more
   transpositions). Does not depend on any other item.
4. **Before iter 4** — #3 (training fp16/compile) to shorten Phase B.
5. **When above plateau** — #6 (TensorRT + INT8). 2–4× throughput.
   ~1–2 days of work. Skip unless everything above plateaus and you need
   another step-change.

**Active TODO — self-play quality (model architecture):**

6. **Iter 5+ in parallel** — #5a (deeper policy head, warm-starts from
   existing `spdata/`). Compare in arena every 3 iters.
7. **When current lineage plateaus** — #5 (larger model) and/or #5b
   (KataGo global pooling) retrain. Both pair well together.
