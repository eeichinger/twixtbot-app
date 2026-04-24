# Further Training Performance Improvements

Ideas to improve self-play + training throughput beyond the current
`--compile --fp16` + `async_calls=32` configuration, ranked by effort-to-reward.

**Current baseline** (as of 2026-04-24, `--num_clones=24 --async_calls=32`):
~14,000 evals/sec, batch size ~480, ~5,600–7,000 games/hour at `trials=100`.
NNS is GPU-busy ~96% of wall clock; model (1.9M params, `num_filters=64`,
`num_blocks=8`) is memory-bandwidth limited on the 5070 Ti.

---

## Quick wins (low effort, low risk)

### 1. Increase `--num_clones` from 16 to 20–24

More parallel MCTS tree traversers means a smoother query stream to NNS,
growing batch size without the burstiness seen at `async_calls=48`.

```bash
python src/pmany.py \
  --num_clones 24 \
  --log_dir logs/sp_gen \
  -- \
  python src/battle.py \
    --white "asn_player:location=/tmp/twixtbot_nns,trials=100,async_calls=32" \
    --black "asn_player:location=/tmp/twixtbot_nns,trials=100,async_calls=32" \
    --num_games 42 \
    --threads 2 \
    --training_file spdata/iter1_%n%.bin
```

Bump NNS `--capacity` to `3072` (24 × 2 × 2 × 32).

**Expected gain:** batch size ~420, throughput 13k–14k evals/sec.
**Why it works:** CPU is at ~55%, so oversubscribing 8 cores with 48 threads
is fine — MCTS workers are I/O-bound on NNS replies, not compute-bound.

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

**Where:** `src/asn_player.py`, around `_expand_leaf()`. Before calling
`self.client.write_query`, check a `LRUCache` keyed on a canonical board hash.
**Size:** ~1M entries ≈ 100–200 MB RAM per worker.
**Empirical benefit on TwixT:** 20–40% fewer NN calls (depends on `trials`;
higher trials = more transpositions hit). Effectively multiplies throughput
without any GPU work.

Implementation note: make the hash invariant under the two symmetries
(board mirror + colour swap) for maximum hit rate.

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

### 6. TensorRT or ONNX Runtime CUDA EP for inference

Export the PyTorch model to ONNX and run NNS against TensorRT (or ORT CUDA EP).
TensorRT on Blackwell with a 1.9M-param conv net typically gives **2–3×
inference throughput** over PyTorch + `torch.compile` because of:
- Aggressive cross-layer fusion (conv + BN + activation + residual-add into
  a single kernel)
- Calibrated kernel autotuning (benchmarks multiple implementations per
  shape, picks the fastest)
- Layout optimisations beyond what TorchInductor generates

**What it requires:**
- ONNX export step in the NNS startup (already have `tools/export_onnx.py`)
- TensorRT engine build step (one-time per checkpoint, ~30 seconds)
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

1. **Next run** — try #2 (channels-last) on the current running config.
   Throughput is already at 14k/sec from #1 (24 clones). Free upside if it
   lands, revert if not.
2. **Before iter 4** — do #3 (training fp16/compile) to shorten Phase B.
3. **Iter 5+ in parallel** — branch a #5a lineage (deeper policy head,
   warm-starts from existing `spdata/`). Compare in arena every 3 iters.
4. **When current lineage plateaus** — switch to #5 (larger model) and/or
   commit to a #5b (KataGo global pooling) retrain. Both pair well together.
5. **Any time** — add #4 (position cache). Big throughput win with minimal
   complexity, doesn't depend on any of the above.
6. **Only if willing to invest a weekend** — #6 (TensorRT). Skip unless
   everything above plateaus and you need another step-change.
