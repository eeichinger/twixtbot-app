# Further Training Performance Improvements

Ideas to improve self-play + training throughput beyond the current
`--compile --fp16` + `async_calls=32` configuration, ranked by effort-to-reward.

**Current baseline** (as of 2026-04-23): ~12,100 evals/sec, ~4,000–5,000 games/hour
at `trials=100`. NNS is GPU-busy 94% of wall clock; model (1.9M params,
`num_filters=64`, `num_blocks=8`) is memory-bandwidth limited on the 5070 Ti.

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

### 8. Automate the iteration loop

Currently you manually kill NNS, copy models, run train.py, restart NNS.
A single script could run unattended overnight:

```bash
# Pseudocode
for iter in {1..10}:
  run_selfplay  # pmany, blocks until num_games done
  kill_nns
  cp models/v$((iter-1)).pt models/v$iter.pt
  run_training  # train.py
  restart_nns models/v$iter.pt
  run_arena v$iter vs v$((iter-1))
  if win_rate < 55%: promote previous model
```

Not a performance win per se but **eliminates human-in-the-loop latency**,
which is often 12–24 hours per iteration in practice.

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

1. **Next run** — try #1 (more clones) + #2 (channels-last) together.
   Validate throughput uplift to 14k+ evals/sec.
2. **After iter 1 succeeds** — do #3 (training fp16/compile) to shorten
   Phase B before iter 2.
3. **Iter 3+** — add #4 (position cache). Big throughput win with
   minimal complexity.
4. **When iter 5 plateaus** — #5 (bigger model) for stronger targets.
5. **Only if willing to invest a weekend** — #6 (TensorRT). Skip unless the
   above plateau and you need another step-change.
