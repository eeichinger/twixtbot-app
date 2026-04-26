# 02 — Model architecture scaling

**Status:** scoped

## Question

When does it pay off to scale the network up — more residual blocks
(depth), more filters (width), bigger heads — and what's the
inference-cost / strength trade-off curve for our specific hardware
and deployment constraints?

## Why it matters for us specifically

- **Two deployment targets** with very different cost sensitivity:
  - **Webapp PWA on iOS Safari** (`webapp/`): WASM ORT, no MPS/CUDA,
    ~12MB WASM heap budget after our work in `CLAUDE.md` notes. Bigger
    model = larger ONNX weights = bigger heap pressure.
  - **Self-play / arena on 5070 Ti**: ~13,200 pos/s at the current
    20×48 ResNet. A bigger model would slow self-play proportionally
    to FLOPs, doubling iteration wall-time potentially.
- **Position cache hit rate is dropping** (iter 6 ~55%, iter 8 ~0%) as
  the model sharpens its policy, so each future iter does more NN work
  per game. A bigger model amplifies that cost.
- We've also documented existing thinking under feature IDs: B7
  (deeper policy head), B5 (attention/non-local — future). Section 5,
  5a, 5c in `further_training_improvements.md` cover progressive
  scaling rationale.

## What we already know / pointers

- `src/model.py` — the `TwixNet` class. Configurable
  `num_filters`, `num_blocks`, `num_value_hidden`, `value_reductions`,
  `value_padding`, `activation`. Current production: 48 / 20 / 80 / 4 /
  same / abs.
- `docs/improvements.md`:
  - **B7** — deeper policy head: 1×1 bottleneck → 3×3→3×3→1×1→528.
    Modest parameter add (~14K), one-shot retrain compatible.
  - **B5** — attention / non-local blocks. Marked P4 / Future. Profile
    after B9a first.
- `docs/further_training_improvements.md`:
  - §5 ("Larger model, once iter 2-3 is stable") — recommends scaling
    once basic loop is stable.
  - §5c ("Progressive network scaling") — train the smaller model first,
    then graft a bigger one and continue training. Saves cold-start
    iterations.

## Open questions

1. **What does our cost curve actually look like?** Need a small
   benchmarking matrix: (blocks, filters) ∈ {(20, 48), (24, 64),
   (24, 96), (30, 96), …} → (params, FLOPs, NN throughput on 5070 Ti,
   peak WASM heap, ONNX file size).
2. **Where are we on the strength-vs-size curve?** Do existing iter
   8 self-play games show signs of capacity ceiling (e.g., training
   loss plateauing, value-head accuracy capping)?
3. **Activation choice:** abs() is a legacy quirk from the TF1
   ancestor. GELU is `model.py`'s default for new training.
   Worth a clean A/B test on a small iteration?
4. **Progressive scaling mechanics:** how do we initialize the new
   weights? Zero-init the additions and copy the rest? Random-init
   and warm-start? Has anyone in the AlphaZero ecosystem published a
   recipe?
5. **iOS budget revisit:** with INT8 quantization (B1 done), the
   current model is ~3.2 MB on disk and fits comfortably. What's the
   ceiling at INT8? 4 MB? 8 MB?

## References to gather

- KataGo paper (Wu 2019) — specifically their "small / medium / large"
  network sweep and its training-cost curves.
- Lc0 community benchmarks of T78 / T82 net sizes.
- AlphaZero paper §Methods — net size vs strength.
- `model.py:` re-read to confirm options & defaults.

## Where we left off

(Just scoped. Cross-references gathered to existing docs but no
benchmarking done yet.)

## Next action

Build the benchmark matrix. Smallest version: pick 3 candidate
(blocks, filters) combinations, instantiate each via `model.py`, run a
fixed-batch-size forward pass on the 5070 Ti, and record params/FLOPs
plus throughput. Output: a table in this file under "Cost matrix".
This is feasible without touching training; can run on the Mac with
MPS for a relative ordering before committing GPU time on the PC.
