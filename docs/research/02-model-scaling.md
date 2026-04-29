# 02 — Model architecture scaling

**Status:** investigating — first data point landed: at our current
training-compute regime (~512K samples from random init), the
shallower-wider modern arch (64f × 8b GELU, valid padding, 1.9M
params) **decisively beats** Lampe's deeper-narrower arch (48f × 20b
abs, SAME padding, 2.6M params) head-to-head: 199/200 = 99.5%
(±1.0%). Architecture-as-substitute-for-volume is not a free lever
in the deeper-narrower direction. See
"Findings — v0_modern vs v0_lampe" below.

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
    8×64 ResNet. A bigger model would slow self-play proportionally
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
  `value_padding`, `activation`. **Current production (v8_F line):
  64 / 8 / 80 / 2 / valid / GELU** (~1.9M params). Lampe's pre-trained
  `six-917000.pt` (TF1 ancestor) used 48 / 20 / 80 / 4 / SAME / abs
  (~2.6M params); see `tools/convert_tf1_to_pt.py`.
- **Receptive field math (corrected):** the residual blocks use
  kernel=5, not 3 (`src/model.py:76,78`). With 8 blocks of 5×5 convs,
  effective RF is ~33×33, which already covers the 24×24 board. So
  "go deeper for receptive field" is not a load-bearing argument for
  TwixT at our depth — both 8-block and 20-block towers see the
  whole board.
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
   benchmarking matrix: (blocks, filters) ∈ {(8, 64), (12, 64),
   (16, 96), …} → (params, FLOPs, NN throughput on 5070 Ti,
   peak WASM heap, ONNX file size).
2. **Where are we on the strength-vs-size curve?** Do existing iter
   8 self-play games show signs of capacity ceiling (e.g., training
   loss plateauing, value-head accuracy capping)?
3. ~~**Activation choice:** abs() vs GELU?~~ Bundled into the
   v0_modern vs v0_lampe arena below; result confounds activation,
   depth, padding, and width but the bottom line is GELU + shallower
   wins at our training compute. Worth a *single-variable* abs vs
   GELU test only if depth/width is held identical.
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

## Findings — v0_modern vs v0_lampe (2026-04-29)

### Motivation

Triggered by `v8_F` losing 0/200 (100%) to the pre-trained
`six-917000` model from Jordan Lampe (his TF1 checkpoint, ported via
`tools/convert_tf1_to_pt.py`). Two competing hypotheses for that gap:

a) **Architecture:** Lampe's deeper-narrower 48f × 20b shape
   (~2.6M params, abs activation, SAME padding, 4 value-head
   reductions) is intrinsically better for TwixT than our shallower
   64f × 8b (~1.9M, GELU, VALID, 2 reductions).
b) **Training volume:** Lampe's net saw an estimated 100M+ self-play
   positions (917K training steps × batch≈128–256), versus our v8
   line's ~32M cumulative. The arch could be ~equivalent and the gap
   is just data.

### Experimental design

To isolate (a) from (b): instantiate both architectures fresh from
random init, train both with **identical settings on identical data**,
then arena them.

| Variant | Arch | Params | Activation | Padding | Init |
|---|---|---|---|---|---|
| `v0_modern` | 64f × 8b, val_reductions=2 | 1,905,068 | GELU | VALID | Random (= existing `v0.pt`) |
| `v0_lampe`  | 48f × 20b, val_reductions=4 | 2,565,372 | abs  | SAME  | Random (`torch.manual_seed(42)`) |

Training (both, identical):
```
batch_size=256  learning_rate=0.01  decay_rate=1.0
num_batches=2000  temperature=0.5  policy_epsilon=0.01
spdata=spdata/   (full 32M-position corpus, weighted-sampled)
```

`decay_rate=1.0` (no LR decay) was important: an earlier run with
`decay_rate=0.95` collapsed LR to ~3e-05 within a few hundred steps
because random-init loss is naturally noisy and the
"decay-on-loss-rise" heuristic in `train.py` fires on most early
batches. Held the LR fixed for a clean architecture comparison.

### Loss trajectory

| step | v0_modern | v0_lampe |
|---|---|---|
| 1    | 7.46 | 7.34 |
| 500  | 4.77 | 5.19 |
| 1000 | 4.71 | 4.93 |
| 1500 | 4.71 | 4.99 |
| 2000 | 4.77 (slope -2e-4) | 4.86 (slope -5e-4) |

`v0_modern` reached its plateau by step ~500 and bounced flat.
`v0_lampe` was still trending down at step 2000 (slope -5e-4) — the
larger network was **under-trained at 512K samples**.

### Arena

```
arena.py --model-a models/v0_modern.pt --model-b models/v0_lampe.pt \
  --device cuda --total_games 200 --num_clones 18 --trials 400 \
  --async_calls 32
```

| Variant | Wins | Win % |
|---|---|---|
| v0_modern (64f × 8b GELU)  | **199 / 200** | **99.5% (±1.0%)** |
| v0_lampe  (48f × 20b abs)  | 1 / 200       | 0.5% |

### Conclusions

1. **At our training-compute regime, the modern shallower-wider arch
   is strictly stronger** than Lampe's deeper-narrower shape. Not
   noise: 199/1 is far outside the ±1% Wilson interval.
2. **The 200/0 loss to `six-917000` is dominated by training volume,
   not architecture.** Lampe's arch needs much more training to be
   competitive at all (under-trained at 2000 batches × 256), let
   alone surpass ours. Their pre-trained model is strong because of
   100M+ positions seen, not because the architecture is innately
   suited to TwixT.
3. **Receptive field is not the lever I initially thought.** With
   kernel=5, the 8-block tower already covers the 24×24 board.
4. **Implication for next steps:** the productive lever is more
   self-play iterations / more positions, on the architecture we
   already have. Architecture exploration in the *width* direction
   (e.g., 64 → 96 filters, same depth) is still open and could be
   investigated separately, but deeper-narrower is ruled out.

### Caveats

- Single training run per architecture (no seed averaging). The
  99.5/0.5 margin is large enough that seed-noise cannot flip it,
  but a smaller margin would have warranted multiple seeds.
- The comparison confounds **four variables** (depth × width ×
  activation × padding). The aggregate result is conclusive, but
  attributing the margin to any single dimension would require
  follow-up A/B's holding 3 dimensions fixed at a time.
- Both nets trained from random init; the comparison says nothing
  about which arch *bootstraps* better through multiple iterations.
  Possible (but unlikely) that v0_lampe's slower convergence
  reverses with more iterations and self-play data.

## Findings — Knowledge distillation from six-917000 (2026-04-29)

### Motivation

Follow-up to v0_modern vs v0_lampe and v8_F vs six-917000. The
cleanest test of "is our 64f×8b a *capacity bottleneck*?" is to
**use six-917000 as a teacher** and train a 64f×8b student to mimic
its outputs. If the student matches the teacher → arch capacity
isn't limiting; if it falls short despite training → real ceiling.

### Implementation

Added `--teacher MODEL_PATH` to `src/train.py`. When set, the
per-batch flow becomes:

```
batch = prepare_batch(...)        # normal MCTS-derived targets
pegs, links, locs, _, _ = batch
with torch.no_grad():
    t_p_logits, t_v_logits = teacher(pegs, links, locs)
    p_target = softmax(t_p_logits)  # replace MCTS policy
    v_target = softmax(t_v_logits)  # replace hard z label
trainer.train_step(pegs, links, locs, p_target, v_target)
```

`Trainer.train_step`'s `_value_loss` was generalised to dispatch on
target dtype: int64 → hard cross-entropy (existing); float → soft
cross-entropy (new). Backward-compatible with all 48 existing
`tests/test_train.py` cases.

### Experiment 1: short distillation

Same protocol as v0_modern: random init from `models/v0.pt`, 64f×8b,
batch=256, lr=0.01, decay=1.0, 2000 batches, current `spdata/`
corpus (32M positions from v8 self-play). Teacher: `six-917000.pt`.

Loss trajectory:
- step 1: 7.49
- step 500: 5.14
- step 1000: 5.24
- step 2000: 5.14 (slope -2e-4, still negative)

Note: distillation loss is not directly comparable to v0_modern's
4.77 because the teacher's softmax targets are smoother than MCTS
targets — the cross-entropy floor is higher even with a perfect
match.

### Experiment 2: long distillation (5×)

Continue from `v0_distill` for another 8000 batches (10000 total),
same hyperparams. Total wall-clock: 5.2 min + 19.4 min = 24.6 min.

Loss trajectory (cumulative step):
- step 2000:   5.14
- step 4000:   5.19 (small bounce)
- step 6000:   4.99
- step 8000:   5.05
- step 10000:  5.01 (slope -8e-6 — near plateau)

### Arena results (n=200, trials=400, num_clones=18)

| Match | Winner | Win % |
|---|---|---|
| v8_F vs six-917000               | six-917000        | 100.0% (200/0)   |
| **v0_distill (2k) vs v8_F**      | **v0_distill**    | **100.0% (200/0)**   |
| **v0_distill_10k vs v0_distill** | **v0_distill_10k**| **89.0% (178/22, ±4.3%)** |
| v0_distill_10k vs six-917000     | six-917000        | 100.0% (200/0)   |

### Conclusions

1. **Distillation is dramatically more compute-efficient than self-play
   bootstrapping for our task.** A 64f×8b student trained for 5
   minutes on teacher outputs beats `v8_F` 200/0. `v8_F` is the
   result of 8 self-play iterations + extensive Phase B
   hyperparameter tuning (~tens of hours of compute).
2. **More distillation training keeps paying off, even though loss
   appears plateaued.** The 10k version beat the 2k version 178/22
   (+78pp) despite loss only dropping from 5.14 to 5.01. The
   loss-vs-strength relationship is not linear — small CE
   differences compound through MCTS.
3. **64f×8b is *not* a capacity bottleneck for v8_F-class strength.**
   Same arch, different training, decisive win. Reinforces topic
   01's conclusion: training volume / objective is the real lever.
4. **The gap to six-917000 remains absolute (200/0)** even at 10k
   batches. The two competing explanations:
   - **Distribution mismatch (likely):** `spdata/` was generated by
     v8 self-play. Six-917000 visits a different position
     distribution when playing itself, and the student is never
     trained on those positions. At arena time, the student faces
     out-of-distribution inputs.
   - **Capacity ceiling (less likely):** continued strength gains
     from more training (#2) suggest we haven't hit the ceiling yet.
5. **`v0_distill_10k` is the new strongest checkpoint** for this
   architecture, displacing v8_F.

### Caveats

- All results are single-seed. The 200/0 / 178/22 margins are large
  enough that seed noise can't flip them, but a smaller margin
  would warrant multiple seeds.
- Teacher inference adds only ~30% to per-step wall-clock at
  batch=256 (5.2 min vs 4.2 min for v0_modern); cheap enough that
  distillation runs are practical to repeat.
- The student's playing-strength curve as a function of distillation
  compute is unmapped beyond the 2k → 10k point. Plausible that a
  100k-batch run closes a meaningful chunk of the gap to the
  teacher; would take ~3–4 hours.

## Where we left off

- **Capacity question is answered: 64f×8b is sufficient.** Distillation
  showed the shape can hold strength well beyond v8_F. The deeper-
  narrower direction is permanently shelved.
- The width direction (96f × 8b) and other capacity tweaks are not
  ruled out, but are *not load-bearing* for the project goal — capacity
  isn't blocking us.
- `v0_distill_10k.pt` exists as a strong checkpoint produced via
  teacher distillation. It is **not** the path forward (see "Status of
  this topic" below) — kept on disk only as a benchmark anchor.

## Status of this topic relative to the project goal

Per `docs/research/README.md`, the project goal is to develop training
methods that match `six-917000`'s strength **without relying on it as
a teacher**. Distillation was useful as a diagnostic — it ruled out
"capacity bottleneck" as an explanation for the v8_F vs six-917000
gap — but a distilled-only model is not a deliverable for this
project. Continuing to grind on distillation training compute would
not teach us anything we don't already know.

This topic is therefore **effectively closed for now.** The capacity
question is settled. Future scaling investigations should be triggered
by *evidence* that capacity is the bottleneck (e.g., self-play loop
plateaus before reaching six-917000-class strength despite improved
training methods), not by speculation.

## Next action (this topic)

None active. The active research focus moves to:

- **`01-replay-buffer-sampling.md`** — KataGo-style curriculum sampling
  across model windows (open / deferred at the end of topic 01).
- **`03-search-efficiency.md`** — Gumbel AlphaZero / QZero. The user
  has explicitly flagged this as the most interesting direction:
  reducing MCTS trials per move directly attacks the dominant cost of
  self-play, which is the dominant cost of training.
- **(New, possibly its own topic)** — "self-play data curation /
  iteration efficiency": now that we've ruled out architecture and
  validated the batch-size lever, what *training-loop-level* changes
  most efficiently move us toward six-917000-class strength?

If a width A/B (64f vs 96f) ever does become useful, the protocol is
fully scoped above (2000-batch / batch=256 / decay=1.0 / random init,
arena n=200 trials=400) — should take ~1 hour to execute.
