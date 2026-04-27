# 01 — Replay buffer + minibatch sampling

**Status:** investigating — batch-size lever active across two
doublings: A→B (256→512) won 66/34, B→D-clean (512→1024) won 68/32,
both at fixed compute (n=200, ±~6.5%). C (2× steps at batch=512) was
within noise vs B — extra steps don't help at the saturation knee.
Next: Variant E pushes batch to 2048 to find where the lever stops
working.

## Question

AlphaZero-style training uses a fixed-size **replay buffer** of
self-play positions and trains by **stochastic minibatch sampling** —
some positions are visited many times, some not at all in a given
training run. Our current pipeline (`src/train.py`) reads the
accumulated `spdata/*.bin` files. We don't know yet whether that
read is full-pass / round-robin / random / weighted.

The investigation: characterise what we do today, compare to
AlphaZero's design, and decide whether changing it improves training
signal per training step.

## Why it matters for us specifically

- **Volume:** iter 8 is at ~4M positions per iter, ~32M positions
  cumulative across 8 iters. Full passes through that volume per iter
  would be 32M / (2000 batches × 256 batch_size) ≈ 62 epochs over the
  recent corpus, which seems excessive for a single iteration.
- **Recency vs diversity:** `USE_WEIGHTED_SAMPLING` already biases
  recent iterations 4× over older ones (0.8/0.2 across a 3-iter
  window). A buffer-with-replacement scheme might subsume or replace
  this mechanism.
- **Effective capacity:** repeated training on the same positions can
  overfit short-term self-play tactics; a sliding window with random
  sampling tends to reduce this.

## What we already know / pointers

- `train_loop.py` constants: `BATCH_SIZE=256`, `LEARNING_RATE=0.01`,
  iter-cadence batches scale 500/1500/2000.
- `USE_WEIGHTED_SAMPLING` in `train_loop.py` rotates spdata files into
  `w=0.8/` (recent) and `w=0.2/` (older) directories. The actual
  sampling logic must be inside `src/train.py`.
- `naf.LearningState.NUM_BYTES = 1789` — fixed-size records, so random
  access into a `.bin` file by record index is `seek(i*1789)`. No
  parsing penalty.

## Open questions

1. How does `src/train.py` actually consume the spdata files? Single
   pass per `--num_batches`? Random offsets? Round-robin? — *first
   thing to read on this topic.*
2. What does AlphaZero's replay buffer paper actually specify? (Sutton
   & Barto Chapter 17? Silver et al. 2017 §Methods? Lc0 docs?)
3. If we're under-sampling, can we measure it? E.g., position-level
   visit counter during a training run.
4. Does the answer change with corpus size? An 8-iter corpus has
   different optimal sampling than a 50-iter one.

## References to gather

- Silver et al. "Mastering Chess and Shogi by Self-Play with a General
  Reinforcement Learning Algorithm" (AlphaZero), §Methods.
- Lc0 documentation on training pipeline / replay buffer.
- KataGo paper (Wu 2019) — has a more elaborate sampling scheme worth
  comparing, especially their use of Polyak averaging across model
  windows.

## Current behavior (as of 2026-04-26, iter 8)

Our pipeline already implements **uniform random sampling with tier
weighting**. Specifically:

- `src/train.py:82-112` (`load_selector`) walks `spdata/` recursively.
  Each `.bin` file becomes a "basket" in a `WeightedRandomSelector` with
  weight 0.8 or 0.2 depending on which directory tier it lives in. The
  `w=0.8/` and `w=0.2/` directories are managed by
  `rotate_spdata_tiers` in `train_loop.py` (TR1 work).
- `src/wrs.py:26-35` (`WeightedRandomSelector.random_item`) draws a
  basket with probability proportional to `weight × records_in_basket`,
  then a uniform-random index inside that basket.
  → **`P(record) ∝ tier weight`, uniform within a tier.**
- `src/train.py:115-135` (`sample_learning_state`) seeks into the file
  at `record_index × 1789`, reads one record, retries on corrupt rows
  (all-zero N), applies one of 4 rotations as data augmentation, returns
  the LearningState.
- `src/train.py:328-333` builds each minibatch with `args.batch_size`
  (=256) **independent** draws from the selector. Not epoch-based, not
  round-robin, not in-game-order.

So we already get the i.i.d. / state-decorrelation properties from
random sampling. The hypothesis "random sampling reduces
state-dependent / trajectory-memorisation learning" is correct; this
is in fact the published rationale (see "AlphaZero references" below).

## AlphaZero's published sampling design

From Silver et al. 2017 (AGZ "Mastering the game of Go without human
knowledge"; AZ "Mastering Chess and Shogi by Self-Play with a General
Reinforcement Learning Algorithm"):

- **Replay buffer** of the most recent 500K self-play games (~25M
  positions for chess at ~50 moves/game).
- **Minibatches of 2048 positions** drawn **uniformly at random** from
  the buffer.
- Continuous learning: training and self-play run concurrently; new
  games push old games out of the buffer (hard sliding window).

Their stated reasons (from the papers and the DQN work they cite —
Mnih et al. 2013/2015 introduced experience replay for deep RL):

1. **Decorrelate consecutive samples.** SGD's convergence assumes
   i.i.d.; within-game positions share almost all of the board state,
   so they are extremely correlated.
2. **Decorrelate value targets.** Every position in a single game
   shares the same z ∈ {-1, 0, +1}. Game-ordered training would feed
   50-400 consecutive examples with identical z, biasing the optimizer.
3. **Avoid recency bias / catastrophic forgetting.** Without a buffer,
   the network would only ever train on what it's currently playing.
4. **Avoid trajectory memorisation.** With sequential game data, the
   network can learn "after move sequence X→Y→Z the value is +1" —
   recognising trajectories instead of positions. Random sampling
   forces it to recognise patterns in *positions* as inputs to a
   function, not as nodes in a sequence. **This is the "less prone
   to learn from state-dependent moves" point.**

KataGo inherits this scheme and elaborates with curriculum-style
sampling across model windows; not yet investigated here.

## Comparison

| Aspect | AlphaZero | Ours |
|---|---|---|
| Sampling | Uniform random | Tier-weighted random (0.8 / 0.2) |
| Window | Last 500K games (hard cutoff) | Last 3 iters at 0.8 + everything else at 0.2 (soft) |
| Batch size | 2048 | 256 |
| Train:games ratio | ~1:1 (continuous) | ~1:8 per iter (2000 × 256 vs 4M positions) |
| Symmetry augmentation | 8-fold (Go's D₄, all same-player) | 4-fold (D₂, full same-player set for our representation) |

(The previous version of this section claimed "4-fold rotations only,
missing reflections" and proposed an easy 2× experiment. That was
wrong — corrected below in "Augmentation correction".)

## Augmentation correction (2026-04-26)

Previous session claimed our 4-fold scheme was "rotations only, missing
reflections" and that adding reflections would 2× the data. **Wrong on
both counts.** Reading `naf.py` more carefully:

- `NUM_ROTATIONS = 4`, `HFLIP_BIT = 1`, `VFLIP_BIT = 2`. `r ∈ {0..3}`
  indexes a *bitmask* of {hflip, vflip}, not 4 rotation angles.
  - `r=0` identity, `r=1` h-flip only, `r=2` v-flip only, `r=3` h+v
    (= 180°). That's the full **Klein four-group D₂**.
- The reason Go's 8-fold doesn't translate to TwixT: TwixT
  **canonicalises player perspective at encoding time**.
  `NetInputs.init_from_game_black` (`naf.py:49-59`) transposes the
  board and swaps colours, so the network always sees a "white-to-move"
  canonical view. The 4 D₄ elements that aren't in our 4-fold set are
  the player-swapping ones (90°, 270°, transposes), and they're
  already absorbed by this canonicalisation.

**So we are not missing geometric augmentation.** D₂ is the right
group for this representation; adding the player-swap elements would
just produce already-existing black-to-move positions viewed from the
other side — no new data.

The "easy 2× data" experiment is dropped from candidate next-actions.

(Side note: there's a subtle correctness check worth doing one day —
the same `r` is applied to both the NetInputs board and the policy
target via `hflip_policy_array`/`vflip_policy_array`, which assume the
canonical white-to-move encoding. Looks consistent in the code I
read, but the policy `hflip` flips the major axis of `(S-2, S)` reshape
and the board `hflip` flips axis 0 of `(S, S, 11)` — these are only
equivalent because of canonicalisation. Worth a directed test if we
ever suspect off-by-one symmetry bugs. Not blocking.)

## Side-thread: batch size choice (256 vs alternatives)

Surfaced when re-reading `train.py`. Quick capture so it doesn't get
lost.

**What we have today.** `train.py:272` defaults `--batch_size=256`;
`train_loop.py:70` uses the same. No detailed rationale in the
codebase. Two short references hint that 256 was set conservatively:

- `TRAINING.md:545` — "Increase to 512 if GPU VRAM allows."
- `docs/further_training_improvements.md:50` — "current 256 leaves
  VRAM headroom."

**Two effects of changing batch size**:

1. *Gradient noise.* Larger batch = lower-variance gradient per step,
   but the noise itself is regularisation. Keskar et al. 2017 ("On
   Large-Batch Training") showed too-large batches converge to "sharp
   minima" that generalise worse. Effect grows with batch size; at
   256 → 512 probably small, at 256 → 4096 noticeable.
2. *Effective throughput.* GPU forward/backward at 2× batch is ≈
   1.3-1.6× the wall time, so larger batch fits more samples per unit
   compute. AlphaZero's 2048 reflects this — TPU pods made it cheap.

**Confound: learning rate.** Goyal et al. 2017 linear scaling rule —
scaling batch k× requires scaling LR k× for a fair comparison. Most
"large batch hurts" results in the literature are confounded by LR
mismatch.

### Proposed experiment

Cheap (≈ a few hours total) and cleanly informative. Three variants,
all started from `v7.pt`, all training against the existing accumulated
`spdata/`:

| Variant | batch_size | num_batches | learning_rate | What it measures |
|---|---|---|---|---|
| **A** (control) | 256 | 2000 | 0.01 | reproduces v8 |
| **B** (LR-scaled) | 512 | 1000 | 0.02 | same compute, scaled LR — "is batch size a lever at all?" |
| **C** (more samples) | 512 | 2000 | 0.02 | 2× samples seen — "are we under-trained at the current cadence?" |

Arena each pairwise at `trials=400, num_games≈100`:

- **A vs B** is the headline question: at fixed compute, does scaled
  larger batch help, hurt, or wash?
- **A vs C** secondary: would more training help at all?

Recommended: run A vs B first. If it washes, batch size isn't worth
chasing further; move on. If B wins materially, consider larger
batches still.

### Mechanics

Direct `train.py` invocation, bypass `train_loop.py` (no need for
markers, NNS, etc. — Phase B only):

```bash
cp models/v7.pt models/v8_B.pt
python src/train.py --model models/v8_B.pt --device cuda \
  --num_batches 1000 --batch_size 512 --learning_rate 0.02 \
  --decay_rate 0.95 --temperature 0.5 --policy_epsilon 0.01 \
  --save_after 200 spdata/
```

Then arena with arena.py against the existing `models/v8.pt`.

## Findings — Variant A vs B (2026-04-27)

Trained `v8_B.pt` from `v7.pt` with the proposed Variant B settings
(`--num_batches 1000 --batch_size 512 --learning_rate 0.02`) and ran a
200-game arena against `v8.pt` (Variant A: batch 256, lr 0.01,
2000 batches — same wall-clock training compute).

```
arena.py --model-a models/v8.pt --model-b models/v8_B.pt --device cuda \
  --total_games 200 --num_clones 18 --trials 400 --async_calls 32
```

| Variant | batch | num_batches | lr | Wins | Win % |
|---|---|---|---|---|---|
| A — `v8.pt`   | 256 | 2000 | 0.01 | 68 / 200 | 34.0% (±6.6%) |
| B — `v8_B.pt` | 512 | 1000 | 0.02 | **132 / 200** | **66.0%** |

Decisive in favour of B. Difference is ~32 percentage points at
n=200 — well outside the ±6.6% Wilson interval, so this is not noise.
At equal training compute, **batch=512 with linearly-scaled LR (0.02)
trains a clearly stronger network** than batch=256 / lr=0.01.

Caveat — this confounds two changes (batch size and number of update
steps). `v8_B` saw the same total samples but ran half as many
optimizer steps. The "fair-compute" reading is "B wins"; the question
"would 2000 batches at 512/0.02 win by even more?" is exactly what
Variant C is for.

### Bug noticed and fixed in arena.py

The raw `Arena complete` line printed `model-b: 0.0%`. Cause:
score-routing bug — `socket_a = "/tmp/twixtbot_nns_v8"` is a substring
of `socket_b = "/tmp/twixtbot_nns_v8_B"`, so the `socket_a in spec`
check matched B-runs first and routed every B-score into the A bucket
(while B itself stayed at 0). Worked around for this run by re-summing
the per-clone final scores from `00.log..17.log` (model-a=68,
model-b=132).

Fixed in `b7ebf49` ("fix bug in arena when calculating summary"):
match against `location=<path>,` (with the trailing comma) instead of
the bare socket path, so prefix-overlapping paths can't cross-match
(`src/arena.py:197-215`). Future arena runs across prefix-related
model names should print correct totals directly.

## Findings — Variant B vs C (2026-04-27)

`v8_C` trained with the proposed Variant C settings (from `v7.pt`,
`batch_size=512`, `num_batches=2000`, `lr=0.02` — same batch and LR as
B, 2× the optimizer steps, 2× the samples seen). Arena `v8_B` vs
`v8_C` at the same n=200, trials=400, num_clones=18:

```
arena.py --model-a models/v8_B.pt --model-b models/v8_C.pt --device cuda \
  --total_games 200 --num_clones 18 --trials 400 --async_calls 32
```

| Variant | batch | num_batches | lr | Samples seen | Wins | Win % |
|---|---|---|---|---|---|---|
| B — `v8_B.pt` | 512 | 1000 | 0.02 |  512K | 108 / 200 | 54.0% (±6.9%) |
| C — `v8_C.pt` | 512 | 2000 | 0.02 | 1024K | 92 / 200  | 46.0% |

**Result is within noise.** The 8-percentage-point margin is smaller
than the ±6.9% Wilson interval (95% CI for B: [47.1%, 60.9%], spans
50%). Honest reading: "no detectable difference at n=200." A bigger
n might tease a small edge out, but it's not worth chasing.

**Practical takeaway:** at our current corpus size and at the new
batch=512 / lr=0.02 operating point, **doubling optimizer steps from
1000 to 2000 does not produce a meaningful additional gain.** Which
in turn means B's big win over A was almost certainly the
**batch-size lever itself**, not "A was under-trained." 1000 steps at
batch=512 sits at or past the saturation knee for this corpus.

Minor caveat: `--decay_rate 0.95` over 2000 steps drops C's effective
average LR below B's, so C's "more steps" came with a slightly lower
effective LR. Probably small relative to the noise here; not worth a
re-run.

## Findings — Variant B vs D-clean (2026-04-27)

`v8_D` trained as the "D-clean" flavour from the previous next-action
plan: from `v7.pt`, `batch_size=1024`, `num_batches=500`, `lr=0.04`.
**Same 512K total samples as A and B** — pure batch-size-lever
isomorphic continuation. Arena n=200, trials=400, num_clones=18:

```
arena.py --model-a models/v8_B.pt --model-b models/v8_D.pt --device cuda \
  --total_games 200 --num_clones 18 --trials 400 --async_calls 32
```

| Variant | batch | num_batches | lr | Samples seen | Wins | Win % |
|---|---|---|---|---|---|---|
| B  — `v8_B.pt` | 512  | 1000 | 0.02 | 512K | 64 / 200  | 32.0% (±6.5%) |
| D  — `v8_D.pt` | 1024 |  500 | 0.04 | 512K | **136 / 200** | **68.0%** |

Decisive — margin (36 pp) is far outside the ±6.5% Wilson interval.
**The batch-size lever is still very much active at the 512→1024
doubling**, with linear LR scaling (Goyal). At fixed compute,
`v8_D` is clearly stronger than `v8_B`.

`v8_D` is the new strongest checkpoint.

## Where we left off

- **`v8_D` is the strongest checkpoint** so far. Trajectory:
  `v8` (A, batch=256) ≪ `v8_B` ≈ `v8_C` (batch=512) ≪ `v8_D`
  (batch=1024).
- The batch-size lever has produced **two consecutive decisive wins**
  at fixed compute (256→512, 512→1024), each ~66/34. The natural next
  question: where does the lever stop working?
- The "more steps" lever (B vs C at batch=512) is exhausted at our
  current corpus size — additional optimizer steps at the saturation
  knee don't help.
- arena.py socket-name matching tightened in `b7ebf49`; no further
  workaround needed.

## Next action

**Variant E — push the batch lever from 1024 to 2048.** Same
isomorphic D-clean shape: same total samples as A/B/D, batch
doubled, LR doubled per Goyal linear-scaling rule.

| Variant | batch | num_batches | lr | Samples | What it tests |
|---|---|---|---|---|---|
| E (D-clean continuation) | **2048** | **250** | **0.08** | 512K | "Does the batch lever still work at 2048?" |

Train and arena commands:

```bash
# E (from v7, isomorphic with A/B/D-clean)
cp models/v7.pt models/v8_E.pt
python src/train.py --model models/v8_E.pt --device cuda \
  --num_batches 250 --batch_size 2048 --learning_rate 0.08 \
  --decay_rate 0.95 --temperature 0.5 --policy_epsilon 0.01 \
  --save_after 50 spdata/

# Head-to-head vs the current best (D)
python src/arena.py --model-a models/v8_D.pt --model-b models/v8_E.pt \
  --device cuda --total_games 200 --num_clones 18 --trials 400 \
  --async_calls 32 --progress_interval 60
```

**Things to watch during the E training run:**

1. **VRAM.** 5070 Ti has 16GB; the model is small (~1.9M params), so
   batch=2048 should fit comfortably, but worth one `nvidia-smi`
   check during the run. If OOM, fall back to batch=1536 (and
   lr ≈ 0.06) or use gradient accumulation.
2. **Loss spike at start.** The Goyal linear-scaling result holds up
   to ~8K batch *with a short warmup* — at lr=0.08 from step 1, the
   first few updates can blow up. If the first 5–10 loss values
   diverge instead of dropping, that's why; remediation is a 25-step
   linear LR warmup. With only 250 total steps, an unwarmed run is
   the riskiest piece of this experiment.

**Decision rule after E arena:**

- E ≫ D (e.g. ≥60% over 200 games): lever still active; queue
  Variant F at batch=4096, lr=0.16.
- E ≈ D (within ±~7% of 50%): **1024 is the saturation knee.** Adopt
  D's settings (batch=1024 / lr=0.04) as the default operating point
  and stop pushing this lever.
- E ≪ D: batch=2048 is past the knee or hit a sharp-minima / warmup
  problem. If a warmup-equipped re-run still loses, 1024 is the knee.

**Adopt-as-default housekeeping:** independent of E, the A→B→D
trajectory already justifies flipping `train_loop.py`'s defaults
from `BATCH_SIZE=256` / `LEARNING_RATE=0.01` to **`1024` / `0.04`**
(skip the intermediate 512 step — D is now the strongest result).
Small standalone commit. Worth doing before the next full
`train_loop.py` iter regardless of how E lands.

Open / deferred: KataGo's curriculum sampling across model windows;
investigate after batch-size experiments close.
