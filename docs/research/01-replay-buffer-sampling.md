# 01 — Replay buffer + minibatch sampling

**Status:** investigating — current behavior characterised; one concrete gap
identified (4-fold vs 8-fold symmetry augmentation).

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
| Symmetry augmentation | **8-fold** (4 rotations × 2 reflections) | **4-fold** (rotations only) |

The only row that suggests a concrete cheap improvement is the last:
`sample_learning_state` calls `r = random.randint(0, 3)` (`train.py:132`)
and applies `nips.rotate(r)` + `rotate_policy_array(N, r)`. There's no
horizontal/vertical reflection. Adding a reflection bit would 2× the
effective dataset per actual self-play game with **no extra training
cost** — same forward/backward pass, just a different augmentation
sample. Worth following up as a small candidate experiment.

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

## Where we left off

Current behavior characterised. AlphaZero rationale documented.
**Two concrete experiment-ready gaps:**

1. **4-fold → 8-fold symmetry augmentation** (rotations only today;
   add reflections). Likely a 1-line edit in `sample_learning_state`
   if naf.py reflection helpers exist. 2× effective dataset size for
   free.
2. **Batch size 256 → 512 with scaled LR**. Cheap A/B test described
   in "Side-thread: batch size choice" above.

The other AlphaZero differences (window shape, train:games ratio) are
either hardware-bound or arguably better in our pipeline already.

## Next action

Two candidate experiments, in order of cheapness:

1. **Batch size A/B (Variant A vs B above).** Direct `train.py`
   invocation, no orchestration needed. Total cost: ~3 min training +
   1-2 hour arena. Decisive about whether 256 is a real bottleneck.

2. **8-fold symmetry augmentation.** Verify `naf.py` reflection
   helpers exist (constants `NUM_ROTATIONS = 4`, `HFLIP_BIT = 1` near
   line 458 suggest yes). 1-line edit in `sample_learning_state` if
   so. Same A-vs-B style arena to confirm no regression.

Whichever runs first, capture the arena result in this file under a
new "Findings" section before moving on.

Open / deferred: KataGo's curriculum sampling across model windows;
investigate after the cheap experiments close.
