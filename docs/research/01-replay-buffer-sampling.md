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

## Where we left off

Current behavior characterised. AlphaZero rationale documented. One
concrete gap surfaced: **4-fold vs 8-fold symmetry augmentation** —
we're missing reflections. The other differences (batch size, window
shape, train:games ratio) are either hardware-bound or arguably
better in our pipeline already.

## Next action

Investigate the symmetry-augmentation gap concretely:

1. Confirm TwixT actually has 8-fold symmetry. Rotation by 0/90/180/270°
   may swap which player is "vertical-connect" vs "horizontal-connect"
   — the 90° rotations are colour-swapping, the 180° is not. Reflections
   need similar care.
2. Check `naf.py` for existing reflection helpers (the constants
   `NUM_ROTATIONS = 4` and `HFLIP_BIT = 1` near line 458 suggest the
   data layout already supports an 8-fold scheme even though
   `sample_learning_state` only uses 4-fold).
3. If the reflection helpers exist, the change is a 1-line edit in
   `sample_learning_state` plus matching application to the policy
   target. Train one iteration as a baseline; arena vs the
   parent at proper trials count to confirm no regression.

Open: KataGo's curriculum sampling (sampling across older model
windows beyond simple recency tiers). Defer; investigate after the
augmentation gap is closed.
