# 01 — Replay buffer + minibatch sampling

**Status:** scoped

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

## Where we left off

(Just scoped the topic. Nothing investigated yet.)

## Next action

Read `src/train.py` end-to-end and document its current sampling
behaviour. Output: a sub-section in this file titled "Current behavior"
with file:line refs and a one-paragraph summary.
