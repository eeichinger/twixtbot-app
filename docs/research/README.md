# Training-effectiveness research notes

Running log of research and small experiments aimed at making twixtbot
training **more effective** — stronger models per unit of compute, not
just faster iterations. Throughput / orchestration improvements live in
`docs/improvements.md` (A-prefix algorithm, B-prefix model) and
`docs/further_training_improvements.md`; this directory is for the
exploratory phase that comes *before* a topic gets a feature ID.

These notes are designed for hobby-pace work: pick a topic, advance the
file, commit. When we resume after a gap, the topic's "Where we left
off" / "Next action" sections rebuild the context fast.

## Topics

| File | Topic | Status | Cross-refs |
|---|---|---|---|
| [01-replay-buffer-sampling.md](01-replay-buffer-sampling.md) | Replay buffer + minibatch sampling vs full-pass training | investigating | (none yet) |
| [02-model-scaling.md](02-model-scaling.md) | Scaling depth / width / heads | scoped | B7, B5 in `improvements.md`; §5–5c in `further_training_improvements.md` |
| [03-search-efficiency.md](03-search-efficiency.md) | Reducing trials per move (Gumbel AlphaZero, "QZero") | scoped | A4 (cpuct tuning) in `improvements.md` |
| [04-global-board-features.md](04-global-board-features.md) | End-to-end connection awareness (KataGo-style global pooling) | scoped | **B9a** in `improvements.md`; §5b in `further_training_improvements.md` |

**Status legend:**
- `scoped` — question framed, references identified, no investigation done yet
- `investigating` — actively reading / prototyping / measuring
- `decided` — outcome reached; either graduated to a feature ID or shelved
- `shelved` — explored and parked; reasons recorded in the file

## How to use this directory

1. **One topic per file.** Cross-link freely when topics interact (e.g.,
   bigger model + global pooling).
2. **End each session with a commit** that updates the topic file's
   "Where we left off" and "Next action" sections, even if no progress.
3. **Decision log.** When a finding is concrete enough to act on, give
   it a feature ID in `docs/planned-features.md` and link both files.
4. **Honest uncertainty.** If we don't know what a paper or term means
   yet, say so in the file — don't guess. The first task on any topic
   that cites a reference is to verify the reference.

## Current model / training context (for quick recall)

As of iter 7 → iter 8 (April 2026):
- **Model:** 20-block × 48-filter ResNet (`src/model.py`). ~1.9M parameters.
  Activation: `abs()` (legacy from TF1 ancestor; could be GELU). 528 policy
  outputs (24×22 inner grid). 3-class value head (loss/draw/win).
- **Training cadence:** 10,000 self-play games per iter at trials=200; 2,000
  training batches at batch=256.
- **Hardware:** 7800X3D + 5070 Ti. Iter wall-time ≈ 13 hours self-play +
  3 minutes training.
- **NN throughput:** ~13,200 positions/s at batch=619 (compile + fp16).
- **Self-play data volume:** ~4M positions per iter (~7 GB `.bin` files).
  ~410 moves/game, decreasing slowly with model strength.
- **Replay weighting:** `USE_WEIGHTED_SAMPLING = True`, recent tier (last 3
  iters) at weight 0.8, older at 0.2.

The "iter 8 plays itself in self-play with much fewer position-cache hits
than iter 6" observation (NN evals per game ~doubled from ~88 to ~200 per
move) is a baseline data point for any future search-efficiency work.
