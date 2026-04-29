# Training-effectiveness research notes

## Goal

**The objective is to develop training methods that *match* the strength
of the pre-trained `models/six-917000.pt` (Lampe's TF1 ancestor),
ideally on commodity hardware (Ryzen 7800X3D + RTX 5070 Ti, 16 GB
VRAM), with optional offload to AWS GPU spot instances acceptable.**

The objective is **not** to surpass `six-917000` — it is already much
stronger than most humans and is the strength target for this work.

Two underlying motivations:

1. **Educational** — learn end-to-end what it takes to train a strong
   AlphaZero-style model from (near-)scratch.
2. **Methodological exploration** — try newer / cheaper approaches than
   vanilla AlphaZero, particularly ones that **reduce the
   computational cost of MCTS-driven self-play**. Examples: Gumbel
   AlphaZero / QZero (see `03-search-efficiency.md`), curriculum
   sampling across model windows (KataGo), better data curation, etc.

**What this implies for prioritisation:**

- Knowledge distillation from `six-917000` is a *teacher shortcut*.
  Useful diagnostically (e.g., it answered "does our arch have enough
  capacity?" — yes, see `02-model-scaling.md`) but not on the critical
  path. We do not intend to ship a distilled-only model as the final
  result, because it doesn't teach us what we want to learn.
- Wall-clock and GPU-hours matter. A method that gets us 80% of the way
  in 1/10th the compute is more interesting than one that gets us 100%
  with the standard cost.
- Single arena-strength numbers are means, not ends. Understanding *why*
  a technique works is the deliverable.

## How this directory is organised

Running log of research and small experiments. Throughput /
orchestration improvements live in `docs/improvements.md` (A-prefix
algorithm, B-prefix model) and `docs/further_training_improvements.md`;
this directory is for the exploratory phase that comes *before* a topic
gets a feature ID.

These notes are designed for hobby-pace work: pick a topic, advance the
file, commit. When we resume after a gap, the topic's "Where we left
off" / "Next action" sections rebuild the context fast.

## Topics

| File | Topic | Status | Cross-refs |
|---|---|---|---|
| [01-replay-buffer-sampling.md](01-replay-buffer-sampling.md) | Replay buffer + minibatch sampling vs full-pass training | closed (batch-size lever); KataGo curriculum deferred | (none yet) |
| [02-model-scaling.md](02-model-scaling.md) | Scaling depth / width / heads | closed (capacity sufficient) | B7, B5 in `improvements.md`; §5–5c in `further_training_improvements.md` |
| **[03-search-efficiency.md](03-search-efficiency.md)** | **Reducing trials per move (Gumbel AlphaZero, "QZero")** | **next** | A4 (cpuct tuning) in `improvements.md` |
| [04-global-board-features.md](04-global-board-features.md) | End-to-end connection awareness (KataGo-style global pooling) | scoped | **B9a** in `improvements.md`; §5b in `further_training_improvements.md` |

**Status legend:**
- `scoped` — question framed, references identified, no investigation done yet
- `next` — promoted as the active focus; pick this up first when resuming
- `investigating` — actively reading / prototyping / measuring
- `closed` — outcome reached on the question's main branch; further work,
  if any, is captured in the file's "Where we left off"
- `shelved` — explored and parked; reasons recorded in the file

**Quick session-resume index:** active topic is **03 (search efficiency)**.
Topics 01 and 02 are closed for now; their experiment artifacts and
findings are in the respective files. The "Goal" section above is the
north star — re-read it before starting new experiments.

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

As of v0_distill_10k (April 29, 2026):
- **Model:** 8-block × 64-filter ResNet (`src/model.py`). ~1.9M parameters.
  Activation: GELU. 528 policy outputs (24×22 inner grid). 3-class value
  head (loss/draw/win). VALID padding in the value head, 2 stride-2
  reductions. Verified stronger than the deeper-narrower 48f×20b "Lampe"
  shape at our training compute (see `02-model-scaling.md`).
- **Strongest checkpoint:** `models/v0_distill_10k.pt` — random init
  distilled from `six-917000.pt` for 10k batches × 256 (~25 min). Beats
  `v8_F.pt` decisively (transitively via v0_distill 200/0; v0_distill_10k
  beats v0_distill 178/22).
- **Training cadence:** 10,000 self-play games per iter at trials=200; per
  topic 01's batch-size investigation, Phase B is now 125 batches at
  batch=4096 / lr=0.16 / `--warmup_steps 25` (16× larger batches than the
  original 2000×256, same total samples).
- **Distillation:** `train.py --teacher PATH` enables soft-target
  distillation (replaces MCTS/z labels with teacher's softmax outputs).
  Compatible with all other flags. ~30% extra wall-clock per step.
- **Hardware:** 7800X3D + 5070 Ti (16 GB). batch=8192 OOMs; 4096 is the
  practical training-batch ceiling on this card.
- **NN throughput:** ~13,200 positions/s at batch=619 (compile + fp16).
- **Self-play data volume:** ~4M positions per iter (~7 GB `.bin` files).
  ~410 moves/game, decreasing slowly with model strength.
- **Replay weighting:** `USE_WEIGHTED_SAMPLING = True`, recent tier (last 3
  iters) at weight 0.8, older at 0.2.
- **Reference baseline:** `models/six-917000.pt` (Lampe's TF1 ancestor,
  ported via `tools/convert_tf1_to_pt.py`). Still beats v0_distill_10k
  200/0 — the remaining gap is likely distribution mismatch
  (`spdata/` was generated by weak v8 self-play, doesn't cover the
  positions a strong agent visits).

The "iter 8 plays itself in self-play with much fewer position-cache hits
than iter 6" observation (NN evals per game ~doubled from ~88 to ~200 per
move) is a baseline data point for any future search-efficiency work.
