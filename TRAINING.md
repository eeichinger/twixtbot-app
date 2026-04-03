# TwixBot Training Guide

Target hardware: AMD Ryzen 7800X3D (8C/16T), NVIDIA RTX 5070 Ti, 64 GB RAM.

---

## Overview

The training pipeline has three components that run in sequence:

1. **Self-play data generation** — neural net inference server (`nns.py`) + parallel game workers (`battle.py` / `pmany.py`) produce binary training files.
2. **Training** — `train.py` consumes the binary files and updates the model in-place.
3. **Evaluation** — two `nns.py` servers (one per model) + `battle.py` / `pmany.py` produce win-rate statistics.

Repeat steps 1–2 until the model stops improving, then run step 3 to compare candidates.

---

## Prerequisites

```bash
cd /home/user/twixtbot-app
pip install torch torchvision numpy pytest   # if not already installed
export PYTHONPATH=$PWD/src
```

All commands below assume you are in the repo root and `PYTHONPATH` is set.

---

## Part A: Self-Play Training

### Step 1 — Create an initial model

Run this once to produce a randomly-initialised checkpoint:

```python
# create_model.py  (run with: python create_model.py)
import torch, sys
sys.path.insert(0, 'src')
from model import TwixNet

# Reasonable size for one RTX 5070 Ti (~1.5 M parameters)
net = TwixNet(num_filters=64, num_blocks=8)
torch.save(net, 'models/v0.pt')
print("Saved models/v0.pt")
```

```bash
mkdir -p models spdata
python create_model.py
```

Adjust `num_filters` / `num_blocks` to trade compute vs. strength:

| Config | Parameters | Approx. GPU time/batch |
|---|---|---|
| `filters=40, blocks=6` | ~0.5 M | fast (good for early iterations) |
| `filters=64, blocks=8` | ~1.5 M | moderate |
| `filters=128, blocks=12` | ~8 M | slow (production strength) |


### Step 2 — Start the GPU inference server

`nns.py` loads the model onto the GPU and serves batched inference requests over a Unix socket.
Run this in a dedicated terminal (keep it running throughout self-play generation):

```bash
python src/nns.py \
  --location /tmp/twixtbot_nns \
  --model models/v0.pt \
  --capacity 200
```

`--capacity 200` is the request-queue depth; the default is fine for 16 worker threads.

To stop the server cleanly when you are done:

```bash
python src/nns.py --location /tmp/twixtbot_nns --kill
```


### Step 3 — Generate self-play games

Each `battle.py` invocation plays games against itself (both players use the same model)
and appends binary training records to a file via `--training_file`.

**Single-process (quick test):**

```bash
python src/battle.py \
  --resource "nnclient:location=/tmp/twixtbot_nns,name=a" \
  --white    "nnmplayer:resource=a,trials=200,add_noise=0.25,temperature=0.5" \
  --black    "nnmplayer:resource=a,trials=200,add_noise=0.25,temperature=0.5" \
  --num_games 20 \
  --threads 8 \
  --training_file spdata/games_000.bin
```

Key flags:
- `--threads 8` — run 8 games in parallel (the 7800X3D has 16 threads; 8 leaves headroom for the GPU server process).
- `trials=200` — MCTS playouts per move. Increase for stronger play / higher-quality data; decrease for faster generation.
- `add_noise=0.25` — Dirichlet exploration noise at the MCTS root. Required for diverse self-play.
- `temperature=0.5` — Move selection temperature (sqrt of visit counts). Produces softer targets than greedy `0.0`.

**Parallel multi-process (recommended for bulk generation):**

`pmany.py` launches `--num_clones` independent copies of the battle command.
The `%n%` token is replaced with a zero-padded clone index so each process writes its own file:

```bash
python src/pmany.py \
  --num_clones 4 \
  --log_dir logs/sp_gen \
  python src/battle.py \
    --resource "nnclient:location=/tmp/twixtbot_nns,name=a" \
    --white    "nnmplayer:resource=a,trials=200,add_noise=0.25,temperature=0.5" \
    --black    "nnmplayer:resource=a,trials=200,add_noise=0.25,temperature=0.5" \
    --num_games 100 \
    --threads 4 \
    --training_file spdata/games_%n%.bin
```

This runs 4 processes × 4 threads = 16 concurrent game workers, saturating the CPU.
Total games: 4 × 100 = 400. Each game produces ~40–60 training records (one per move).
Output: `spdata/games_00.bin` … `spdata/games_03.bin`, logs in `logs/sp_gen/`.

Aim for at least **10 000–50 000 games** (several hundred MB of `.bin` files) before training.


### Step 4 — Train the model

Stop or leave the `nns.py` server running (it holds the model in GPU memory but
`train.py` works independently on CPU/GPU; they do not share state).

```bash
python src/train.py \
  --model      models/v0.pt \
  --num_batches 2000 \
  --batch_size  256 \
  --learning_rate 0.01 \
  --decay_rate    0.95 \
  --temperature   0.5 \
  --policy_epsilon 0.01 \
  spdata/
```

`train.py` modifies `models/v0.pt` in-place. Back it up first if you want to keep the old weights:

```bash
cp models/v0.pt models/v0_backup.pt
python src/train.py --model models/v0.pt ...
```

Key flags:
- `--num_batches 2000` — number of gradient steps. One epoch over ~500 K records at batch 256 ≈ 2 000 batches.
- `--decay_rate 0.95` — multiply LR by 0.95 whenever the batch loss goes up (soft annealing).
- `--temperature 0.5` — must match the temperature used during self-play generation.
- `--policy_epsilon 0.01` — small label smoothing; prevents overconfident policy targets.
- `spdata/` — path to a directory (or individual `.bin` files). Subdirectories named `w=<float>/` get that sampling weight (see *Weighted sampling* below).

**Save intermediate checkpoints** to guard against crashes:

```bash
python src/train.py \
  --model models/v0.pt \
  --num_batches 2000 \
  --save_after 500 \     # save every 500 batches as models/v0.pt.500, .1000, etc.
  ...
```

**Holdout evaluation** — measure generalisation on games not used for training:

```bash
mkdir -p spdata_holdout
# Move or copy some .bin files to spdata_holdout/ before training, then:
python src/train.py \
  --model      models/v0.pt \
  --holdout    spdata_holdout/ \
  --num_batches 2000 \
  ...
```

Training prints per-batch: `[batch] total_loss  policy_loss  value_loss  lr`.


### Step 5 — Restart the server and repeat

After training, kill the old server and restart it with the updated model:

```bash
python src/nns.py --location /tmp/twixtbot_nns --kill
python src/nns.py --location /tmp/twixtbot_nns --model models/v0.pt --capacity 200
```

Then go back to Step 3. A typical iteration:

| Iteration | games generated | training batches |
|---|---|---|
| 1 | 1 000 | 500 |
| 2–5 | 5 000 each | 1 000 each |
| 6+ | 10 000+ each | 2 000+ each |

Save each iteration as a new checkpoint (`v1.pt`, `v2.pt`, …) rather than overwriting,
so you can roll back:

```bash
cp models/v0.pt models/v1.pt
python src/train.py --model models/v1.pt ...
```


### Weighted sampling of training data

If you keep self-play data from multiple iterations, weight recent data more heavily
using `w=<float>/` subdirectories:

```
spdata/
  w=0.2/          ← older games (20% sampling weight)
    iter1_*.bin
  w=0.8/          ← recent games (80% sampling weight)
    iter5_*.bin
```

```bash
python src/train.py --model models/v5.pt spdata/
```

---

## Part B: Evaluating Two Models

Run each model in its own `nns.py` server on a **different socket path**, then point
`battle.py` at both.

### Step 1 — Start two inference servers

Open two terminals:

```bash
# Terminal A — model under evaluation
python src/nns.py --location /tmp/twixtbot_nns_a --model models/v5.pt --capacity 100

# Terminal B — baseline / challenger
python src/nns.py --location /tmp/twixtbot_nns_b --model models/v4.pt --capacity 100
```

Lower `--capacity` to 100 since the two servers share the same GPU; 100 each is
enough for 8 concurrent games.


### Step 2 — Run arena games

```bash
python src/battle.py \
  --resource "nnclient:location=/tmp/twixtbot_nns_a,name=model_a" \
  --resource "nnclient:location=/tmp/twixtbot_nns_b,name=model_b" \
  --white "nnmplayer:resource=model_a,trials=400" \
  --black "nnmplayer:resource=model_b,trials=400" \
  --num_games 200 \
  --threads 8
```

`battle.py` automatically alternates colours every game for fairness.
The final output shows wins/draws/losses and win-rate percentages for each player spec.

**Parallel arena (faster):**

```bash
python src/pmany.py \
  --num_clones 4 \
  --log_dir logs/arena_v5_vs_v4 \
  python src/battle.py \
    --resource "nnclient:location=/tmp/twixtbot_nns_a,name=model_a" \
    --resource "nnclient:location=/tmp/twixtbot_nns_b,name=model_b" \
    --white "nnmplayer:resource=model_a,trials=400" \
    --black "nnmplayer:resource=model_b,trials=400" \
    --num_games 50 \
    --threads 4
```

4 processes × 4 threads = 16 concurrent games, 200 total.
Logs per clone in `logs/arena_v5_vs_v4/00.log` … `03.log`.

**Interpreting results:**
- Win-rate > 55% is a meaningful improvement at n=200 games (±7% statistical margin).
- Win-rate > 60% at n=400 is a strong signal to promote the new model.


### Step 3 — Stop the servers

```bash
python src/nns.py --location /tmp/twixtbot_nns_a --kill
python src/nns.py --location /tmp/twixtbot_nns_b --kill
```

---

## Reference: Key Parameters

| Parameter | Where | Default | Notes |
|---|---|---|---|
| `trials` | `nnmplayer` | 100 | MCTS playouts/move. 200 for self-play, 400+ for evaluation |
| `add_noise` | `nnmplayer` | 0.0 | Dirichlet noise. Set 0.25 for self-play, 0.0 for evaluation |
| `temperature` | `nnmplayer` | 0.0 | Move selection. 0.5 for self-play, 0.0 for evaluation |
| `--batch_size` | `train.py` | 256 | Increase to 512 if GPU VRAM allows |
| `--learning_rate` | `train.py` | 0.01 | Reduce to 0.001 for fine-tuning late-stage models |
| `--capacity` | `nns.py` | 200 | Request queue depth. 200 for one server, 100 each for two |
| `--threads` | `battle.py` | 0 (serial) | Parallel game workers. 8–12 on a 7800X3D |
| `--num_clones` | `pmany.py` | — | Independent processes. 4 for 16-thread CPU |
| `num_filters` | `TwixNet` | 40 | Larger = stronger but slower; 64 is a good balance |
| `num_blocks` | `TwixNet` | 12 | Residual depth; 8 with larger filters is often better |
