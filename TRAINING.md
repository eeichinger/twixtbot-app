# TwixBot Training Guide

Target hardware: AMD Ryzen 7800X3D (8C/16T), NVIDIA RTX 5070 Ti, 64 GB RAM,
Windows 11 + WSL2 Ubuntu 24.04.

The commands below assume that runtime environment. Adapt the `--device` flags
and paths for a different OS or hardware.

---

## Overview

The training pipeline has three components that run in sequence per iteration:

1. **Self-play data generation** — neural net inference server (`nns.py`) +
   parallel game workers (`battle.py` / `pmany.py`) produce binary training files.
2. **Training** — `train.py` consumes the binary files and updates the model in-place.
3. **Evaluation** — two `nns.py` servers (one per model) + `battle.py` / `pmany.py`
   produce win-rate statistics.

Repeat steps 1–2 until the model stops improving, then run step 3 to compare
candidates. A meaningful training run from a random init takes roughly 1–2 weeks
of intermittent sessions on the target hardware.

---

## Prerequisites

### WSL2 + CUDA setup (Windows host only)

1. Install WSL2 with Ubuntu 24.04:
   ```powershell
   wsl --set-default-version 2
   wsl --install -d Ubuntu-24.04
   ```
   Reboot when prompted.

2. Install Windows NVIDIA driver ≥ 570 (Blackwell / `sm_120`).
   **Do not** install a Linux driver inside WSL — WSL2 passes the Windows
   driver through via `/dev/dxg` automatically.

3. Verify inside Ubuntu:
   ```bash
   nvidia-smi
   ```

4. Move the repo into the WSL2 native filesystem (not `/mnt/c/...` — NTFS
   passthrough is slow and self-play writes hundreds of MB of `.bin` files):
   ```bash
   cp -r /mnt/c/path/to/twixtbot-app ~/twixtbot-app
   cd ~/twixtbot-app
   ```

5. Tune WSL2 resource limits. Create or edit `C:\Users\<you>\.wslconfig`:
   ```ini
   [wsl2]
   memory=52GB
   swap=0
   pageReporting=false
   ```
   Do **not** set `processors=` — the 7800X3D has SMT disabled (8 physical cores)
   and WSL2 uses all available cores by default. Restart WSL to apply:
   `wsl --shutdown` from PowerShell.

### System packages

```bash
apt-get install -y python3.12-dev
```

`python3.12-dev` is required because `torch.compile` uses Triton, which
JIT-compiles a C extension at runtime that `#include`s `Python.h`. Without
this package the first NNS startup with `--compile` fails with
`fatal error: Python.h: No such file or directory`.

### Python environment

```bash
cd ~/twixtbot-app
python3 -m venv .venv
source .venv/bin/activate

# Override the CPU-only index pinned in requirements.txt — install CUDA 12.8 build
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
pip install numpy pytest

export PYTHONPATH=$PWD/src
```

Persist the PYTHONPATH:
```bash
echo 'export PYTHONPATH=$HOME/twixtbot-app/src' >> ~/.bashrc
```

Verify CUDA is visible:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: True  NVIDIA GeForce RTX 5070 Ti
```

Confirm the baseline test suite passes:
```bash
pytest src/
```

All commands below assume you are in the repo root with the venv active and
`PYTHONPATH` set.

---

## Part A: Self-Play Training

### Step 1 — Create an initial model

Run this once to produce a randomly-initialised checkpoint:

```python
# create_model.py  (run with: python create_model.py)
import torch, sys
sys.path.insert(0, 'src')
from model import TwixNet

# Reasonable size for one RTX 5070 Ti (~1.9 M parameters)
net = TwixNet(num_filters=64, num_blocks=8)
torch.save(net, 'models/v0.pt')
print("Saved models/v0.pt")
```

```bash
mkdir -p models spdata
python create_model.py
cp models/v0.pt models/v0_backup.pt
```

Adjust `num_filters` / `num_blocks` to trade compute vs. strength:

| Config | Parameters | Approx. GPU time/batch |
|---|---|---|
| `filters=40, blocks=6` | ~0.5 M | fast (good for early iterations) |
| `filters=64, blocks=8` | ~1.9 M | moderate (default recommendation) |
| `filters=128, blocks=12` | ~8 M | slow (production strength) |


### Step 2 — Start the GPU inference server

`nns.py` loads the model onto the GPU and serves batched inference requests
over a Unix socket. Run this in a dedicated terminal (keep it running
throughout self-play generation):

```bash
python src/nns.py \
  --location /tmp/twixtbot_nns \
  --device cuda \
  --model models/v0.pt \
  --capacity 2048 \
  --compile \
  --fp16
```

Flag notes:
- `--device cuda` — run inference on GPU (default `cpu`).
- `--capacity 2048` — request-queue depth; must equal
  `num_clones × threads × 2 × async_calls` on the client side or slot allocation
  will fail silently.
- `--compile` — wraps the model with `torch.compile(mode='default')`, fusing
  conv+BN+activation kernels. First batch is slow (~30 s JIT); steady-state
  throughput improves ~20–40%.
- `--fp16` — float16 autocast on conv and matmul; ~1.5–2× throughput on
  Blackwell's fp16 hardware. Model weights stay fp32 on disk; silently ignored
  on `--device cpu`.

Stop the server cleanly:

```bash
python src/nns.py --location /tmp/twixtbot_nns --kill
```


### Step 3 — Generate self-play games

`battle.py` plays games against itself (both players use the same model) and
appends binary training records to a file via `--training_file`.

**Parallel multi-process (recommended for bulk generation):**

`pmany.py` launches `--num_clones` independent copies of the battle command.
The `%n%` token is replaced with a zero-padded clone index so each process
writes its own output file. The `--` separator is required — without it,
pmany's argparse consumes `--white`, `--black`, etc. meant for battle.py:

```bash
rm -rf logs/sp_gen && python src/pmany.py \
  --num_clones 16 \
  --log_dir logs/sp_gen \
  -- \
  python src/battle.py \
    --white "asn_player:location=/tmp/twixtbot_nns,trials=100,async_calls=32,add_noise=0.25,temperature=0.5" \
    --black "asn_player:location=/tmp/twixtbot_nns,trials=100,async_calls=32,add_noise=0.25,temperature=0.5" \
    --num_games 63 \
    --threads 2 \
    --training_file spdata/iter1_%n%.bin
```

This runs 16 processes × 2 threads × 2 players × 32 async queries =
2 048 concurrent NN queries in flight — hence `--capacity 2048` on NNS.
Total games: 16 × 63 ≈ 1 008. Each game produces ~40–60 training records
(one per move). Output: `spdata/iter1_00.bin` … `spdata/iter1_15.bin`,
logs in `logs/sp_gen/`.

Key parameters:
- `--num_clones 16` — independent OS processes.
- `--threads 2` — game threads per clone (two concurrent games per clone).
- `trials=100` — MCTS playouts per move. 100 is fast; raise to 200 in later
  iterations once the model knows basic tactics.
- `async_calls=32` — concurrent NN queries per player. With 32 clients × 32 in
  flight, the NNS receives ~320-position batches and stays GPU-busy ~97% of
  the time. See Appendix B for why going higher hurts throughput.
- `add_noise=0.25` — Dirichlet exploration noise at each expanded leaf's
  policy. Essential for self-play data diversity; set to `0` for evaluation.
- `temperature=0.5` — Move selection: sample proportional to visit-count² (soft
  but biased toward the best move). `1.0` = linear sample, `0.0` = greedy
  argmax. Must match `train.py --temperature`.
- `position_cache=1` — Enable per-worker transposition cache (Zobrist hash →
  cached policy/value). Skips NN queries for previously-seen board states.
  10–20% hit rate at `trials=100`, higher at `trials=200+`. Set to `0` or
  omit to disable. `train_loop.py` controls this via `POSITION_CACHE`.

Aim for at least **10 000–50 000 games** (several hundred MB of `.bin` files)
before training, split across a few iterations.

**Two self-play player implementations exist:**

- **`asn_player`** (asynchronous) — pipelines up to `async_calls` NN queries
  concurrently; required for GPU saturation. Use for bulk self-play.
- **`nnmplayer`** (synchronous) — one NN query per MCTS leaf, blocking.
  Simpler and better for debugging, but leaves most of the GPU idle. Accepts
  `add_noise=0.25,temperature=0.5` (Dirichlet root noise and soft move sampling
  for training diversity):
  ```bash
  python src/battle.py \
    --resource "nnclient:location=/tmp/twixtbot_nns,name=a" \
    --white "nnmplayer:resource=a,trials=200,add_noise=0.25,temperature=0.5" \
    --black "nnmplayer:resource=a,trials=200,add_noise=0.25,temperature=0.5" \
    --num_games 20 --threads 8 \
    --training_file spdata/games_000.bin
  ```


### Step 4 — Train the model

Kill the `nns.py` server to free the GPU, copy the model to the next version
(`train.py` saves in-place — copying first preserves the prior checkpoint),
then run:

```bash
python src/nns.py --location /tmp/twixtbot_nns --kill
cp models/v0.pt models/v1.pt

python src/train.py \
  --model models/v1.pt \
  --device cuda \
  --num_batches 2000 \
  --batch_size 256 \
  --learning_rate 0.01 \
  --decay_rate 0.95 \
  --temperature 0.5 \
  --policy_epsilon 0.01 \
  --save_after 500 \
  spdata/
```

Key flags:
- `--device cuda` — run training on GPU.
- `--num_batches 2000` — one epoch over ~500 K records at batch 256 ≈ 2 000 batches.
- `--decay_rate 0.95` — multiply LR by 0.95 whenever the batch loss goes up (soft annealing).
- `--temperature 0.5` — must match the temperature used during self-play.
- `--policy_epsilon 0.01` — label smoothing; prevents overconfident policy targets.
- `--save_after 500` — save every 500 batches as `models/v1.pt.500`, `.1000`, etc.
  Guards against crashes on long runs.
- `spdata/` — directory (recursively scanned) or individual `.bin` files.
  Subdirectories named `w=<float>/` get that sampling weight (see *Weighted sampling*).

Training prints per-batch `loss`, `slope`, `policy`, `value`, plus a 30-second
progress summary (`progress 147/1000 | elapsed 30s | 4.89 batch/s | ETA 174s`)
and a total-duration line at the end.

**Holdout evaluation** — measure generalisation on games not used for training:

```bash
mkdir -p spdata_holdout
# Move or copy some .bin files to spdata_holdout/ before training, then:
python src/train.py \
  --model models/v1.pt --device cuda \
  --holdout spdata_holdout/ \
  --num_batches 2000 \
  ...
```


### Step 5 — Restart the server and repeat

After training, restart the server with the updated model:

```bash
python src/nns.py \
  --location /tmp/twixtbot_nns \
  --device cuda \
  --model models/v1.pt \
  --capacity 2048 \
  --compile --fp16
```

Then go back to Step 3 with `--training_file spdata/iter2_%n%.bin`.
A typical iteration cadence:

| Iteration | games generated | training batches |
|---|---|---|
| 1 | 1 000 | 500 |
| 2–5 | 5 000 each | 1 000 each |
| 6+ | 10 000+ each | 2 000+ each |

Save each trained model as a new file (`v1.pt`, `v2.pt`, …) rather than
overwriting, so you can roll back or run arena comparisons later.


### Weighted sampling of training data

If you keep self-play data from multiple iterations, weight recent data more
heavily using `w=<float>/` subdirectories. `train.py` recursively scans the
directory tree and adjusts each file's basket weight when it encounters a
subdirectory whose name matches `w=<float>`:

```
spdata/
  w=0.2/          ← older games (20% sampling weight)
    iter1_*.bin
    iter2_*.bin
  w=0.8/          ← recent games (80% sampling weight)
    iter4_*.bin
    iter5_*.bin
```

```bash
python src/train.py --model models/v5.pt --device cuda ... spdata/
```

---

## Part B: Evaluating Two Models

Run each model in its own `nns.py` server on a **different socket path**, then
point `battle.py` at both. The arena uses stronger MCTS (`trials=400`) and no
exploration noise so results reflect model strength.

### Step 1 — Start two inference servers

Open two terminals:

```bash
# Terminal A — model under evaluation
python src/nns.py --location /tmp/twixtbot_nns_a --device cuda \
  --model models/v5.pt --capacity 512 --compile --fp16

# Terminal B — baseline / challenger
python src/nns.py --location /tmp/twixtbot_nns_b --device cuda \
  --model models/v4.pt --capacity 512 --compile --fp16
```

Lower `--capacity` (512 each) since the two servers share the same GPU;
enough for 8 clones × 2 threads × 32 async_calls per server.


### Step 2 — Run arena games

```bash
rm -rf logs/arena && python src/pmany.py \
  --num_clones 8 \
  --log_dir logs/arena \
  -- \
  python src/battle.py \
    --white "asn_player:location=/tmp/twixtbot_nns_a,trials=400,async_calls=32" \
    --black "asn_player:location=/tmp/twixtbot_nns_b,trials=400,async_calls=32" \
    --num_games 50 \
    --threads 2
```

8 clones × 50 games = 400 games total. `battle.py` automatically alternates
colours every game for fairness. The final output shows wins/draws/losses and
win-rate percentages per player spec.

**Interpreting results:**
- Win-rate > 55% at n=200 is a meaningful improvement (±7% margin at p=0.05).
- Win-rate > 60% at n=400 is a strong signal to promote the new model.


### Step 3 — Stop the servers

```bash
python src/nns.py --location /tmp/twixtbot_nns_a --kill
python src/nns.py --location /tmp/twixtbot_nns_b --kill
```

---

## Part C: Deploying the Trained Model

Ship a new model to the webapp:

```bash
# Export to ONNX (float32, BN folded into conv layers)
python tools/export_onnx.py --model models/v5.pt --out webapp/public/model.onnx

# INT8 dynamic quantization (in-place; saves fp32 backup as model.fp32.onnx)
# Reduces model size ~75%, cuts peak WASM heap on iOS
python tools/quantize_model.py

# Rebuild the webapp
cd webapp && npm run build
```

Play-test in the browser (including iOS if possible) before committing and pushing.

---

## Automating the Full Loop

`train_loop.py` orchestrates iterations of Phase A + Phase B + server restart
so you can run unattended overnight training:

```bash
# Run iterations 1..5 starting from v0.pt
python train_loop.py models/v0.pt

# Run iterations 4..8, resuming after a failed iter 3 → v3.pt
python train_loop.py models/v3.pt --start_iter 4 --total_iters 8
```

All tunables (num_clones, trials, async_calls, cadence table, weighted sampling)
are constants at the top of `train_loop.py` — edit those for experiments.
Logs land in `logs/train_loop_YYYYMMDD_HHMMSS.log`.

The script:
- Organises `spdata/` into `w=0.8/` (recent) and `w=0.2/` (older) tiers and
  physically moves iteration files across tiers as they age out of the
  `RECENCY_WINDOW` (default 3 iterations).
- Streams `train.py` output to both terminal and log file; emits a 60-second
  heartbeat during the self-play phase.
- Stops NNS cleanly between phases via `--kill`; cleans up stale `.sock` /
  `.shm` files before each new NNS start.

---

## Verification Smoke Test

Run this end-to-end check before committing to a multi-hour iteration.
Three terminals inside WSL2, each with `source .venv/bin/activate` and
`export PYTHONPATH=$PWD/src`.

**Terminal 1 — start NNS:**
```bash
python src/nns.py \
  --location /tmp/twixtbot_nns --device cuda \
  --model models/v0.pt --capacity 64
```

**Terminal 2 — run 2 self-play games:**
```bash
python src/battle.py \
  --white "asn_player:location=/tmp/twixtbot_nns,trials=10,async_calls=8" \
  --black "asn_player:location=/tmp/twixtbot_nns,trials=10,async_calls=8" \
  --num_games 2 --threads 1
```

**Terminal 2 (after games done) — run 5 training batches:**
```bash
python src/train.py \
  --model models/v0.pt --device cuda \
  --num_batches 5 --batch_size 16
```

**Terminal 1 — kill NNS:**
```bash
python src/nns.py --location /tmp/twixtbot_nns --kill
```

If all three steps succeed without errors, the pipeline is healthy.

---

## Reference: Key Parameters

| Parameter | Where | Default | Notes |
|---|---|---|---|
| `trials` | `nnmplayer`, `asn_player` | 100 | MCTS playouts/move. 100 for self-play iter 1–3, 200+ later, 400+ for evaluation |
| `async_calls` | `asn_player` | 8 | Concurrent NN queries per player. 32 is the sweet spot on 8 cores |
| `add_noise` | `nnmplayer`, `asn_player` | 0.0 | Dirichlet root noise. 0.25 for self-play, 0.0 for evaluation |
| `temperature` | `nnmplayer`, `asn_player` | 0.0 | Move selection. 0.5 for self-play, 0.0 for evaluation |
| `use_swap` | `nnmplayer` (0), `asn_player` (1) | — | Consult swap-rule model for move 1 |
| `random_rotation` | `nnmplayer`, `asn_player` | 1 | Random 0–3 rotation of NN input per leaf. Leave on |
| `--device` | `nns.py`, `train.py` | `cpu` | `cuda` for GPU |
| `--compile` | `nns.py` | off | `torch.compile` kernel fusion |
| `--fp16` | `nns.py` | off | float16 autocast inference |
| `--capacity` | `nns.py` | 200 | Request queue depth. Must match total in-flight: `num_clones × threads × 2 × async_calls` |
| `--batch_size` | `train.py` | 256 | Increase to 512 if GPU VRAM allows |
| `--learning_rate` | `train.py` | 0.01 | Reduce to 0.001 for fine-tuning late-stage models |
| `--threads` | `battle.py` | 0 (serial) | Parallel game threads per process. 2 on 7800X3D |
| `--num_clones` | `pmany.py` | — | Independent processes. 16 for 7800X3D at `async_calls=32` |
| `num_filters` | `TwixNet` | 40 | Larger = stronger but slower; 64 is a good balance |
| `num_blocks` | `TwixNet` | 12 | Residual depth; 8 with 64 filters is a good balance |

---

## Appendix A — Dual-GPU variant

If a second GPU is added later, the changes are local to Part A and Part B.
The loop overlaps self-play and training instead of serialising them.

- **GPU0** (primary) — NNS for self-play, running continuously:
  ```bash
  CUDA_VISIBLE_DEVICES=0 python src/nns.py \
    --location /tmp/twixtbot_nns --device cuda \
    --model models/v0.pt --capacity 2048 --compile --fp16
  ```

- **GPU1** (secondary) — `train.py` in parallel with self-play:
  ```bash
  CUDA_VISIBLE_DEVICES=1 python src/train.py \
    --model models/v1.pt --device cuda ...
  ```

NNS does **not** need to be stopped between training runs: training writes a
new checkpoint, then you SIGTERM NNS and relaunch against the latest model.
GPU0 stays at near-constant utilisation.

For the arena, run two NNS servers on either GPU at `--capacity 512` each —
VRAM is not the bottleneck for inference-only on a 1.9 M-param model.

No code changes needed. GPU selection is by `CUDA_VISIBLE_DEVICES`.

Expected wall-clock speedup: **~1.5–1.7×** per iteration vs single-GPU serial
loop, limited by how well the CPU can feed both GPUs simultaneously.

---

## Appendix B — GPU is the Bottleneck: Hardware Analysis

A representative NNS milestone during steady-state self-play on the reference
hardware (`--compile --fp16`, `async_calls=32`, 16 clones):

```
waiting:      N=30753  T=12.8    avg=0.000417
preprocessing: N=30753  T=12.3    avg=0.000400
gpu:          N=30753  T=1068.3  W=9.84e+06  avg=-0.000322 + 0.000110*W
```

Derived metrics:

| Metric | Value | Interpretation |
|---|---|---|
| Average batch size W/N | **320 positions/batch** | CPU workers are feeding NNS well |
| GPU busy fraction T/wallclock | **~97%** | NNS is GPU-saturated, not CPU-starved |
| Sustained throughput | **~12 000 evals/sec** | → ~4 000–5 000 games/hour at `trials=100` |
| nvidia-smi GPU utilisation | ~35–40% | Reflects memory-bandwidth-limited SM occupancy |

The nvidia-smi reading does **not** indicate the NNS is underfed. The
convolutions are memory-bandwidth bound (24×22 spatial grid, 64 filters), so
SMs are idle waiting on memory I/O even while the kernel is running. Adding
more CPU workers does not help once the NNS is already 97% busy — the model
itself is the limit.

### Why `async_calls=32` (not higher)

Measured progression on the reference hardware:

| Config | Batch size | NNS waiting/cycle | Throughput |
|---|---|---|---|
| No compile, no fp16, async=32 | 320 | 0.42 ms | 8 960 evals/s |
| `--compile --fp16`, async=32 | 320 | 1.24 ms | 12 093 evals/s |
| `--compile --fp16`, async=48 | 399 | 8.14 ms | 11 441 evals/s |

Going from 32 → 48 async made batches slightly larger but broke the
smooth-stream invariant: 3 072 simultaneous replies arrive at once, all
workers do tree traversal in the same CPU burst, and NNS sits idle 23% of
wall clock waiting for the next query burst. Net throughput dropped ~5%.
`async_calls=32` is the sweet spot for 8 physical cores.

### Why `--capacity` must match

The NNS capacity formula:
```
capacity = num_clones × threads × 2 × async_calls
```
If capacity is too low, `smmpp` slot allocation fails silently — a worker
connects, tries to reserve slots, the server returns an error, and the client
hangs. With the recommended config: 16 × 2 × 2 × 32 = 2048.

### CPU upgrade: 7800X3D vs 9950X3D

Because the GPU is the bottleneck, not the CPU, doubling core count from 8
(7800X3D) to 16 (9950X3D, one 3D V-Cache CCD + one standard CCD) gives
diminishing returns:

| Phase | Expected speedup | Rationale |
|---|---|---|
| Self-play (single-GPU) | **~1.1–1.2×** | GPU already 97% busy; extra CPU only modestly grows average batch size |
| Training | **~1.0×** | GPU-bound; CPU only feeds batches |
| Arena | **~1.1–1.3×** | Two NNS share one GPU — extra cores reduce idle gaps slightly |
| **Full iteration (dual-GPU)** | **~1.5–1.7×** | Training parallelism helps; extra CPU sustains both GPUs |

Bottom line: a 9950X3D is worth it mainly for dual-GPU training. For
single-GPU serial iterations, `--compile --fp16` on NNS is a bigger lever
than a CPU upgrade.

### Windows-side tips (either CPU)

- Disable **Core Isolation / Memory Integrity (HVCI)** in Windows Security —
  costs 10–15% CPU under WSL2.
- Set Windows power plan to **Best Performance** during training runs.
- Close Chrome/Edge — background timers degrade 3D V-Cache hit rates.
