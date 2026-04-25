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
- A flat ~50/50 result is *not* automatically "training failed" — see
  **Appendix E** for the diagnostic checklist before drawing that conclusion.


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

### Pausing to free the GPU

Long-running self-play and the things you actually want your PC for (gaming,
video calls, anything GPU-heavy) compete for the same hardware. The 5070 Ti
isn't preemptible the way CPU is — the only reliable way to free it is to
shut down the NNS process that holds it.

Use the convenience script:

```bash
scripts/pause-self-play.sh
```

It SIGINTs `arena.py` if running (which lets it cleanly terminate clones and
SUICIDE-kill its NNS instances), SIGTERMs any stray `battle.py` / `pmany.py`
clones, then sends `--kill` to any remaining NNS processes by parsing their
`--location` from the command line. The GPU is free within ~10 seconds.

To resume:

- **Self-play training:** re-run the command you used to start it
  (`train_loop.py models/vN.pt --start_iter K`, or your manual sequence).
  The orchestrator reads the latest checkpoint; you lose at most one
  in-progress game per worker (completed games are flushed to `spdata/` per-game).
- **Arena (`arena.py`):** the script stops it cleanly, but `arena.py` runs
  are *not* mid-flight resumable — restarting begins a fresh batch of games.
  If you needed the partial results, copy `logs/arena/` aside before pausing.

The script does **not** touch `train.py` (the per-iteration training step).
Training a single iteration takes minutes-to-hours; if you really need the
GPU mid-training, kill it manually and re-run the iteration. Otherwise
let it finish — its output checkpoint is what the next iteration's
self-play uses.

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

---

## Appendix C — Running on Apple Silicon (Mac)

The full training/arena pipeline runs on Apple M1/M2/M3 Macs using the **MPS**
backend (Metal Performance Shaders). Intended use case: small-scale iteration,
sanity checks, and running the arena locally against a candidate model while
the 5070 Ti PC is busy with self-play. For serious training volume, SSH into
the PC — the Mac is 5–10× slower on this workload.

### Prerequisites

- Python 3.11 (the repo's `requirements.txt` targets 3.11). Newer minor
  versions (3.12+) may also work; Python 3.14 will not — `tensorflow` has no
  wheels, blocking `tools/convert_tf1_to_pt.py`. Install via Homebrew:
  ```bash
  brew install python@3.11
  /opt/homebrew/bin/python3.11 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```
- PyTorch with MPS support ships in the standard `torch` wheel — no extra
  install needed.
- `tools/convert_tf1_to_pt.py` additionally needs TensorFlow. On macOS the
  package is `tensorflow` (not `tensorflow-cpu`, which is Linux-only) — a
  one-off `pip install tensorflow` in the venv is enough.

### Device selection

Use `--device mps`. The MPS backend is **5-8× faster than CPU** for this
model; use CPU only as a correctness fallback when diagnosing MPS issues.

```bash
# Fast (MPS)
python src/nns.py --location /tmp/twixtbot_nns_v3 --device mps \
  --model models/v3.pt --capacity 1024 --compile

# Slow fallback (CPU) — only if MPS has a known problem
python src/nns.py --location /tmp/twixtbot_nns_v3 --device cpu \
  --model models/v3.pt --capacity 1024
```

`--fp16` has no effect on MPS (disabled for CPU too; see `src/nneval.py:34`).
Leave it off.

### `--compile` on MPS

When `--compile` is passed, `NNEvaluater` automatically uses
`torch.compile(..., dynamic=True)` on MPS. This is important: without
`dynamic=True`, Inductor recompiles for every new batch size, inflating
per-call overhead 3× (see measurements below). On CUDA, shape specialization
is the right default and `dynamic=False` stays active.

Net effect of `--compile` on MPS at steady state:
- `a` (fixed per-call overhead): 21.7 ms → **15.0 ms** (−30%)
- `b` (per-example cost): 0.457 ms → 0.593 ms (+30%)
- Breakeven batch: ~100 positions. At observed batch sizes 70-130, a 3-5%
  throughput win.

Small gain, but free once the flag is wired. Recommended on by default.

### Arena on Mac

The simplest path is `arena.py` — one command, supervises both NNS instances
and all battle clones, prints aggregated progress and a final summary with
confidence interval:

```bash
python src/arena.py \
  --model-a models/v2.pt --model-b models/v3.pt \
  --device mps \
  --total_games 400 --num_clones 8 \
  --trials 200 --async_calls 32
```

Defaults: `--threads 2` (the sweet spot per Appendix B), `--compile` enabled,
logs to `logs/arena/`. Sockets are derived from model basenames (e.g.
`models/v2.pt` → `/tmp/twixtbot_nns_v2.sock`). NNS `--capacity` is computed
automatically from `num_clones × threads × async_calls × 2`.

The lower-level `pmany + battle.py` form is still available for anything
that arena.py doesn't cover — for example, running with `--training_file`,
custom init moves, or a mix of player types. See `src/pmany.py --help`
and `src/battle.py --help`.

Notes on the parameters:
- `trials=200` is a balance: enough search depth for arena results to be
  trustworthy, but lower than the reference `trials=400` to keep wall-clock
  reasonable on Mac. Use 50-100 only for very quick smoke tests; below
  ~150 trials, low-search noise can dominate small model strength
  differences (see Appendix E).
- `async_calls=32` is the per-client in-flight limit. On MPS, **higher
  values do not help** if the trials count is already 200+ — the
  query inflow rate from MCTS at higher trials is what drives larger
  batches at the NNS, not the per-client cap.
- `--capacity 1024` per NNS is fine; both share unified memory.

### Expected throughput

Throughput on MPS depends strongly on the average batch size, which in turn
depends on the `trials` count (more trials → more queries per client per
second → larger batches at the NNS). Two distinct regimes to expect:

| Config | Typical `W/N` | `b` (ms/pos) | `a/b` | **Throughput** |
|---|---|---|---|---|
| `trials=50, async_calls=64` (smoke test) | ~110 | 0.58 | ~19 | ~1450 pos/s |
| **`trials=200, async_calls=32` (proper arena)** | **~170** | **0.36** | **~66** | **~2000 pos/s** |

`b` is *not* constant on MPS. There's a batch-size knee where the GPU
transitions from being underutilized (per-example cost dominated by
dispatch overhead) to being properly fed (per-example cost approaches
the silicon's compute limit). For this 20-block × 48-filter ResNet that
knee sits somewhere around `W/N ≈ 130-150`. Push batches across the
knee and per-example cost drops sharply.

Practical implication: the `1/b` asymptote you can compute from any single
NNS milestone reading is only a local prediction. Batch regimes well
above the data range can show a *different* `b`, and therefore a higher
ceiling. Don't conclude "the Mac is at its throughput ceiling" from a
short, low-trials run.

For reference: the 5070 Ti with `--compile --fp16` processes ~10,000+
positions/sec on the same model. If the Mac number looks far lower than
the table values, check: (a) battery not on low-power mode, (b) nothing
else saturating the GPU, (c) let it run 10+ minutes before reading stats —
`torch.compile` warmup can take several minutes to fully amortize.

### macOS-specific gotchas

- **Multiprocessing start method** — `src/nns.py` relies on the `fork`
  start method. macOS defaults to `spawn` since Python 3.8, which cannot
  transport `mmap` objects to child processes. If you see
  `TypeError: cannot pickle 'mmap.mmap' object`, set `fork` explicitly
  at the top of `src/nns.py`:
  ```python
  import multiprocessing
  multiprocessing.set_start_method('fork', force=True)
  ```
- **Socket accept backlog** — `src/smmpp.py` uses `listen(128)` which
  matches macOS's `kern.somaxconn`. Earlier versions used `listen(5)`
  which caused `Connection refused` under arena load on Mac. If rolling
  back, keep the bump.
- **Training on Mac** — `src/train.py` works on MPS but is not
  throughput-competitive with the PC. Acceptable for debugging training
  code; not for real iterations.

### When to use the Mac vs the PC

| Task | Where |
|---|---|
| Code iteration, debugging, unit tests | Mac |
| Small arena smoke test (n=20-50, trials=50) | Mac |
| Full arena evaluation (n=400, trials=400) | PC (5070 Ti) |
| Self-play training | PC (5070 Ti) |
| Model conversion + ONNX export + quantization | CI (Ubuntu) |

See `docs/wsl2-ssh-setup.md` for the SSH workflow for offloading to the PC.

---

## Appendix D — How to calculate throughput from NNS GPU stats

NNS prints a milestone stats block every `--milestone_step` positions. The
`gpu:` line is the one that matters for throughput. Example from a real run:

```
gpu: N=4110 T=358.384 W=500303 W/N=121.7 W/T=1396/s avg=0.015033 + 0.000593*W a/b=25.4
```

Two ways to compute throughput from this. Both give the same answer for the
run you measured. The simpler one is just division.

### Method 1 (simple): direct ratio from raw counts

The NNS counts every position evaluated (`W`) and every second spent in GPU
forward passes (`T`):

```
positions / second = W / T = 500303 / 358.384 = 1395.8 pos/s
```

That's the actual throughput observed during the run. No model, no math
beyond division. This value is also printed directly as `W/T=1396/s` in the
milestone output, so usually you don't need to compute it yourself.

### Method 2 (modeled): plug observed batch size into the regression

The line `avg = 0.015033 + 0.000593*W` is a linear regression fit:

```
time_per_batch (seconds) = a + b · W
                         = 0.015033 + 0.000593 · W
```

where `W` here is the batch size (number of positions in that one call). At
the observed average batch size `W = 121.7`:

```
time_per_batch = 0.015033 + 0.000593 · 121.7
               = 0.015033 + 0.072168
               = 0.087201 s
```

That's the predicted time for one batch of 121.7 positions. Positions per
second:

```
positions_per_second = batch_size / time_per_batch
                     = 121.7 / 0.087201
                     = 1395.4 pos/s
```

Same number (within rounding) as Method 1. They have to match — the
regression is fit on the same data that produced `T` and `W`.

### Why bother with Method 2?

Because once you have `a` and `b`, you can predict throughput at batch sizes
you didn't actually observe. That's the whole point of the regression.
Examples:

| hypothetical batch | predicted batch time | predicted pos/s |
|---|---|---|
| 32 | 0.015033 + 32·0.000593 = 0.0340 s | 941 |
| 64 | 0.015033 + 64·0.000593 = 0.0530 s | 1208 |
| **121.7 (observed)** | **0.0872 s** | **1395** |
| 256 | 0.015033 + 256·0.000593 = 0.1668 s | 1535 |
| 1000 | 0.015033 + 1000·0.000593 = 0.6080 s | 1645 |

So the regression lets you answer "what if I could push batch sizes higher?"
without actually running the experiment. The asymptote (very large W)
approaches `1/b = 1/0.000593 ≈ 1686 pos/s` — the maximum throughput
predicted *by this fit*, achievable only if you could amortize the fixed
overhead `a` across infinite work.

**Caveat: `b` is only locally constant.** The regression assumes a linear
time-vs-work relationship, which holds well within the range of batch sizes
you actually observed. Real GPUs (especially integrated ones like MPS) have
a *batch-size knee* — below it, compute units are underutilized and per-
example cost is high; above it, the device is properly fed and `b` drops
sharply. So extrapolating `1/b` from a low-batch run can underestimate the
true ceiling. If you push batch size into a different regime (e.g. by
raising `trials`), refit and you may see a meaningfully smaller `b` and
higher asymptote. See Appendix C for an example: at `W/N ≈ 110` MPS shows
`b ≈ 0.58 ms` (asymptote ~1700 pos/s); at `W/N ≈ 170` the same hardware
shows `b ≈ 0.36 ms` (asymptote ~2800 pos/s).

### Quick recap of the symbols

| symbol | meaning |
|---|---|
| `N` | total batches (forward passes) so far |
| `T` | total seconds in those forward passes |
| `W` | total positions evaluated across all batches |
| `W/N` | average batch size — what you actually ran at |
| `a` | fixed per-batch overhead (kernel launch / dispatch); ~15 ms here |
| `b` | per-position compute cost; ~0.6 ms here |
| `a/b` | crossover batch — at this batch size, fixed cost = compute cost |

Two quick rules of thumb:
- **Are batches big enough?** Compare `W/N` to `a/b`. Example: 121.7 vs 25.4
  → batches are ~5× the crossover, so well-batched.
- **Real throughput?** Just `W / T`. You don't need the regression for that.

---

## Appendix E — Assessing training progress (diagnosing flat arena results)

When a freshly-trained model goes head-to-head with its predecessor and the
arena returns ~50% wins for each side, the natural reaction is "training did
nothing." That conclusion is often wrong. This appendix is a diagnostic guide
for what to check before deciding whether the training pipeline failed,
whether the arena setup hid the improvement, or whether you genuinely need
more iterations.

### Why a flat result can be misleading

400 games is statistically informative — the 95% confidence interval on a
50/50 result is roughly ±5%. So the *measurement* is precise. But what that
measurement *means* depends critically on the `trials=N` setting in
`battle.py`'s asn_player spec.

At `trials=50`, MCTS does very little search relative to the network's raw
policy. The arena is essentially measuring "which network's untreated policy
is better." Several effects compete:

- Network differences get **amplified** if the new policy is meaningfully
  different — a few extra rollouts won't matter, the policy itself decides.
- Search **can't rescue** an improved network whose advantage only manifests
  in positions that need 200+ rollouts to evaluate correctly.
- At low trials, search noise is high enough that small but real strength
  differences can be drowned out.

So a 50/50 at `trials=50` can mean:
1. v(N) and v(N+1) really are equal strength — training had no effect.
2. v(N+1) is better but its advantage only shows up at higher search depth.
3. The networks differ but in symmetric ways that cancel at low trials.

The reference Part B workflow specifies **`trials=400`** for arena exactly
to avoid this ambiguity. If you ran a quick smoke arena at `trials=50` (as
recommended for the Mac in Appendix C), the 50/50 result is **not yet** a
strong signal of training failure. It's a preliminary indicator that warrants
a proper-trials follow-up before drawing conclusions.

### Diagnostic checklist (in order of effort)

Run through these before concluding training failed. The order is roughly
cheapest-first.

#### 1. Are the two model files actually different?

```bash
md5sum models/v2.pt models/v3.pt
```

If the hashes match, you're playing the same model against itself. Common
ways this happens: forgetting to run training, training crashing silently,
or copying the wrong source file when staging the new arena.

#### 2. Did the training loss actually decrease?

Look at the log output from the `train.py` step that produced the new model.
The expected pattern is total loss trending downward over the training
epochs (with normal short-term wobble). If loss is flat from start to end,
the network never learned anything from the self-play data — and 50/50 in
the arena is the correct outcome. The fix is in training, not arena: check
learning rate, batch size, optimizer settings, or whether the self-play
data file was loaded at all.

#### 3. Re-run a smaller arena at the canonical trials count

Even 50 games at `trials=400` (≈ 1 hour on the 5070 Ti, longer on Mac) is
much more diagnostic than 400 games at `trials=50`:

- Win rate ≥ 60% for the new model → training worked, low-trials arena was
  hiding the improvement. Promote the new model and continue.
- Win rate still ~50% at `trials=400` → the result is genuinely flat;
  proceed to checks 4 and 5.

This is the single most-informative step. Do it before concluding training
failed.

#### 4. Eyeball a few games

Either run `battle.py --show_moves` interactively, or save the games via the
`--training_file` mechanism and inspect with `python src/one.py` or by
loading positions in the webapp. Quick sanity checks:

- **Move counts:** typical TwixT games end in 20-60 moves. If games end in
  <10 moves or run to the board limit, something structural is broken
  (e.g., one side resigning immediately, the network outputting NaNs, or
  an MCTS bug producing illegal moves).
- **Move quality:** are the moves recognizably reasonable to a human, or
  do you see obvious blunders the same way both models? Both-blunder
  patterns suggest a shared training-data issue rather than a real
  comparison.
- **Resignations:** if both sides resign rapidly in symmetric positions,
  the resign threshold may be too aggressive — see Part A on resign tuning.

#### 5. Check self-play data quality and quantity

How many self-play games went into the new model? At what `trials` count?
Reference workflow guidelines:

- **Self-play trials matter more than arena trials** for training signal
  quality. Rough rule: arena trials should be ≥ 4× self-play trials,
  because arena is judging not generating. Self-play at trials=50 produces
  noisy training targets; the resulting network has no clean policy to
  learn from.
- **Volume matters.** A single iteration with a few thousand self-play
  games is rarely enough to show measurable arena improvement, especially
  early in training. The reference workflow runs many iterations.
- **Hyperparameter consistency.** If the v2→v3 step used different
  Dirichlet noise, temperature schedule, or exploration constant from
  earlier iterations, comparing them directly may not be apples-to-apples.

### What "no improvement in one iteration" really means

AlphaZero-style self-play improvement is famously **slow per iteration at
the start**. The original AlphaGo Zero paper showed flat measured
performance for several iterations before progress kicked in. Plateaus
between checkpoints are normal — what matters is the trend across many
iterations, not a single comparison.

Practical rule: don't conclude "training is broken" from a single arena
plateau. Conclude it from:
- Multiple consecutive iterations all returning ~50% in proper-trials arena, **and**
- Training loss not decreasing across those iterations, **or**
- Self-play game quality visibly stagnant or regressing.

If only one of those signals fires, you're more likely looking at a slow
iteration than a broken pipeline.

### Recommended order of operations when arena is flat

1. `md5sum` the model files. (Seconds — rules out the dumbest mistake.)
2. Check training loss curve from the offending training step. (Minutes.)
3. Run a `trials=400, num_games=50-100` arena on the PC. (~1-2 hours.)
4. If still flat: inspect a handful of games for sanity. (Hour.)
5. If still flat: investigate self-play data quality and run at least one
   more training iteration before deciding the pipeline needs a real
   overhaul. (Days.)

Each step is cheap relative to the next, and each one resolves a different
class of root cause. Skipping ahead — e.g. immediately blaming the model
architecture — usually wastes more time than it saves.
