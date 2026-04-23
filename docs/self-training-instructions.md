# TwixBot Self-Training Guide (Windows + RTX 5070 Ti)

Target hardware: **Windows 11, AMD Ryzen 7 7800X3D, NVIDIA RTX 5070 Ti, 64 GB RAM**.
Runtime environment: **WSL2 Ubuntu** (required — the Unix socket IPC layer in `smmpp.py` does not work natively on Windows).

---

## Overview

Running the full training pipeline produces a `model.onnx` file that you drop into the
webapp and rebuild. A meaningful first training run (enough to beat the random-init
baseline by a wide margin) takes **roughly 1–2 weeks of intermittent sessions**, or
a few days of continuous running.

The pipeline has three repeating phases:

| Phase | What runs | Where |
|---|---|---|
| **A — Self-play** | NNS (GPU inference) + pmany/battle (CPU MCTS) | WSL2 terminal pair |
| **B — Training** | train.py consumes `.bin` files, updates the model | WSL2 (GPU) |
| **C — Restart** | Kill old NNS, relaunch on new model | WSL2 |

---

## Step 1 — WSL2 + CUDA Setup

### 1a. Install WSL2

In a Windows PowerShell (Admin):

```powershell
wsl --set-default-version 2
wsl --install -d Ubuntu-24.04
```

Reboot when prompted.

### 1b. NVIDIA driver

Install Windows NVIDIA driver **≥ 570** (Blackwell / `sm_120`).
**Do NOT install a separate Linux driver inside WSL** — WSL2 passes the Windows
driver through via `/dev/dxg` automatically.

Verify inside Ubuntu:

```bash
nvidia-smi
```

You should see the 5070 Ti listed with driver version and CUDA version.

### 1c. Move the repo to WSL2 ext4

Self-play writes hundreds of `.bin` files; NTFS passthrough (`/mnt/c/...`) is slow.
Clone or copy the repo into the WSL2 native filesystem:

```bash
cp -r /mnt/c/path/to/twixtbot-app ~/twixtbot-app
cd ~/twixtbot-app
```

All commands below assume you are in `~/twixtbot-app`.

### 1d. Tune WSL2 resource limits

Create or edit `C:\Users\<you>\.wslconfig`:

```ini
[wsl2]
memory=52GB
swap=0
pageReporting=false
```

- `memory=52GB` — leaves ~12 GB for Windows and the GPU driver.
- `swap=0` — no swap file; avoids silent disk I/O under memory pressure.
- `pageReporting=false` — prevents WSL from returning pages to Windows mid-session.
- Do **not** set `processors=` — the 7800X3D has SMT disabled (8 physical cores);
  WSL2 will use all available cores by default. Verify with `nproc` inside Ubuntu.

Restart WSL to apply: `wsl --shutdown` in PowerShell, then reopen Ubuntu.

---

## Step 2 — Python Environment

```bash
cd ~/twixtbot-app
python3 -m venv .venv
source .venv/bin/activate

# PyTorch CUDA 12.8 build (overrides the CPU-only index in requirements.txt)
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision

pip install numpy pytest
export PYTHONPATH=$PWD/src
```

Add the export to `~/.bashrc` so it survives new shells:

```bash
echo 'export PYTHONPATH=$HOME/twixtbot-app/src' >> ~/.bashrc
```

Verify CUDA is visible:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: True  NVIDIA GeForce RTX 5070 Ti
```

Run the test suite to confirm a clean baseline:

```bash
pytest src/
```

---

## Step 3 — Create the Initial Model

Run once to produce a randomly-initialised checkpoint:

```bash
mkdir -p models spdata
python create_model.py
cp models/v0.pt models/v0_backup.pt
```

`create_model.py` saves a `TwixNet(num_filters=64, num_blocks=8)` model (~1.9 M parameters,
~7.7 MB on disk, fits comfortably in 16 GB VRAM with room for activations and optimizer state).

---

## Step 4 — Verification Smoke Test

Run this end-to-end check before committing to a multi-hour iteration.
You need **three terminals** inside WSL2, each with `source .venv/bin/activate`
and `export PYTHONPATH=$PWD/src`.

**Terminal 1 — start NNS:**

```bash
python src/nns.py \
  --location /tmp/twixtbot_nns \
  --device cuda \
  --model models/v0.pt \
  --capacity 64
```

Wait until the server prints its ready message (no error on startup).

**Terminal 2 — run 2 self-play games:**

```bash
python src/battle.py \
  --white "asn_player:location=/tmp/twixtbot_nns,trials=10,async_calls=8" \
  --black "asn_player:location=/tmp/twixtbot_nns,trials=10,async_calls=8" \
  --num_games 2 \
  --threads 1
```

Expected: prints game results (scores and model names) within ~30 seconds.

**Terminal 2 — run 5 training batches:**

```bash
python src/train.py \
  --model models/v0.pt \
  --device cuda \
  --num_batches 5 \
  --batch_size 16
```

Expected: prints 5 batch-loss lines and saves the model.

**Terminal 1 — kill NNS:**

```bash
python src/nns.py --location /tmp/twixtbot_nns --kill
```

If all three steps succeed without errors, you are ready for a full iteration.

---

## Step 5 — Single-GPU Iteration Loop

### Phase A — Self-play

Open two terminals.

**Terminal 1 — GPU inference server:**

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
- `--capacity 2048` — must equal `num_clones × threads × 2 × async_calls`
  (16 × 2 × 2 × 32 = 2048). Too low causes silent slot-allocation failures that
  hang worker threads.
- `--compile` — wraps the model with `torch.compile(mode='reduce-overhead')`,
  fusing conv+BN+activation kernels. First batch is slow (~30 s compilation);
  steady-state throughput improves ~20–40%.
- `--fp16` — runs inference in float16, halving memory-bandwidth demand. On
  Blackwell fp16 hardware this gives ~1.5–2× inference throughput. Silently
  ignored if `--device cpu`.

**Terminal 2 — self-play workers:**

```bash
mkdir -p spdata
rm -rf logs/sp_gen && python src/pmany.py \
  --num_clones 16 \
  --log_dir logs/sp_gen \
  -- \
  python src/battle.py \
    --white "asn_player:location=/tmp/twixtbot_nns,trials=100,async_calls=32" \
    --black "asn_player:location=/tmp/twixtbot_nns,trials=100,async_calls=32" \
    --num_games 63 \
    --threads 2 \
    --training_file spdata/iter1_%n%.bin
```

The `--` separator is required; without it, pmany's argparse intercepts the
`--white`, `--black`, etc. flags meant for battle.py.

Parameter notes:
- `--num_clones 16` — 16 independent processes. Each runs 2 game threads
  (via `--threads 2`), totalling 32 concurrent games. The 7800X3D (SMT disabled,
  8 physical cores) handles this comfortably because MCTS workers spend most of
  their time blocked on NNS socket replies, not burning CPU.
- `async_calls=32` — `asn_player` keeps 32 in-flight NNS queries per player,
  maintaining a large GPU batch. With 32 game threads and 32 async_calls,
  NNS receives ~320 positions per batch and is GPU-busy ~97% of wall clock.
- `trials=100` — MCTS playouts per move. 100 is fast and produces reasonable
  training signal; raise to 200 for later iterations once the model is stronger.
- `--num_games 63` — games per clone. 16 × 63 ≈ **1 008 games per run** (~60 K
  training records). Adjust to taste; aim for 5 000–10 000 games in later iterations.
- `%n%` — replaced by zero-padded clone index (00–15), giving separate output files
  (`spdata/iter1_00.bin` … `spdata/iter1_15.bin`).

**Monitoring:**

Watch `logs/sp_gen/master.log` for progress:

```bash
tail -f logs/sp_gen/master.log
```

Watch NNS terminal for GPU stats (printed every 10 000 evaluations):

```
gpu: N=30753 T=1068.28 W=9.84018e+06 avg=-0.000322 + 0.000110*W
```

- Average batch size = W/N ≈ **320** (good; larger is better)
- GPU busy fraction = T_gpu / (T_gpu + T_wait) ≈ **97%** (NNS is GPU-saturated)
- The nvidia-smi reading of ~60% reflects memory-bandwidth-limited SM occupancy,
  not NNS starvation. Adding more CPU workers does not improve throughput once
  the NNS is GPU-saturated; use `--compile` and `--fp16` instead.

**Expected throughput:** ~4 000–5 000 games/hour at `trials=100` on this hardware.

### Phase B — Training

Kill the NNS to free the GPU, then train:

```bash
# Terminal 1
python src/nns.py --location /tmp/twixtbot_nns --kill
```

```bash
# Terminal 2 (or same terminal after pmany finishes)
cp models/v0.pt models/v1.pt

python src/train.py \
  --model models/v1.pt \
  --device cuda \
  --num_batches 1000 \
  --batch_size 256 \
  --learning_rate 0.01 \
  --decay_rate 0.95 \
  --temperature 0.5 \
  --policy_epsilon 0.01 \
  --save_after 200 \
  spdata/
```

Flag notes:
- Copy to `v1.pt` first — `train.py` modifies the model file in-place.
  `v0.pt` remains as the prior checkpoint for arena comparison.
- `spdata/` — pass the directory; train.py scans all `.bin` files recursively.
  See *Weighted sampling* below for multi-iteration data organisation.
- `--num_batches 1000` — for iteration 1 (~60 K records at batch 256 ≈ 234 batches
  per epoch, so 1000 batches ≈ 4 epochs). Scale up as the dataset grows.
- `--decay_rate 0.95` — multiplies the learning rate by 0.95 whenever batch loss
  increases (soft annealing). Set to 1.0 to disable.
- `--save_after 200` — writes intermediate checkpoints every 200 batches
  (`v1.pt.200`, `v1.pt.400`, …). Guards against crashes on long runs.
- `--temperature 0.5` — must match the temperature used during self-play generation.

Training prints per-batch: `loss=…  slope=…  policy=…  value=…`.
A downward-trending loss slope confirms learning is happening.

### Phase C — Restart on New Model

```bash
python src/nns.py \
  --location /tmp/twixtbot_nns \
  --device cuda \
  --model models/v1.pt \
  --capacity 2048 \
  --compile \
  --fp16
```

Then repeat Phase A with `--training_file spdata/iter2_%n%.bin` and
`--model models/v1.pt` in the NNS command.

### Iteration Cadence

| Iteration | Games to generate | Training batches | Notes |
|---|---|---|---|
| 1 | 1 000 | 500–1 000 | Validate pipeline; model improves quickly from random init |
| 2–4 | 3 000–5 000 each | 1 000–1 500 | Model starts making recognisable moves |
| 5+ | 10 000+ each | 2 000+ | Use weighted sampling to down-weight old data |

Save each trained model as a new file (`v1.pt`, `v2.pt`, …) rather than
overwriting, so you can roll back or run arena comparisons at any point.

### Weighted Sampling of Training Data

Keep data from multiple iterations but down-weight older games:

```
spdata/
  w=0.2/          ← older games (20% sampling weight)
    iter1_*.bin
    iter2_*.bin
  w=0.8/          ← recent games (80% sampling weight)
    iter5_*.bin
```

```bash
python src/train.py --model models/v5.pt --device cuda ... spdata/
```

train.py picks up the `w=<float>/` convention automatically from the directory names.

---

## Step 6 — Evaluate Model vs Model (Arena)

Run arena comparisons to decide whether to promote a newly trained model.

Two NNS servers share the single GPU — one per model under evaluation.

**Terminal 1 — NNS for new model (candidate):**

```bash
python src/nns.py \
  --location /tmp/twixtbot_nns_new \
  --device cuda \
  --model models/v1.pt \
  --capacity 512 \
  --fp16 --compile
```

**Terminal 2 — NNS for baseline model:**

```bash
python src/nns.py \
  --location /tmp/twixtbot_nns_old \
  --device cuda \
  --model models/v0.pt \
  --capacity 512 \
  --fp16 --compile
```

`--capacity 512` per server (8 clones × 2 threads × 1 player per server × 32
async_calls = 512).

**Terminal 3 — arena games:**

```bash
rm -rf logs/arena && python src/pmany.py \
  --num_clones 8 \
  --log_dir logs/arena \
  -- \
  python src/battle.py \
    --white "asn_player:location=/tmp/twixtbot_nns_new,trials=400,async_calls=32" \
    --black "asn_player:location=/tmp/twixtbot_nns_old,trials=400,async_calls=32" \
    --num_games 50 \
    --threads 2
```

8 clones × 50 games = **400 games total**. battle.py automatically alternates
colours every game for fairness. The final score shows `--black` (old model) vs
`--white` (new model); look at the white-side percentage.

**Promotion threshold:**
- Win-rate > **55%** at n=200 — meaningful improvement (±7% margin at p=0.05).
- Win-rate > **60%** at n=400 — strong signal; promote without hesitation.

**Kill servers after arena:**

```bash
python src/nns.py --location /tmp/twixtbot_nns_new --kill
python src/nns.py --location /tmp/twixtbot_nns_old --kill
```

---

## Step 7 — Deploy

When the model is good enough to ship:

```bash
# Export to ONNX (float32, BN folded into conv layers)
python tools/export_onnx.py --model models/v1.pt --out webapp/public/model.onnx

# INT8 dynamic quantization (in-place; saves fp32 backup as model.fp32.onnx)
# Reduces model size ~75%, cuts peak WASM heap on iOS — do not skip this step
python tools/quantize_model.py

# Rebuild the webapp
cd webapp && npm run build
```

Play-test in the browser (including on iOS if possible) before committing and pushing.

---

## Appendix A — Dual-GPU Variant

If a second GPU is added later, the only changes are in Steps 5 and 6.
The loop overlaps self-play and training instead of serialising them:

- **GPU0** (5070 Ti) — NNS for self-play, running continuously:
  ```bash
  CUDA_VISIBLE_DEVICES=0 python src/nns.py \
    --location /tmp/twixtbot_nns \
    --device cuda --model models/v0.pt \
    --capacity 2048 --compile --fp16
  ```

- **GPU1** (secondary) — train.py in parallel with self-play:
  ```bash
  CUDA_VISIBLE_DEVICES=1 python src/train.py \
    --model models/v1.pt --device cuda ...
  ```

With two GPUs, NNS does **not** need to be stopped between training runs. Instead:
write a new checkpoint (`v1.pt`, `v2.pt`, …), then SIGTERM NNS and relaunch
against the latest checkpoint. GPU0 stays at near-constant utilisation.

Arena: run two NNS servers on either GPU at `--capacity 512` each — VRAM is
not the bottleneck for inference on a 1.9 M-param model.

No code changes beyond the existing `--device` flags. GPU selection is by
`CUDA_VISIBLE_DEVICES`.

Expected wall-clock speedup: **~2× per iteration** (training no longer blocks
self-play), provided the CPU keeps GPU0 fed — see Appendix B.

---

## Appendix B — GPU is the Bottleneck: Hardware Analysis

### What the data shows

During a real self-play run, NNS reports cumulative statistics at every 10 000
evaluations. A representative snapshot:

```
waiting:     N=30753  T=12.8    avg=0.000417
preprocessing: N=30753  T=12.3    avg=0.000400
gpu:         N=30753  T=1068.3  W=9.84e+06  avg=-0.000322 + 0.000110*W
pp_shmem:    N=30753  T=4.2
pp_socket:   N=30753  T=0.8
```

Key derived metrics:

| Metric | Value | Interpretation |
|---|---|---|
| Average batch size W/N | **320 positions/batch** | Workers are keeping NNS well-fed |
| GPU time per batch T/N | **34.7 ms** | Dominated by memory-bandwidth, not compute |
| GPU busy fraction | **97%** of wall clock | NNS is GPU-saturated, not CPU-starved |
| nvidia-smi GPU utilisation | **~60%** | SM occupancy is memory-bandwidth limited |

**Critical finding: the GPU, not the CPU, is the bottleneck.**

The NNS spends 97% of its wall-clock time running GPU inference. Adding more CPU
workers (more clones) does not improve throughput — the NNS can already barely keep
up with the CPU. The 60% nvidia-smi reading does not indicate starvation; it reflects
the GPU running at 60% SM occupancy because convolution over a 24×22 spatial grid
with 64 filters is **memory-bandwidth limited**, not compute-limited, on Blackwell.

The levers that actually improve throughput are on the inference side:

| Lever | Flag | Expected gain | Mechanism |
|---|---|---|---|
| Kernel fusion | `--compile` | 20–40% | Eliminates per-layer dispatch overhead via TorchInductor |
| Half-precision | `--fp16` | 40–90% | Halves memory bandwidth; Blackwell has dedicated fp16 hardware |
| Larger model | Rebuild with 128 filters | Plateaus sooner | More FLOPs per memory access → better SM occupancy |

### AMD Ryzen 7 7800X3D (current)

- **8 physical cores, SMT disabled.** All cores on a single 3D V-Cache CCD;
  uniform L3 latency ideal for MCTS tree traversal.
- **WSL2**: do not set `processors=` in `.wslconfig` — WSL2 sees 8 logical cores
  automatically (verify with `nproc`). Setting `processors=16` causes a boot error.
- **Recommended config**: `--num_clones 16 --threads 2 --async_calls 32`
  (32 concurrent game threads, each player keeping 32 in-flight NNS queries).
  Total in-flight: 16 × 2 × 2 × 32 = **2048** → `--capacity 2048` on NNS.
- **Why 16 clones on 8 cores**: MCTS workers are I/O-bound (blocking on NNS
  replies), not compute-bound. Oversubscribing cores is beneficial — while one
  thread waits for a NNS reply, another can do tree traversal. 32 game threads on
  8 cores achieves ~97% NNS GPU utilisation in practice.
- **Measured throughput**: ~4 000–5 000 games/hour at `trials=100`.
- **WSL2 tips**:
  - Ensure **Core Isolation / Memory Integrity (HVCI)** is off in Windows Security
    → can cost 10–15% CPU throughput under WSL2.
  - Set Windows power plan to **Best Performance**.
  - Close Chrome/Edge — background timers degrade 3D V-Cache hit rates.

### AMD Ryzen 9 9950X3D (upgrade scenario)

The 9950X3D has **16 cores / 32 threads**, but 3D V-Cache is on **one CCD only
(cores 0–7)**; the second CCD (cores 8–15) is standard Zen 5 without the extra L3.

**Because the GPU is the bottleneck, not the CPU, adding cores gives diminishing returns.**
The 7800X3D already keeps NNS at 97% GPU utilisation. More CPU only helps at the margin
by enabling larger async batches and slightly larger instantaneous batch sizes at NNS.

**Revised performance estimate vs. 7800X3D under the same 5070 Ti:**

| Phase | Speedup | Rationale |
|---|---|---|
| Self-play (Phase A) | **1.1–1.2×** | GPU is already saturated at 97%; more CPU mainly increases average batch size from 320 → 450, yielding modest SM-occupancy gains. Not a 2× improvement. |
| Training (Phase B) | **~1.0×** | GPU-bound; CPU only feeds batches and is not the bottleneck. |
| Arena (Step 6) | **1.1–1.3×** | Two NNS sharing one GPU caps speedup; extra cores reduce inter-batch idle time slightly. |
| **Full iteration (single-GPU serial)** | **~1.1–1.2×** | Self-play dominates time; GPU saturation limits CPU scaling. |
| **Full iteration (dual-GPU, Appendix A)** | **~1.5–1.7×** | Training parallelism adds ~2× when GPU-bound training runs concurrently; CPU headroom helps sustain GPU0. |

**Recommended config on 9950X3D:**
```
--num_clones 24 --threads 2 --async_calls 32
# Total in-flight: 24 × 2 × 2 × 32 = 3072 → --capacity 3072 on NNS
```

Optional CCD pinning for arena (not self-play):
```bash
taskset -c 0-7 python src/pmany.py ...
```

Pins arena workers to the V-Cache CCD for more consistent per-game MCTS latency.
For bulk self-play, let the scheduler spread freely — aggregate throughput matters
more than per-game latency.

**Bottom line:** A 9950X3D is a meaningful upgrade **only when running dual-GPU**
(Appendix A) — the extra CPU cores sustain both GPU0 (self-play NNS) and GPU1
(training) simultaneously without one starving the other. For single-GPU serial
training, the 7800X3D is adequate; the GPU is the binding constraint and
`--compile --fp16` on NNS are far more impactful than a CPU upgrade.
