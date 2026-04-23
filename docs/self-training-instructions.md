## Goal

Deliver a new developer-facing doc, `docs/self-training-instructions.md`, that walks the user through training the TwixBot network themselves from a random init. Primary scenario: **Windows host + single RTX 5070 Ti**. Two appendices cover (A) how to extend the flow to **dual GPUs**, and (B) **CPU tuning** for the user's AMD 7800X3D with an upgrade estimate for a 9950X3D.

No code changes are made by this plan itself — the doc references two tiny device-flag patches to `src/nns.py` and `src/train.py` that the user will apply in a follow-up commit. All runtime commands in the doc are the same ones already used in `TRAINING.md` with `--device cuda` added.

## Constraints confirmed with the user

- **Windows only** — no Linux dual-boot, no repartitioning. ⇒ use WSL2 Ubuntu (Unix sockets in `src/smmpp.py` / `src/nns.py` require a POSIX environment; WSL2 passes the NVIDIA Windows driver through via `/dev/dxg` with no separate Linux driver install).
- **Single 5070 Ti** — second card may not fit. ⇒ serialise self-play and training on the same GPU; run arena with two NNS servers sharing the GPU at lower `--capacity`.
- **Random init** — start from `models/v0.pt` produced by a `TwixNet(num_filters=64, num_blocks=8)` + `torch.save` recipe (see `TRAINING.md` lines 37–47).

## Files referenced (read-only)

- `src/nns.py` — socket path `/tmp/twixtbot_nns` wired through line 49; `nneval.NNEvaluater` constructed at line 33 with hard-coded device.
- `src/nneval.py:32` — `NNEvaluater(model, device='cpu')` — already takes a device arg; nns.py just needs to forward it.
- `src/train.py` — `Trainer(...)` constructed at line 275; `prepare_batch(..., device=...)` called at line 315; both currently default to `cpu`.
- `src/model.py` — `TwixNet` is plain PyTorch, works on CUDA unchanged.
- `src/battle.py`, `src/pmany.py`, `src/asn_player.py` — CPU-only self-play workers; talk to NNS over Unix socket; no code change needed.
- `tools/export_onnx.py`, `tools/quantize_model.py` — unchanged deployment path.
- `TRAINING.md` — upstream canonical training doc; the new doc is a platform-specific practical companion, not a replacement.
- `docs/planned-features.md` — unchanged by this task (this is docs-only, not a feature from the tracker).

## Target file structure

`docs/self-training-instructions.md`, roughly 400–600 lines, with these sections:

1. **Overview & prerequisites** — what you'll produce (a trained `model.onnx` for the webapp), who the doc is for (repo maintainer on Windows + 5070 Ti), total time estimate for a meaningful run (~1–2 weeks of intermittent training to reach a model that beats `v0` by a wide margin).
2. **Step 1 — WSL2 + CUDA setup**
   - `wsl --install -d Ubuntu-24.04`, `--set-default-version 2`, reboot.
   - Windows NVIDIA driver ≥ 570 (Blackwell / `sm_120`); **do not** install a Linux driver inside WSL.
   - Verify with `nvidia-smi` inside Ubuntu.
   - Move repo into `~/twixtbot-app` (ext4) rather than `/mnt/c/...` — self-play writes hundreds of MB of `.bin` files, NTFS passthrough is slow.
3. **Step 2 — Python env inside WSL2**
   - `python3 -m venv .venv && source .venv/bin/activate`
   - `pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision` (override the CPU index pinned in `requirements.txt`).
   - `pip install numpy pytest`
   - `export PYTHONPATH=$PWD/src`
   - `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`
   - Run `pytest src/` to confirm baseline passes.
4. **Step 3 — Patch `--device` support**
   - `src/nns.py`: add `-d/--device` argparse flag (default `"cpu"` for back-compat); forward to `nneval.NNEvaluater(model, device=args.device)` at line 33.
   - `src/train.py`: add `--device` argparse flag (default `"cpu"`); forward to `Trainer(model, ..., device=args.device)` at line 275 and `prepare_batch(..., device=args.device)` at line 315.
   - No other code changes; tests remain CPU by default.
5. **Step 4 — Create the initial model**
   - Shell snippet creating `create_model.py` and saving `models/v0.pt` with `num_filters=64, num_blocks=8` (~1.5M params, fits 16GB with room for training activations).
   - `cp models/v0.pt models/v0_backup.pt`.
6. **Step 5 — Single-GPU iteration loop (core of the doc)**
   - **Phase A — self-play**: NNS holds GPU (`--device cuda`, `--capacity 200`), `pmany` spawns CPU-bound MCTS workers using the socket.
   - **Phase B — stop NNS, then train** on the freed GPU (`--device cuda`, `--batch_size 256`, 500 batches for iter 1 scaling to 2000 later).
   - **Phase C — restart NNS** on the new model; loop.
   - Include the weighted-sampling `w=<float>/` subdir convention from `TRAINING.md` lines 214–229 and the cadence table (1k / 5k / 10k games per iter).
7. **Step 6 — Evaluate model-vs-model**
   - Two NNS servers sharing the one GPU at `--capacity 100`; arena via `battle.py` with `add_noise=0.0`, `temperature=0.0`, 200–400 games; promote on >55% at n=200.
8. **Step 7 — Deploy**
   - `python tools/export_onnx.py --model models/vN.pt --out webapp/public/model.onnx`
   - `python tools/quantize_model.py`
   - `cd webapp && npm run build`
   - Play-test in the browser before committing.
9. **Verification** — end-to-end smoke test (NNS + 2-game battle + 5-batch train run) before committing to a multi-hour iteration.

## Appendix A — Dual-GPU variant

If the user later adds a second GPU, the changes are local to Steps 5 and 6 only. The loop overlaps self-play and training:

- **GPU0** (primary, e.g. 5070 Ti): **NNS for self-play inference**, running continuously. `CUDA_VISIBLE_DEVICES=0 python src/nns.py --device cuda --capacity 400 ...`
- **GPU1** (secondary): **`train.py` in parallel** with self-play — reads the growing `spdata/` dir, updates model checkpoints. `CUDA_VISIBLE_DEVICES=1 python src/train.py --device cuda ...`
- NNS does **not** need to be restarted after each training run; instead, training writes a new `models/vN.pt` and the dual-GPU cadence is "every K hours, SIGTERM NNS and relaunch against the latest checkpoint". Keeps GPU0 at near-constant utilisation.
- For **arena**: run two NNS servers on the same GPU (either one) at `--capacity 200` each — VRAM is not the bottleneck for inference-only at batch 200 on a 1.5M-param model.
- No code changes beyond the Step 3 `--device` patches. GPU selection is by `CUDA_VISIBLE_DEVICES` env var.
- Expected wall-clock speedup: **~2× per iteration** vs single-GPU serial loop (training no longer blocks self-play), provided the CPU can keep both GPUs fed (see Appendix B).

## Appendix B — CPU tuning: 7800X3D vs 9950X3D

Self-play is CPU-bound: MCTS tree traversal runs in Python (per-thread), and `pmany` orchestrates multiple processes each with multiple threads. The GPU-side NNS only works as fast as the CPU produces batched leaf positions. On a 1.5M-param TwixNet the 5070 Ti easily outpaces the CPU; **CPU cores are the self-play bottleneck**.

### AMD Ryzen 7 7800X3D (current)

- **8 cores / 16 threads**, all on a single 3D V-cache CCD (uniform L3 latency, ideal for cache-heavy workloads like MCTS tree traversal).
- **Recommended `pmany` config**: `--num_clones 4 --threads 4` (16 threads total), matching the 16 SMT threads.
- **NNS `--capacity`**: 200 is ample — leaf positions arrive slower than inference can consume them.
- **Batch sizes**: 256 for training is comfortable; GPU spends most time waiting anyway.
- **WSL2 considerations**: ensure WSL2 is allowed to use all cores (`.wslconfig` with `processors=16`); by default it uses all logical processors but worth verifying with `nproc` inside WSL.
- **Self-play rate ballpark**: ~2,500–3,500 games/hour at `trials=200` on a 5070 Ti + 7800X3D (empirical from comparable TF/PyTorch MCTS setups; actual numbers should be logged in iteration 1).

### AMD Ryzen 9 9950X3D (upgrade scenario)

- **16 cores / 32 threads**, but 3D V-cache is only on **one CCD (cores 0–7)**; the second CCD (cores 8–15) is vanilla Zen 5 without the extra L3. MCTS benefits heavily from L3, so layout matters.
- **Recommended `pmany` config**: `--num_clones 8 --threads 4` (32 threads total). Each process is independent and talks only to the NNS over the socket — no inter-process chatter that would suffer from cross-CCD latency.
- **Optional CCD pinning**: if per-game throughput is more important than aggregate throughput (e.g. during arena evaluation), pin the arena processes to cores 0–7 with `taskset -c 0-7` so they sit on the V-cache CCD. For bulk self-play, let the scheduler spread across both CCDs.
- **NNS `--capacity`**: raise to 400 — twice as many MCTS workers means twice the batch arrival rate.
- **Training phase**: unchanged (GPU-bound, not CPU-bound).

### Relative performance estimate (9950X3D vs 7800X3D)

Under the same 5070 Ti:

| Phase               | Speedup        | Rationale |
|---------------------|----------------|-----------|
| Self-play (Phase A) | **1.5–1.8×**   | 2× core count, but non-V-cache CCD is slower for MCTS; GPU starts to saturate around 24–28 MCTS threads on a 1.5M-param model, limiting scaling below 2×. |
| Training (Phase B)  | ~1.0×          | GPU-bound; CPU only feeds batches, which is not the bottleneck. |
| Arena (Step 6)      | 1.3–1.5×       | Two NNS sharing one GPU caps speedup; more cores mostly reduce the idle time between MCTS batches. |
| **Full iteration wall-clock (single-GPU serial loop)** | **~1.3–1.5×** | Weighted by time share: ~60% self-play + ~30% train + ~10% arena. |
| **Full iteration wall-clock (dual-GPU, Appendix A)**   | **~1.7–1.9×** | Training is parallel with self-play ⇒ self-play speedup dominates. |

Bottom line: a 9950X3D is a meaningful upgrade only if the user plans **heavy** training (many iterations, large game counts). For a first end-to-end validation run on the 7800X3D, the existing CPU is adequate; the 5070 Ti is not being starved at `--num_clones 4 --threads 4`.

Windows-side tips either CPU:
- Ensure **Core Isolation / Memory Integrity** (HVCI) is off in Windows Security — it can cost 10–15% CPU under WSL2.
- Set the Windows power plan to **Best Performance** during training runs.
- Close Chrome/Edge — each tab's background timer wakes up cores and hurts the 3D V-cache hit rate.

## Implementation checklist (after ExitPlanMode approval)

1. Create `docs/self-training-instructions.md` with the structure above, populating each step with the commands shown in my earlier plan draft.
2. Verify the doc renders cleanly on GitHub (no broken relative links; `TRAINING.md` cross-references resolve).
3. No code, test, or `docs/planned-features.md` changes required.

## Out of scope

- Actually applying the `--device` patches to `src/nns.py` / `src/train.py` — the doc tells the user to do this; doing it is a separate commit.
- Performing a training run.
- Automating any of the iteration loop (multi-phase scripts).