#!/usr/bin/env python3
"""
train_loop.py — Automated TwixBot training loop.

Usage:
    python train_loop.py models/v0.pt
    python train_loop.py models/v3.pt --start_iter 4 --total_iters 8

Each iteration runs Phase A (self-play) then Phase B (training):
  - Phase A: starts NNS on the current model, runs `pmany.py` + `battle.py`
    workers to generate `spdata/iterN_*.bin` files, then stops NNS.
  - Phase B: copies the current model to `models/vN.pt`, trains it in-place
    via `train.py` on the accumulated spdata.

Games per iteration and training batches per iteration scale with the
iteration index according to ITERATION_CADENCE below (matches the cadence
table in docs/self-training-instructions.md Step 5).

All tunables are constants at the top of the file — edit those for experiments.

This script *orchestrates* src/nns.py, src/pmany.py, src/battle.py, and
src/train.py via subprocess; it contains no training logic of its own.
"""

import argparse
import os
import re
import shutil
import signal
import subprocess
import sys
import time


# =============================================================================
# Configuration — edit these constants for experiments
# =============================================================================

# --- Paths ----------------------------------------------------------------
NNS_SOCKET = "/tmp/twixtbot_nns"       # unix socket path used by NNS/workers
SPDATA_DIR = "spdata"                  # where .bin training files accumulate
LOGS_DIR = "logs"                      # all logs (main + per-phase) go here

# --- NNS (inference server) ------------------------------------------------
NNS_DEVICE = "cuda"
NNS_CAPACITY = 2048                    # = NUM_CLONES * THREADS_PER_CLONE * 2 * ASYNC_CALLS
NNS_USE_COMPILE = True                 # --compile flag on src/nns.py
NNS_USE_FP16 = True                    # --fp16 flag on src/nns.py
NNS_READY_TIMEOUT_SEC = 60             # how long to wait for socket after spawn

# --- Self-play (Phase A) ---------------------------------------------------
NUM_CLONES = 16                        # pmany --num_clones
THREADS_PER_CLONE = 2                  # battle.py --threads
ASYNC_CALLS = 32                       # asn_player async_calls=
TRIALS = 100                           # asn_player trials= (MCTS playouts/move)

# --- Training (Phase B) ----------------------------------------------------
TRAIN_DEVICE = "cuda"
BATCH_SIZE = 256
LEARNING_RATE = 0.01
DECAY_RATE = 0.95
TEMPERATURE = 0.5                      # must match self-play temperature
POLICY_EPSILON = 0.01
SAVE_AFTER = 200                       # intermediate checkpoint every N batches

# --- Iteration cadence -----------------------------------------------------
# List of (upper_iter_bound_inclusive, games_this_tier, batches_this_tier).
# Entries are scanned in order; the first one whose bound covers the current
# iteration index wins. The last entry's bound acts as the default for higher
# iterations.
ITERATION_CADENCE = [
    (1,  1000,   500),   # iter 1
    (4,  5000,  1500),   # iter 2-4
    (99, 10000, 2000),   # iter 5+
]

# --- Weighted sampling of self-play data -----------------------------------
# Organises spdata/ into two weighted tiers so train.py samples recent games
# more often than old ones (see "Weighted sampling" in self-training-instructions.md).
# At the start of iteration N, files produced by iterations older than the
# recency window are physically moved from the RECENT tier to the OLDER tier.
USE_WEIGHTED_SAMPLING = True
RECENT_WEIGHT = 0.8                    # sampling weight for recent-tier files
OLDER_WEIGHT = 0.2                     # sampling weight for older-tier files
RECENCY_WINDOW = 3                     # iters N-RECENCY_WINDOW+1 .. N count as "recent"

# --- Run control -----------------------------------------------------------
TOTAL_ITERATIONS = 5                   # how many iterations to run by default
SELF_PLAY_HEARTBEAT_SEC = 60           # how often to print "still running" during Phase A


# =============================================================================
# Helpers
# =============================================================================

def ts():
    return time.strftime("%Y%m%d %H:%M:%S")


class Logger:
    """Write to stdout AND to a consolidated log file."""

    def __init__(self, path):
        self.path = path
        self.f = open(path, 'a', buffering=1)  # line-buffered

    def __call__(self, msg):
        line = f"{ts()} {msg}"
        print(line, flush=True)
        self.f.write(line + '\n')

    def raw_stream_from(self, proc):
        """Stream every line of proc.stdout to terminal + log file."""
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            self.f.write(line)
        self.f.flush()

    def close(self):
        self.f.close()


def get_cadence(iteration):
    for bound, games, batches in ITERATION_CADENCE:
        if iteration <= bound:
            return games, batches
    return ITERATION_CADENCE[-1][1:]


_ITER_RE = re.compile(r'iter(\d+)_\d+\.bin$')


def rotate_spdata_tiers(iteration, log):
    """Ensure weighted-tier directory layout and demote aged-out files.

    Before self-play for iteration N writes new files to the RECENT tier,
    move any files from iterations <= (N - RECENCY_WINDOW) out of RECENT and
    into OLDER. Returns the directory path where new iteration-N files
    should be written.

    If USE_WEIGHTED_SAMPLING is False, returns SPDATA_DIR (flat layout).
    """
    if not USE_WEIGHTED_SAMPLING:
        return SPDATA_DIR

    recent_dir = os.path.join(SPDATA_DIR, f"w={RECENT_WEIGHT}")
    older_dir = os.path.join(SPDATA_DIR, f"w={OLDER_WEIGHT}")
    os.makedirs(recent_dir, exist_ok=True)
    os.makedirs(older_dir, exist_ok=True)

    cutoff = iteration - RECENCY_WINDOW  # iterations <= cutoff are "old"
    if cutoff < 1:
        return recent_dir

    moved = 0
    for fname in sorted(os.listdir(recent_dir)):
        m = _ITER_RE.match(fname)
        if not m:
            continue
        file_iter = int(m.group(1))
        if file_iter <= cutoff:
            src = os.path.join(recent_dir, fname)
            dst = os.path.join(older_dir, fname)
            shutil.move(src, dst)
            moved += 1
    if moved:
        log(f"spdata rotation: moved {moved} files (iters <= {cutoff}) "
            f"from w={RECENT_WEIGHT}/ to w={OLDER_WEIGHT}/")
    return recent_dir


# =============================================================================
# NNS lifecycle
# =============================================================================

def start_nns(model_path, iter_log_path, log):
    """Spawn the NNS as a background process. Returns (Popen, file handle)."""
    cmd = [
        sys.executable, 'src/nns.py',
        '--location', NNS_SOCKET,
        '--device', NNS_DEVICE,
        '--model', model_path,
        '--capacity', str(NNS_CAPACITY),
    ]
    if NNS_USE_COMPILE:
        cmd.append('--compile')
    if NNS_USE_FP16:
        cmd.append('--fp16')

    log(f"NNS cmd: {' '.join(cmd)}")
    nns_log_f = open(iter_log_path, 'w', buffering=1)
    p = subprocess.Popen(cmd, stdout=nns_log_f, stderr=subprocess.STDOUT)
    return p, nns_log_f


def wait_for_nns_ready(nns_proc, log):
    socket_path = NNS_SOCKET + '.sock'
    start = time.time()
    while time.time() - start < NNS_READY_TIMEOUT_SEC:
        if nns_proc.poll() is not None:
            raise RuntimeError(
                f"NNS exited during startup (code {nns_proc.returncode}); "
                f"check its log file")
        if os.path.exists(socket_path):
            log(f"NNS ready (socket at {socket_path})")
            return
        time.sleep(0.5)
    raise TimeoutError(f"NNS socket {socket_path} did not appear within "
                       f"{NNS_READY_TIMEOUT_SEC}s")


def stop_nns(nns_proc, nns_log_f, log):
    if nns_proc.poll() is not None:
        log(f"NNS already exited (code {nns_proc.returncode})")
        nns_log_f.close()
        return
    log("stopping NNS via --kill")
    try:
        subprocess.run(
            [sys.executable, 'src/nns.py',
             '--location', NNS_SOCKET, '--kill'],
            check=False, timeout=30)
    except subprocess.TimeoutExpired:
        log("NNS --kill command timed out")
    try:
        nns_proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        log("NNS did not exit within 20s; terminating")
        nns_proc.terminate()
        try:
            nns_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            log("NNS did not respond to SIGTERM; sending SIGKILL")
            nns_proc.kill()
            nns_proc.wait()
    nns_log_f.close()


# =============================================================================
# Phase A — self-play
# =============================================================================

def run_self_play(iteration, games_target, output_dir, log):
    sp_log_dir = os.path.join(LOGS_DIR, f"sp_iter{iteration}")
    if os.path.isdir(sp_log_dir):
        shutil.rmtree(sp_log_dir)

    games_per_clone = max(1, games_target // NUM_CLONES)
    total_planned = games_per_clone * NUM_CLONES

    asn_spec = (f"asn_player:location={NNS_SOCKET},"
                f"trials={TRIALS},async_calls={ASYNC_CALLS}")

    cmd = [
        sys.executable, 'src/pmany.py',
        '--num_clones', str(NUM_CLONES),
        '--log_dir', sp_log_dir,
        '--',
        sys.executable, 'src/battle.py',
        '--white', asn_spec,
        '--black', asn_spec,
        '--num_games', str(games_per_clone),
        '--threads', str(THREADS_PER_CLONE),
        '--training_file', os.path.join(output_dir, f"iter{iteration}_%n%.bin"),
    ]

    log(f"self-play start: {NUM_CLONES} clones x {games_per_clone} games "
        f"= ~{total_planned} total (logs in {sp_log_dir}/)")
    log(f"pmany cmd: {' '.join(cmd)}")

    phase_start = time.time()
    # pmany redirects its own stdout to its master.log, so we inherit ours and
    # drive the heartbeat from here.
    p = subprocess.Popen(cmd)
    last_hb = phase_start
    try:
        while p.poll() is None:
            time.sleep(5)
            now = time.time()
            if now - last_hb >= SELF_PLAY_HEARTBEAT_SEC:
                elapsed = now - phase_start
                log(f"self-play running... elapsed {elapsed:.0f}s "
                    f"({elapsed/60:.1f} min); tail {sp_log_dir}/master.log for detail")
                last_hb = now
    except KeyboardInterrupt:
        log("KeyboardInterrupt: terminating pmany")
        p.terminate()
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()
        raise

    duration = time.time() - phase_start
    if p.returncode != 0:
        raise RuntimeError(f"pmany failed with exit code {p.returncode}")
    log(f"self-play done in {duration:.0f}s ({duration/60:.1f} min) "
        f"for iter {iteration}")
    return duration


# =============================================================================
# Phase B — training
# =============================================================================

def run_training(iteration, current_model, next_model, batches, log):
    log(f"copy {current_model} -> {next_model}")
    shutil.copy2(current_model, next_model)

    cmd = [
        sys.executable, '-u',               # -u: unbuffered stdout for real-time stream
        'src/train.py',
        '--model', next_model,
        '--device', TRAIN_DEVICE,
        '--num_batches', str(batches),
        '--batch_size', str(BATCH_SIZE),
        '--learning_rate', str(LEARNING_RATE),
        '--decay_rate', str(DECAY_RATE),
        '--temperature', str(TEMPERATURE),
        '--policy_epsilon', str(POLICY_EPSILON),
        '--save_after', str(SAVE_AFTER),
        SPDATA_DIR + '/',
    ]

    log(f"training start: {batches} batches -> {next_model}")
    log(f"train cmd: {' '.join(cmd)}")

    phase_start = time.time()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        log.raw_stream_from(p)
        p.wait()
    except KeyboardInterrupt:
        log("KeyboardInterrupt: terminating train.py")
        p.terminate()
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()
        raise

    duration = time.time() - phase_start
    if p.returncode != 0:
        raise RuntimeError(f"train.py failed with exit code {p.returncode}")
    log(f"training done in {duration:.0f}s ({duration/60:.1f} min) "
        f"for iter {iteration}")
    return duration


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Automated TwixBot self-play + training loop.')
    parser.add_argument('starting_model',
                        help='Path to the model used at the start of --start_iter '
                             '(e.g. models/v0.pt)')
    parser.add_argument('--start_iter', type=int, default=1,
                        help='Iteration index to start at (default: 1). Use this '
                             'to resume after a failed iteration.')
    parser.add_argument('--total_iters', type=int, default=TOTAL_ITERATIONS,
                        help=f'Last iteration to run (default: {TOTAL_ITERATIONS}).')
    args = parser.parse_args()

    if not os.path.isfile(args.starting_model):
        sys.exit(f"Error: starting model not found: {args.starting_model}")
    if args.start_iter < 1 or args.start_iter > args.total_iters:
        sys.exit(f"Error: --start_iter must be in [1, {args.total_iters}]")

    os.makedirs(SPDATA_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    run_tag = time.strftime("%Y%m%d_%H%M%S")
    main_log_path = os.path.join(LOGS_DIR, f"train_loop_{run_tag}.log")
    log = Logger(main_log_path)

    log(f"=== train_loop start (log: {main_log_path}) ===")
    log(f"starting model: {args.starting_model}")
    log(f"iterations: {args.start_iter}..{args.total_iters}")
    log(f"cadence: {ITERATION_CADENCE}")
    log(f"self-play: {NUM_CLONES} clones x {THREADS_PER_CLONE} threads, "
        f"trials={TRIALS}, async_calls={ASYNC_CALLS}")
    log(f"NNS: device={NNS_DEVICE}, capacity={NNS_CAPACITY}, "
        f"compile={NNS_USE_COMPILE}, fp16={NNS_USE_FP16}")
    if USE_WEIGHTED_SAMPLING:
        log(f"weighted sampling: recent={RECENT_WEIGHT} older={OLDER_WEIGHT} "
            f"recency_window={RECENCY_WINDOW} iters")
    else:
        log("weighted sampling: disabled (flat spdata/ layout)")

    model_dir = os.path.dirname(args.starting_model) or '.'
    overall_start = time.time()
    current_model = args.starting_model

    try:
        for iteration in range(args.start_iter, args.total_iters + 1):
            games, batches = get_cadence(iteration)
            log(f"--- iter {iteration} start: games={games}, "
                f"batches={batches}, model={current_model} ---")
            iter_start = time.time()

            output_dir = rotate_spdata_tiers(iteration, log)
            nns_log_path = os.path.join(LOGS_DIR, f"nns_iter{iteration}.log")
            nns_proc, nns_log_f = start_nns(current_model, nns_log_path, log)
            try:
                wait_for_nns_ready(nns_proc, log)
                sp_dur = run_self_play(iteration, games, output_dir, log)
            finally:
                stop_nns(nns_proc, nns_log_f, log)

            next_model = os.path.join(model_dir, f"v{iteration}.pt")
            train_dur = run_training(iteration, current_model,
                                     next_model, batches, log)

            iter_dur = time.time() - iter_start
            log(f"--- iter {iteration} done: self-play {sp_dur:.0f}s, "
                f"train {train_dur:.0f}s, total {iter_dur:.0f}s "
                f"({iter_dur/60:.1f} min) ---")

            current_model = next_model

    except KeyboardInterrupt:
        log("=== train_loop interrupted by user ===")
        log(f"last completed model: {current_model}")
        sys.exit(130)
    except Exception as e:
        log(f"=== train_loop failed: {type(e).__name__}: {e} ===")
        log(f"last completed model: {current_model}")
        raise

    total = time.time() - overall_start
    n = args.total_iters - args.start_iter + 1
    log(f"=== train_loop done: {n} iterations in {total:.0f}s "
        f"({total/60:.1f} min, {total/3600:.2f} hrs) ===")
    log(f"final model: {current_model}")
    log.close()


if __name__ == '__main__':
    main()
