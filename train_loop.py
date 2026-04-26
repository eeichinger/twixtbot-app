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
import socket
import subprocess
import sys
import time

# Live progress helpers shared with arena.py.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from progress_stats import CloneLogTail, fmt_hms, latest_gpu_line  # noqa: E402

try:
    import psutil
except ImportError:
    psutil = None  # CPU% will be omitted from the heartbeat block


# =============================================================================
# Configuration — edit these constants for experiments
# =============================================================================

# --- Paths ----------------------------------------------------------------
NNS_SOCKET = "/tmp/twixtbot_nns"       # unix socket path used by NNS/workers
SPDATA_DIR = "spdata"                  # where .bin training files accumulate
LOGS_DIR = "logs"                      # all logs (main + per-phase) go here

# --- Self-play (Phase A) ---------------------------------------------------
NUM_CLONES = 24                        # pmany --num_clones
THREADS_PER_CLONE = 2                  # battle.py --threads
ASYNC_CALLS = 32                       # asn_player async_calls=
ADD_NOISE = 0.25                       # asn_player add_noise= (Dirichlet root noise; 0 disables)
POSITION_CACHE = True                  # asn_player position_cache= (transposition table)

# --- NNS (inference server) ------------------------------------------------
NNS_DEVICE = "cuda"
NNS_CAPACITY = NUM_CLONES * THREADS_PER_CLONE * 2 * ASYNC_CALLS   # = NUM_CLONES * THREADS_PER_CLONE * 2 * ASYNC_CALLS
NNS_USE_COMPILE = True                 # --compile flag on src/nns.py
NNS_USE_FP16 = True                    # --fp16 flag on src/nns.py
NNS_READY_TIMEOUT_SEC = 60             # how long to wait for socket after spawn

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

# --- Per-iteration MCTS trials scaling -------------------------------------
# List of (upper_iter_bound_inclusive, trials).
# Weak models don't benefit from deep search; strong models do. Scaling
# trials with iteration saves GPU time early and improves data quality late.
TRIALS_CADENCE = [
    (2,  50),    # iter 1-2: weak model, fast games
    (4,  100),   # iter 3-4: developing model
    (99, 200),   # iter 5+: strong model, high-quality data
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

# --- Resume / state markers ------------------------------------------------
# After each successful Phase A and Phase B, train_loop drops a sentinel file
# in SPDATA_DIR. On restart, the per-iteration loop skips phases whose marker
# already exists, and partially-completed Phase A iterations resume by counting
# positions in iter{N}_*.bin (each LearningState record is fixed-size).
# See docs/specs/tr1-train-loop-resumable.md.
LEARNING_STATE_BYTES = 1789            # naf.LearningState.NUM_BYTES for board=24
AVG_MOVES_PER_GAME_DEFAULT = 410       # fallback when no prior iteration is on disk
                                       # (e.g. resuming iter 1). Normally we estimate
                                       # avg moves/game live from iter N-1's data, see
                                       # estimate_avg_moves_per_game(). Calibrated from
                                       # iters 4-6 (420/413/403); games shorten as the
                                       # model gets more decisive (iter 8 was ~289).


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
    games, batches = None, None
    for bound, g, b in ITERATION_CADENCE:
        if iteration <= bound:
            games, batches = g, b
            break
    if games is None:
        games, batches = ITERATION_CADENCE[-1][1], ITERATION_CADENCE[-1][2]

    trials = TRIALS_CADENCE[-1][1]
    for bound, t in TRIALS_CADENCE:
        if iteration <= bound:
            trials = t
            break
    return games, batches, trials


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
# Resume markers
# =============================================================================

def marker_path_phase_a_done(iteration):
    return os.path.join(SPDATA_DIR, f"iter{iteration}.phase_a_done")


def marker_path_done(iteration):
    return os.path.join(SPDATA_DIR, f"iter{iteration}.done")


def positions_in_iteration(iteration, search_dir):
    """Sum complete LearningState records across iter{N}_*.bin files."""
    pattern = re.compile(rf'^iter{iteration}_\d+\.bin$')
    if not os.path.isdir(search_dir):
        return 0
    total = 0
    for fname in os.listdir(search_dir):
        if pattern.match(fname):
            size = os.path.getsize(os.path.join(search_dir, fname))
            total += size // LEARNING_STATE_BYTES
    return total


def _spdata_search_dirs():
    """All directories where iter{N}_*.bin files may live: flat root + weighted tiers."""
    dirs = [SPDATA_DIR]
    if USE_WEIGHTED_SAMPLING:
        dirs.append(os.path.join(SPDATA_DIR, f"w={RECENT_WEIGHT}"))
        dirs.append(os.path.join(SPDATA_DIR, f"w={OLDER_WEIGHT}"))
    return dirs


def positions_for_iteration_anywhere(iteration):
    """Sum positions for iter{N} across flat-root + both weighted-tier dirs."""
    return sum(positions_in_iteration(iteration, d) for d in _spdata_search_dirs())


def planned_games_for_iteration(iteration):
    """Number of games pmany actually targets for iter N (cadence rounded to NUM_CLONES)."""
    games_target, _, _ = get_cadence(iteration)
    games_per_clone = max(1, games_target // NUM_CLONES)
    return games_per_clone * NUM_CLONES


def estimate_avg_moves_per_game(iteration, log):
    """Estimate avg moves/game for iter N's resume calculation.

    Uses iter (N-1)'s on-disk data: avg ≈ positions(N-1) / planned_games(N-1).
    Self-calibrating — game length drifts down as the model improves and a hard-coded
    constant goes stale (iters 4-6 ≈ 410, iter 8 ≈ 289). Falls back to the default
    when no prior-iter data is available (iter 1, or first run after a wipe).
    """
    if iteration <= 1:
        return float(AVG_MOVES_PER_GAME_DEFAULT)

    prev = iteration - 1
    prev_positions = positions_for_iteration_anywhere(prev)
    prev_planned = planned_games_for_iteration(prev)

    if prev_positions <= 0 or prev_planned <= 0:
        log(f"  no iter {prev} data on disk; using default "
            f"avg moves/game = {AVG_MOVES_PER_GAME_DEFAULT}")
        return float(AVG_MOVES_PER_GAME_DEFAULT)

    avg = prev_positions / prev_planned
    log(f"  iter {prev}: {prev_positions:,} positions / "
        f"{prev_planned:,} planned games = {avg:.0f} avg moves/game (auto-calibrated)")
    return avg


def write_marker_atomic(path, content):
    """Write a sentinel file with crash-safe semantics: tmp + fsync + rename."""
    tmp = path + ".partial"
    with open(tmp, "w") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# =============================================================================
# NNS lifecycle
# =============================================================================

def start_nns(model_path, iter_log_path, log):
    """Spawn the NNS as a background process. Returns (Popen, file handle).

    Removes any stale .sock / .shm files left by a previous NNS run before
    launching — otherwise wait_for_nns_ready could race against the old
    socket file during NNS's torch import / model load window.
    """
    for suffix in ('.sock', '.shm'):
        stale = NNS_SOCKET + suffix
        if os.path.exists(stale):
            log(f"removing stale {stale}")
            try:
                os.remove(stale)
            except OSError as e:
                log(f"  could not remove {stale}: {e}")

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
    """Wait until NNS actually accepts unix-socket connections.

    Checking os.path.exists() alone is unreliable: a stale .sock file from a
    prior crashed NNS satisfies the check instantly. Here we try a real
    connect() every second so we only succeed when NNS is truly listening.
    """
    socket_path = NNS_SOCKET + '.sock'
    start = time.time()
    while time.time() - start < NNS_READY_TIMEOUT_SEC:
        if nns_proc.poll() is not None:
            raise RuntimeError(
                f"NNS exited during startup (code {nns_proc.returncode}); "
                f"check its log file")
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(1.0)
        try:
            s.connect(socket_path)
            log(f"NNS ready (accepting connections at {socket_path})")
            return
        except (FileNotFoundError, ConnectionRefusedError, OSError):
            pass  # not ready yet; retry
        finally:
            s.close()
        time.sleep(1.0)
    raise TimeoutError(f"NNS did not accept connections within "
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

def _discover_clone_logs(sp_log_dir):
    """Per-clone log file paths under sp_log_dir (excluding master.log)."""
    if not os.path.isdir(sp_log_dir):
        return []
    return sorted(
        os.path.join(sp_log_dir, f)
        for f in os.listdir(sp_log_dir)
        if f.endswith('.log') and f != 'master.log'
    )


def _refresh_clone_tails(sp_log_dir, clones):
    """Discover any new clone log files and update() all known tails."""
    existing = {c.log_path for c in clones}
    for path in _discover_clone_logs(sp_log_dir):
        if path not in existing:
            clones.append(CloneLogTail(path))
    for c in clones:
        c.update()


def _emit_self_play_heartbeat(iteration, total_planned, sp_log_dir,
                              nns_log_path, phase_start, clones, log):
    _refresh_clone_tails(sp_log_dir, clones)
    games_done = sum(c.games_done for c in clones)
    elapsed = time.time() - phase_start
    pct = 100.0 * games_done / total_planned if total_planned else 0
    gpm = games_done * 60 / elapsed if elapsed > 0 else 0
    eta = (elapsed * (total_planned - games_done) / games_done
           if games_done > 0 else 0)
    log(f"[elapsed {fmt_hms(elapsed)} | iter {iteration} self-play | "
        f"{games_done:,}/{total_planned:,} games ({pct:.0f}%) | "
        f"{gpm:.1f} games/min | ETA {fmt_hms(eta)}]")
    if psutil is not None:
        cpu_pct = psutil.cpu_percent(interval=None)
        n_cores = psutil.cpu_count(logical=True) or 1
        log(f"  CPU: {cpu_pct:.1f}% of {n_cores} cores")
    nns_line = latest_gpu_line(nns_log_path)
    if nns_line:
        log(f"  NNS: {nns_line}")


def run_self_play(iteration, games_target, trials, output_dir, log):
    sp_log_dir = os.path.join(LOGS_DIR, f"sp_iter{iteration}")
    if os.path.isdir(sp_log_dir):
        shutil.rmtree(sp_log_dir)
    nns_log_path = os.path.join(LOGS_DIR, f"nns_iter{iteration}.log")

    games_per_clone = max(1, games_target // NUM_CLONES)
    total_planned = games_per_clone * NUM_CLONES

    cache_flag = ",position_cache=1" if POSITION_CACHE else ""
    asn_spec = (f"asn_player:location={NNS_SOCKET},"
                f"trials={trials},async_calls={ASYNC_CALLS},"
                f"add_noise={ADD_NOISE},temperature={TEMPERATURE}"
                f"{cache_flag}")

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
    # pmany redirects its own stdout to its master.log; we drive the
    # heartbeat from here by tailing per-clone logs and the NNS log.
    p = subprocess.Popen(cmd)
    clones = []
    if psutil is not None:
        psutil.cpu_percent(interval=None)  # warm-up the per-process sampler
    last_hb = phase_start
    try:
        while p.poll() is None:
            time.sleep(5)
            now = time.time()
            if now - last_hb >= SELF_PLAY_HEARTBEAT_SEC:
                _emit_self_play_heartbeat(iteration, total_planned, sp_log_dir,
                                          nns_log_path, phase_start, clones, log)
                last_hb = now
    except KeyboardInterrupt:
        log("KeyboardInterrupt: terminating pmany")
        p.terminate()
        try:
            # 30s is generous: pmany propagates SIGTERM to N battle clones
            # which each tear down their NNS connections and flush training
            # files. Better to wait than to kill -9 mid-flush.
            p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            p.kill()
        raise

    duration = time.time() - phase_start
    if p.returncode != 0:
        raise RuntimeError(f"pmany failed with exit code {p.returncode}")
    log(f"self-play done in {duration:.0f}s ({duration/60:.1f} min) "
        f"for iter {iteration}")

    # Enriched post-Phase-A summary: total positions, real avg moves/game from
    # the actual game count summed across clones, and the final NNS gpu line.
    _refresh_clone_tails(sp_log_dir, clones)
    actual_games = sum(c.games_done for c in clones)
    positions = positions_in_iteration(iteration, output_dir)
    if actual_games > 0 and positions > 0:
        avg_mpg = positions / actual_games
        log(f"  positions: {positions:,} "
            f"(avg {avg_mpg:.0f} moves/game across {actual_games:,} games)")
    nns_final = latest_gpu_line(nns_log_path)
    if nns_final:
        log(f"  NNS final: {nns_final}")
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
            # 30s lets train.py finish the in-flight optimizer step and write
            # the partial checkpoint instead of getting kill -9'd mid-batch.
            p.wait(timeout=30)
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
    log(f"trials cadence: {TRIALS_CADENCE}")
    log(f"self-play: {NUM_CLONES} clones x {THREADS_PER_CLONE} threads, "
        f"async_calls={ASYNC_CALLS}, "
        f"add_noise={ADD_NOISE}, temperature={TEMPERATURE}, "
        f"position_cache={POSITION_CACHE}")
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
            games, batches, trials = get_cadence(iteration)
            next_model = os.path.join(model_dir, f"v{iteration}.pt")

            # If this iteration is fully complete from a previous run, skip it
            # entirely and advance current_model. (See docs/specs/tr1-train-loop-resumable.md.)
            if os.path.exists(marker_path_done(iteration)):
                log(f"--- iter {iteration} already complete (marker found), "
                    f"skipping; current_model = {next_model} ---")
                current_model = next_model
                continue

            log(f"--- iter {iteration} start: games={games}, "
                f"batches={batches}, trials={trials}, "
                f"model={current_model} ---")
            iter_start = time.time()

            output_dir = rotate_spdata_tiers(iteration, log)

            # Phase A — skip if its marker exists (interrupted between Phase A
            # and Phase B). Otherwise count any partial data and run only the
            # remaining games.
            if os.path.exists(marker_path_phase_a_done(iteration)):
                log(f"iter {iteration} Phase A already complete (marker found), "
                    f"skipping to Phase B")
                sp_dur = 0.0
            else:
                positions_so_far = positions_in_iteration(iteration, output_dir)
                if positions_so_far > 0:
                    avg_mpg = estimate_avg_moves_per_game(iteration, log)
                    estimated_games_done = int(positions_so_far / avg_mpg)
                    games_remaining = max(0, games - estimated_games_done)
                    log(f"iter {iteration} Phase A resuming: "
                        f"{positions_so_far:,} positions / {avg_mpg:.0f} avg "
                        f"≈ {estimated_games_done:,} games already done, "
                        f"running {games_remaining:,} more")
                else:
                    games_remaining = games

                if games_remaining <= 0:
                    log(f"iter {iteration} Phase A target already met "
                        f"({positions_so_far:,} positions), skipping")
                    sp_dur = 0.0
                else:
                    nns_log_path = os.path.join(LOGS_DIR,
                                                f"nns_iter{iteration}.log")
                    nns_proc, nns_log_f = start_nns(current_model,
                                                    nns_log_path, log)
                    try:
                        wait_for_nns_ready(nns_proc, log)
                        sp_dur = run_self_play(iteration, games_remaining,
                                               trials, output_dir, log)
                    finally:
                        stop_nns(nns_proc, nns_log_f, log)

                final_positions = positions_in_iteration(iteration, output_dir)
                write_marker_atomic(
                    marker_path_phase_a_done(iteration),
                    f"target_games={games} positions_collected={final_positions} "
                    f"timestamp={time.strftime('%Y-%m-%dT%H:%M:%S')}\n",
                )

            train_dur = run_training(iteration, current_model,
                                     next_model, batches, log)

            write_marker_atomic(
                marker_path_done(iteration),
                f"model={next_model} "
                f"timestamp={time.strftime('%Y-%m-%dT%H:%M:%S')}\n",
            )

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
