#!/usr/bin/env python3
"""arena.py — head-to-head arena between two TwixT models.

Supervises two NNS instances and a configurable number of parallel
battle.py clones, aggregates per-clone scores and progress, and prints a
final summary. One-shot replacement for the manual pmany+battle pipeline
when you just want to compare two models.

Example:
    python src/arena.py \\
        --model-a models/v2.pt --model-b models/v4.pt \\
        --device mps --total_games 400 --num_clones 8 \\
        --trials 200 --async_calls 32
"""

import argparse
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

try:
    import psutil
except ImportError:
    sys.exit("arena.py requires psutil. Install with:  pip install psutil")


SCRIPT_DIR = Path(__file__).resolve().parent
NNS_PY = SCRIPT_DIR / "nns.py"
BATTLE_PY = SCRIPT_DIR / "battle.py"

NNS_READY_TIMEOUT = 60.0
NNS_KILL_TIMEOUT = 15.0


# ---------------------------------------------------------------------------
# CLI / pre-flight
# ---------------------------------------------------------------------------

def autodetect_device():
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def parse_args():
    p = argparse.ArgumentParser(
        description="Head-to-head TwixT arena between two models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model-a", required=True, type=Path,
                   help="Path to model A (.pt).")
    p.add_argument("--model-b", required=True, type=Path,
                   help="Path to model B (.pt).")
    p.add_argument("--device", required=True, choices=["cpu", "mps", "cuda"],
                   help="Device for both NNS instances.")
    p.add_argument("--total_games", required=True, type=int,
                   help="Total games across all clones.")
    p.add_argument("--num_clones", required=True, type=int,
                   help="Number of parallel battle.py processes.")
    p.add_argument("--trials", required=True, type=int,
                   help="MCTS trials per move for both players.")
    p.add_argument("--async_calls", required=True, type=int,
                   help="Per-game in-flight NN evaluations.")
    p.add_argument("--threads", type=int, default=2,
                   help="Game threads per battle.py process.")
    p.add_argument("--log_dir", type=Path, default=Path("logs/arena"),
                   help="Output directory; cleaned at start.")
    p.add_argument("--progress_interval", type=float, default=30.0,
                   help="Seconds between progress status prints.")
    p.add_argument("--no-compile", dest="compile", action="store_false",
                   help="Disable torch.compile in NNS (default: enabled).")
    return p.parse_args()


def preflight(args):
    """Validate inputs. Returns (socket_a, socket_b)."""
    for label, path in [("--model-a", args.model_a), ("--model-b", args.model_b)]:
        if not path.is_file():
            sys.exit(f"error: {label} {path} does not exist")

    detected = autodetect_device()
    if args.device != detected:
        print(f"note: --device {args.device}; auto-detected {detected}",
              file=sys.stderr)

    if args.num_clones < 1:
        sys.exit("error: --num_clones must be >= 1")
    if args.num_clones > args.total_games:
        sys.exit(f"error: --num_clones {args.num_clones} > --total_games "
                 f"{args.total_games}; each clone needs at least 1 game")

    socket_a = f"/tmp/twixtbot_nns_{args.model_a.stem}"
    socket_b = f"/tmp/twixtbot_nns_{args.model_b.stem}"
    if socket_a == socket_b:
        sys.exit("error: model basenames collide; rename one model")
    for sock in (socket_a, socket_b):
        if os.path.exists(sock + ".sock"):
            sys.exit(f"error: {sock}.sock already exists. Stop the existing NNS first:\n"
                     f"  python {NNS_PY} --location {sock} --kill")
    return socket_a, socket_b


def distribute_games(total, n):
    """Distribute total games across n clones; first (total % n) clones get +1."""
    base, extra = divmod(total, n)
    return [base + (1 if i < extra else 0) for i in range(n)]


# ---------------------------------------------------------------------------
# NNS supervision
# ---------------------------------------------------------------------------

def launch_nns(args, socket, model, log_path):
    capacity = args.num_clones * args.threads * args.async_calls * 2
    cmd = [
        sys.executable, "-u", str(NNS_PY),
        "--location", socket,
        "--device", args.device,
        "--model", str(model),
        "--capacity", str(capacity),
    ]
    if args.compile:
        cmd.append("--compile")
    f = open(log_path, "w")
    p = subprocess.Popen(cmd, stdout=f, stderr=f)
    return p, f


def wait_for_nns_ready(log_path, label, timeout=NNS_READY_TIMEOUT):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with open(log_path) as f:
                for line in f:
                    if "Ready for connections" in line:
                        return
        except FileNotFoundError:
            pass
        time.sleep(0.3)
    raise TimeoutError(f"{label} did not become ready within {timeout:.0f}s; check {log_path}")


def kill_nns(socket):
    try:
        subprocess.run(
            [sys.executable, str(NNS_PY), "--location", socket, "--kill"],
            timeout=NNS_KILL_TIMEOUT,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        pass


# ---------------------------------------------------------------------------
# Battle clone spawn + log parsing
# ---------------------------------------------------------------------------

def asn_player_spec(socket, trials, async_calls):
    return f"asn_player:location={socket},trials={trials},async_calls={async_calls}"


def launch_clone(args, num_games, spec_a, spec_b, log_path):
    cmd = [
        sys.executable, "-u", str(BATTLE_PY),
        "--white", spec_a,
        "--black", spec_b,
        "--num_games", str(num_games),
        "--threads", str(args.threads),
    ]
    f = open(log_path, "w")
    p = subprocess.Popen(cmd, stdout=f, stderr=f)
    return p, f


_AFTER_LINE = re.compile(r'After (\d+)/\d+ games?')
_SCORE_LINE = re.compile(r'^:\s+([\d.]+)\s+\(\s*[\d.]+%\)\s+(.+)\s*$')
_GPU_LINE = re.compile(
    r'gpu:\s+N=\d+\s+T=[\d.]+\s+W=[\d.eE+-]+\s+W/N=([\d.]+)\s+W/T=(\d+)/s'
)


class CloneState:
    def __init__(self, clone_id, log_path, expected_games, socket_a, socket_b):
        self.clone_id = clone_id
        self.log_path = log_path
        self.expected_games = expected_games
        self.socket_a = socket_a
        self.socket_b = socket_b
        self.byte_offset = 0
        self.games_done = 0
        self.score_a = 0.0
        self.score_b = 0.0

    def update(self):
        try:
            with open(self.log_path, "rb") as f:
                f.seek(self.byte_offset)
                new = f.read()
                self.byte_offset = f.tell()
        except FileNotFoundError:
            return
        for raw in new.splitlines():
            line = raw.decode("utf-8", errors="replace")
            m = _AFTER_LINE.search(line)
            if m:
                self.games_done = max(self.games_done, int(m.group(1)))
                continue
            m = _SCORE_LINE.match(line)
            if m:
                score = float(m.group(1))
                spec = m.group(2)
                if self.socket_a in spec:
                    self.score_a = score
                elif self.socket_b in spec:
                    self.score_b = score


def latest_gpu_summary(log_path):
    """Return (W/N, W/T) tuple from the latest gpu: line, or None."""
    try:
        with open(log_path) as f:
            content = f.read()
    except FileNotFoundError:
        return None
    last = None
    for m in _GPU_LINE.finditer(content):
        last = m
    if last is None:
        return None
    return float(last.group(1)), int(last.group(2))


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def fmt_hms(seconds):
    s = max(0, int(seconds))
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def confidence_interval_pct(wins, total):
    """95% CI half-width on win rate, expressed as percentage points."""
    if total <= 0:
        return 0.0
    p = wins / total
    se = math.sqrt(p * (1 - p) / total)
    return 1.96 * se * 100


def print_status(args, states, nns_a_log, nns_b_log, total_games, elapsed):
    for s in states:
        s.update()
    games_done = sum(s.games_done for s in states)
    score_a = sum(s.score_a for s in states)
    score_b = sum(s.score_b for s in states)
    played = score_a + score_b
    pct_a = 100.0 * score_a / played if played else 50.0
    pct_b = 100.0 - pct_a if played else 50.0

    pct_done = 100.0 * games_done / total_games if total_games else 0
    eta = elapsed * (total_games - games_done) / games_done if games_done > 0 else 0.0
    cpu_pct = psutil.cpu_percent(interval=None)
    n_cores = psutil.cpu_count(logical=True) or 1

    print()
    print(f"[{fmt_hms(elapsed)} elapsed | {games_done}/{total_games} games "
          f"({pct_done:.0f}%) | ETA {fmt_hms(eta)}]")
    print(f"  : {score_a:6.1f} ({pct_a:5.1f}%) model-a ({args.model_a})")
    print(f"  : {score_b:6.1f} ({pct_b:5.1f}%) model-b ({args.model_b})")
    print(f"  CPU: {cpu_pct:5.1f}% of {n_cores} cores")

    a_summary = latest_gpu_summary(nns_a_log)
    b_summary = latest_gpu_summary(nns_b_log)
    if a_summary:
        wn, wt = a_summary
        print(f"  NNS-A ({args.model_a.stem}): {wt} pos/s @ batch {wn:.0f}")
    if b_summary:
        wn, wt = b_summary
        print(f"  NNS-B ({args.model_b.stem}): {wt} pos/s @ batch {wn:.0f}")
    sys.stdout.flush()


def build_summary(args, states, nns_a_log, nns_b_log, total_elapsed):
    """Return the final summary as a list of lines."""
    score_a = sum(s.score_a for s in states)
    score_b = sum(s.score_b for s in states)
    games_done = sum(s.games_done for s in states)
    played = score_a + score_b
    pct_a = 100.0 * score_a / played if played else 0.0
    pct_b = 100.0 - pct_a if played else 0.0
    ci_a = confidence_interval_pct(score_a, played)
    gpm = (games_done / total_elapsed * 60) if total_elapsed > 0 else 0

    lines = []
    lines.append("=" * 64)
    lines.append(f"Arena complete in {fmt_hms(total_elapsed)} — "
                 f"{games_done} games at {gpm:.1f} games/min")
    lines.append("-" * 64)
    lines.append(f"  model-a ({args.model_a}):  {score_a:6.1f} wins  "
                 f"({pct_a:5.1f}%, ±{ci_a:.1f}%)")
    lines.append(f"  model-b ({args.model_b}):  {score_b:6.1f} wins  "
                 f"({pct_b:5.1f}%)")
    lines.append("=" * 64)
    a_summary = latest_gpu_summary(nns_a_log)
    b_summary = latest_gpu_summary(nns_b_log)
    if a_summary:
        wn, wt = a_summary
        lines.append(f"NNS-A ({args.model_a.stem}) final: {wt} pos/s @ batch {wn:.0f}")
    if b_summary:
        wn, wt = b_summary
        lines.append(f"NNS-B ({args.model_b.stem}) final: {wt} pos/s @ batch {wn:.0f}")
    return lines


def write_summary_file(args, summary_lines, summary_path):
    """Write the summary to a file with run metadata header for archival."""
    started = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    header = [
        f"# arena.py summary — {started}",
        f"# command: {' '.join(sys.argv)}",
        f"# model-a: {args.model_a}",
        f"# model-b: {args.model_b}",
        f"# device={args.device}, total_games={args.total_games}, "
        f"num_clones={args.num_clones}, trials={args.trials}, "
        f"async_calls={args.async_calls}, threads={args.threads}, "
        f"compile={args.compile}",
        "",
    ]
    with open(summary_path, "w") as f:
        f.write("\n".join(header + summary_lines) + "\n")


# ---------------------------------------------------------------------------
# Main supervisor
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    socket_a, socket_b = preflight(args)

    if args.log_dir.exists():
        shutil.rmtree(args.log_dir)
    args.log_dir.mkdir(parents=True)

    nns_a_log = args.log_dir / f"nns_{args.model_a.stem}.log"
    nns_b_log = args.log_dir / f"nns_{args.model_b.stem}.log"

    per_clone_games = distribute_games(args.total_games, args.num_clones)
    actual_total = sum(per_clone_games)
    print(f"arena.py: {actual_total} games across {args.num_clones} clones "
          f"(per-clone: {per_clone_games})")
    print(f"  model-a: {args.model_a}  ->  {socket_a}")
    print(f"  model-b: {args.model_b}  ->  {socket_b}")
    print(f"  device={args.device}, trials={args.trials}, "
          f"async_calls={args.async_calls}, threads={args.threads}, "
          f"compile={args.compile}")

    nns_procs = []
    clone_procs = []
    clone_logfiles = []
    clone_states = []
    interrupted = False

    def cleanup():
        # Stop battle clones first (they hold NNS connections).
        for p, _ in clone_procs:
            if p.poll() is None:
                p.terminate()
        for p, _ in clone_procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
        for f in clone_logfiles:
            try: f.close()
            except Exception: pass
        # Then kill NNS via the protocol's SUICIDE message.
        for sock in (socket_a, socket_b):
            kill_nns(sock)
        for p, f, _ in nns_procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
            try: f.close()
            except Exception: pass

    print(f"arena.py: starting NNS-A ({args.model_a.stem}) ...")
    pa, fa = launch_nns(args, socket_a, args.model_a, nns_a_log)
    nns_procs.append((pa, fa, socket_a))
    print(f"arena.py: starting NNS-B ({args.model_b.stem}) ...")
    pb, fb = launch_nns(args, socket_b, args.model_b, nns_b_log)
    nns_procs.append((pb, fb, socket_b))

    try:
        wait_for_nns_ready(nns_a_log, f"NNS-A ({args.model_a.stem})")
        wait_for_nns_ready(nns_b_log, f"NNS-B ({args.model_b.stem})")
    except TimeoutError as e:
        print(f"error: {e}", file=sys.stderr)
        cleanup()
        sys.exit(1)

    spec_a = asn_player_spec(socket_a, args.trials, args.async_calls)
    spec_b = asn_player_spec(socket_b, args.trials, args.async_calls)
    digits = max(1, len(str(args.num_clones - 1)))
    print(f"arena.py: launching {args.num_clones} battle clones ...")
    for i in range(args.num_clones):
        log_path = args.log_dir / f"{i:0{digits}d}.log"
        p, f = launch_clone(args, per_clone_games[i], spec_a, spec_b, log_path)
        clone_procs.append((p, f))
        clone_logfiles.append(f)
        clone_states.append(CloneState(i, log_path, per_clone_games[i],
                                       socket_a, socket_b))

    psutil.cpu_percent(interval=None)  # warm sample
    start = time.monotonic()
    last_status = 0.0
    try:
        while True:
            still_running = sum(1 for p, _ in clone_procs if p.poll() is None)
            now = time.monotonic()
            elapsed = now - start

            if now - last_status >= args.progress_interval:
                print_status(args, clone_states, nns_a_log, nns_b_log,
                             actual_total, elapsed)
                last_status = now

            if still_running == 0:
                break
            time.sleep(min(2.0, max(0.5, args.progress_interval / 10)))
    except KeyboardInterrupt:
        interrupted = True
        print("\narena.py: interrupted, shutting down children ...")
    finally:
        cleanup()

    for s in clone_states:
        s.update()
    failed = [(p, i) for i, (p, _) in enumerate(clone_procs) if p.returncode not in (0, None)]
    if failed:
        print(f"\nwarning: {len(failed)} clone(s) exited with non-zero status: "
              f"{[i for _, i in failed]} — see {args.log_dir}")
    summary_lines = build_summary(args, clone_states, nns_a_log, nns_b_log,
                                  time.monotonic() - start)
    print()
    for line in summary_lines:
        print(line)
    summary_path = args.log_dir / "summary.txt"
    write_summary_file(args, summary_lines, summary_path)
    print(f"\nsummary written to {summary_path}")
    sys.exit(1 if interrupted or failed else 0)


if __name__ == "__main__":
    main()
