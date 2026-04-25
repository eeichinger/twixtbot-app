#!/usr/bin/env bash
# scripts/pause-self-play.sh — gracefully stop all self-play / arena
# processes so the GPU is free for other use (e.g. gaming).
#
# Order matters:
#   1. train_loop.py (if running): send SIGINT and let its own
#      KeyboardInterrupt handlers terminate pmany, the per-iteration NNS,
#      and any in-progress train.py. This is the canonical path —
#      train_loop does its own cleanup and we just give it time.
#   2. arena.py (if running): same idea, send SIGINT and let it run its
#      cleanup() — terminates battle clones, SUICIDE-kills its NNS pair.
#   3. Stray battle.py / pmany.py clones (if launched directly without
#      a parent orchestrator).
#   4. NNS instances still around — find their --location from the
#      command line and shut them down via the protocol's SUICIDE_CODE
#      so the .sock and .shm files are unlinked.
#
# Does NOT touch the inner train.py (single-iteration training). It is
# typically only 2-3 minutes per iteration; let it finish or, if
# train_loop.py is supervising, train_loop's KeyboardInterrupt handler
# terminates train.py for us in step 1.
#
# To resume after this script:
#   - train_loop.py: re-run your original command. TR1 markers
#     (iter{N}.phase_a_done / iter{N}.done) plus position counting
#     in iter{N}_*.bin let it pick up mid-Phase-A; ≤ 1 in-progress
#     game per clone is lost.
#   - arena.py: NOT mid-flight resumable; restarting begins a fresh
#     batch. Copy logs/arena/*.log aside first if you need the partials.

set -u

PATTERN_TRAIN_LOOP='python.*train_loop\.py'
PATTERN_ARENA='python.*src/arena\.py'
PATTERN_CLONES='python.*src/(battle|pmany)\.py'
PATTERN_NNS='python.*src/nns\.py'
PATTERN_ALL='python.*(train_loop\.py|src/(arena|battle|pmany|nns)\.py)'

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NNS_PY="${SCRIPT_DIR}/../src/nns.py"

# Step 1: stop train_loop.py — its SIGINT handler terminates pmany
# (up to 30s wait), runs the per-iteration finally to SUICIDE the NNS,
# and exits cleanly. Give it ~40s to wind down.
if pgrep -f "$PATTERN_TRAIN_LOOP" > /dev/null; then
    echo "stopping train_loop.py ..."
    pkill -INT -f "$PATTERN_TRAIN_LOOP" || true
    sleep 40
fi

# Step 2: stop arena.py — its SIGINT handler does its own cleanup.
if pgrep -f "$PATTERN_ARENA" > /dev/null; then
    echo "stopping arena.py ..."
    pkill -INT -f "$PATTERN_ARENA" || true
    # arena.py terminates clones (waits up to 5s each) then kills NNS.
    sleep 8
fi

# Step 3: any stray battle/pmany clones (launched directly without a
# parent orchestrator, or that survived steps 1-2).
if pgrep -f "$PATTERN_CLONES" > /dev/null; then
    echo "stopping battle / pmany clones ..."
    pkill -TERM -f "$PATTERN_CLONES" || true
    sleep 3
fi

# Step 4: SUICIDE remaining NNS instances. Sending SIGTERM works too,
# but SUICIDE lets the NNS unlink its .sock and .shm files cleanly.
pgrep -af "$PATTERN_NNS" | while read -r _pid rest; do
    loc=$(echo "$rest" | grep -oE -- '--location [^ ]+' | awk '{print $2}')
    if [ -n "$loc" ]; then
        echo "stopping NNS at $loc ..."
        python "$NNS_PY" --location "$loc" --kill 2>/dev/null || true
    fi
done

# Step 5: report stragglers but don't escalate to SIGKILL — if SIGTERM
# wasn't enough, you have a real stuck process and should investigate.
sleep 2
remaining=$(pgrep -af "$PATTERN_ALL" || true)
if [ -n "$remaining" ]; then
    echo
    echo "warning: some processes did not exit cleanly:"
    echo "$remaining"
    echo "if needed, kill them manually with:  kill <pid>  (or kill -9 as last resort)"
    exit 1
fi

echo "self-play paused. GPU should now be free."
echo "to resume: re-run your original arena.py or training command."
