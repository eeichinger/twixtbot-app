#!/usr/bin/env bash
# scripts/pause-self-play.sh — gracefully stop all self-play / arena
# processes so the GPU is free for other use (e.g. gaming).
#
# Order matters:
#   1. arena.py (if running): send SIGINT and let its own cleanup
#      terminate battle clones and SUICIDE-kill the NNS instances.
#   2. Stray battle.py / pmany.py clones (if launched directly).
#   3. NNS instances still around — find their --location from the
#      command line and shut them down via the protocol's SUICIDE_CODE
#      so the .sock and .shm files are unlinked.
#
# Does NOT touch train.py — training is per-iteration and you typically
# want it to finish on its own. Kill it manually if needed.
#
# To resume after this script:
#   - Self-play training: re-run your training/orchestrator command;
#     it resumes from the latest checkpoint. At most one in-progress
#     game per worker is lost (their results would not have been
#     written yet — completed games flush per-game).
#   - Arena (arena.py): NOT resumable; restarting begins a fresh batch.
#     If you needed the partial results, copy logs/arena/*.log first.

set -u

PATTERN_ARENA='python.*src/arena\.py'
PATTERN_CLONES='python.*src/(battle|pmany)\.py'
PATTERN_NNS='python.*src/nns\.py'
PATTERN_ALL='python.*src/(arena|battle|pmany|nns)\.py'

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NNS_PY="${SCRIPT_DIR}/../src/nns.py"

# Step 1: stop arena.py — its SIGINT handler does its own cleanup.
if pgrep -f "$PATTERN_ARENA" > /dev/null; then
    echo "stopping arena.py ..."
    pkill -INT -f "$PATTERN_ARENA" || true
    # arena.py terminates clones (waits up to 5s each) then kills NNS.
    sleep 8
fi

# Step 2: any stray battle/pmany clones (launched directly without arena.py).
if pgrep -f "$PATTERN_CLONES" > /dev/null; then
    echo "stopping battle / pmany clones ..."
    pkill -TERM -f "$PATTERN_CLONES" || true
    sleep 3
fi

# Step 3: SUICIDE remaining NNS instances. Sending SIGTERM works too,
# but SUICIDE lets the NNS unlink its .sock and .shm files cleanly.
pgrep -af "$PATTERN_NNS" | while read -r _pid rest; do
    loc=$(echo "$rest" | grep -oE -- '--location [^ ]+' | awk '{print $2}')
    if [ -n "$loc" ]; then
        echo "stopping NNS at $loc ..."
        python "$NNS_PY" --location "$loc" --kill 2>/dev/null || true
    fi
done

# Step 4: report stragglers but don't escalate to SIGKILL — if SIGTERM
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
