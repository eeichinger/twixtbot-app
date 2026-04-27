# TR1 — Resumable train_loop.py

**Status:** Done — implementation in `train_loop.py`, unit tests in `tests/test_train_loop.py`.
**Priority:** P1
**Files affected:** `train_loop.py` only (~50 lines added). No changes to
`battle.py`, `train.py`, or `pmany.py`.

---

## Problem

A single self-play iteration on the 5070 Ti reference hardware takes
**4.5–7.8 hours**, dominated entirely by Phase A (self-play game generation).
Phase B (training) is just 2–3 minutes. When the user pauses self-play to free
the GPU (gaming, etc.) via `scripts/pause-self-play.sh`, restarting
`train_loop.py` re-runs the entire interrupted iteration from scratch — wasting
up to 8 hours of GPU time per pause.

Today's `--start_iter` argument lets you skip iterations explicitly, but only
at iteration boundaries. There's no way to resume a partially-completed
Phase A.

## Goal

After `pause-self-play.sh` interrupts a `train_loop.py` run, restarting it
should pick up exactly where it left off without manual `--start_iter`
calculation. Wasted work on resume bounded to:

- At most one in-progress game per clone (not yet flushed to disk).
- At most a few seconds of Phase B if interrupted mid-training.

## Approach

Use filesystem state as the single source of truth — no databases or external
state stores. Two tiny sentinel files per iteration plus the existing
`iter{N}_*.bin` data files give complete information about progress.

Key insight: `LearningState.to_bytes()` produces a fixed-size 1789-byte
record (`naf.LearningState.NUM_BYTES` for board size 24). Counting positions
in a `.bin` file is `file_size // 1789`. Combined with the per-record `b'JTwx'`
magic at offset 0, file integrity is verifiable cheaply.

Per-move flush in `battle.py:251-258` ensures the file always ends on a
complete record boundary at any kill point — at worst, an in-progress game
contributed *some* of its moves before interruption (still valid training
data).

## LearningState format reference

| Bytes | Field |
|---|---|
| 4 | `b'JTwx'` magic header |
| 8 | last 4 moves (2 bytes each) |
| 720 | board features (10 channels × 24² bits, packed) |
| 1056 | visit-count target (528 moves × uint16) |
| 1 | z value (offset by +1: 0=loss, 1=draw, 2=win) |
| **1789** | **total** = `naf.LearningState.NUM_BYTES` |

## New filesystem markers

Located in `SPDATA_DIR` next to the data files:

| Marker | Meaning | Contents |
|---|---|---|
| `iter{N}.phase_a_done` | Phase A finished successfully for iter N | `target_games={X} positions_collected={Y} timestamp={ISO}` |
| `iter{N}.done` | Phase B finished successfully (iteration fully complete) | `model={path} timestamp={ISO}` |

Markers are written atomically — write to `tmp.partial`, fsync, rename — so
a crash mid-write can't leave a half-written marker.

## Behavior change

For each iteration `N` in `range(start_iter, total_iters + 1)`:

```
1. If iter{N}.done exists:
     log "iter {N} already complete, skipping"
     current_model = models/v{N}.pt   (path from marker; default if absent)
     continue

2. Compute games_target from cadence as today.

3. If iter{N}.phase_a_done exists:
     log "iter {N} Phase A already complete, skipping to Phase B"
   else:
     positions_so_far = sum (file_size // 1789) over iter{N}_*.bin
     if positions_so_far > 0:
         estimated_games_done = positions_so_far // AVG_MOVES_PER_GAME
         games_remaining = max(0, target_games - estimated_games_done)
         log "iter {N} Phase A resuming: {positions_so_far} positions ≈ "
             "{estimated_games_done} games already done, running {games_remaining} more"
     else:
         games_remaining = target_games

     start NNS, wait for ready
     run pmany for games_remaining games
     stop NNS
     write iter{N}.phase_a_done

4. Run Phase B (training) — unchanged.
5. Write iter{N}.done.
6. current_model = next_model.
```

## New helpers in `train_loop.py`

```python
LEARNING_STATE_BYTES = 1789       # naf.LearningState.NUM_BYTES; hardcoded for resilience
AVG_MOVES_PER_GAME_DEFAULT = 410  # fallback only — used for iter 1 or when no prior
                                  # iter data exists. Otherwise avg moves/game is
                                  # estimated live from iter (N-1)'s on-disk positions
                                  # (see estimate_avg_moves_per_game()).

def positions_in_iteration(iteration, output_dir):
    pattern = re.compile(rf'^iter{iteration}_\d+\.bin$')
    total = 0
    for f in os.listdir(output_dir):
        if pattern.match(f):
            total += os.path.getsize(os.path.join(output_dir, f)) // LEARNING_STATE_BYTES
    return total

def write_marker_atomic(path, content):
    tmp = path + ".partial"
    with open(tmp, "w") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def marker_path_phase_a_done(iteration):
    return os.path.join(SPDATA_DIR, f"iter{iteration}.phase_a_done")

def marker_path_done(iteration):
    return os.path.join(SPDATA_DIR, f"iter{iteration}.done")
```

## Affected functions

- **`run_self_play`** — gains an optional `games_already_done` arg; the call
  site computes `games_remaining` from positions count and passes it. Inside,
  `games_target` becomes `games_remaining`. ~5 lines.
- **`main()` per-iteration loop** — three new branches: skip-if-done,
  skip-Phase-A-if-marker, count-positions-for-partial-resume. ~20 lines.
- **End of Phase A** — write `phase_a_done` marker. 1 line.
- **End of Phase B** — write `done` marker. 1 line.

Total new code: ~40 lines, all in `train_loop.py`.

## Resume scenarios

| State on disk before resume | Detected | Action |
|---|---|---|
| No markers, no spdata | Fresh start | Start at `--start_iter` (default 1), run normally |
| `iter{N}.done` exists, nothing for `N+1` | Iter N fully done | Skip iter N, start iter N+1 fresh |
| `iter{N}.done` + `iter{N+1}_*.bin` partial | Phase A of N+1 was running | Count positions, run remaining games |
| `iter{N}.done` + `iter{N+1}.phase_a_done` | Phase A done, Phase B was interrupted | Skip Phase A, run Phase B |
| `iter{N}.phase_a_done` only (no `done`) | Phase B of N was interrupted | Skip Phase A of N, redo Phase B (≤3 min waste) |
| `iter{N}_*.bin` partial, no markers | Phase A of N was interrupted | Count positions, run remaining games |

## Edge cases / assumptions

1. **Avg moves/game accuracy.** Affects only resume estimation precision.
   If we underestimate, we run a few more games than needed (harmless). If we
   overestimate, we run a few fewer (less data than nominal target — also fine;
   cadence numbers are heuristic). Computed live by
   `estimate_avg_moves_per_game()` as `positions(N-1) / planned_games(N-1)` —
   self-calibrating, since game length drifts down as the model gets more
   decisive (iter 4-6 ≈ 410, iter 8 ≈ 289). Falls back to
   `AVG_MOVES_PER_GAME_DEFAULT = 410` for iter 1 or when prior-iter data is
   absent.

2. **Torn final record protection.** If a battle clone is killed mid-write
   (between writes to a single record), the file could end with a partial
   1789-byte record. The integer division `file_size // 1789` discards it —
   that record is just lost, no corruption propagates. The next pmany invocation
   appends fresh records starting at the byte boundary.

3. **`iter{N}_*.bin` rotation.** Once iter N gets a `phase_a_done` marker,
   the data files freeze. The `rotate_spdata_tiers` machinery still moves
   them between `w=0.8/` and `w=0.2/` based on age, but never modifies them.
   Marker location stays in `SPDATA_DIR/` (the parent), so rotation doesn't
   affect markers.

4. **`--start_iter` interaction.** If user passes explicit `--start_iter K`,
   it overrides marker-based resume — but the iter loop still respects
   markers within iterations from K onward. If user wants to truly *redo*
   iter K (not resume), they need to `rm spdata/iter{K}.*` markers and
   `iter{K}_*.bin` files manually. Could add `--force_restart_iter K` later
   if it becomes annoying; not in this spec.

5. **Multiple parallel `train_loop.py` runs against the same `SPDATA_DIR`.**
   Not supported — markers would race. Not new behavior; existing code also
   doesn't support this.

## Out of scope (intentionally)

- **Phase B mid-batch resume.** As established, Phase B is 2–3 minutes; not
  worth the complexity. If interrupted during Phase B, the iteration's
  training restarts from the post-Phase-A checkpoint (i.e., `current_model`
  copied to `next_model` and trained from scratch). Up to 3 minutes wasted.
- **Auto-detection of `--start_iter` from highest `iter*.done` marker.**
  Possible future enhancement but adds complexity around model-file consistency
  checking. For now, user passes `--start_iter` explicitly; the per-iteration
  markers handle within-iteration partial state.

## Testing

Manual test sequence (cheap to run):

1. Smoke run with very small cadence (e.g., 50 games, 50 batches, 2 clones)
   to iter 3.
2. Verify markers appear at `spdata/iter1.done`, `iter2.done`, `iter3.done`.
3. Re-run from iter 1; verify each iter is detected as `.done` and skipped.
4. Modify a marker to test "phase_a_done but not done" path:
   `mv iter2.done iter2.phase_a_done`, re-run, verify only Phase B runs for iter 2.
5. Mid-Phase-A interrupt: kick off iter 3 fresh, kill via `pause-self-play.sh`
   after some games complete, count positions, re-run, verify
   "{N} games already done, running {M} more" log line and that final spdata
   aligns with target.
6. Validate file integrity by checking that every `iter*_*.bin` file's size
   is divisible by 1789 (no torn records persisted).

## Related

- `show_bin_info.py` — utility to inspect `.bin` files (added with this spec)
  for verifying record counts and integrity.
- `scripts/pause-self-play.sh` — the pause mechanism this spec makes safe.
