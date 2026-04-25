# TR2 — Live progress stats for `train_loop.py` self-play

**Status:** Done — implementation in `train_loop.py` + `src/progress_stats.py`,
unit tests in `tests/test_progress_stats.py`.
**Priority:** P2
**Files affected:** `src/progress_stats.py` (new), `src/arena.py` (refactor),
`train_loop.py`, `tests/test_progress_stats.py` (new).

---

## Problem

`train_loop.py`'s self-play phase is the long-running phase in this codebase
(4–8 hours per iteration). The previous heartbeat output told you the
orchestrator was alive but nothing about how far along it was:

```
self-play running... elapsed 4500s (75.0 min); tail logs/sp_iter6/master.log for detail
```

Meanwhile `arena.py` already prints rich live stats — games done, games/min,
ETA, NNS GPU line, CPU% — every progress interval. The same treatment should
apply to self-play.

## Solution

Extract the progress-tracking helpers from `arena.py` into a small shared
module (`src/progress_stats.py`) and consume them from both `arena.py` and
`train_loop.py`.

### `src/progress_stats.py` — shared helpers

- `_AFTER_LINE` regex — parses `After N/M games?` lines from `battle.py`.
- `_GPU_RAW_LINE` regex — matches a full `gpu:` milestone line.
- `fmt_hms(seconds)` — formats seconds as `HH:MM:SS`.
- `latest_gpu_line(log_path)` — returns the raw text of the latest `gpu:` line
  in an NNS log, or `None`.
- `class CloneLogTail` — tails one clone log via byte-offset reads, parses
  `games_done` from `After X/Y games` lines. Exposes a `_handle_line()` hook
  that subclasses use to parse additional patterns from the same stream.

### `arena.py` refactor

`CloneState` becomes a subclass of `CloneLogTail` adding score parsing via
`_SCORE_LINE`. The duplicated `latest_gpu_line` / `fmt_hms` / regex
definitions are removed and imported from the shared module instead.

### `train_loop.py` heartbeat replacement

`run_self_play` now:

1. Maintains a list of `CloneLogTail` objects, growing as `pmany.py` spawns
   per-clone log files (lazy discovery so we don't race against pmany's
   spawn loop).
2. Every `SELF_PLAY_HEARTBEAT_SEC` (60s default) prints:
   ```
   [elapsed HH:MM:SS | iter N self-play | A/B games (P%) | X.X games/min | ETA HH:MM:SS]
     CPU: X.X% of N cores
     NNS: gpu: N=... W/T=.../s ...
   ```
3. After Phase A completes, enriches the existing "self-play done" line with:
   ```
     positions: X,XXX,XXX (avg N moves/game across X,XXX games)
     NNS final: gpu: ...
   ```
   The avg moves/game is computed from real data
   (`positions / sum(c.games_done)`), not from `AVG_MOVES_PER_GAME`.

## Reuse summary

| Helper | Location | Used by |
|---|---|---|
| `CloneLogTail` | `src/progress_stats.py` | `train_loop.py` directly; `arena.py` via `CloneState` subclass |
| `fmt_hms` | `src/progress_stats.py` | both |
| `latest_gpu_line` | `src/progress_stats.py` | both |
| `positions_in_iteration` | `train_loop.py` (TR1) | post-Phase-A summary |

`psutil` (added with arena.py) is reused for system CPU sampling. If not
installed, the heartbeat omits the CPU line rather than failing.

## Out of scope

- **Cross-iteration ETA** ("iter 7/10 done, ~9h remaining"). Per-iteration
  durations vary with cadence and model strength, so projection is noisy.
  Defer to a later TR.
- **Live stats during Phase B.** Phase B is 2–3 minutes per iteration; the
  cost of adding a parallel parser for `train.py` output exceeds the value.

## Tests

`tests/test_progress_stats.py` — 16 unit tests covering:
- `fmt_hms`: zero, negative, sub-minute, sub-hour, hours, fractional truncation.
- `latest_gpu_line`: missing file, empty file, no gpu line, single gpu line,
  multiple gpu lines (latest is returned).
- `CloneLogTail`: missing file, parses After-N/M lines, incremental
  byte-offset reads only consume new bytes, monotonic `games_done`,
  late-appearing log file, subclass `_handle_line` hook.

Existing tests (`tests/test_arena.py`, `tests/test_train_loop.py`) continue
to pass after the refactor.

## Verification

1. `python -m pytest tests/test_progress_stats.py tests/test_train_loop.py -v`
   — all green.
2. `python src/arena.py --help` — CLI still parses (smoke test).
3. Live: kick off `train_loop.py` with a tiny cadence and watch the
   heartbeat block appear with progress numbers that climb. Verify they
   match a `tail` of the per-clone logs and the latest `gpu:` line in the
   NNS log.
