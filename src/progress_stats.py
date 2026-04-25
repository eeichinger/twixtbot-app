"""Shared progress-tracking helpers.

Used by both `arena.py` (head-to-head model evaluation) and `train_loop.py`
(self-play training) to:
  - tail per-clone log files via byte offset and parse `After X/Y games`
    progress lines from `battle.py` output;
  - extract the latest `gpu:` milestone line from an NNS log;
  - format seconds as HH:MM:SS for status output.

Score parsing is arena-specific (self-play has white = black = same model)
and stays in `arena.py` as a subclass of `CloneLogTail`.
"""
import os
import re


_AFTER_LINE = re.compile(r'After (\d+)/\d+ games?')
_GPU_RAW_LINE = re.compile(r'^gpu:\s+.*$', re.MULTILINE)


def fmt_hms(seconds):
    """Format seconds as HH:MM:SS. Negative values clamp to zero."""
    s = max(0, int(seconds))
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def latest_gpu_line(log_path):
    """Return the raw text of the latest `gpu:` line from an NNS log, or None.

    The line format comes from `timestat.WorkTimeStat.__str__` printed by
    `nns.py` at every `--milestone_step` positions.
    """
    try:
        with open(log_path) as f:
            content = f.read()
    except FileNotFoundError:
        return None
    last = None
    for m in _GPU_RAW_LINE.finditer(content):
        last = m
    return last.group(0).strip() if last else None


class CloneLogTail:
    """Tail one clone's log file via byte-offset reads, parsing progress.

    `update()` reads any new bytes since the previous call and updates
    `games_done` from `After X/Y games` lines. Subclasses can extend
    `_handle_line()` to parse additional patterns from the same stream.
    """

    def __init__(self, log_path):
        self.log_path = log_path
        self.byte_offset = 0
        self.games_done = 0

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
            self._handle_line(line)

    def _handle_line(self, line):
        """Hook for subclasses; default is a no-op."""
        pass
