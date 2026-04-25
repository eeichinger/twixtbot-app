"""
Tests for train_loop.py — resume marker helpers and position counting.

Only the pure helpers added by TR1 are unit-tested here. The full
orchestration (Phase A / Phase B / NNS lifecycle) requires subprocess
spawning and is exercised by the manual test sequence in
docs/specs/tr1-train-loop-resumable.md.
"""
import os
import sys

# train_loop.py lives at project root, not under src/. Add root to path.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import train_loop  # noqa: E402


# ---------------------------------------------------------------------------
# positions_in_iteration
# ---------------------------------------------------------------------------

class TestPositionsInIteration:
    def test_empty_directory(self, tmp_path):
        assert train_loop.positions_in_iteration(1, str(tmp_path)) == 0

    def test_missing_directory(self, tmp_path):
        missing = tmp_path / "does_not_exist"
        assert train_loop.positions_in_iteration(1, str(missing)) == 0

    def test_counts_full_records(self, tmp_path):
        rec = train_loop.LEARNING_STATE_BYTES
        (tmp_path / "iter4_00.bin").write_bytes(b'X' * rec * 100)
        (tmp_path / "iter4_01.bin").write_bytes(b'X' * rec * 50)
        assert train_loop.positions_in_iteration(4, str(tmp_path)) == 150

    def test_torn_trailing_record_is_discarded(self, tmp_path):
        """A partial trailing record must be ignored, not counted."""
        rec = train_loop.LEARNING_STATE_BYTES
        (tmp_path / "iter4_00.bin").write_bytes(b'X' * (rec * 30 + 17))
        assert train_loop.positions_in_iteration(4, str(tmp_path)) == 30

    def test_ignores_other_iterations(self, tmp_path):
        rec = train_loop.LEARNING_STATE_BYTES
        (tmp_path / "iter4_00.bin").write_bytes(b'X' * rec * 10)
        (tmp_path / "iter5_00.bin").write_bytes(b'X' * rec * 999)
        (tmp_path / "iter40_00.bin").write_bytes(b'X' * rec * 999)  # not iter 4
        assert train_loop.positions_in_iteration(4, str(tmp_path)) == 10

    def test_ignores_unrelated_filenames(self, tmp_path):
        rec = train_loop.LEARNING_STATE_BYTES
        (tmp_path / "iter4_00.bin").write_bytes(b'X' * rec * 10)
        (tmp_path / "iter4_meta.txt").write_bytes(b'X' * rec * 10)
        (tmp_path / "noise.bin").write_bytes(b'X' * rec * 10)
        (tmp_path / "iter4.done").write_bytes(b'marker')
        assert train_loop.positions_in_iteration(4, str(tmp_path)) == 10


# ---------------------------------------------------------------------------
# write_marker_atomic
# ---------------------------------------------------------------------------

class TestWriteMarkerAtomic:
    def test_creates_file_with_content(self, tmp_path):
        path = str(tmp_path / "iter1.done")
        train_loop.write_marker_atomic(path, "model=v1.pt timestamp=2026-01-01\n")
        assert open(path).read() == "model=v1.pt timestamp=2026-01-01\n"

    def test_no_partial_leftover(self, tmp_path):
        path = str(tmp_path / "iter1.done")
        train_loop.write_marker_atomic(path, "x")
        assert not os.path.exists(path + ".partial")

    def test_overwrites_existing(self, tmp_path):
        path = str(tmp_path / "iter1.done")
        train_loop.write_marker_atomic(path, "first")
        train_loop.write_marker_atomic(path, "second")
        assert open(path).read() == "second"


# ---------------------------------------------------------------------------
# Marker path helpers
# ---------------------------------------------------------------------------

class TestMarkerPaths:
    def test_phase_a_done_path(self):
        p = train_loop.marker_path_phase_a_done(7)
        assert p.endswith("iter7.phase_a_done")
        assert train_loop.SPDATA_DIR in p

    def test_done_path(self):
        p = train_loop.marker_path_done(7)
        assert p.endswith("iter7.done")
        assert train_loop.SPDATA_DIR in p

    def test_distinct_paths(self):
        a = train_loop.marker_path_phase_a_done(3)
        b = train_loop.marker_path_done(3)
        assert a != b
