"""
Tests for src/progress_stats.py — shared progress-tracking helpers used by
arena.py and train_loop.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from progress_stats import CloneLogTail, fmt_hms, latest_gpu_line  # noqa: E402


# ---------------------------------------------------------------------------
# fmt_hms
# ---------------------------------------------------------------------------

class TestFmtHms:
    def test_zero(self):
        assert fmt_hms(0) == "00:00:00"

    def test_negative_clamps_to_zero(self):
        assert fmt_hms(-100) == "00:00:00"

    def test_sub_minute(self):
        assert fmt_hms(45) == "00:00:45"

    def test_sub_hour(self):
        assert fmt_hms(125) == "00:02:05"

    def test_with_hours(self):
        assert fmt_hms(3725) == "01:02:05"

    def test_many_hours(self):
        assert fmt_hms(36000) == "10:00:00"

    def test_truncates_fractional(self):
        assert fmt_hms(45.9) == "00:00:45"


# ---------------------------------------------------------------------------
# latest_gpu_line
# ---------------------------------------------------------------------------

class TestLatestGpuLine:
    def test_missing_file(self, tmp_path):
        assert latest_gpu_line(str(tmp_path / "noexist.log")) is None

    def test_empty_file(self, tmp_path):
        p = tmp_path / "nns.log"
        p.write_text("")
        assert latest_gpu_line(str(p)) is None

    def test_no_gpu_line(self, tmp_path):
        p = tmp_path / "nns.log"
        p.write_text("Ready for connections on /tmp/x.sock\n"
                     "gpu side going!\n"
                     "opened connection. cons=2 free_slots=1024\n")
        assert latest_gpu_line(str(p)) is None

    def test_single_gpu_line(self, tmp_path):
        p = tmp_path / "nns.log"
        p.write_text("noise\n"
                     "gpu: N=100 T=10 W=1000 W/N=10.0 W/T=100/s avg=0.001 + 0.0001*W a/b=10\n"
                     "more noise\n")
        line = latest_gpu_line(str(p))
        assert line is not None
        assert line.startswith("gpu: N=100")

    def test_returns_latest_among_many(self, tmp_path):
        p = tmp_path / "nns.log"
        p.write_text(
            "gpu: N=100 T=10 W=1000 W/N=10.0 W/T=100/s avg=0 + 0.0001*W a/b=10\n"
            "----\n"
            "gpu: N=200 T=20 W=2000 W/N=10.0 W/T=100/s avg=0 + 0.0001*W a/b=10\n"
            "----\n"
            "gpu: N=300 T=30 W=3000 W/N=10.0 W/T=100/s avg=0 + 0.0001*W a/b=10\n"
        )
        line = latest_gpu_line(str(p))
        assert "N=300" in line
        assert "N=100" not in line and "N=200" not in line


# ---------------------------------------------------------------------------
# CloneLogTail
# ---------------------------------------------------------------------------

class TestCloneLogTail:
    def test_missing_file(self, tmp_path):
        c = CloneLogTail(str(tmp_path / "noexist.log"))
        c.update()  # must not raise
        assert c.games_done == 0

    def test_parses_after_lines(self, tmp_path):
        p = tmp_path / "00.log"
        p.write_text(
            "20260425 12:00:00 After 1/50 games (10s elapsed)...\n"
            ":  1.0 (100.0%) asn_player:location=/x,trials=10\n"
            "20260425 12:01:00 After 5/50 games (60s elapsed)...\n"
            ":  3.0 ( 60.0%) asn_player:location=/x,trials=10\n"
        )
        c = CloneLogTail(str(p))
        c.update()
        assert c.games_done == 5

    def test_incremental_reads(self, tmp_path):
        """Each update() should only consume *new* bytes."""
        p = tmp_path / "00.log"
        p.write_text("20260425 12:00:00 After 3/50 games\n")
        c = CloneLogTail(str(p))
        c.update()
        assert c.games_done == 3

        # Append more lines and update again
        with open(p, "a") as f:
            f.write("20260425 12:01:00 After 7/50 games\n")
        c.update()
        assert c.games_done == 7

    def test_does_not_decrease(self, tmp_path):
        """Out-of-order or stale lines must not lower the games_done count."""
        p = tmp_path / "00.log"
        p.write_text("After 7/50 games\nAfter 3/50 games\n")
        c = CloneLogTail(str(p))
        c.update()
        assert c.games_done == 7

    def test_handles_missing_file_then_appears(self, tmp_path):
        """File may not exist when CloneLogTail is constructed (clone hasn't
        spawned yet)."""
        path = tmp_path / "00.log"
        c = CloneLogTail(str(path))
        c.update()  # file missing
        assert c.games_done == 0

        path.write_text("After 2/10 games\n")
        c.update()
        assert c.games_done == 2

    def test_subclass_handle_line_hook(self, tmp_path):
        """Subclasses can extend parsing without re-implementing the file tail."""
        captured = []

        class MyTail(CloneLogTail):
            def _handle_line(self, line):
                if line.startswith("ZZ:"):
                    captured.append(line)

        p = tmp_path / "00.log"
        p.write_text("After 1/10 games\nZZ: hello\nZZ: world\n")
        c = MyTail(str(p))
        c.update()
        assert c.games_done == 1
        assert captured == ["ZZ: hello", "ZZ: world"]
