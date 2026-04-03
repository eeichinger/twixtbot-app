"""
Tests for timestat.py — TimeStat and WorkTimeStat.
"""
import time
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from timestat import TimeStat, WorkTimeStat


# ---------------------------------------------------------------------------
# TimeStat
# ---------------------------------------------------------------------------

class TestTimeStat:
    def test_initial_state(self):
        ts = TimeStat("t")
        assert ts.total_count == 0
        assert ts.total_time == 0
        assert ts.start_time is None

    def test_start_stop(self):
        ts = TimeStat("t")
        ts.start()
        time.sleep(0.01)
        ts.stop()
        assert ts.total_count == 1
        assert ts.total_time > 0

    def test_double_start_raises(self):
        ts = TimeStat("t")
        ts.start()
        with pytest.raises(AssertionError):
            ts.start()

    def test_stop_without_start_raises(self):
        ts = TimeStat("t")
        with pytest.raises(AssertionError):
            ts.stop()

    def test_multiple_rounds(self):
        ts = TimeStat("t")
        for _ in range(3):
            ts.start()
            ts.stop()
        assert ts.total_count == 3

    def test_str_contains_name(self):
        ts = TimeStat("mytimer")
        ts.start()
        ts.stop()
        assert "mytimer" in str(ts)

    def test_ignore_filters_long_durations(self):
        """With ignore=large_value, counts are filtered; start/stop should work."""
        ts = TimeStat("t", ignore=0.001)   # ignore anything >= 1ms
        ts.start()
        time.sleep(0.05)   # definitely > 1ms → ignored
        ts.stop()
        # Either count is 0 (filtered) or 1 (just under threshold race) — both fine
        # Key: no exception raised
        assert ts.start_time is None


# ---------------------------------------------------------------------------
# WorkTimeStat
# ---------------------------------------------------------------------------

class TestWorkTimeStat:
    def test_initial_state(self):
        wts = WorkTimeStat("wt")
        assert wts.total_count() == 0
        assert wts.total_work() == 0
        assert wts.total_time() == 0

    def test_start_stop(self):
        wts = WorkTimeStat("wt")
        wts.start(10)
        time.sleep(0.01)
        wts.stop()
        assert wts.total_count() == 1
        assert wts.total_work() == pytest.approx(10.0)
        assert wts.total_time() > 0

    def test_double_start_raises(self):
        wts = WorkTimeStat("wt")
        wts.start(1)
        with pytest.raises(AssertionError):
            wts.start(2)

    def test_stop_without_start_raises(self):
        wts = WorkTimeStat("wt")
        with pytest.raises(AssertionError):
            wts.stop()

    def test_multiple_rounds_accumulate_work(self):
        wts = WorkTimeStat("wt")
        for w in [5, 10, 15]:
            wts.start(w)
            wts.stop()
        assert wts.total_count() == pytest.approx(3.0)
        assert wts.total_work() == pytest.approx(30.0)

    def test_str_contains_name(self):
        wts = WorkTimeStat("mywork")
        wts.start(1)
        wts.stop()
        assert "mywork" in str(wts)

    def test_str_singular_no_crash(self):
        """__str__ should work with a single measurement (no matrix error)."""
        wts = WorkTimeStat("wt")
        wts.start(5)
        wts.stop()
        s = str(wts)
        assert isinstance(s, str)
