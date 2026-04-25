"""
Tests for pmany.py — process manager utilities.

Only pure functions are testable without spawning subprocesses.
"""
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


# Import only the pure functions; argparse runs at module level so we
# must guard against it by monkeypatching sys.argv before import.
import importlib
import unittest.mock as mock

# pmany.py calls argparse.parse_args() and os.mkdir() at module level.
# Patch those so the module can be imported in tests.
with mock.patch('sys.argv', ['pmany.py', '--num_clones', '1', '--log_dir', '/dev/null', 'echo']), \
     mock.patch('os.mkdir'), \
     mock.patch('builtins.open', mock.mock_open()):
    import pmany


# ---------------------------------------------------------------------------
# mini_log10
# ---------------------------------------------------------------------------

class TestMiniLog10:
    def test_one(self):
        assert pmany.mini_log10(1) == 1

    def test_nine(self):
        assert pmany.mini_log10(9) == 1

    def test_ten(self):
        assert pmany.mini_log10(10) == 2

    def test_99(self):
        assert pmany.mini_log10(99) == 2

    def test_100(self):
        assert pmany.mini_log10(100) == 3

    def test_1000(self):
        assert pmany.mini_log10(1000) == 4

    def test_zero_raises(self):
        with pytest.raises(AssertionError):
            pmany.mini_log10(0)

    def test_negative_raises(self):
        with pytest.raises(AssertionError):
            pmany.mini_log10(-5)


# ---------------------------------------------------------------------------
# when
# ---------------------------------------------------------------------------

class TestWhen:
    def test_returns_string(self):
        assert isinstance(pmany.when(), str)

    def test_format(self):
        import re
        s = pmany.when()
        # Expected: "YYYYMMDD HH:MM:SS"
        assert re.match(r'^\d{8} \d{2}:\d{2}:\d{2}$', s), f"Unexpected format: {s!r}"


# ---------------------------------------------------------------------------
# search_replace_cmd
# ---------------------------------------------------------------------------

class TestSearchReplaceCmd:
    def test_single_substitution(self):
        result = pmany.search_replace_cmd(['prog', '--name=%n%'], '42')
        assert result == ['prog', '--name=42']

    def test_no_placeholder(self):
        result = pmany.search_replace_cmd(['echo', 'hello'], '99')
        assert result == ['echo', 'hello']

    def test_multiple_occurrences(self):
        result = pmany.search_replace_cmd(['%n%', '%n%'], 'abc')
        assert result == ['abc', 'abc']

    def test_empty_cmdline(self):
        result = pmany.search_replace_cmd([], '0')
        assert result == []

    def test_name_with_digits(self):
        result = pmany.search_replace_cmd(['log-%n%.txt'], '007')
        assert result == ['log-007.txt']


# ---------------------------------------------------------------------------
# shutdown_children — signal handler that terminates all live child
# instances when pmany itself receives SIGINT/SIGTERM.
# ---------------------------------------------------------------------------

import signal


def _make_proc(pid, ident=0):
    """Build a fake ProcInfo whose underlying Popen has a settable terminate."""
    p = mock.Mock()
    p.pid = pid
    p.terminate = mock.Mock()
    return pmany.ProcInfo(p, ident, mock.Mock())


class TestShutdownChildren:
    def test_terminates_each_child(self, monkeypatch):
        procs = [_make_proc(pid=1000 + i, ident=i) for i in range(3)]
        monkeypatch.setattr(pmany, 'procmap', {p.p.pid: p for p in procs})
        monkeypatch.setattr(pmany, '_shutdown_requested', False)

        pmany.shutdown_children(signal.SIGINT, None)

        for proc in procs:
            proc.p.terminate.assert_called_once()

    def test_sets_shutdown_requested_flag(self, monkeypatch):
        monkeypatch.setattr(pmany, 'procmap', {})
        monkeypatch.setattr(pmany, '_shutdown_requested', False)

        pmany.shutdown_children(signal.SIGTERM, None)

        assert pmany._shutdown_requested is True

    def test_continues_after_terminate_error(self, monkeypatch):
        """A failing terminate() on one child must not prevent the others."""
        proc_dead = _make_proc(pid=1, ident=0)
        proc_live1 = _make_proc(pid=2, ident=1)
        proc_live2 = _make_proc(pid=3, ident=2)
        proc_dead.p.terminate.side_effect = ProcessLookupError("already gone")

        monkeypatch.setattr(pmany, 'procmap', {
            proc_dead.p.pid: proc_dead,
            proc_live1.p.pid: proc_live1,
            proc_live2.p.pid: proc_live2,
        })
        monkeypatch.setattr(pmany, '_shutdown_requested', False)

        pmany.shutdown_children(signal.SIGINT, None)

        proc_dead.p.terminate.assert_called_once()
        proc_live1.p.terminate.assert_called_once()
        proc_live2.p.terminate.assert_called_once()

    def test_empty_procmap_is_noop(self, monkeypatch):
        """Handler must not crash if no children are registered yet."""
        monkeypatch.setattr(pmany, 'procmap', {})
        monkeypatch.setattr(pmany, '_shutdown_requested', False)

        pmany.shutdown_children(signal.SIGINT, None)  # must not raise

        assert pmany._shutdown_requested is True

    def test_handlers_registered(self):
        """signal.signal must have installed the handler for both signals."""
        assert signal.getsignal(signal.SIGINT) is pmany.shutdown_children
        assert signal.getsignal(signal.SIGTERM) is pmany.shutdown_children
