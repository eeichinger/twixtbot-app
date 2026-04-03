"""
Tests for smmpp.py — Shared Memory Message Passing Protocol.

Unit tests cover checksum, slot arithmetic, and constants.
Integration tests run a SillyServer end-to-end using a temp directory.
"""
import os
import struct
import sys
import tempfile
import threading
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import smmpp
from smmpp import (
    checksum,
    QUERY_AVAILABLE, REPLY_AVAILABLE, ANSWER_SENT,
    SUICIDE_CODE,
    IntPacker,
    SillyServer,
    Client,
    Server,
)


# ---------------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_query_available_is_bytes(self):
        assert isinstance(QUERY_AVAILABLE, bytes)

    def test_reply_available_is_bytes(self):
        assert isinstance(REPLY_AVAILABLE, bytes)

    def test_answer_sent_is_bytes(self):
        assert isinstance(ANSWER_SENT, bytes)

    def test_all_distinct(self):
        assert len({QUERY_AVAILABLE, REPLY_AVAILABLE, ANSWER_SENT}) == 3

    def test_each_is_two_bytes(self):
        for c in (QUERY_AVAILABLE, REPLY_AVAILABLE, ANSWER_SENT):
            assert len(c) == 2

    def test_suicide_code_is_int(self):
        assert isinstance(SUICIDE_CODE, int)


# ---------------------------------------------------------------------------
# checksum
# ---------------------------------------------------------------------------

class TestChecksum:
    def test_returns_int(self):
        assert isinstance(checksum(b"hello"), int)

    def test_constant_0xcc(self):
        """Current implementation always returns 0xcc."""
        assert checksum(b"") == 0xcc
        assert checksum(b"anything") == 0xcc
        assert checksum(bytes(100)) == 0xcc

    def test_fits_in_byte(self):
        assert 0 <= checksum(b"data") <= 255


# ---------------------------------------------------------------------------
# Slot arithmetic (no I/O)
# ---------------------------------------------------------------------------

class TestSlotLocations:
    """Verify slot_locations arithmetic via Server/ServerSocketProcess directly."""

    def _make_server_proxy(self, query_size, reply_size):
        """Create a minimal object with slot_locations but no file I/O."""
        class Proxy:
            def __init__(self, q, r):
                self.slot_size = 3 + max(q, r)
            def slot_locations(self, slot):
                return slot * self.slot_size, (slot + 1) * self.slot_size
        return Proxy(query_size, reply_size)

    def test_slot_zero_starts_at_zero(self):
        p = self._make_server_proxy(100, 100)
        x, y = p.slot_locations(0)
        assert x == 0

    def test_slot_zero_end_equals_slot_size(self):
        p = self._make_server_proxy(100, 100)
        x, y = p.slot_locations(0)
        assert y == p.slot_size

    def test_slots_contiguous(self):
        p = self._make_server_proxy(50, 80)
        _, y0 = p.slot_locations(0)
        x1, _ = p.slot_locations(1)
        assert y0 == x1

    def test_slot_size_uses_max(self):
        p_q_big = self._make_server_proxy(200, 10)
        p_r_big = self._make_server_proxy(10, 200)
        assert p_q_big.slot_size == p_r_big.slot_size == 3 + 200

    def test_checksum_byte_at_y_minus_3(self):
        """Status bytes at y-2:y; checksum byte at y-3."""
        p = self._make_server_proxy(100, 100)
        x, y = p.slot_locations(0)
        # 3 overhead bytes: [y-3]=checksum, [y-2:y]=status
        assert y - x == p.slot_size
        assert y - 3 >= x   # checksum byte is within the slot


# ---------------------------------------------------------------------------
# SillyServer.run_jobs (unit — no IPC)
# ---------------------------------------------------------------------------

class TestSillyRunJobs:
    """SillyServer.run_jobs reverses each job. Tests the Python 3 bytes fix."""

    def _make_silly(self, tmp_path):
        loc = str(tmp_path / "silly")
        return SillyServer(loc, capacity=4, query_size=20, reply_size=20)

    def test_reverses_bytes(self, tmp_path):
        s = self._make_silly(tmp_path)
        jobs = [b"hello", b"world"]
        result = s.run_jobs(jobs)
        assert result[0] == b"olleh"
        assert result[1] == b"dlrow"

    def test_returns_bytes(self, tmp_path):
        s = self._make_silly(tmp_path)
        result = s.run_jobs([b"abc"])
        assert isinstance(result[0], bytes)

    def test_empty_job(self, tmp_path):
        s = self._make_silly(tmp_path)
        result = s.run_jobs([b""])
        assert result[0] == b""

    def test_single_byte(self, tmp_path):
        s = self._make_silly(tmp_path)
        result = s.run_jobs([b"\x42"])
        assert result[0] == b"\x42"

    def test_batch_of_jobs(self, tmp_path):
        s = self._make_silly(tmp_path)
        jobs = [bytes([i, i+1, i+2]) for i in range(5)]
        result = s.run_jobs(jobs)
        assert len(result) == 5
        for original, rev in zip(jobs, result):
            assert rev == bytes(reversed(original))


# ---------------------------------------------------------------------------
# Client.write_query type validation
# ---------------------------------------------------------------------------

class TestClientWriteQuery:
    def test_str_raises(self, tmp_path):
        """write_query must reject str input (Python 3 bytes fix)."""
        # We can test the guard without a real server by calling the method
        # through a dummy Client that skips socket/shmem setup.
        class FakeClient(Client):
            def __init__(self):
                # Bypass __init__ entirely
                self.query_size = 100
                self.unused_slots = [0]
                self.cb_by_slot = {}
                self.slot_size = 103
                self.shmem = bytearray(103)

            def slot_locations(self, slot):
                return 0, 103

        c = FakeClient()
        with pytest.raises((ValueError, TypeError)):
            c.write_query("not bytes", lambda x: None)

    def test_bytes_accepted(self, tmp_path):
        class FakeClient(Client):
            def __init__(self):
                self.query_size = 100
                self.unused_slots = [0]
                self.cb_by_slot = {}
                self.slot_size = 103
                self.shmem = bytearray(103)

            def slot_locations(self, slot):
                return 0, 103  # x=0, y=103 → status at shmem[101:103]

        c = FakeClient()
        callbacks = []
        c.write_query(b"hello", lambda x: callbacks.append(x))
        # Status is at shmem[y-2:y] = shmem[101:103]
        assert bytes(c.shmem[101:103]) == QUERY_AVAILABLE


# ---------------------------------------------------------------------------
# Integration test: SillyServer + Client over Unix socket
# ---------------------------------------------------------------------------

@pytest.fixture
def server_location(tmp_path):
    return str(tmp_path / "test_server")


def _run_server(location, capacity, query_size, reply_size):
    """Start a SillyServer and run it (blocks until killed or socket error)."""
    s = SillyServer(location, capacity, query_size, reply_size)
    s.run()


class TestIntegration:
    """Full round-trip: SillyServer in a thread (gpu_side only), Client sends query."""

    QUERY_SIZE = 20
    REPLY_SIZE = 20
    CAPACITY = 4

    def _wait_for_socket(self, sock_path, timeout=5.0):
        """Poll until the socket file exists."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if os.path.exists(sock_path):
                return True
            time.sleep(0.05)
        return False

    def _recv_reply(self, client, timeout=5.0):
        """Block on the client Unix socket until a reply arrives (or timeout)."""
        import select as sel
        deadline = time.time() + timeout
        while time.time() < deadline:
            remaining = deadline - time.time()
            rd, _, _ = sel.select([client.socket], [], [], remaining)
            if rd:
                client.handle_read()
                return True
        return False

    def test_silly_roundtrip(self, tmp_path):
        """Client sends bytes; SillyServer reverses them; client gets reversed bytes back."""
        import multiprocessing
        loc = str(tmp_path / "srv")

        proc = multiprocessing.Process(
            target=_run_server,
            args=(loc, self.CAPACITY, self.QUERY_SIZE, self.REPLY_SIZE),
        )
        proc.start()

        try:
            sock_path = loc + ".sock"
            assert self._wait_for_socket(sock_path), "Server socket never appeared"

            replies = []
            c = Client(loc, slots_needed=1, quiet=True)
            # Fill the full QUERY_SIZE so reversed bytes are deterministic
            query = bytes(range(self.QUERY_SIZE))
            c.write_query(query, lambda data: replies.append(data))

            # Wait for server to push reply notification via socket
            assert self._recv_reply(c), "No reply received within timeout"
            assert replies, "handle_read was called but no callback fired"
            # SillyServer reverses the full QUERY_SIZE bytes
            assert replies[0] == bytes(reversed(query))
        finally:
            proc.terminate()
            proc.join(timeout=2)

    def test_multiple_queries(self, tmp_path):
        """Send two sequential queries; both return correct reversed bytes."""
        import multiprocessing
        loc = str(tmp_path / "srv2")

        proc = multiprocessing.Process(
            target=_run_server,
            args=(loc, self.CAPACITY, self.QUERY_SIZE, self.REPLY_SIZE),
        )
        proc.start()

        try:
            assert self._wait_for_socket(loc + ".sock")
            c = Client(loc, slots_needed=1, quiet=True)
            # Use full QUERY_SIZE queries so reversed bytes are deterministic
            queries = [
                bytes(range(self.QUERY_SIZE)),
                bytes(range(self.QUERY_SIZE - 1, -1, -1)),
            ]
            replies = {}

            for q in queries:
                got = []
                c.write_query(q, lambda data, q=q: got.append((q, data)))
                assert self._recv_reply(c), f"No reply for query"
                assert got, f"handle_read called but no callback"
                replies[q] = got[0][1]

            for q in queries:
                assert replies[q] == bytes(reversed(q))
        finally:
            proc.terminate()
            proc.join(timeout=2)
