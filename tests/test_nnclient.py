"""
Tests for nnclient.py — Resource reply-parsing logic.

The full IPC path requires a running smmpp server, so we test:
  - reply-byte decoding in isolation (by constructing fake reply bytes)
  - Resource construction validation
"""
import os
import sys
import struct
import numpy
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import twixt
import naf
from naf import LearningState

NUM_MOVES = LearningState.NUM_MOVES  # 528
SIZE = twixt.Game.SIZE


# ---------------------------------------------------------------------------
# Helper: build the reply bytes that nns.py/NNServer.run_jobs produces
# ---------------------------------------------------------------------------

def make_reply_bytes(pws_array, mls_array):
    """Encode (pws, mls) as the bytes nns.py sends back per query.

    pws_array: 1-D float32 array of length pwin_size (1 or 3)
    mls_array: 1-D float32 array of length NUM_MOVES (528)
    """
    b  = numpy.array([pws_array], dtype=numpy.float32).tobytes()
    b += mls_array.astype(numpy.float32).tobytes()
    return b


def parse_reply(reply_bytes):
    """Replicate the parsing logic in Resource.eval()."""
    p0 = numpy.frombuffer(reply_bytes, dtype=numpy.float32)
    nml = SIZE * (SIZE - 2)
    if p0.shape[0] == nml + 1:
        pwin = p0[0]
        movelogits = p0[1:]
    elif p0.shape[0] == nml + 3:
        pwin = p0[0:3]
        movelogits = p0[3:]
    else:
        raise TypeError("Unexpected shape:", p0.shape)
    return pwin, movelogits


# ---------------------------------------------------------------------------
# Reply-byte encoding (mirrors nns.py NNServer.run_jobs)
# ---------------------------------------------------------------------------

class TestReplyEncoding:
    def test_pwin3_total_length(self):
        pws = numpy.array([0.1, 0.6, 0.3], dtype=numpy.float32)
        mls = numpy.zeros(NUM_MOVES, dtype=numpy.float32)
        b = make_reply_bytes(pws, mls)
        assert len(b) == 4 * (3 + NUM_MOVES)

    def test_pwin1_total_length(self):
        pws = numpy.array([0.7], dtype=numpy.float32)
        mls = numpy.zeros(NUM_MOVES, dtype=numpy.float32)
        b = make_reply_bytes(pws, mls)
        assert len(b) == 4 * (1 + NUM_MOVES)

    def test_uses_tobytes_not_tostring(self):
        """tobytes() and tostring() are identical; verify bytes type."""
        pws = numpy.array([0.5, 0.3, 0.2], dtype=numpy.float32)
        mls = numpy.zeros(NUM_MOVES, dtype=numpy.float32)
        b = make_reply_bytes(pws, mls)
        assert isinstance(b, bytes)


# ---------------------------------------------------------------------------
# Reply-byte parsing (mirrors Resource.eval())
# ---------------------------------------------------------------------------

class TestReplyParsing:
    def test_pwin3_pwin_shape(self):
        pws = numpy.array([0.1, 0.6, 0.3], dtype=numpy.float32)
        mls = numpy.arange(NUM_MOVES, dtype=numpy.float32)
        b = make_reply_bytes(pws, mls)
        pwin, movelogits = parse_reply(b)
        assert pwin.shape == (3,)

    def test_pwin3_movelogits_shape(self):
        pws = numpy.array([0.1, 0.6, 0.3], dtype=numpy.float32)
        mls = numpy.arange(NUM_MOVES, dtype=numpy.float32)
        b = make_reply_bytes(pws, mls)
        pwin, movelogits = parse_reply(b)
        assert movelogits.shape == (NUM_MOVES,)

    def test_pwin3_values_preserved(self):
        pws = numpy.array([-1.5, 0.0, 2.3], dtype=numpy.float32)
        mls = numpy.zeros(NUM_MOVES, dtype=numpy.float32)
        b = make_reply_bytes(pws, mls)
        pwin, _ = parse_reply(b)
        assert numpy.allclose(pwin, pws, atol=1e-6)

    def test_movelogits_values_preserved(self):
        pws = numpy.array([0.1, 0.2, 0.7], dtype=numpy.float32)
        mls = numpy.arange(NUM_MOVES, dtype=numpy.float32)
        b = make_reply_bytes(pws, mls)
        _, movelogits = parse_reply(b)
        assert numpy.allclose(movelogits, mls, atol=1e-6)

    def test_pwin1_pwin_scalar(self):
        pws = numpy.array([0.8], dtype=numpy.float32)
        mls = numpy.zeros(NUM_MOVES, dtype=numpy.float32)
        b = make_reply_bytes(pws, mls)
        pwin, movelogits = parse_reply(b)
        assert numpy.isclose(float(pwin), 0.8, atol=1e-6)
        assert movelogits.shape == (NUM_MOVES,)

    def test_unexpected_size_raises(self):
        bad = numpy.zeros(5, dtype=numpy.float32).tobytes()
        with pytest.raises(TypeError):
            parse_reply(bad)

    def test_pwin3_three_to_one(self):
        """pwin from a 3-class reply can be converted to a scalar via three_to_one."""
        pws = numpy.array([0.0, 0.0, 1.0], dtype=numpy.float32)  # pure win logit
        mls = numpy.zeros(NUM_MOVES, dtype=numpy.float32)
        b = make_reply_bytes(pws, mls)
        pwin, _ = parse_reply(b)
        score = naf.three_to_one(pwin)
        assert isinstance(score, float)
        assert score > 0  # Win logit dominant → positive score


# ---------------------------------------------------------------------------
# nnclient.Resource type check
# ---------------------------------------------------------------------------

class TestResourceTypeCheck:
    def test_eval_rejects_non_netinputs(self):
        """Resource.eval raises TypeError for non-NetInputs inputs."""
        import nnclient

        class FakeClient:
            def write_query(self, *a): pass
            def handle_read(self): pass

        class FakeResource(nnclient.Resource):
            def __init__(self):
                self.client = FakeClient()

        r = FakeResource()
        with pytest.raises(TypeError):
            r.eval("not a NetInputs")

    def test_eval_accepts_netinputs(self):
        """Resource.eval accepts a NetInputs without raising TypeError early."""
        import nnclient

        replies = []

        class FakeClient:
            reply_bytes = make_reply_bytes(
                numpy.array([0.1, 0.2, 0.7], dtype=numpy.float32),
                numpy.zeros(NUM_MOVES, dtype=numpy.float32)
            )
            def write_query(self, data, cb):
                # Simulate immediate callback with fake reply
                cb(self.reply_bytes)
            def handle_read(self):
                pass  # callback already fired in write_query

        class FakeResource(nnclient.Resource):
            def __init__(self):
                self.client = FakeClient()

        r = FakeResource()
        g = twixt.Game()
        nips = naf.NetInputs(g)
        pwin, movelogits = r.eval(nips)
        assert pwin.shape == (3,)
        assert movelogits.shape == (NUM_MOVES,)
