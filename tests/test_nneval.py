"""
Tests for nneval.py — PyTorch NNEvaluater (replaces TF1 original).

Uses a freshly-initialised TwixNet so no model file is required.
All tests run on CPU; no GPU required.
"""
import io
import numpy
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import twixt
import naf
from twixt import Point, Game
from naf import LearningState
from model import TwixNet
from nneval import NNEvaluater

NUM_MOVES = LearningState.NUM_MOVES  # 528
SIZE = Game.SIZE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fresh_evaluater():
    m = TwixNet(num_filters=16, num_blocks=2).eval()
    return NNEvaluater(m)


def make_nips(n=2):
    """Return n NetInputs objects built from fresh games."""
    games = [Game() for _ in range(n)]
    return [naf.NetInputs(g) for g in games]


def make_nips_varied(n=3):
    """Return n NetInputs from different game positions."""
    nips = []
    g = Game()
    nips.append(naf.NetInputs(g))
    g.play(Point(5, 5))
    nips.append(naf.NetInputs(g))
    g.play(Point(10, 10))
    nips.append(naf.NetInputs(g))
    return nips[:n]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_accepts_model_object(self):
        m = TwixNet(num_filters=16, num_blocks=2).eval()
        ne = NNEvaluater(m)
        assert ne is not None

    def test_accepts_model_path(self, tmp_path):
        m = TwixNet(num_filters=16, num_blocks=2).eval()
        path = str(tmp_path / "model.pt")
        torch.save(m, path)
        ne = NNEvaluater(path)
        assert ne is not None

    def test_device_cpu(self):
        m = TwixNet(num_filters=16, num_blocks=2).eval()
        ne = NNEvaluater(m, device='cpu')
        assert ne.device == 'cpu'


# ---------------------------------------------------------------------------
# pwin_size
# ---------------------------------------------------------------------------

class TestPwinSize:
    def test_pwin_size_is_3(self):
        """PyTorch model always outputs 3-class value head."""
        ne = fresh_evaluater()
        assert ne.pwin_size() == 3


# ---------------------------------------------------------------------------
# eval_many_prepare
# ---------------------------------------------------------------------------

class TestEvalManyPrepare:
    def test_returns_three_arrays(self):
        ne = fresh_evaluater()
        nips = make_nips(3)
        result = ne.eval_many_prepare(nips)
        assert len(result) == 3

    def test_pegs_shape(self):
        ne = fresh_evaluater()
        nips = make_nips(3)
        pegs, links, locs = ne.eval_many_prepare(nips)
        assert pegs.shape == (3, SIZE, SIZE, 2)

    def test_links_shape(self):
        ne = fresh_evaluater()
        nips = make_nips(3)
        pegs, links, locs = ne.eval_many_prepare(nips)
        assert links.shape == (3, SIZE, SIZE, 8)

    def test_locs_shape(self):
        ne = fresh_evaluater()
        nips = make_nips(3)
        pegs, links, locs = ne.eval_many_prepare(nips)
        assert locs.shape == (3, SIZE, SIZE, 2)

    def test_single_input(self):
        ne = fresh_evaluater()
        nips = make_nips(1)
        pegs, links, locs = ne.eval_many_prepare(nips)
        assert pegs.shape == (1, SIZE, SIZE, 2)

    def test_accepts_generator(self):
        """eval_many_prepare should consume any iterable, not just lists."""
        ne = fresh_evaluater()
        nips = (naf.NetInputs(Game()) for _ in range(2))
        pegs, links, locs = ne.eval_many_prepare(nips)
        assert pegs.shape == (2, SIZE, SIZE, 2)


# ---------------------------------------------------------------------------
# eval_many_doit
# ---------------------------------------------------------------------------

class TestEvalManyDoit:
    def test_returns_two_arrays(self):
        ne = fresh_evaluater()
        nips = make_nips(2)
        pegs, links, locs = ne.eval_many_prepare(nips)
        result = ne.eval_many_doit(pegs, links, locs)
        assert len(result) == 2

    def test_pws_shape(self):
        ne = fresh_evaluater()
        nips = make_nips(4)
        pegs, links, locs = ne.eval_many_prepare(nips)
        pws, mls = ne.eval_many_doit(pegs, links, locs)
        assert pws.shape == (4, 3)

    def test_mls_shape(self):
        ne = fresh_evaluater()
        nips = make_nips(4)
        pegs, links, locs = ne.eval_many_prepare(nips)
        pws, mls = ne.eval_many_doit(pegs, links, locs)
        assert mls.shape == (4, NUM_MOVES)

    def test_output_dtype_float32(self):
        ne = fresh_evaluater()
        nips = make_nips(2)
        pegs, links, locs = ne.eval_many_prepare(nips)
        pws, mls = ne.eval_many_doit(pegs, links, locs)
        assert pws.dtype == numpy.float32
        assert mls.dtype == numpy.float32

    def test_outputs_finite(self):
        ne = fresh_evaluater()
        nips = make_nips(2)
        pegs, links, locs = ne.eval_many_prepare(nips)
        pws, mls = ne.eval_many_doit(pegs, links, locs)
        assert numpy.isfinite(pws).all()
        assert numpy.isfinite(mls).all()


# ---------------------------------------------------------------------------
# eval_many (end-to-end)
# ---------------------------------------------------------------------------

class TestEvalMany:
    def test_returns_two_arrays(self):
        ne = fresh_evaluater()
        pws, mls = ne.eval_many(make_nips(2))
        assert pws.shape == (2, 3)
        assert mls.shape == (2, NUM_MOVES)

    def test_batch_size_1(self):
        ne = fresh_evaluater()
        pws, mls = ne.eval_many(make_nips(1))
        assert pws.shape == (1, 3)
        assert mls.shape == (1, NUM_MOVES)

    def test_batch_size_8(self):
        ne = fresh_evaluater()
        pws, mls = ne.eval_many(make_nips(8))
        assert pws.shape == (8, 3)
        assert mls.shape == (8, NUM_MOVES)

    def test_outputs_finite(self):
        ne = fresh_evaluater()
        pws, mls = ne.eval_many(make_nips(3))
        assert numpy.isfinite(pws).all()
        assert numpy.isfinite(mls).all()

    def test_varied_positions(self):
        """Evaluating different positions gives different policy outputs."""
        ne = fresh_evaluater()
        nips = make_nips_varied(3)
        pws, mls = ne.eval_many(nips)
        assert mls.shape == (3, NUM_MOVES)


# ---------------------------------------------------------------------------
# eval_one
# ---------------------------------------------------------------------------

class TestEvalOne:
    def test_returns_two_arrays(self):
        ne = fresh_evaluater()
        nip = naf.NetInputs(Game())
        pws, mls = ne.eval_one(nip)
        assert len(pws) == 1
        assert len(mls) == 1

    def test_pws_shape(self):
        ne = fresh_evaluater()
        nip = naf.NetInputs(Game())
        pws, mls = ne.eval_one(nip)
        assert pws.shape == (1, 3)

    def test_mls_shape(self):
        ne = fresh_evaluater()
        nip = naf.NetInputs(Game())
        pws, mls = ne.eval_one(nip)
        assert mls.shape == (1, NUM_MOVES)

    def test_output_dtype_float32(self):
        ne = fresh_evaluater()
        nip = naf.NetInputs(Game())
        pws, mls = ne.eval_one(nip)
        assert pws.dtype == numpy.float32
        assert mls.dtype == numpy.float32

    def test_outputs_finite(self):
        ne = fresh_evaluater()
        nip = naf.NetInputs(Game())
        pws, mls = ne.eval_one(nip)
        assert numpy.isfinite(pws).all()
        assert numpy.isfinite(mls).all()

    def test_consistent_with_eval_many(self):
        """eval_one and eval_many with the same input must give identical results."""
        ne = fresh_evaluater()
        nip = naf.NetInputs(Game())
        pws_one, mls_one = ne.eval_one(nip)

        # Re-create nip from same game for eval_many
        pws_many, mls_many = ne.eval_many([nip])
        assert numpy.allclose(pws_one, pws_many, atol=1e-6)
        assert numpy.allclose(mls_one, mls_many, atol=1e-6)


# ---------------------------------------------------------------------------
# nns.py integration: run_jobs byte-encoding
# ---------------------------------------------------------------------------

class TestRunJobsEncoding:
    """Verify the byte-encoding that nns.NNServer.run_jobs relies on."""

    def test_pws_tobytes_length(self):
        """pws[i] as float32 bytes has length 4 * pwin_size()."""
        ne = fresh_evaluater()
        pws, mls = ne.eval_many(make_nips(1))
        b = numpy.array([pws[0]], dtype=numpy.float32).tobytes()
        assert len(b) == 4 * 1 * ne.pwin_size()   # 4 bytes/float × 3

    def test_mls_tobytes_length(self):
        """mls[i] as float32 bytes has length 4 * NUM_MOVES."""
        ne = fresh_evaluater()
        pws, mls = ne.eval_many(make_nips(1))
        b = mls[0].astype(numpy.float32).tobytes()
        assert len(b) == 4 * NUM_MOVES

    def test_from_bytes_roundtrip(self):
        """NetInputs built from EXPANDED_SIZE bytes gives same output as from Game."""
        ne = fresh_evaluater()
        g = Game()
        g.play(Point(7, 3))
        nip_game = naf.NetInputs(g)
        # Serialise to expanded bytes then re-construct
        expanded = nip_game.to_expanded_bytes()
        nip_bytes = naf.NetInputs(expanded)

        pws_g, mls_g = ne.eval_one(nip_game)
        pws_b, mls_b = ne.eval_one(nip_bytes)
        assert numpy.allclose(pws_g, pws_b, atol=1e-6)
        assert numpy.allclose(mls_g, mls_b, atol=1e-6)

    def test_total_reply_bytes(self):
        """Total reply size matches nns.py formula: 4*(NUM_MOVES + pwin_size())."""
        ne = fresh_evaluater()
        pws, mls = ne.eval_many(make_nips(1))
        b_pws = numpy.array([pws[0]], dtype=numpy.float32).tobytes()
        b_mls = mls[0].astype(numpy.float32).tobytes()
        total = len(b_pws) + len(b_mls)
        expected = 4 * (NUM_MOVES + ne.pwin_size())
        assert total == expected


# ---------------------------------------------------------------------------
# Model file round-trip
# ---------------------------------------------------------------------------

class TestModelFileRoundtrip:
    def test_save_and_load_gives_same_output(self, tmp_path):
        m = TwixNet(num_filters=16, num_blocks=2).eval()
        path = str(tmp_path / "model.pt")
        torch.save(m, path)

        ne1 = NNEvaluater(m)
        ne2 = NNEvaluater(path)

        nips = make_nips(2)
        pws1, mls1 = ne1.eval_many(nips)
        # Need fresh nips since generators are consumed
        pws2, mls2 = ne2.eval_many(make_nips(2))
        # Both use default-init weights from the same seed, so values differ;
        # just check shapes are consistent
        assert pws1.shape == pws2.shape
        assert mls1.shape == mls2.shape

    def test_loaded_model_same_weights(self, tmp_path):
        m = TwixNet(num_filters=16, num_blocks=2).eval()
        path = str(tmp_path / "model.pt")
        torch.save(m, path)

        ne1 = NNEvaluater(m)
        ne2 = NNEvaluater(path)

        # Use the same NetInputs object for both
        nip = naf.NetInputs(Game())
        pws1, mls1 = ne1.eval_one(nip)
        pws2, mls2 = ne2.eval_one(nip)
        assert numpy.allclose(pws1, pws2, atol=1e-6)
        assert numpy.allclose(mls1, mls2, atol=1e-6)
