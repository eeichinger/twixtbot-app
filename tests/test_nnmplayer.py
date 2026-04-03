"""
Tests for nnmplayer.py — Player wrapping NeuralMCTS with a neural net.

Uses a tiny TwixNet (8 filters, 1 block) as the model to keep tests fast.
"""
import os
import sys
import numpy
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import twixt
import naf
from twixt import Point, Game
from naf import LearningState
from model import TwixNet
from nnmplayer import Player

NUM_MOVES = LearningState.NUM_MOVES  # 528


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_model(tmp_path):
    """Save a tiny TwixNet to disk and return its path."""
    m = TwixNet(num_filters=8, num_blocks=1).eval()
    path = str(tmp_path / "tiny.pt")
    torch.save(m, path)
    return path


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestPlayerConstruction:
    def test_from_model_path(self, tmp_path):
        path = make_model(tmp_path)
        p = Player(model=path)
        assert p is not None

    def test_from_model_object(self, tmp_path):
        m = TwixNet(num_filters=8, num_blocks=1).eval()
        p = Player(model=m)
        assert p is not None

    def test_bad_temperature_raises(self, tmp_path):
        path = make_model(tmp_path)
        with pytest.raises(ValueError):
            Player(model=path, temperature=0.3)

    def test_no_model_no_resource_raises(self):
        with pytest.raises(Exception):
            Player()

    def test_default_trials(self, tmp_path):
        path = make_model(tmp_path)
        p = Player(model=path)
        assert p.num_trials == 100

    def test_custom_trials(self, tmp_path):
        path = make_model(tmp_path)
        p = Player(model=path, trials=10)
        assert p.num_trials == 10

    def test_temperature_zero(self, tmp_path):
        path = make_model(tmp_path)
        p = Player(model=path, temperature=0)
        assert p.temperature == 0.0

    def test_temperature_half(self, tmp_path):
        path = make_model(tmp_path)
        p = Player(model=path, temperature=0.5)
        assert p.temperature == 0.5


# ---------------------------------------------------------------------------
# pick_move — basic
# ---------------------------------------------------------------------------

class TestPickMove:
    def _player(self, tmp_path, **kw):
        path = make_model(tmp_path)
        return Player(model=path, trials=4, **kw)

    def test_returns_point_or_str(self, tmp_path):
        p = self._player(tmp_path)
        g = Game()
        result = p.pick_move(g)
        # Returns either (Point, N) tuple or a str/Point for forced moves
        if isinstance(result, tuple):
            move, N = result
            assert isinstance(move, Point)
        else:
            assert isinstance(result, (str, Point))

    def test_returns_policy_array(self, tmp_path):
        p = self._player(tmp_path)
        g = Game()
        result = p.pick_move(g)
        if isinstance(result, tuple):
            move, N = result
            assert N.shape == (NUM_MOVES,)

    def test_move_is_legal(self, tmp_path):
        p = self._player(tmp_path)
        g = Game()
        result = p.pick_move(g)
        if isinstance(result, tuple):
            move, _ = result
            assert move in g.legal_plays()

    def test_sets_report(self, tmp_path):
        p = self._player(tmp_path)
        g = Game()
        p.pick_move(g)
        assert hasattr(p, 'report')

    def test_temperature_zero(self, tmp_path):
        p = self._player(tmp_path, temperature=0.0)
        g = Game()
        result = p.pick_move(g)
        assert result is not None

    def test_temperature_half(self, tmp_path):
        p = self._player(tmp_path, temperature=0.5)
        g = Game()
        result = p.pick_move(g)
        assert result is not None

    def test_temperature_one(self, tmp_path):
        p = self._player(tmp_path, temperature=1.0)
        g = Game()
        result = p.pick_move(g)
        assert result is not None

    def test_mid_game(self, tmp_path):
        """pick_move works after several moves have been played."""
        p = self._player(tmp_path)
        g = Game()
        g.play(Point(5, 5))
        g.play(Point(10, 10))
        g.play(Point(7, 3))
        result = p.pick_move(g)
        assert result is not None


# ---------------------------------------------------------------------------
# swap model integration
# ---------------------------------------------------------------------------

class TestSwapModelIntegration:
    def test_use_swap_first_move_returns_point(self, tmp_path):
        path = make_model(tmp_path)
        p = Player(model=path, trials=2, use_swap=1)
        g = Game()
        result = p.pick_move(g)
        # swapmodel.choose_first_move() returns a Point
        assert isinstance(result, Point)
        assert p.report == "swapmodel"

    def test_use_swap_second_move_strong_centre(self, tmp_path):
        """After a very central first move, Player with use_swap may return 'swap'."""
        path = make_model(tmp_path)
        p = Player(model=path, trials=2, use_swap=1)
        g = Game()
        # Play a central first move that should trigger swap
        import swapmodel
        from twixt import Point as P
        # Find a point that want_swap returns True for
        centre = P(twixt.Game.SIZE // 2 - 1, twixt.Game.SIZE // 2 - 2)
        if swapmodel.want_swap(centre):
            g.play(centre)
            result = p.pick_move(g)
            assert result == "swap"
            assert p.report == "swapmodel"


# ---------------------------------------------------------------------------
# nnfunc output shape sanity (unit test of the closure)
# ---------------------------------------------------------------------------

class TestNNFunc:
    def test_nnfunc_returns_scalar_and_policy(self, tmp_path):
        """The nnfunc closure must return (scalar, 528-array) for NeuralMCTS."""
        path = make_model(tmp_path)
        import nneval
        ne = nneval.NNEvaluater(torch.load(path, weights_only=False))

        g = Game()
        nips = naf.NetInputs(g)
        pws, mls = ne.eval_one(nips)
        pw = naf.three_to_one(pws[0])   # (1,3) → (3,) → scalar
        ml = mls[0]                      # (1,528) → (528,)

        assert isinstance(pw, float)
        assert ml.shape == (NUM_MOVES,)
        rotated = naf.rotate_policy_array(ml, 0)
        assert rotated.shape == (NUM_MOVES,)
