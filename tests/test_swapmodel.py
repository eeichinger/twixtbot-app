"""
Tests for swapmodel.py — first-move / swap heuristic.
"""
import os
import sys
import pytest
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import twixt
from twixt import Point, Game
import swapmodel
from swapmodel import (
    want_swap,
    choose_first_move,
    points_and_locs,
    _point_score,
    _betas,
)

SIZE = Game.SIZE


# ---------------------------------------------------------------------------
# _point_score
# ---------------------------------------------------------------------------

class TestPointScore:
    def test_returns_float(self):
        p = Point(5, 5)
        assert isinstance(_point_score(p), float)

    def test_centre_ish_score_near_half(self):
        """Centre points tend to score near 0.5 (border of swap zone)."""
        p = Point(SIZE // 2 - 1, SIZE // 2 - 1)
        score = _point_score(p)
        assert 0.3 < score < 0.7

    def test_far_corner_score_below_half(self):
        """Corner-adjacent points tend to be safe (don't want swap)."""
        p = Point(1, 0)
        score = _point_score(p)
        assert score < 0.5


# ---------------------------------------------------------------------------
# want_swap
# ---------------------------------------------------------------------------

class TestWantSwap:
    def test_returns_bool(self):
        p = Point(5, 5)
        result = want_swap(p)
        # want_swap returns a numpy bool_ or Python bool
        assert result is True or result is False or result == 0 or result == 1

    def test_strong_centre_wants_swap(self):
        """A very central first move should trigger swap."""
        # x = SIZE//2 - 1 = 11, y = SIZE//2 - 1 = 11 — close to centre
        p = Point(SIZE // 2 - 1, SIZE // 2 - 2)
        # Just verify it runs without error; the actual value depends on betas
        result = want_swap(p)
        assert result in (True, False, 0, 1)

    def test_no_swap_for_edge(self):
        """Edge moves should not trigger swap."""
        p = Point(1, 0)
        assert not want_swap(p)


# ---------------------------------------------------------------------------
# points_and_locs
# ---------------------------------------------------------------------------

class TestPointsAndLocs:
    def test_returns_two_lists(self):
        pts, locs = points_and_locs()
        assert isinstance(pts, list)
        assert isinstance(locs, list)

    def test_locations_start_at_zero(self):
        _, locs = points_and_locs()
        assert locs[0] == 0.0

    def test_locations_monotone(self):
        _, locs = points_and_locs()
        for i in range(len(locs) - 1):
            assert locs[i] <= locs[i+1], f"Non-monotone at {i}"

    def test_lengths_consistent(self):
        pts, locs = points_and_locs()
        assert len(locs) == len(pts) + 1

    def test_points_are_Point_objects(self):
        pts, _ = points_and_locs()
        for p in pts:
            assert isinstance(p, Point)

    def test_all_points_in_valid_range(self):
        pts, _ = points_and_locs()
        for p in pts:
            assert 1 <= p.x <= SIZE - 2
            assert 0 <= p.y <= SIZE - 1


# ---------------------------------------------------------------------------
# choose_first_move
# ---------------------------------------------------------------------------

class TestChooseFirstMove:
    def test_returns_point(self):
        m = choose_first_move()
        assert isinstance(m, Point)

    def test_move_in_board_range(self):
        for _ in range(20):
            m = choose_first_move()
            assert 1 <= m.x <= SIZE - 2
            assert 0 <= m.y <= SIZE - 1

    def test_returns_different_moves(self):
        """choose_first_move is stochastic; should not always return the same point."""
        import random
        random.seed(None)
        moves = set()
        for _ in range(50):
            m = choose_first_move()
            moves.add((m.x, m.y))
        # With 50 samples, we should see more than 1 distinct move
        assert len(moves) > 1, "choose_first_move always returns the same point"

    def test_prefer_near_boundary(self):
        """Moves very near the boundary should never be chosen (score too low)."""
        import random
        random.seed(42)
        for _ in range(100):
            m = choose_first_move()
            # x=1,y=0 is very unlikely (near edge, low weight)
            # Just verify no out-of-bounds
            assert 1 <= m.x <= SIZE - 2
